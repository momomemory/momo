mod api;
mod config;
mod db;
mod embeddings;
mod error;
mod intelligence;
mod llm;
mod mcp;
mod migration;
mod models;
mod ocr;
mod processing;
mod search;
mod services;
mod transcription;

use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "momo")]
#[command(about = "Open-source, self-hostable AI memory system")]
struct Args {
    /// Force rebuild embeddings when dimension mismatch detected
    #[arg(long)]
    rebuild_embeddings: bool,
}

use std::sync::Arc;

use crate::api::{create_router, AppState};
use crate::config::Config;
use crate::db::{Database, DatabaseBackend, LibSqlBackend};
use crate::embeddings::{EmbeddingProvider, RerankerProvider};
use crate::intelligence::InferenceEngine;
use crate::llm::LlmProvider;
use crate::ocr::OcrProvider;
use crate::transcription::TranscriptionProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    dotenvy::dotenv().ok();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "momo=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env();

    if config.server.api_keys.is_empty() {
        tracing::warn!(
            "MOMO_API_KEYS is not set — admin endpoints are locked. Set MOMO_API_KEYS to enable /admin/* routes."
        );
    }

    tracing::info!("Initializing database...");
    let raw_db = Database::new(&config.database).await?;
    let db_backend = LibSqlBackend::new(raw_db);
    // Wrap in Arc<dyn DatabaseBackend> immediately so we can clone it
    let db: Arc<dyn DatabaseBackend> = Arc::new(db_backend);

    tracing::info!("Loading embedding model: {}...", config.embeddings.model);
    let embeddings = EmbeddingProvider::new(&config.embeddings)?;

    // Pass &*db to dereference Arc<dyn DatabaseBackend> into &dyn DatabaseBackend
    match migration::check_dimension_compatibility(&*db, &embeddings, args.rebuild_embeddings)
        .await?
    {
        migration::MigrationDecision::NotNeeded => {}
        migration::MigrationDecision::Approved => {
            migration::trigger_reembedding(&*db, embeddings.dimensions()).await?;
            tracing::info!("Migration started. Documents will be re-embedded in background.");
        }
        migration::MigrationDecision::Rejected => {
            tracing::error!("Migration rejected. Cannot start with dimension mismatch.");
            return Err(anyhow::anyhow!(
                "Embedding dimension mismatch - use --rebuild-embeddings flag to force migration"
            ));
        }
    }

    tracing::info!("Initializing OCR provider: {}...", config.ocr.model);
    let ocr = OcrProvider::new(&config.ocr)?;
    if !ocr.is_available() {
        tracing::warn!("OCR unavailable - image processing will be skipped");
    }

    tracing::info!(
        "Initializing transcription provider: {}...",
        config.transcription.model
    );
    let transcription = TranscriptionProvider::new(&config.transcription)?;
    if !transcription.is_available() {
        tracing::warn!("Transcription unavailable - audio processing will be skipped");
    }

    if let Some(llm_config) = &config.llm {
        tracing::info!("Initializing LLM provider: {}...", llm_config.model);
    }
    let llm = LlmProvider::new(config.llm.as_ref());
    if !llm.is_available() {
        tracing::warn!("LLM unavailable - LLM features will be disabled");
    }

    let reranker = if let Some(reranker_config) = &config.reranker {
        if reranker_config.enabled {
            tracing::info!("Initializing reranker: {}...", reranker_config.model);
            match RerankerProvider::new_async(reranker_config).await {
                Ok(provider) => {
                    tracing::info!("Reranker initialized successfully");
                    Some(provider)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize reranker: {} - continuing without reranking",
                        e
                    );
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    let state = AppState::new(
        config.clone(),
        db,
        embeddings,
        reranker,
        ocr,
        transcription,
        llm,
    );

    let cancel_token = CancellationToken::new();

    tracing::info!("Starting background processing...");
    let pipeline = state.pipeline.clone();
    let token = cancel_token.child_token();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    tracing::info!("Background processing shutting down...");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                    if let Err(e) = pipeline.process_pending().await {
                        tracing::error!("Background processing error: {}", e);
                    }
                }
            }
        }
    });

    tracing::info!("Starting forgetting manager...");
    let manager = services::ForgettingManager::new(
        state.db.clone(),
        state.config.memory.forgetting_check_interval_secs,
    );
    let token = cancel_token.child_token();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    tracing::info!("Forgetting manager shutting down...");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(manager.interval_secs())) => {
                    if let Err(e) = manager.run_once().await {
                        tracing::error!("Forgetting manager error: {}", e);
                    }
                }
            }
        }
    });

    tracing::info!(
        "Starting episode decay manager... (threshold={}, grace_days={})",
        state.config.memory.episode_decay_threshold,
        state.config.memory.episode_forget_grace_days
    );
    let decay_manager = services::EpisodeDecayManager::new(
        state.db.clone(),
        state.config.memory.episode_decay_threshold,
        state.config.memory.episode_forget_grace_days,
        state.config.memory.episode_decay_days,
        state.config.memory.episode_decay_factor,
    );
    let token = cancel_token.child_token();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = token.cancelled() => {
                    tracing::info!("Episode decay manager shutting down...");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(decay_manager.interval_secs())) => {
                    if let Err(e) = decay_manager.run_once().await {
                        tracing::error!("Episode decay manager error: {}", e);
                    }
                }
            }
        }
    });

    // Inference engine (opt-in)
    if state.config.memory.inference.enabled {
        tracing::info!(
            "Starting inference engine... (interval={}s)",
            state.config.memory.inference.interval_secs
        );
        let engine = InferenceEngine::new(
            state.db.clone(),
            state.llm.clone(),
            state.embeddings.clone(),
            state.config.memory.inference.clone(),
        );

        let token = cancel_token.child_token();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => {
                        tracing::info!("Inference engine shutting down...");
                        break;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(engine.interval_secs())) => {
                        if let Err(e) = engine.run_once().await {
                            tracing::error!("Inference engine error: {}", e);
                        }
                    }
                }
            }
        });
    }

    if state.llm.is_available() {
        tracing::info!(
            "Starting profile refresh manager... (interval={}s)",
            state.config.memory.profile_refresh_interval_secs
        );
        let profile_refresh = services::ProfileRefreshManager::new(
            state.db.clone(),
            state.llm.clone(),
            state.config.memory.profile_refresh_interval_secs,
        );
        let token = cancel_token.child_token();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => {
                        tracing::info!("Profile refresh manager shutting down...");
                        break;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(profile_refresh.interval_secs())) => {
                        if let Err(e) = profile_refresh.run_once().await {
                            tracing::error!("Profile refresh error: {}", e);
                        }
                    }
                }
            }
        });
    }

    let app = create_router(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Momo starting on http://{}", addr);
    tracing::info!("  Health check: http://{}/api/v1/health", addr);
    tracing::info!("  API docs:     http://{}/api/v1/docs", addr);
    tracing::info!("  OpenAPI spec: http://{}/api/v1/openapi.json", addr);
    if config.mcp.enabled {
        tracing::info!("  MCP endpoint: http://{}{}", addr, config.mcp.path);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(cancel_token))
        .await?;

    Ok(())
}

async fn shutdown_signal(cancel_token: CancellationToken) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, cancelling background tasks...");
    cancel_token.cancel();
}
