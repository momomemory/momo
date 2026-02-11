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
use std::process::Stdio;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "momo")]
#[command(about = "Open-source, self-hostable AI memory system")]
struct Args {
    /// Force rebuild embeddings when dimension mismatch detected
    #[arg(long)]
    rebuild_embeddings: bool,

    /// Runtime mode: all, api, or worker
    #[arg(long)]
    mode: Option<String>,

    /// Run API and workers in one process when mode=all
    #[arg(long)]
    single_process: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeMode {
    All,
    Api,
    Worker,
}

impl RuntimeMode {
    fn parse(raw: Option<&str>) -> Self {
        let value = raw
            .map(std::string::ToString::to_string)
            .or_else(|| std::env::var("MOMO_RUNTIME_MODE").ok())
            .map(|v| v.trim().to_lowercase());

        match value.as_deref() {
            Some("api") => Self::Api,
            Some("worker") => Self::Worker,
            Some("all") | None => Self::All,
            Some(other) => {
                tracing::warn!(
                    value = %other,
                    "Invalid MOMO_RUNTIME_MODE/--mode; falling back to 'all'"
                );
                Self::All
            }
        }
    }

    fn runs_api(self) -> bool {
        matches!(self, Self::All | Self::Api)
    }

    fn runs_worker(self) -> bool {
        matches!(self, Self::All | Self::Worker)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Api => "api",
            Self::Worker => "worker",
        }
    }
}

#[derive(Debug, Clone)]
struct ReadReplicaSettings {
    database: crate::config::DatabaseConfig,
    sync_interval_secs: u64,
}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(
                    variable = name,
                    value = %raw,
                    error = %error,
                    "Invalid numeric env value; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => {
                tracing::warn!(
                    variable = name,
                    value = %raw,
                    "Invalid boolean env value; using default"
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn should_supervise_subprocesses(runtime_mode: RuntimeMode, single_process: bool) -> bool {
    matches!(runtime_mode, RuntimeMode::All) && !single_process
}

fn read_replica_settings(
    write_config: &crate::config::DatabaseConfig,
) -> Option<ReadReplicaSettings> {
    let read_url = std::env::var("DATABASE_READ_URL").ok();
    let read_auth_token = std::env::var("DATABASE_READ_AUTH_TOKEN").ok();
    let read_local_path = std::env::var("DATABASE_READ_LOCAL_PATH").ok();
    let sync_interval_secs = parse_env_u64("DATABASE_READ_SYNC_INTERVAL_SECS", 2).max(1);

    build_read_replica_settings(
        write_config,
        read_url,
        read_auth_token,
        read_local_path,
        sync_interval_secs,
    )
}

fn build_read_replica_settings(
    write_config: &crate::config::DatabaseConfig,
    read_url: Option<String>,
    read_auth_token: Option<String>,
    read_local_path: Option<String>,
    sync_interval_secs: u64,
) -> Option<ReadReplicaSettings> {
    if read_url.is_none() && read_auth_token.is_none() && read_local_path.is_none() {
        return None;
    }

    Some(ReadReplicaSettings {
        database: crate::config::DatabaseConfig {
            url: read_url.unwrap_or_else(|| write_config.url.clone()),
            auth_token: read_auth_token.or_else(|| write_config.auth_token.clone()),
            local_path: read_local_path.or_else(|| write_config.local_path.clone()),
        },
        sync_interval_secs,
    })
}

fn build_child_command(
    executable: &std::path::Path,
    mode: RuntimeMode,
    args: &Args,
) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(executable);
    command
        .arg("--mode")
        .arg(mode.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

    if args.rebuild_embeddings {
        command.arg("--rebuild-embeddings");
    }

    command
}

async fn terminate_child(name: &str, child: &mut tokio::process::Child) {
    match child.try_wait() {
        Ok(Some(status)) => {
            tracing::info!(process = name, %status, "Subprocess already exited");
            return;
        }
        Ok(None) => {}
        Err(error) => {
            tracing::warn!(process = name, error = %error, "Failed to inspect subprocess state");
        }
    }

    match child.kill().await {
        Ok(()) => tracing::info!(process = name, "Subprocess terminated"),
        Err(error) => {
            tracing::warn!(process = name, error = %error, "Failed to terminate subprocess")
        }
    }
}

async fn wait_for_shutdown_signal() {
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
}

async fn run_subprocess_supervisor(args: &Args) -> anyhow::Result<()> {
    let executable = std::env::current_exe()?;
    tracing::info!(path = %executable.display(), "Starting all-mode subprocess supervisor");

    let mut api_child = build_child_command(&executable, RuntimeMode::Api, args).spawn()?;
    let mut worker_child = build_child_command(&executable, RuntimeMode::Worker, args).spawn()?;

    tracing::info!(pid = api_child.id(), "Spawned API subprocess");
    tracing::info!(pid = worker_child.id(), "Spawned worker subprocess");

    tokio::select! {
        _ = wait_for_shutdown_signal() => {
            tracing::info!("Shutdown signal received, terminating subprocesses...");
            terminate_child("api", &mut api_child).await;
            terminate_child("worker", &mut worker_child).await;
            Ok(())
        }
        status = api_child.wait() => {
            let status = status?;
            tracing::error!(%status, "API subprocess exited unexpectedly");
            terminate_child("worker", &mut worker_child).await;
            Err(anyhow::anyhow!("API subprocess exited unexpectedly: {status}"))
        }
        status = worker_child.wait() => {
            let status = status?;
            tracing::error!(%status, "Worker subprocess exited unexpectedly");
            terminate_child("api", &mut api_child).await;
            Err(anyhow::anyhow!("Worker subprocess exited unexpectedly: {status}"))
        }
    }
}

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

    let runtime_mode = RuntimeMode::parse(args.mode.as_deref());
    let single_process = args.single_process || parse_env_bool("MOMO_SINGLE_PROCESS", false);

    tracing::info!(
        mode = runtime_mode.as_str(),
        single_process,
        "Runtime mode selected"
    );

    if should_supervise_subprocesses(runtime_mode, single_process) {
        return run_subprocess_supervisor(&args).await;
    }

    if matches!(runtime_mode, RuntimeMode::All) && single_process {
        tracing::info!("Single-process all-mode enabled");
    }

    let config = Config::from_env();

    if config.server.api_keys.is_empty() {
        tracing::warn!(
            "MOMO_API_KEYS is not set — admin endpoints are locked. Set MOMO_API_KEYS to enable /admin/* routes."
        );
    }

    tracing::info!("Initializing write database...");
    let write_raw_db = Database::new(&config.database).await?;
    let write_db_backend = LibSqlBackend::new(write_raw_db);
    let write_db: Arc<dyn DatabaseBackend> = Arc::new(write_db_backend);

    let (read_db, read_sync_interval_secs) =
        if let Some(replica) = read_replica_settings(&config.database) {
            tracing::info!(
                url = %replica.database.url,
                local_path = ?replica.database.local_path,
                "Initializing dedicated read database"
            );
            let read_raw_db = Database::new(&replica.database).await?;
            let read_backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(read_raw_db));
            (read_backend, Some(replica.sync_interval_secs))
        } else {
            tracing::info!("Using primary database for reads and writes");
            (write_db.clone(), None)
        };

    tracing::info!("Loading embedding model: {}...", config.embeddings.model);
    let embeddings = EmbeddingProvider::new(&config.embeddings)?;

    // Pass &*write_db to dereference Arc<dyn DatabaseBackend> into &dyn DatabaseBackend
    match migration::check_dimension_compatibility(&*write_db, &embeddings, args.rebuild_embeddings)
        .await?
    {
        migration::MigrationDecision::NotNeeded => {}
        migration::MigrationDecision::Approved => {
            migration::trigger_reembedding(&*write_db, embeddings.dimensions()).await?;
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
        write_db,
        read_db,
        embeddings,
        reranker,
        ocr,
        transcription,
        llm,
    );

    let cancel_token = CancellationToken::new();
    if runtime_mode.runs_worker() {
        let processing_interval_secs = parse_env_u64("PROCESSING_POLL_INTERVAL_SECS", 10).max(1);
        tracing::info!(
            interval_secs = processing_interval_secs,
            "Starting background processing"
        );
        let pipeline = state.pipeline.clone();
        let token = cancel_token.child_token();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => {
                        tracing::info!("Background processing shutting down...");
                        break;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(processing_interval_secs)) => {
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
    } else {
        tracing::info!("Worker tasks disabled in API-only mode");
    }

    if runtime_mode.runs_api() {
        if let Some(interval_secs) = read_sync_interval_secs {
            tracing::info!(interval_secs, "Starting read-replica sync loop");
            let read_db = state.read_db.clone();
            let token = cancel_token.child_token();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = token.cancelled() => {
                            tracing::info!("Read-replica sync loop shutting down...");
                            break;
                        }
                        _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {
                            if let Err(e) = read_db.sync().await {
                                tracing::warn!(error = %e, "Read-replica sync failed");
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

        return Ok(());
    }

    tracing::info!("Worker mode active; HTTP server disabled");
    shutdown_signal(cancel_token).await;
    Ok(())
}

async fn shutdown_signal(cancel_token: CancellationToken) {
    wait_for_shutdown_signal().await;
    tracing::info!("Shutdown signal received, cancelling background tasks...");
    cancel_token.cancel();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_mode_parse_values() {
        assert_eq!(RuntimeMode::parse(Some("all")), RuntimeMode::All);
        assert_eq!(RuntimeMode::parse(Some("api")), RuntimeMode::Api);
        assert_eq!(RuntimeMode::parse(Some("worker")), RuntimeMode::Worker);
        assert_eq!(RuntimeMode::parse(Some("unknown")), RuntimeMode::All);
    }

    #[test]
    fn should_supervise_only_when_all_and_not_single_process() {
        assert!(should_supervise_subprocesses(RuntimeMode::All, false));
        assert!(!should_supervise_subprocesses(RuntimeMode::All, true));
        assert!(!should_supervise_subprocesses(RuntimeMode::Api, false));
        assert!(!should_supervise_subprocesses(RuntimeMode::Worker, false));
    }

    #[test]
    fn parse_env_bool_handles_supported_values() {
        assert!(parse_env_bool_from_raw("true", false));
        assert!(parse_env_bool_from_raw("1", false));
        assert!(parse_env_bool_from_raw("yes", false));
        assert!(!parse_env_bool_from_raw("false", true));
        assert!(!parse_env_bool_from_raw("0", true));
        assert!(!parse_env_bool_from_raw("no", true));
        assert!(parse_env_bool_from_raw("invalid", true));
        assert!(!parse_env_bool_from_raw("invalid", false));
    }

    #[test]
    fn build_read_replica_settings_none_when_no_overrides() {
        let write_cfg = crate::config::DatabaseConfig {
            url: "file:momo.db".to_string(),
            auth_token: None,
            local_path: None,
        };

        let settings = build_read_replica_settings(&write_cfg, None, None, None, 2);
        assert!(settings.is_none());
    }

    #[test]
    fn build_read_replica_settings_uses_write_defaults() {
        let write_cfg = crate::config::DatabaseConfig {
            url: "libsql://primary.turso.io".to_string(),
            auth_token: Some("primary-token".to_string()),
            local_path: Some("primary-local.db".to_string()),
        };

        let settings = build_read_replica_settings(
            &write_cfg,
            Some("libsql://read.turso.io".to_string()),
            None,
            None,
            5,
        )
        .expect("read replica should be configured");

        assert_eq!(settings.database.url, "libsql://read.turso.io");
        assert_eq!(
            settings.database.auth_token,
            Some("primary-token".to_string())
        );
        assert_eq!(
            settings.database.local_path,
            Some("primary-local.db".to_string())
        );
        assert_eq!(settings.sync_interval_secs, 5);
    }

    fn parse_env_bool_from_raw(raw: &str, default: bool) -> bool {
        match raw.trim().to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        }
    }
}
