use libsql::{Builder, Connection};
use std::sync::Arc;

use crate::config::DatabaseConfig;
use crate::error::Result;

use super::schema;

pub struct Database {
    pub(crate) db: Arc<libsql::Database>,
    pub(crate) busy_timeout_ms: u64,
    pub(crate) journal_mode: String,
    pub(crate) synchronous: String,
}

impl Database {
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        let busy_timeout_ms = std::env::var("DATABASE_BUSY_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5000);
        let journal_mode = normalize_journal_mode(
            &std::env::var("DATABASE_JOURNAL_MODE").unwrap_or_else(|_| "WAL".to_string()),
        )
        .to_string();
        let synchronous = normalize_synchronous(
            &std::env::var("DATABASE_SYNCHRONOUS").unwrap_or_else(|_| "NORMAL".to_string()),
        )
        .to_string();

        let db = if config.url.starts_with("libsql://") || config.url.starts_with("https://") {
            if let Some(ref local_path) = config.local_path {
                Builder::new_remote_replica(
                    local_path,
                    config.url.clone(),
                    config.auth_token.clone().unwrap_or_default(),
                )
                .build()
                .await?
            } else {
                Builder::new_remote(
                    config.url.clone(),
                    config.auth_token.clone().unwrap_or_default(),
                )
                .build()
                .await?
            }
        } else if config.url == ":memory:" {
            Builder::new_local(":memory:").build().await?
        } else {
            let path = config.url.strip_prefix("file:").unwrap_or(&config.url);
            Builder::new_local(path).build().await?
        };

        let database = Self {
            db: Arc::new(db),
            busy_timeout_ms,
            journal_mode,
            synchronous,
        };
        database.configure_database().await?;
        database.init_schema().await?;

        Ok(database)
    }

    pub fn connect(&self) -> Result<Connection> {
        Ok(self.db.connect()?)
    }

    async fn configure_database(&self) -> Result<()> {
        let conn = self.connect()?;

        let busy_timeout_sql = format!("PRAGMA busy_timeout = {}", self.busy_timeout_ms);
        if let Err(error) = conn.execute_batch(&busy_timeout_sql).await {
            tracing::warn!(
                busy_timeout_ms = self.busy_timeout_ms,
                error = %error,
                "Failed to set SQLite busy_timeout"
            );
        }

        let journal_sql = format!("PRAGMA journal_mode = {}", self.journal_mode);
        if let Err(error) = conn.execute_batch(&journal_sql).await {
            tracing::warn!(
                mode = %self.journal_mode,
                error = %error,
                "Failed to set SQLite journal_mode"
            );
        }

        let synchronous_sql = format!("PRAGMA synchronous = {}", self.synchronous);
        if let Err(error) = conn.execute_batch(&synchronous_sql).await {
            tracing::warn!(
                mode = %self.synchronous,
                error = %error,
                "Failed to set SQLite synchronous pragma"
            );
        }

        Ok(())
    }

    async fn init_schema(&self) -> Result<()> {
        let conn = self.connect()?;
        schema::init_schema(&conn).await?;
        Ok(())
    }

    pub async fn sync(&self) -> Result<()> {
        if let Ok(sync) = self.db.sync().await {
            tracing::info!("Database synced: {:?}", sync);
        }
        Ok(())
    }
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            busy_timeout_ms: self.busy_timeout_ms,
            journal_mode: self.journal_mode.clone(),
            synchronous: self.synchronous.clone(),
        }
    }
}

fn normalize_journal_mode(value: &str) -> &'static str {
    match value.trim().to_uppercase().as_str() {
        "DELETE" => "DELETE",
        "TRUNCATE" => "TRUNCATE",
        "PERSIST" => "PERSIST",
        "MEMORY" => "MEMORY",
        "WAL" => "WAL",
        "OFF" => "OFF",
        _ => "WAL",
    }
}

fn normalize_synchronous(value: &str) -> &'static str {
    match value.trim().to_uppercase().as_str() {
        "OFF" => "OFF",
        "NORMAL" => "NORMAL",
        "FULL" => "FULL",
        "EXTRA" => "EXTRA",
        _ => "NORMAL",
    }
}
