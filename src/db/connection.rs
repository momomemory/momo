use libsql::{Builder, Connection};
use std::sync::Arc;

use crate::config::DatabaseConfig;
use crate::error::Result;

use super::schema;

pub struct Database {
    pub(crate) db: Arc<libsql::Database>,
}

impl Database {
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
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

        let database = Self { db: Arc::new(db) };
        database.init_schema().await?;

        Ok(database)
    }

    pub fn connect(&self) -> Result<Connection> {
        Ok(self.db.connect()?)
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
        }
    }
}
