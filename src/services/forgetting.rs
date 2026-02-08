use std::sync::Arc;

use crate::db::DatabaseBackend;
use crate::error::Result;
use chrono::Utc;
use tracing::{debug, error, info};

/// Manager responsible for automatic forgetting of expired memories
#[derive(Clone)]
pub struct ForgettingManager {
    db: Arc<dyn DatabaseBackend>,
    interval_secs: u64,
}

impl ForgettingManager {
    /// Create a new ForgettingManager
    pub fn new(db: Arc<dyn DatabaseBackend>, interval_secs: u64) -> Self {
        Self { db, interval_secs }
    }

    /// Run a single pass of the forgetting process
    ///
    /// Gets all expired memories and marks them as forgotten.
    /// Continues processing even if individual forgets fail.
    /// Returns the number of memories successfully forgotten.
    pub async fn run_once(&self) -> Result<u64> {
        info!("Starting forgetting process");

        let now = Utc::now();

        // Get candidates
        let candidates = self.db.get_forgetting_candidates(now).await?;
        let count = candidates.len();

        if count == 0 {
            info!("No expired memories to forget");
            return Ok(0);
        }

        debug!("Found {} expired memories to forget", count);

        let mut forgotten_count = 0u64;
        let mut error_count = 0;

        for memory in candidates {
            debug!("Forgetting memory: id={}", memory.id);

            match self
                .db
                .forget_memory(&memory.id, Some("auto-forgotten: expired"))
                .await
            {
                Ok(_) => {
                    forgotten_count += 1;
                }
                Err(e) => {
                    error!("Failed to forget memory {}: {}", memory.id, e);
                    error_count += 1;
                }
            }
        }

        info!(
            "Forgetting process complete: {} forgotten, {} errors out of {} candidates",
            forgotten_count, error_count, count
        );

        Ok(forgotten_count)
    }

    /// Get the configured interval in seconds
    pub fn interval_secs(&self) -> u64 {
        self.interval_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Database, LibSqlBackend};
    use chrono::Duration;
    use libsql::Connection;
    use tempfile::NamedTempFile;

    async fn setup_test_db() -> (Connection, Arc<dyn DatabaseBackend>, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let inner_db = libsql::Builder::new_local(path).build().await.unwrap();

        let conn = inner_db.connect().unwrap();

        conn.execute(
            r#"
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                space_id TEXT NOT NULL,
                container_tag TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                is_latest INTEGER NOT NULL DEFAULT 1,
                parent_memory_id TEXT,
                root_memory_id TEXT,
                memory_relations TEXT NOT NULL DEFAULT '{}',
                source_count INTEGER NOT NULL DEFAULT 0,
                is_inference INTEGER NOT NULL DEFAULT 0,
                is_forgotten INTEGER NOT NULL DEFAULT 0,
                is_static INTEGER NOT NULL DEFAULT 0,
                forget_after TEXT,
                forget_reason TEXT,
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                last_accessed TEXT,
                confidence REAL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embedding F32_BLOB(384)
            )
            "#,
            (),
        )
        .await
        .unwrap();

        let db = Database {
            db: Arc::new(inner_db),
        };
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));

        (conn, backend, temp_file)
    }

    async fn insert_memory(
        conn: &Connection,
        id: &str,
        memory: &str,
        forget_after: Option<&str>,
        is_forgotten: bool,
    ) {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO memories (
                id, memory, space_id, container_tag, version, is_latest,
                memory_relations, source_count, is_inference, is_forgotten,
                is_static, forget_after, memory_type, metadata,
                created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, 1, 1, '{}', 0, 0, ?5, 0, ?6, 'episodic', '{}', ?7, ?7)
            "#,
            (
                id,
                memory,
                "test-space",
                "test-tag",
                is_forgotten as i32,
                forget_after,
                now.as_str(),
            ),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_run_once_forgets_expired_memories() {
        // Given a database with an expired memory
        let (conn, db, _temp) = setup_test_db().await;
        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        insert_memory(&conn, "mem1", "expired memory", Some(&past), false).await;

        let manager = ForgettingManager::new(db.clone(), 3600);

        // When run_once is called
        let result = manager.run_once().await;

        // Then it should succeed
        assert!(result.is_ok());

        // And the memory should be marked as forgotten
        let row: i32 = conn
            .query("SELECT is_forgotten FROM memories WHERE id = 'mem1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(row, 1);
    }

    #[tokio::test]
    async fn test_run_once_skips_future_memories() {
        // Given a database with a future-expiring memory
        let (conn, db, _temp) = setup_test_db().await;
        let future = (Utc::now() + Duration::hours(1)).to_rfc3339();
        insert_memory(&conn, "mem1", "future memory", Some(&future), false).await;

        let manager = ForgettingManager::new(db.clone(), 3600);

        // When run_once is called
        let result = manager.run_once().await;

        // Then it should succeed
        assert!(result.is_ok());

        // And the memory should NOT be forgotten
        let row: i32 = conn
            .query("SELECT is_forgotten FROM memories WHERE id = 'mem1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(row, 0);
    }

    #[tokio::test]
    async fn test_run_once_skips_already_forgotten() {
        // Given a database with an already-forgotten memory
        let (conn, db, _temp) = setup_test_db().await;
        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        insert_memory(&conn, "mem1", "already forgotten", Some(&past), true).await;

        let manager = ForgettingManager::new(db.clone(), 3600);

        // When run_once is called
        let result = manager.run_once().await;

        // Then it should succeed (no candidates)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_once_handles_no_expired_memories() {
        // Given an empty database
        let (_conn, db, _temp) = setup_test_db().await;
        let manager = ForgettingManager::new(db, 3600);

        // When run_once is called
        let result = manager.run_once().await;

        // Then it should succeed
        if let Err(e) = &result {
            eprintln!("Error: {e:?}");
        }
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_once_forgets_multiple_memories() {
        // Given a database with multiple expired memories
        let (conn, db, _temp) = setup_test_db().await;
        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        insert_memory(&conn, "mem1", "expired 1", Some(&past), false).await;
        insert_memory(&conn, "mem2", "expired 2", Some(&past), false).await;
        insert_memory(&conn, "mem3", "expired 3", Some(&past), false).await;

        let manager = ForgettingManager::new(db.clone(), 3600);

        // When run_once is called
        let result = manager.run_once().await;

        // Then it should succeed
        assert!(result.is_ok());

        // And all memories should be forgotten
        let count: i32 = conn
            .query("SELECT COUNT(*) FROM memories WHERE is_forgotten = 1", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_run_once_sets_forget_reason() {
        // Given a database with an expired memory
        let (conn, db, _temp) = setup_test_db().await;
        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        insert_memory(&conn, "mem1", "expired memory", Some(&past), false).await;

        let manager = ForgettingManager::new(db.clone(), 3600);

        // When run_once is called
        manager.run_once().await.unwrap();

        // Then the forget_reason should be set
        let reason: String = conn
            .query("SELECT forget_reason FROM memories WHERE id = 'mem1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(reason, "auto-forgotten: expired");
    }

    #[tokio::test]
    async fn test_manager_clone() {
        // Given a ForgettingManager
        let (_conn, db, _temp) = setup_test_db().await;
        let manager = ForgettingManager::new(db, 7200);

        // When cloned
        let cloned = manager.clone();

        // Then the clone should have the same interval
        assert_eq!(cloned.interval_secs(), 7200);
    }

    #[tokio::test]
    async fn test_interval_secs() {
        // Given a ForgettingManager with interval
        let (_conn, db, _temp) = setup_test_db().await;
        let manager = ForgettingManager::new(db, 3600);

        // When interval_secs is called
        let interval = manager.interval_secs();

        // Then it should return the configured interval
        assert_eq!(interval, 3600);
    }
}
