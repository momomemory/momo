use std::sync::Arc;

use crate::db::DatabaseBackend;
use crate::error::Result;
use crate::models::Memory;
use chrono::{Duration, Utc};
use tracing::{debug, error, info};

/// Manager responsible for scheduling low-relevance Episode memories for forgetting
#[derive(Clone)]
pub struct EpisodeDecayManager {
    db: Arc<dyn DatabaseBackend>,
    threshold: f64,
    grace_days: u32,
    decay_days: f64,
    decay_factor: f64,
}

impl EpisodeDecayManager {
    /// Create a new EpisodeDecayManager
    pub fn new(
        db: Arc<dyn DatabaseBackend>,
        threshold: f64,
        grace_days: u32,
        decay_days: f64,
        decay_factor: f64,
    ) -> Self {
        Self {
            db,
            threshold,
            grace_days,
            decay_days,
            decay_factor,
        }
    }

    /// Run a single pass of the decay process. Finds episode memories with relevance below
    /// threshold and schedules them for forgetting by setting forget_after to now + grace_days.
    pub async fn run_once(&self) -> Result<u64> {
        info!(
            threshold = self.threshold,
            "Starting episode decay run_once"
        );

        let candidates = self.db.get_episode_decay_candidates().await?;

        let mut scheduled = 0u64;

        for candidate in candidates {
            // Build a minimal Memory to use existing relevance calculation
            let mut m = Memory::new(candidate.id.clone(), candidate.memory, candidate.space_id);

            // parse created_at / last_accessed safely
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&candidate.created_at) {
                m.created_at = dt.with_timezone(&Utc);
            }

            if let Some(s) = candidate.last_accessed {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
                    m.last_accessed = Some(dt.with_timezone(&Utc));
                }
            }

            // Ensure memory_type is Episode so calculate uses decay formula
            m.memory_type = crate::models::MemoryType::Episode;

            let relevance = m.calculate_episode_relevance(self.decay_days, self.decay_factor);
            debug!(id = candidate.id.as_str(), relevance, "Episode relevance calculated");

            if relevance < self.threshold {
                let forget_after = Utc::now() + Duration::days(self.grace_days as i64);
                match self.db.set_memory_forget_after(&candidate.id, forget_after).await {
                    Ok(affected) => {
                        if affected > 0 {
                            scheduled += 1;
                            info!(id = candidate.id.as_str(), "Scheduled episode for forgetting");
                        }
                    }
                    Err(e) => {
                        error!(id = candidate.id.as_str(), "Failed to schedule forget_after: {}", e);
                    }
                }
            }
        }

        info!(scheduled, "Episode decay run complete");
        Ok(scheduled)
    }

    /// Get the configured interval in seconds
    pub fn interval_secs(&self) -> u64 {
        // Default to once per day
        86400
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Database, LibSqlBackend};
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
                memory_type TEXT NOT NULL DEFAULT 'fact',
                last_accessed TEXT,
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

        let db = Database { db: Arc::new(inner_db) };
        let backend: Arc<dyn DatabaseBackend> = Arc::new(LibSqlBackend::new(db));
        (conn, backend, temp_file)
    }

    async fn insert_memory(
        conn: &Connection,
        id: &str,
        memory: &str,
        memory_type: &str,
        last_accessed: Option<&str>,
        is_static: bool,
        is_forgotten: bool,
    ) {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO memories (
                id, memory, space_id, container_tag, version, is_latest,
                memory_relations, source_count, is_inference, is_forgotten,
                is_static, forget_after, memory_type, last_accessed, metadata,
                created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, 1, 1, '{}', 0, 0, ?5, ?6, NULL, ?7, ?8, '{}', ?9, ?9)
            "#,
            (
                id,
                memory,
                "space1",
                "tag",
                is_forgotten as i32,
                is_static as i32,
                memory_type,
                last_accessed,
                now.as_str(),
            ),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_run_once_empty_db() {
        let (_conn, db, _tmp) = setup_test_db().await;
        let mgr = EpisodeDecayManager::new(db, 0.5, 7, 30.0, 0.9);
        let res = mgr.run_once().await.unwrap();
        assert_eq!(res, 0);
    }

    #[tokio::test]
    async fn test_static_episodes_excluded() {
        let (conn, db, _tmp) = setup_test_db().await;

        // old episode but marked static
        let past = (Utc::now() - Duration::days(365)).to_rfc3339();
        insert_memory(
            &conn,
            "e1",
            "old episode",
            "episode",
            Some(&past),
            true,
            false,
        )
        .await;

        let mgr = EpisodeDecayManager::new(Arc::clone(&db), 0.5, 7, 30.0, 0.9);
        let scheduled = mgr.run_once().await.unwrap();
        assert_eq!(scheduled, 0);

        // verify forget_after still null
        let row = conn
            .query("SELECT forget_after FROM memories WHERE id = 'e1'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();

        let f: Option<String> = row.get(0).unwrap();
        assert!(f.is_none());
    }

    #[tokio::test]
    async fn test_high_relevance_not_scheduled() {
        let (conn, db, _tmp) = setup_test_db().await;

        // recent episode, high relevance
        insert_memory(
            &conn,
            "e2",
            "recent episode",
            "episode",
            Some(&Utc::now().to_rfc3339()),
            false,
            false,
        )
        .await;

        let mgr = EpisodeDecayManager::new(Arc::clone(&db), 0.9, 7, 30.0, 0.9);
        let scheduled = mgr.run_once().await.unwrap();
        assert_eq!(scheduled, 0);

        let row = conn
            .query("SELECT forget_after FROM memories WHERE id = 'e2'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();

        let f: Option<String> = row.get(0).unwrap();
        assert!(f.is_none());
    }

    #[tokio::test]
    async fn test_low_relevance_scheduled() {
        let (conn, db, _tmp) = setup_test_db().await;

        // very old episode to ensure low relevance
        let past = (Utc::now() - Duration::days(365)).to_rfc3339();
        insert_memory(
            &conn,
            "e3",
            "ancient episode",
            "episode",
            Some(&past),
            false,
            false,
        )
        .await;

        let mgr = EpisodeDecayManager::new(Arc::clone(&db), 0.5, 10, 30.0, 0.9);
        let scheduled = mgr.run_once().await.unwrap();
        assert_eq!(scheduled, 1);

        let row = conn
            .query("SELECT forget_after FROM memories WHERE id = 'e3'", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();

        let f: Option<String> = row.get(0).unwrap();
        assert!(f.is_some());
        // ensure it's in future
        let dt = chrono::DateTime::parse_from_rfc3339(&f.unwrap())
            .unwrap()
            .with_timezone(&Utc);
        assert!(dt > Utc::now());
    }
}
