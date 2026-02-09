use chrono::{DateTime, Utc};
use libsql::{params, Connection};
use nanoid::nanoid;

use crate::error::Result;
use crate::models::MemorySource;

pub struct MemorySourcesRepository;

impl MemorySourcesRepository {
    pub async fn create(
        conn: &Connection,
        memory_id: &str,
        document_id: &str,
        chunk_id: Option<&str>,
    ) -> Result<MemorySource> {
        let id = nanoid!();
        let created_at = Utc::now();

        let id_clone = id.clone();
        conn.execute(
            r#"
            INSERT INTO memory_sources (
                id, memory_id, document_id, chunk_id, created_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5
            )
            "#,
            params![
                id_clone,
                memory_id,
                document_id,
                chunk_id,
                created_at.to_rfc3339(),
            ],
        )
        .await?;

        Ok(MemorySource {
            id,
            memory_id: memory_id.to_string(),
            document_id: document_id.to_string(),
            chunk_id: chunk_id.map(|value| value.to_string()),
            created_at,
        })
    }

    pub async fn get_by_memory(conn: &Connection, memory_id: &str) -> Result<Vec<MemorySource>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, memory_id, document_id, chunk_id, created_at
                FROM memory_sources
                WHERE memory_id = ?1
                ORDER BY created_at ASC
                "#,
                params![memory_id],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory_source(&row)?);
        }

        Ok(results)
    }

    #[allow(dead_code)]
    pub async fn get_by_document(
        conn: &Connection,
        document_id: &str,
    ) -> Result<Vec<MemorySource>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, memory_id, document_id, chunk_id, created_at
                FROM memory_sources
                WHERE document_id = ?1
                ORDER BY created_at ASC
                "#,
                params![document_id],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory_source(&row)?);
        }

        Ok(results)
    }

    fn row_to_memory_source(row: &libsql::Row) -> Result<MemorySource> {
        Ok(MemorySource {
            id: row.get(0)?,
            memory_id: row.get(1)?,
            document_id: row.get(2)?,
            chunk_id: row.get(3)?,
            created_at: DateTime::parse_from_rfc3339(&row.get::<String>(4)?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_test_db() -> Connection {
        let conn = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .unwrap()
            .connect()
            .unwrap();

        conn.execute(
            r#"
            CREATE TABLE memory_sources (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT,
                created_at TEXT NOT NULL
            )
            "#,
            (),
        )
        .await
        .unwrap();

        conn
    }

    #[tokio::test]
    async fn test_create_and_get_by_memory() {
        let conn = setup_test_db().await;

        let created = MemorySourcesRepository::create(&conn, "mem1", "doc1", Some("chunk1"))
            .await
            .unwrap();

        let results = MemorySourcesRepository::get_by_memory(&conn, "mem1")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let fetched = &results[0];
        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.memory_id, "mem1");
        assert_eq!(fetched.document_id, "doc1");
        assert_eq!(fetched.chunk_id.as_deref(), Some("chunk1"));
    }

    #[tokio::test]
    async fn test_get_by_document() {
        let conn = setup_test_db().await;

        MemorySourcesRepository::create(&conn, "mem1", "doc1", None)
            .await
            .unwrap();
        MemorySourcesRepository::create(&conn, "mem2", "doc2", Some("chunk2"))
            .await
            .unwrap();

        let results = MemorySourcesRepository::get_by_document(&conn, "doc1")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, "mem1");
        assert_eq!(results[0].document_id, "doc1");
        assert!(results[0].chunk_id.is_none());
    }
}
