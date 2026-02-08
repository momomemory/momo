use crate::error::Result;
use chrono::Utc;
use libsql::Connection;

pub struct MetadataRepository;

impl MetadataRepository {
    pub async fn get(conn: &Connection, key: &str) -> Result<Option<String>> {
        let mut rows = conn
            .query("SELECT value FROM momo_meta WHERE key = ?", [key])
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get::<String>(0)?))
        } else {
            Ok(None)
        }
    }

    pub async fn set(conn: &Connection, key: &str, value: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO momo_meta (key, value, updated_at) VALUES (?, ?, ?)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [key, value, &now],
        )
        .await?;
        Ok(())
    }

    pub async fn get_embedding_dimensions(conn: &Connection) -> Result<Option<usize>> {
        match Self::get(conn, "embedding_dimensions").await? {
            Some(s) => Ok(s.parse().ok()),
            None => Ok(None),
        }
    }

    pub async fn set_embedding_dimensions(conn: &Connection, dims: usize) -> Result<()> {
        Self::set(conn, "embedding_dimensions", &dims.to_string()).await
    }
}
