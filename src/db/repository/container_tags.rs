use chrono::Utc;
use libsql::{params, Connection};

use crate::error::Result;
use crate::models::ContainerFilter;

pub struct ContainerTagsRepository;

impl ContainerTagsRepository {
    /// Retrieve the filter configuration for a specific container tag
    pub async fn get_filter_config(
        conn: &Connection,
        tag: &str,
    ) -> Result<Option<ContainerFilter>> {
        let mut rows = conn
            .query(
                "SELECT tag, should_llm_filter, filter_prompt FROM container_tags WHERE tag = ?1",
                params![tag],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            let tag: String = row.get(0)?;
            let should_llm_filter: i32 = row.get(1)?;
            let filter_prompt: Option<String> = row.get(2)?;

            Ok(Some(ContainerFilter {
                tag,
                should_llm_filter: should_llm_filter != 0,
                filter_prompt,
            }))
        } else {
            Ok(None)
        }
    }

    /// Set or update the filter configuration for a specific container tag
    pub async fn set_filter_config(
        conn: &Connection,
        tag: &str,
        filter: &ContainerFilter,
    ) -> Result<()> {
        let updated_at = Utc::now().to_rfc3339();

        let mut rows = conn
            .query(
                "SELECT tag FROM container_tags WHERE tag = ?1",
                params![tag],
            )
            .await?;

        if rows.next().await?.is_some() {
            conn.execute(
                r#"
                UPDATE container_tags
                SET should_llm_filter = ?2, filter_prompt = ?3, updated_at = ?4
                WHERE tag = ?1
                "#,
                params![
                    tag,
                    filter.should_llm_filter as i32,
                    filter.filter_prompt.clone(),
                    updated_at,
                ],
            )
            .await?;
        } else {
            let created_at = Utc::now().to_rfc3339();
            conn.execute(
                r#"
                INSERT INTO container_tags (
                    tag, metadata, document_count, memory_count,
                    should_llm_filter, filter_prompt, created_at, updated_at
                ) VALUES (?1, '{}', 0, 0, ?2, ?3, ?4, ?5)
                "#,
                params![
                    tag,
                    filter.should_llm_filter as i32,
                    filter.filter_prompt.clone(),
                    created_at.clone(),
                    updated_at,
                ],
            )
            .await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsql::Builder;

    async fn setup_test_db() -> Connection {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS container_tags (
                tag TEXT PRIMARY KEY,
                metadata TEXT DEFAULT '{}',
                document_count INTEGER DEFAULT 0,
                memory_count INTEGER DEFAULT 0,
                should_llm_filter INTEGER DEFAULT 0,
                filter_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .await
        .unwrap();

        conn
    }

    #[tokio::test]
    async fn test_get_filter_config_returns_config() {
        let conn = setup_test_db().await;

        let created_at = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO container_tags (
                tag, metadata, document_count, memory_count,
                should_llm_filter, filter_prompt, created_at, updated_at
            ) VALUES (?1, '{}', 0, 0, ?2, ?3, ?4, ?5)
            "#,
            params![
                "test_tag",
                1,
                Some("Only technical documents"),
                created_at.clone(),
                created_at,
            ],
        )
        .await
        .unwrap();

        let config = ContainerTagsRepository::get_filter_config(&conn, "test_tag")
            .await
            .unwrap();

        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.tag, "test_tag");
        assert_eq!(config.should_llm_filter, true);
        assert_eq!(
            config.filter_prompt,
            Some("Only technical documents".to_string())
        );
    }

    #[tokio::test]
    async fn test_get_filter_config_returns_none_for_nonexistent_tag() {
        let conn = setup_test_db().await;

        let config = ContainerTagsRepository::get_filter_config(&conn, "nonexistent")
            .await
            .unwrap();

        assert!(config.is_none());
    }

    #[tokio::test]
    async fn test_set_filter_config_creates_new_tag() {
        let conn = setup_test_db().await;

        let filter = ContainerFilter {
            tag: "new_tag".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("Filter prompt".to_string()),
        };

        ContainerTagsRepository::set_filter_config(&conn, "new_tag", &filter)
            .await
            .unwrap();

        let config = ContainerTagsRepository::get_filter_config(&conn, "new_tag")
            .await
            .unwrap();

        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.should_llm_filter, true);
        assert_eq!(config.filter_prompt, Some("Filter prompt".to_string()));
    }

    #[tokio::test]
    async fn test_set_filter_config_updates_existing_tag() {
        let conn = setup_test_db().await;

        let created_at = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO container_tags (
                tag, metadata, document_count, memory_count,
                should_llm_filter, filter_prompt, created_at, updated_at
            ) VALUES (?1, '{}', 0, 0, ?2, ?3, ?4, ?5)
            "#,
            params![
                "existing_tag",
                0,
                None::<String>,
                created_at.clone(),
                created_at,
            ],
        )
        .await
        .unwrap();

        let filter = ContainerFilter {
            tag: "existing_tag".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("New filter prompt".to_string()),
        };

        ContainerTagsRepository::set_filter_config(&conn, "existing_tag", &filter)
            .await
            .unwrap();

        let config = ContainerTagsRepository::get_filter_config(&conn, "existing_tag")
            .await
            .unwrap();

        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.should_llm_filter, true);
        assert_eq!(config.filter_prompt, Some("New filter prompt".to_string()));
    }

    #[tokio::test]
    async fn test_set_filter_config_with_all_option_types() {
        let conn = setup_test_db().await;

        let filter1 = ContainerFilter {
            tag: "tag1".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("prompt".to_string()),
        };
        ContainerTagsRepository::set_filter_config(&conn, "tag1", &filter1)
            .await
            .unwrap();

        let filter2 = ContainerFilter {
            tag: "tag2".to_string(),
            should_llm_filter: false,
            filter_prompt: None,
        };
        ContainerTagsRepository::set_filter_config(&conn, "tag2", &filter2)
            .await
            .unwrap();

        let filter3 = ContainerFilter {
            tag: "tag3".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("custom prompt".to_string()),
        };
        ContainerTagsRepository::set_filter_config(&conn, "tag3", &filter3)
            .await
            .unwrap();

        let config1 = ContainerTagsRepository::get_filter_config(&conn, "tag1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(config1.should_llm_filter, true);
        assert_eq!(config1.filter_prompt, Some("prompt".to_string()));

        let config2 = ContainerTagsRepository::get_filter_config(&conn, "tag2")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(config2.should_llm_filter, false);
        assert_eq!(config2.filter_prompt, None);

        let config3 = ContainerTagsRepository::get_filter_config(&conn, "tag3")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(config3.should_llm_filter, true);
        assert_eq!(config3.filter_prompt, Some("custom prompt".to_string()));
    }

    #[tokio::test]
    async fn test_set_filter_config_updates_timestamp() {
        let conn = setup_test_db().await;

        let created_at = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO container_tags (
                tag, metadata, document_count, memory_count,
                should_llm_filter, filter_prompt, created_at, updated_at
            ) VALUES (?1, '{}', 0, 0, ?2, ?3, ?4, ?5)
            "#,
            params!["timestamp_tag", 0, None::<String>, created_at.clone(), created_at,],
        )
        .await
        .unwrap();

        let mut rows = conn
            .query(
                "SELECT updated_at FROM container_tags WHERE tag = ?1",
                params!["timestamp_tag"],
            )
            .await
            .unwrap();
        let initial_timestamp: String = rows.next().await.unwrap().unwrap().get(0).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let filter = ContainerFilter {
            tag: "timestamp_tag".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("Updated".to_string()),
        };
        ContainerTagsRepository::set_filter_config(&conn, "timestamp_tag", &filter)
            .await
            .unwrap();

        let mut rows = conn
            .query(
                "SELECT updated_at FROM container_tags WHERE tag = ?1",
                params!["timestamp_tag"],
            )
            .await
            .unwrap();
        let updated_timestamp: String = rows.next().await.unwrap().unwrap().get(0).unwrap();

        assert_ne!(initial_timestamp, updated_timestamp);
    }

    #[tokio::test]
    async fn test_set_filter_config_invalid_tag() {
        let conn = setup_test_db().await;

        let filter = ContainerFilter {
            tag: "".to_string(),
            should_llm_filter: true,
            filter_prompt: Some("Prompt".to_string()),
        };

        let result = ContainerTagsRepository::set_filter_config(&conn, "", &filter).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_db_backend_filter_methods() {
        use crate::config::DatabaseConfig;
        use crate::db::backends::libsql::LibSqlBackend;
        use crate::db::connection::Database;
        use crate::db::traits::DatabaseBackend;
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let thread_id = std::thread::current().id();

        let config = DatabaseConfig {
            url: format!(
                "file:/tmp/momo_test_db_{:?}_{}?mode=memory&cache=shared",
                thread_id, timestamp
            ),
            auth_token: None,
            local_path: None,
        };
        let db = Database::new(&config).await.expect("Failed to create database");
        let backend = LibSqlBackend::new(db);

        let result = backend.get_container_filter("nonexistent_tag").await.unwrap();
        assert!(result.is_none());
    }
}
