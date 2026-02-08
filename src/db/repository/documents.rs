use chrono::{DateTime, Utc};
use libsql::{params, Connection};

use crate::error::Result;
use crate::models::{
    Document, DocumentSummary, DocumentType, ListDocumentsRequest, Pagination, ProcessingDocument,
    ProcessingStatus,
};

pub struct DocumentRepository;

impl DocumentRepository {
    pub async fn create(conn: &Connection, doc: &Document) -> Result<()> {
        conn.execute(
            r#"
            INSERT INTO documents (
                id, custom_id, connection_id, title, content, summary, url, source,
                doc_type, status, metadata, container_tags, chunk_count, token_count,
                word_count, error_message, created_at, updated_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18
            )
            "#,
            params![
                doc.id.clone(),
                doc.custom_id.clone(),
                doc.connection_id.clone(),
                doc.title.clone(),
                doc.content.clone(),
                doc.summary.clone(),
                doc.url.clone(),
                doc.source.clone(),
                doc.doc_type.to_string(),
                doc.status.to_string(),
                serde_json::to_string(&doc.metadata)?,
                serde_json::to_string(&doc.container_tags)?,
                doc.chunk_count,
                doc.token_count,
                doc.word_count,
                doc.error_message.clone(),
                doc.created_at.to_rfc3339(),
                doc.updated_at.to_rfc3339(),
            ],
        )
        .await?;

        Ok(())
    }

    pub async fn get_by_id(conn: &Connection, id: &str) -> Result<Option<Document>> {
        let mut rows = conn
            .query("SELECT * FROM documents WHERE id = ?1", params![id])
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(Self::row_to_document(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_by_ids(conn: &Connection, ids: &[String]) -> Result<Vec<Document>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut placeholders = String::new();
        for i in 0..ids.len() {
            if i > 0 {
                placeholders.push_str(", ");
            }
            placeholders.push('?');
            placeholders.push_str(&(i + 1).to_string());
        }

        let sql = format!("SELECT * FROM documents WHERE id IN ({placeholders})");
        let params: Vec<libsql::Value> =
            ids.iter().map(|id| libsql::Value::from(id.clone())).collect();

        let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;
        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_document(&row)?);
        }
        Ok(results)
    }

    pub async fn get_by_custom_id(conn: &Connection, custom_id: &str) -> Result<Option<Document>> {
        let mut rows = conn
            .query(
                "SELECT * FROM documents WHERE custom_id = ?1",
                params![custom_id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(Self::row_to_document(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn update(conn: &Connection, doc: &Document) -> Result<()> {
        conn.execute(
            r#"
            UPDATE documents SET
                title = ?2,
                content = ?3,
                summary = ?4,
                url = ?5,
                doc_type = ?6,
                status = ?7,
                metadata = ?8,
                container_tags = ?9,
                chunk_count = ?10,
                token_count = ?11,
                word_count = ?12,
                error_message = ?13,
                updated_at = ?14
            WHERE id = ?1
            "#,
            params![
                doc.id.clone(),
                doc.title.clone(),
                doc.content.clone(),
                doc.summary.clone(),
                doc.url.clone(),
                doc.doc_type.to_string(),
                doc.status.to_string(),
                serde_json::to_string(&doc.metadata)?,
                serde_json::to_string(&doc.container_tags)?,
                doc.chunk_count,
                doc.token_count,
                doc.word_count,
                doc.error_message.clone(),
                doc.updated_at.to_rfc3339(),
            ],
        )
        .await?;

        Ok(())
    }

    pub async fn delete(conn: &Connection, id: &str) -> Result<bool> {
        let rows_affected = conn
            .execute("DELETE FROM documents WHERE id = ?1", params![id])
            .await?;

        Ok(rows_affected > 0)
    }

    pub async fn delete_by_custom_id(conn: &Connection, custom_id: &str) -> Result<bool> {
        let rows_affected = conn
            .execute(
                "DELETE FROM documents WHERE custom_id = ?1",
                params![custom_id],
            )
            .await?;

        Ok(rows_affected > 0)
    }

    pub async fn list(
        conn: &Connection,
        req: &ListDocumentsRequest,
    ) -> Result<(Vec<DocumentSummary>, Pagination)> {
        let limit = req.limit.unwrap_or(10).min(1100);
        let page = req.page.unwrap_or(1).max(1);
        let offset = (page - 1) * limit;
        let order = req.order.as_deref().unwrap_or("desc");
        let sort = req.sort.as_deref().unwrap_or("created_at");

        let order_clause = format!(
            "ORDER BY {} {}",
            match sort {
                "updated_at" => "updated_at",
                _ => "created_at",
            },
            match order {
                "asc" => "ASC",
                _ => "DESC",
            }
        );

        let mut where_clauses = Vec::new();
        let mut tag_params: Vec<libsql::Value> = Vec::new();

        if let Some(ref tags) = req.container_tags {
            if !tags.is_empty() {
                for (i, tag) in tags.iter().enumerate() {
                    where_clauses.push(format!("container_tags LIKE ?{}", i + 1));
                    tag_params.push(libsql::Value::from(format!("%\"{}%", tag)));
                }
            }
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" OR "))
        };

        let count_query = format!("SELECT COUNT(*) FROM documents {}", where_clause);
        let mut count_rows = conn
            .query(&count_query, libsql::params_from_iter(tag_params.clone()))
            .await?;
        let total: i64 = if let Some(row) = count_rows.next().await? {
            row.get(0)?
        } else {
            0
        };

        // LIMIT and OFFSET params come after the tag params
        let limit_idx = tag_params.len() + 1;
        let offset_idx = tag_params.len() + 2;
        let query = format!(
            "SELECT * FROM documents {} {} LIMIT ?{} OFFSET ?{}",
            where_clause, order_clause, limit_idx, offset_idx
        );

        let mut list_params = tag_params;
        list_params.push(libsql::Value::from(limit as i64));
        list_params.push(libsql::Value::from(offset as i64));

        let mut rows = conn
            .query(&query, libsql::params_from_iter(list_params))
            .await?;

        let mut documents = Vec::new();
        while let Some(row) = rows.next().await? {
            let doc = Self::row_to_document(&row)?;
            documents.push(DocumentSummary::from(doc));
        }

        let pagination = Pagination::new(page, limit, total as u32);

        Ok((documents, pagination))
    }

    pub async fn get_processing(conn: &Connection) -> Result<Vec<ProcessingDocument>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, status, title, created_at 
                FROM documents 
                WHERE status NOT IN ('done', 'failed')
                ORDER BY created_at ASC
                "#,
                (),
            )
            .await?;

        let mut docs = Vec::new();
        while let Some(row) = rows.next().await? {
            docs.push(ProcessingDocument {
                id: row.get(0)?,
                status: row
                    .get::<String>(1)?
                    .parse()
                    .unwrap_or(ProcessingStatus::Unknown),
                title: row.get(2)?,
                created_at: DateTime::parse_from_rfc3339(&row.get::<String>(3)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }

        Ok(docs)
    }

    pub async fn update_status(
        conn: &Connection,
        id: &str,
        status: ProcessingStatus,
        error: Option<&str>,
    ) -> Result<()> {
        conn.execute(
            r#"
            UPDATE documents 
            SET status = ?2, error_message = ?3, updated_at = ?4
            WHERE id = ?1
            "#,
            params![id, status.to_string(), error, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    fn row_to_document(row: &libsql::Row) -> Result<Document> {
        Ok(Document {
            id: row.get(0)?,
            custom_id: row.get(1)?,
            connection_id: row.get(2)?,
            title: row.get(3)?,
            content: row.get(4)?,
            summary: row.get(5)?,
            url: row.get(6)?,
            source: row.get(7)?,
            doc_type: row
                .get::<String>(8)?
                .parse()
                .unwrap_or(DocumentType::Unknown),
            status: row
                .get::<String>(9)?
                .parse()
                .unwrap_or(ProcessingStatus::Unknown),
            metadata: serde_json::from_str(&row.get::<String>(10)?).unwrap_or_default(),
            container_tags: serde_json::from_str(&row.get::<String>(11)?).unwrap_or_default(),
            chunk_count: row.get(12)?,
            token_count: row.get(13)?,
            word_count: row.get(14)?,
            error_message: row.get(15)?,
            created_at: DateTime::parse_from_rfc3339(&row.get::<String>(16)?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&row.get::<String>(17)?)
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
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                custom_id TEXT,
                connection_id TEXT,
                title TEXT,
                content TEXT,
                summary TEXT,
                url TEXT,
                source TEXT,
                doc_type TEXT NOT NULL DEFAULT 'text',
                status TEXT NOT NULL DEFAULT 'queued',
                metadata TEXT DEFAULT '{}',
                container_tags TEXT DEFAULT '[]',
                chunk_count INTEGER DEFAULT 0,
                token_count INTEGER,
                word_count INTEGER,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
            (),
        )
        .await
        .unwrap();

        conn
    }

    fn make_doc(id: &str, tags: Vec<String>) -> Document {
        let mut doc = Document::new(id.to_string());
        doc.container_tags = tags;
        doc
    }

    #[tokio::test]
    async fn test_tag_filter_normal() {
        let conn = setup_test_db().await;

        let doc1 = make_doc("d1", vec!["project_a".to_string()]);
        let doc2 = make_doc("d2", vec!["project_b".to_string()]);
        let doc3 = make_doc("d3", vec!["project_a".to_string(), "project_b".to_string()]);
        DocumentRepository::create(&conn, &doc1).await.unwrap();
        DocumentRepository::create(&conn, &doc2).await.unwrap();
        DocumentRepository::create(&conn, &doc3).await.unwrap();

        let req = ListDocumentsRequest {
            container_tags: Some(vec!["project_a".to_string()]),
            ..Default::default()
        };
        let (results, pagination) = DocumentRepository::list(&conn, &req).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(pagination.total_items, 2);

        let ids: Vec<&str> = results.iter().map(|d| d.id.as_str()).collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d3"));
    }

    #[tokio::test]
    async fn test_tag_filter_multiple_tags() {
        let conn = setup_test_db().await;

        let doc1 = make_doc("d1", vec!["alpha".to_string()]);
        let doc2 = make_doc("d2", vec!["beta".to_string()]);
        let doc3 = make_doc("d3", vec!["gamma".to_string()]);
        DocumentRepository::create(&conn, &doc1).await.unwrap();
        DocumentRepository::create(&conn, &doc2).await.unwrap();
        DocumentRepository::create(&conn, &doc3).await.unwrap();

        let req = ListDocumentsRequest {
            container_tags: Some(vec!["alpha".to_string(), "beta".to_string()]),
            ..Default::default()
        };
        let (results, _) = DocumentRepository::list(&conn, &req).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_tag_filter_no_tags_returns_all() {
        let conn = setup_test_db().await;

        DocumentRepository::create(&conn, &make_doc("d1", vec!["x".to_string()]))
            .await
            .unwrap();
        DocumentRepository::create(&conn, &make_doc("d2", vec![]))
            .await
            .unwrap();

        let req = ListDocumentsRequest::default();
        let (results, pagination) = DocumentRepository::list(&conn, &req).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(pagination.total_items, 2);
    }

    #[tokio::test]
    async fn test_tag_filter_injection_does_not_cause_error() {
        let conn = setup_test_db().await;

        let doc = make_doc("d1", vec!["safe_tag".to_string()]);
        DocumentRepository::create(&conn, &doc).await.unwrap();

        let injection_payloads = vec![
            "'; DROP TABLE documents; --".to_string(),
            "\" OR 1=1 --".to_string(),
            "tag' UNION SELECT * FROM documents --".to_string(),
            "%' OR '1'='1".to_string(),
        ];

        for payload in injection_payloads {
            let req = ListDocumentsRequest {
                container_tags: Some(vec![payload.clone()]),
                ..Default::default()
            };
            let result = DocumentRepository::list(&conn, &req).await;
            assert!(
                result.is_ok(),
                "SQL injection payload should not cause error: {}",
                payload
            );
            let (results, _) = result.unwrap();
            assert_eq!(
                results.len(),
                0,
                "injection payload '{}' should match no documents",
                payload
            );
        }

        let verify_req = ListDocumentsRequest::default();
        let (all_docs, _) = DocumentRepository::list(&conn, &verify_req).await.unwrap();
        assert_eq!(
            all_docs.len(),
            1,
            "documents table must still have its data after injection attempts"
        );
        assert_eq!(all_docs[0].id, "d1");
    }

    #[tokio::test]
    async fn test_tag_filter_injection_with_quotes() {
        let conn = setup_test_db().await;

        let doc = make_doc("d1", vec!["normal".to_string()]);
        DocumentRepository::create(&conn, &doc).await.unwrap();

        let req = ListDocumentsRequest {
            container_tags: Some(vec!["tag\"; DROP TABLE documents; --".to_string()]),
            ..Default::default()
        };
        let result = DocumentRepository::list(&conn, &req).await;
        assert!(result.is_ok());
        let (results, _) = result.unwrap();
        assert_eq!(results.len(), 0);

        let (all, _) = DocumentRepository::list(&conn, &ListDocumentsRequest::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 1, "table must survive injection attempt");
    }

    #[tokio::test]
    async fn test_get_by_ids_returns_matching_documents() {
        let conn = setup_test_db().await;

        DocumentRepository::create(&conn, &make_doc("d1", vec![])).await.unwrap();
        DocumentRepository::create(&conn, &make_doc("d2", vec![])).await.unwrap();
        DocumentRepository::create(&conn, &make_doc("d3", vec![])).await.unwrap();

        let ids = vec!["d1".to_string(), "d3".to_string()];
        let results = DocumentRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert_eq!(results.len(), 2);

        let result_ids: Vec<&str> = results.iter().map(|d| d.id.as_str()).collect();
        assert!(result_ids.contains(&"d1"));
        assert!(result_ids.contains(&"d3"));
    }

    #[tokio::test]
    async fn test_get_by_ids_empty_input() {
        let conn = setup_test_db().await;

        DocumentRepository::create(&conn, &make_doc("d1", vec![])).await.unwrap();

        let results = DocumentRepository::get_by_ids(&conn, &[]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_ids_nonexistent_ids() {
        let conn = setup_test_db().await;

        DocumentRepository::create(&conn, &make_doc("d1", vec![])).await.unwrap();

        let ids = vec!["no_such_id".to_string(), "also_missing".to_string()];
        let results = DocumentRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_ids_partial_match() {
        let conn = setup_test_db().await;

        DocumentRepository::create(&conn, &make_doc("d1", vec![])).await.unwrap();
        DocumentRepository::create(&conn, &make_doc("d2", vec![])).await.unwrap();

        let ids = vec!["d1".to_string(), "missing".to_string()];
        let results = DocumentRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "d1");
    }
}
