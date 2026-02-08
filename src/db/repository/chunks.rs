use chrono::{DateTime, Utc};
use libsql::{params, Connection};

use crate::error::Result;
use crate::models::{Chunk, ChunkWithDocument};

/// Build parameterized LIKE clauses for container_tags filtering.
/// Returns (sql_fragment, param_values) where sql_fragment uses positional
/// placeholders starting at `start_idx` (e.g. "d.container_tags LIKE ?4 OR d.container_tags LIKE ?5")
/// and param_values contains the corresponding LIKE patterns.
fn build_tag_filter(
    tags: &[String],
    start_idx: usize,
    column_prefix: &str,
) -> (String, Vec<libsql::Value>) {
    let mut clauses = Vec::with_capacity(tags.len());
    let mut values = Vec::with_capacity(tags.len());
    for (i, tag) in tags.iter().enumerate() {
        clauses.push(format!(
            "{}.container_tags LIKE ?{}",
            column_prefix,
            start_idx + i
        ));
        // LIKE pattern: match JSON array element containing this tag value
        values.push(libsql::Value::from(format!("%\"{}%", tag)));
    }
    (clauses.join(" OR "), values)
}

pub struct ChunkRepository;

impl ChunkRepository {
    pub async fn create(conn: &Connection, chunk: &Chunk) -> Result<()> {
        conn.execute(
            r#"
            INSERT INTO chunks (
                id, document_id, content, embedded_content, position, token_count, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                chunk.id.clone(),
                chunk.document_id.clone(),
                chunk.content.clone(),
                chunk.embedded_content.clone(),
                chunk.position,
                chunk.token_count,
                chunk.created_at.to_rfc3339(),
            ],
        )
        .await?;

        Ok(())
    }

    pub async fn create_batch(conn: &Connection, chunks: &[Chunk]) -> Result<()> {
        for chunk in chunks {
            Self::create(conn, chunk).await?;
        }
        Ok(())
    }

    pub async fn get_by_document_id(conn: &Connection, document_id: &str) -> Result<Vec<Chunk>> {
        let mut rows = conn
            .query(
                "SELECT id, document_id, content, embedded_content, position, token_count, created_at 
                 FROM chunks WHERE document_id = ?1 ORDER BY position",
                params![document_id],
            )
            .await?;

        let mut chunks = Vec::new();
        while let Some(row) = rows.next().await? {
            chunks.push(Chunk {
                id: row.get(0)?,
                document_id: row.get(1)?,
                content: row.get(2)?,
                embedded_content: row.get(3)?,
                position: row.get(4)?,
                token_count: row.get(5)?,
                created_at: DateTime::parse_from_rfc3339(&row.get::<String>(6)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }

        Ok(chunks)
    }

    pub async fn update_embedding(
        conn: &Connection,
        chunk_id: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let embedding_json = serde_json::to_string(embedding)?;

        conn.execute(
            "UPDATE chunks SET embedding = vector32(?2) WHERE id = ?1",
            params![chunk_id, embedding_json],
        )
        .await?;

        Ok(())
    }

    pub async fn update_embeddings_batch(
        conn: &Connection,
        updates: &[(String, Vec<f32>)],
    ) -> Result<()> {
        for (chunk_id, embedding) in updates {
            Self::update_embedding(conn, chunk_id, embedding).await?;
        }
        Ok(())
    }

    pub async fn delete_by_document_id(conn: &Connection, document_id: &str) -> Result<()> {
        conn.execute(
            "DELETE FROM chunks WHERE document_id = ?1",
            params![document_id],
        )
        .await?;

        Ok(())
    }

    pub async fn search_similar(
        conn: &Connection,
        embedding: &[f32],
        limit: u32,
        threshold: f32,
        container_tags: Option<&[String]>,
    ) -> Result<Vec<ChunkWithDocument>> {
        let embedding_json = serde_json::to_string(embedding)?;

        let has_tags = container_tags
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        let (query, tag_values) = if has_tags {
            let tags = container_tags.unwrap();
            // Fixed params: ?1=embedding, ?2=threshold, ?3=limit; tags start at ?4
            let (tag_clause, tag_vals) = build_tag_filter(tags, 4, "d");
            let q = format!(
                r#"
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content as chunk_content,
                    d.title as document_title,
                    d.metadata as document_metadata,
                    1 - vector_distance_cos(c.embedding, vector32(?1)) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND (1 - vector_distance_cos(c.embedding, vector32(?1))) >= ?2
                  AND ({})
                ORDER BY score DESC
                LIMIT ?3
                "#,
                tag_clause
            );
            (q, tag_vals)
        } else {
            (
                r#"
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content as chunk_content,
                    d.title as document_title,
                    d.metadata as document_metadata,
                    1 - vector_distance_cos(c.embedding, vector32(?1)) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND (1 - vector_distance_cos(c.embedding, vector32(?1))) >= ?2
                ORDER BY score DESC
                LIMIT ?3
                "#
                .to_string(),
                Vec::new(),
            )
        };

        let mut param_values: Vec<libsql::Value> = vec![
            libsql::Value::from(embedding_json),
            libsql::Value::from(threshold as f64),
            libsql::Value::from(limit),
        ];
        param_values.extend(tag_values);

        let mut rows = conn
            .query(&query, libsql::params_from_iter(param_values))
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let score = row.get::<f64>(5)? as f32;

            results.push(ChunkWithDocument {
                chunk_id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_content: row.get(2)?,
                document_title: row.get(3)?,
                document_metadata: serde_json::from_str(&row.get::<String>(4)?).unwrap_or_default(),
                score,
            });
        }

        Ok(results)
    }

    pub async fn search_with_vector_index(
        conn: &Connection,
        embedding: &[f32],
        limit: u32,
        container_tags: Option<&[String]>,
    ) -> Result<Vec<ChunkWithDocument>> {
        let embedding_json = serde_json::to_string(embedding)?;

        let base_query = r#"
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content as chunk_content,
                d.title as document_title,
                d.metadata as document_metadata,
                1 - vector_distance_cos(c.embedding, vector32(?1)) as score
            FROM vector_top_k('chunks_embedding_idx', vector32(?1), ?2) as v
            JOIN chunks c ON c.rowid = v.id
            JOIN documents d ON c.document_id = d.id
        "#;

        let (query, tag_values) = if let Some(tags) = container_tags {
            if !tags.is_empty() {
                // Fixed params: ?1=embedding, ?2=limit; tags start at ?3
                let (tag_clause, tag_vals) = build_tag_filter(tags, 3, "d");
                (format!("{} WHERE {}", base_query, tag_clause), tag_vals)
            } else {
                (base_query.to_string(), Vec::new())
            }
        } else {
            (base_query.to_string(), Vec::new())
        };

        let mut param_values: Vec<libsql::Value> = vec![
            libsql::Value::from(embedding_json),
            libsql::Value::from(limit),
        ];
        param_values.extend(tag_values);

        let mut rows = conn
            .query(&query, libsql::params_from_iter(param_values))
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            // match pattern used in memories.rs: score is fetched as f64 then cast to f32
            let score = row.get::<f64>(5)? as f32;
            results.push(ChunkWithDocument {
                chunk_id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_content: row.get(2)?,
                document_title: row.get(3)?,
                document_metadata: serde_json::from_str(&row.get::<String>(4)?).unwrap_or_default(),
                score,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_tag_filter_single_tag() {
        let tags = vec!["mytag".to_string()];
        let (clause, vals) = build_tag_filter(&tags, 4, "d");
        assert_eq!(clause, "d.container_tags LIKE ?4");
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn test_build_tag_filter_multiple_tags() {
        let tags = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let (clause, vals) = build_tag_filter(&tags, 3, "d");
        assert_eq!(
            clause,
            "d.container_tags LIKE ?3 OR d.container_tags LIKE ?4 OR d.container_tags LIKE ?5"
        );
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_build_tag_filter_injection_payload_is_literal() {
        let injection = "'; DROP TABLE documents; --".to_string();
        let tags = vec![injection.clone()];
        let (clause, vals) = build_tag_filter(&tags, 1, "d");

        assert_eq!(clause, "d.container_tags LIKE ?1");
        assert_eq!(vals.len(), 1);
        match &vals[0] {
            libsql::Value::Text(s) => {
                assert!(
                    s.contains("DROP TABLE"),
                    "injection payload should be kept as literal text in the param value"
                );
                assert!(
                    !clause.contains("DROP"),
                    "SQL clause must not contain injection payload"
                );
            }
            other => panic!("Expected Text value, got {:?}", other),
        }
    }

    #[test]
    fn test_build_tag_filter_special_chars_preserved() {
        let tags = vec!["tag with \"quotes\" and % wildcards".to_string()];
        let (clause, vals) = build_tag_filter(&tags, 1, "x");
        assert_eq!(clause, "x.container_tags LIKE ?1");
        match &vals[0] {
            libsql::Value::Text(s) => {
                assert!(s.contains("\"quotes\""));
                assert!(s.starts_with("%\""));
            }
            other => panic!("Expected Text value, got {:?}", other),
        }
    }
}
