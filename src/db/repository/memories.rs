use std::collections::{HashSet, VecDeque};

use chrono::{DateTime, Utc};
use libsql::{params, Connection};

use crate::error::Result;
use crate::models::{
    CachedProfile, Document, GraphData, GraphEdge, GraphEdgeType, Memory, MemoryRelationType,
    MemorySearchHit, ProfileFact, UserProfile,
};

use super::DocumentRepository;

pub struct MemoryRepository;

impl MemoryRepository {
    pub async fn create(conn: &Connection, memory: &Memory) -> Result<()> {
        conn.execute(
            r#"
            INSERT INTO memories (
                id, memory, space_id, container_tag, version, is_latest,
                parent_memory_id, root_memory_id, memory_relations, source_count,
                is_inference, is_forgotten, is_static, forget_after, forget_reason,
                memory_type, last_accessed, confidence, metadata, created_at, updated_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21
            )
            "#,
            params![
                memory.id.clone(),
                memory.memory.clone(),
                memory.space_id.clone(),
                memory.container_tag.clone(),
                memory.version,
                memory.is_latest as i32,
                memory.parent_memory_id.clone(),
                memory.root_memory_id.clone(),
                serde_json::to_string(&memory.memory_relations)?,
                memory.source_count,
                memory.is_inference as i32,
                memory.is_forgotten as i32,
                memory.is_static as i32,
                memory.forget_after.map(|dt| dt.to_rfc3339()),
                memory.forget_reason.clone(),
                memory.memory_type.to_string(),
                memory.last_accessed.map(|dt| dt.to_rfc3339()),
                memory.confidence,
                serde_json::to_string(&memory.metadata)?,
                memory.created_at.to_rfc3339(),
                memory.updated_at.to_rfc3339(),
            ],
        )
        .await?;

        Ok(())
    }

    pub async fn get_by_id(conn: &Connection, id: &str) -> Result<Option<Memory>> {
        let mut rows = conn
            .query(
                "SELECT id, memory, space_id, container_tag, version, is_latest, 
                        parent_memory_id, root_memory_id, memory_relations, source_count,
                        is_inference, is_forgotten, is_static, forget_after, forget_reason,
                        memory_type, last_accessed, confidence, metadata, created_at, updated_at 
                 FROM memories WHERE id = ?1 AND is_forgotten = 0",
                params![id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(Self::row_to_memory(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_by_ids(conn: &Connection, ids: &[String]) -> Result<Vec<Memory>> {
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

        let sql = format!(
            "SELECT id, memory, space_id, container_tag, version, is_latest, \
                    parent_memory_id, root_memory_id, memory_relations, source_count, \
                    is_inference, is_forgotten, is_static, forget_after, forget_reason, \
                    memory_type, last_accessed, confidence, metadata, created_at, updated_at \
             FROM memories WHERE id IN ({placeholders}) AND is_forgotten = 0"
        );
        let params: Vec<libsql::Value> =
            ids.iter().map(|id| libsql::Value::from(id.clone())).collect();

        let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;
        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }
        Ok(results)
    }

    pub async fn get_by_content(
        conn: &Connection,
        content: &str,
        container_tag: &str,
    ) -> Result<Option<Memory>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, memory, space_id, container_tag, version, is_latest, 
                       parent_memory_id, root_memory_id, memory_relations, source_count,
                       is_inference, is_forgotten, is_static, forget_after, forget_reason,
                        memory_type, last_accessed, confidence, metadata, created_at, updated_at
                FROM memories 
                WHERE memory = ?1 AND container_tag = ?2 AND is_latest = 1 AND is_forgotten = 0
                "#,
                params![content, container_tag],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(Self::row_to_memory(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn update_to_not_latest(conn: &Connection, id: &str) -> Result<()> {
        conn.execute(
            "UPDATE memories SET is_latest = 0, updated_at = ?2 WHERE id = ?1",
            params![id, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    pub async fn forget(conn: &Connection, id: &str, reason: Option<&str>) -> Result<()> {
        conn.execute(
            r#"
            UPDATE memories 
            SET is_forgotten = 1, forget_reason = ?2, updated_at = ?3 
            WHERE id = ?1
            "#,
            params![id, reason, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    pub async fn update_last_accessed(conn: &Connection, id: &str) -> Result<()> {
        conn.execute(
            "UPDATE memories SET last_accessed = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, Utc::now().to_rfc3339(), Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }
    pub async fn update_last_accessed_batch(conn: &Connection, ids: &[&str]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }

        // Build placeholders for parameterized IN clause: (?1, ?2, ...)
        // libsql params! macro uses positional params starting at ?1, so we construct
        // a query with the correct number of placeholders and then pass the params array.
        let mut placeholders = String::new();
        for i in 0..ids.len() {
            if i > 0 {
                placeholders.push_str(", ");
            }
            placeholders.push_str("?");
            placeholders.push_str(&(i + 1).to_string());
        }

        let sql = format!(
            "UPDATE memories SET last_accessed = ?{ts_idx}, updated_at = ?{ts2_idx} WHERE id IN ({placeholders}) AND memory_type = 'episode'",
            placeholders = placeholders,
            ts_idx = ids.len() + 1,
            ts2_idx = ids.len() + 2
        );

        // Build params vector: ids..., timestamp, timestamp
        let now = Utc::now().to_rfc3339();
        // libsql::params! macro requires compile-time known number of args, so use params_from_iter
        let mut v: Vec<libsql::Value> = Vec::with_capacity(ids.len() + 2);
        for id in ids {
            v.push(libsql::Value::from(id.to_string()));
        }
        v.push(libsql::Value::from(now.clone()));
        v.push(libsql::Value::from(now));

        let res = conn.execute(&sql, libsql::params_from_iter(v)).await?;
        // libsql::Connection::execute returns u64 (rows affected)
        Ok(res)
    }
    pub async fn update_source_count(conn: &Connection, id: &str, new_count: i32) -> Result<()> {
        conn.execute(
            "UPDATE memories SET source_count = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, new_count, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    pub async fn update_version_chain(
        conn: &Connection,
        id: &str,
        parent_memory_id: &str,
        root_memory_id: &str,
        version: i32,
    ) -> Result<()> {
        conn.execute(
            "UPDATE memories SET parent_memory_id = ?2, root_memory_id = ?3, version = ?4, updated_at = ?5 WHERE id = ?1",
            params![id, parent_memory_id, root_memory_id, version, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    pub async fn update_embedding(
        conn: &Connection,
        memory_id: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let embedding_json = serde_json::to_string(embedding)?;

        conn.execute(
            "UPDATE memories SET embedding = vector32(?2) WHERE id = ?1",
            params![memory_id, embedding_json],
        )
        .await?;

        Ok(())
    }

    pub async fn search_similar(
        conn: &Connection,
        embedding: &[f32],
        limit: u32,
        threshold: f32,
        container_tag: Option<&str>,
        include_forgotten: bool,
    ) -> Result<Vec<MemorySearchHit>> {
        let embedding_json = serde_json::to_string(embedding)?;

        let columns = "m.id, m.memory, m.space_id, m.container_tag, m.version, m.is_latest,
                       m.parent_memory_id, m.root_memory_id, m.memory_relations, m.source_count,
                       m.is_inference, m.is_forgotten, m.is_static, m.forget_after, m.forget_reason,
                       m.memory_type, m.last_accessed, m.confidence, m.metadata, m.created_at, m.updated_at";

        let forget_after_filter = if include_forgotten {
            ""
        } else {
            "AND (m.forget_after IS NULL OR m.forget_after > datetime('now'))"
        };

        let query = if container_tag.is_some() {
            format!(
                r#"
                SELECT {},
                       1 - vector_distance_cos(m.embedding, vector32(?1)) as score
                FROM memories m
                WHERE m.embedding IS NOT NULL
                  AND m.is_latest = 1
                  AND m.is_forgotten = 0
                  AND m.container_tag = ?4
                  AND (1 - vector_distance_cos(m.embedding, vector32(?1))) >= ?2
                  {}
                ORDER BY score DESC
                LIMIT ?3
                "#,
                columns, forget_after_filter
            )
        } else {
            format!(
                r#"
                SELECT {},
                       1 - vector_distance_cos(m.embedding, vector32(?1)) as score
                FROM memories m
                WHERE m.embedding IS NOT NULL
                  AND m.is_latest = 1
                  AND m.is_forgotten = 0
                  AND (1 - vector_distance_cos(m.embedding, vector32(?1))) >= ?2
                  {}
                ORDER BY score DESC
                LIMIT ?3
                "#,
                columns, forget_after_filter
            )
        };

        let mut rows = if container_tag.is_some() {
            conn.query(
                &query,
                params![embedding_json, threshold, limit, container_tag],
            )
            .await?
        } else {
            conn.query(&query, params![embedding_json, threshold, limit])
                .await?
        };

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let memory = Self::row_to_memory(&row)?;
            let score = row.get::<f64>(21)? as f32;
            results.push(MemorySearchHit { memory, score });
        }

        Ok(results)
    }

    pub async fn get_children(conn: &Connection, parent_id: &str) -> Result<Vec<Memory>> {
        let mut rows = conn
            .query(
                "SELECT id, memory, space_id, container_tag, version, is_latest, 
                        parent_memory_id, root_memory_id, memory_relations, source_count,
                        is_inference, is_forgotten, is_static, forget_after, forget_reason,
                        memory_type, last_accessed, confidence, metadata, created_at, updated_at 
                 FROM memories WHERE parent_memory_id = ?1 ORDER BY version DESC",
                params![parent_id],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }

        Ok(results)
    }

    pub async fn get_parents(conn: &Connection, root_id: &str) -> Result<Vec<Memory>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, memory, space_id, container_tag, version, is_latest,
                       parent_memory_id, root_memory_id, memory_relations, source_count,
                       is_inference, is_forgotten, is_static, forget_after, forget_reason,
                       memory_type, last_accessed, confidence, metadata, created_at, updated_at
                FROM memories
                WHERE root_memory_id = ?1 AND is_latest = 0
                ORDER BY version ASC
                "#,
                params![root_id],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }

        Ok(results)
    }

    pub async fn get_forgetting_candidates(
        conn: &Connection,
        before: DateTime<Utc>,
    ) -> Result<Vec<Memory>> {
        let mut rows = conn
            .query(
                r#"
                SELECT id, memory, space_id, container_tag, version, is_latest,
                       parent_memory_id, root_memory_id, memory_relations, source_count,
                       is_inference, is_forgotten, is_static, forget_after, forget_reason,
                       memory_type, last_accessed, confidence, metadata, created_at, updated_at
                FROM memories
                WHERE forget_after IS NOT NULL
                  AND forget_after < ?1
                  AND is_forgotten = 0
                "#,
                params![before.to_rfc3339()],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }

        Ok(results)
    }

    pub async fn get_seed_memories(conn: &Connection, limit: usize) -> Result<Vec<Memory>> {
        // Query: memory_type IN ('fact', 'preference', 'episode'), is_inference=0, is_latest=1, is_forgotten=0
        // Order by created_at DESC, apply LIMIT
        // Note: Episodes are included here and filtered at the application layer if exclude_episodes is set
        let query = r#"
            SELECT id, memory, space_id, container_tag, version, is_latest,
                   parent_memory_id, root_memory_id, memory_relations, source_count,
                   is_inference, is_forgotten, is_static, forget_after, forget_reason,
                   memory_type, last_accessed, confidence, metadata, created_at, updated_at
            FROM memories
            WHERE memory_type IN ('fact', 'preference', 'episode')
              AND is_inference = 0
              AND is_latest = 1
              AND is_forgotten = 0
            ORDER BY created_at DESC
            LIMIT ?1
        "#;

        let mut rows = conn.query(query, params![limit as i64]).await?;
        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }

        Ok(results)
    }

    pub async fn check_inference_exists(conn: &Connection, source_ids: &[String]) -> Result<bool> {
        // Return false for empty input
        if source_ids.is_empty() {
            return Ok(false);
        }

        // Build JSON path checks: json_extract(memory_relations, '$."<id>"') = 'derives'
        let mut where_clauses: Vec<String> = Vec::new();
        let mut params_vec: Vec<libsql::Value> = Vec::new();

        for id in source_ids {
            // Protect quotes inside JSON path
            let escaped = id.replace('"', "\\\"");
            let path = format!("$.\"{}\"", escaped);
            where_clauses.push(format!("json_extract(memory_relations, ?) = 'derives'"));
            params_vec.push(libsql::Value::from(path));
        }

        // Also require the total number of 'derives' relations equals the source set size.
        // Without this, a superset (inference deriving from {A, B, C}) would falsely match
        // a check for {A, B}.
        where_clauses.push(
            "(SELECT COUNT(*) FROM json_each(memory_relations) WHERE json_each.value = 'derives') = ?"
                .to_string(),
        );
        params_vec.push(libsql::Value::from(source_ids.len() as i64));

        let where_joined = where_clauses.join(" AND ");

        let sql = format!(
            "SELECT COUNT(*) FROM memories WHERE is_inference = 1 AND is_forgotten = 0 AND is_latest = 1 AND ({})",
            where_joined
        );

        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params_vec))
            .await?;
        if let Some(row) = rows.next().await? {
            let count: i32 = row.get(0)?;
            Ok(count > 0)
        } else {
            Ok(false)
        }
    }

    pub async fn get_user_profile(
        conn: &Connection,
        container_tag: &str,
        include_dynamic: bool,
        limit: u32,
    ) -> Result<UserProfile> {
        let static_query = r#"
            SELECT memory, confidence, created_at
            FROM memories
            WHERE container_tag = ?1 AND is_static = 1 AND is_latest = 1 AND is_forgotten = 0
            ORDER BY created_at DESC
            LIMIT ?2
        "#;

        let mut static_rows = conn
            .query(static_query, params![container_tag, limit])
            .await?;

        let mut static_facts = Vec::new();
        while let Some(row) = static_rows.next().await? {
            static_facts.push(ProfileFact {
                memory: row.get(0)?,
                confidence: row.get::<Option<f64>>(1)?,
                created_at: DateTime::parse_from_rfc3339(&row.get::<String>(2)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }

        let mut dynamic_facts = Vec::new();
        if include_dynamic {
            let dynamic_query = r#"
                SELECT memory, confidence, created_at
                FROM memories
                WHERE container_tag = ?1 AND is_static = 0 AND is_latest = 1 AND is_forgotten = 0
                ORDER BY created_at DESC
                LIMIT ?2
            "#;

            let mut dynamic_rows = conn
                .query(dynamic_query, params![container_tag, limit])
                .await?;

            while let Some(row) = dynamic_rows.next().await? {
                dynamic_facts.push(ProfileFact {
                    memory: row.get(0)?,
                    confidence: row.get::<Option<f64>>(1)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<String>(2)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                });
            }
        }

        let count_query = r#"
            SELECT COUNT(*) FROM memories
            WHERE container_tag = ?1 AND is_latest = 1 AND is_forgotten = 0
        "#;
        let mut count_rows = conn.query(count_query, params![container_tag]).await?;
        let total_memories: i32 = if let Some(row) = count_rows.next().await? {
            row.get(0)?
        } else {
            0
        };

        let last_updated_query = r#"
            SELECT MAX(updated_at) FROM memories
            WHERE container_tag = ?1 AND is_latest = 1 AND is_forgotten = 0
        "#;
        let mut last_rows = conn
            .query(last_updated_query, params![container_tag])
            .await?;
        let last_updated: DateTime<Utc> = if let Some(row) = last_rows.next().await? {
            row.get::<Option<String>>(0)?
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now)
        } else {
            Utc::now()
        };

        Ok(UserProfile {
            container_tag: container_tag.to_string(),
            narrative: None,
            static_facts,
            dynamic_facts,
            total_memories,
            last_updated,
        })
    }

    pub async fn update_relations(
        conn: &Connection,
        id: &str,
        new_relations: std::collections::HashMap<String, crate::models::MemoryRelationType>,
    ) -> Result<()> {
        let existing = Self::get_by_id(conn, id).await?;
        let memory = existing.ok_or_else(|| {
            crate::error::MomoError::NotFound(format!("Memory not found: {}", id))
        })?;

        let mut merged_relations = memory.memory_relations.clone();
        for (related_id, relation_type) in new_relations {
            merged_relations.entry(related_id).or_insert(relation_type);
        }

        let relations_json = serde_json::to_string(&merged_relations)?;
        conn.execute(
            "UPDATE memories SET memory_relations = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, relations_json, Utc::now().to_rfc3339()],
        )
        .await?;

        Ok(())
    }

    pub async fn add_relation(
        conn: &Connection,
        id: &str,
        related_id: &str,
        relation_type: crate::models::MemoryRelationType,
    ) -> Result<()> {
        let mut relations = std::collections::HashMap::new();
        relations.insert(related_id.to_string(), relation_type);
        Self::update_relations(conn, id, relations).await
    }

    fn relation_to_edge_type(relation: &MemoryRelationType) -> GraphEdgeType {
        match relation {
            MemoryRelationType::Updates => GraphEdgeType::Updates,
            MemoryRelationType::Extends => GraphEdgeType::RelatesTo,
            MemoryRelationType::Derives => GraphEdgeType::DerivedFrom,
        }
    }

    // get_memories_by_ids removed: function was unused. If needed in future, reintroduce with
    // parameterized query using libsql::params_from_iter to avoid SQL injection.

    async fn get_memories_referencing(conn: &Connection, target_id: &str) -> Result<Vec<Memory>> {
        let path = format!("$.\"{}\"", target_id.replace('"', "\\\""));
        let mut rows = conn
            .query(
                r#"SELECT id, memory, space_id, container_tag, version, is_latest,
                          parent_memory_id, root_memory_id, memory_relations, source_count,
                          is_inference, is_forgotten, is_static, forget_after, forget_reason,
                          memory_type, last_accessed, confidence, metadata, created_at, updated_at
                   FROM memories
                   WHERE json_extract(memory_relations, ?1) IS NOT NULL
                     AND is_forgotten = 0"#,
                params![path],
            )
            .await?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            results.push(Self::row_to_memory(&row)?);
        }

        Ok(results)
    }

    async fn get_source_documents(conn: &Connection, memory_ids: &[String]) -> Result<(Vec<Document>, Vec<GraphEdge>)> {
        if memory_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut placeholders = String::new();
        for i in 0..memory_ids.len() {
            if i > 0 {
                placeholders.push_str(", ");
            }
            placeholders.push('?');
            placeholders.push_str(&(i + 1).to_string());
        }

        let sql = format!(
            "SELECT memory_id, document_id FROM memory_sources WHERE memory_id IN ({})",
            placeholders
        );

        let params: Vec<libsql::Value> = memory_ids.iter().map(|id| libsql::Value::from(id.clone())).collect();
        let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;

        let mut edges = Vec::new();
        let mut doc_ids = HashSet::new();

        while let Some(row) = rows.next().await? {
            let mem_id: String = row.get(0)?;
            let doc_id: String = row.get(1)?;
            doc_ids.insert(doc_id.clone());
            edges.push(GraphEdge::new(mem_id, doc_id, GraphEdgeType::Sources));
        }

        let doc_id_vec: Vec<String> = doc_ids.into_iter().collect();
        let mut documents = Vec::new();
        for doc_id in &doc_id_vec {
            if let Some(doc) = DocumentRepository::get_by_id(conn, doc_id).await? {
                documents.push(doc);
            }
        }

        Ok((documents, edges))
    }

    pub async fn get_graph_neighborhood(
        conn: &Connection,
        id: &str,
        depth: u32,
        max_nodes: u32,
        relation_types: Option<&[GraphEdgeType]>,
    ) -> Result<GraphData> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut all_memories: Vec<Memory> = Vec::new();
        let mut all_edges: Vec<GraphEdge> = Vec::new();

        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((id.to_string(), 0));

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if visited.contains(&current_id) {
                continue;
            }
            if visited.len() >= max_nodes as usize {
                break;
            }

            visited.insert(current_id.clone());

            let memory = match Self::get_by_id(conn, &current_id).await? {
                Some(m) => m,
                None => continue,
            };

            for (related_id, relation_type) in &memory.memory_relations {
                let edge_type = Self::relation_to_edge_type(relation_type);

                // Skip edges that don't match the requested relation types
                if let Some(ref types) = relation_types {
                    if !types.contains(&edge_type) {
                        continue;
                    }
                }

                all_edges.push(GraphEdge::new(
                    current_id.clone(),
                    related_id.clone(),
                    edge_type,
                ));

                if current_depth < depth && !visited.contains(related_id) {
                    queue.push_back((related_id.clone(), current_depth + 1));
                }
            }

            if current_depth < depth {
                let referencing = Self::get_memories_referencing(conn, &current_id).await?;
                for ref_memory in &referencing {
                    if !visited.contains(&ref_memory.id) {
                        if let Some(rel_type) = ref_memory.memory_relations.get(&current_id) {
                            let edge_type = Self::relation_to_edge_type(rel_type);
                            if let Some(ref types) = relation_types {
                                if !types.contains(&edge_type) {
                                    continue;
                                }
                            }
                            all_edges.push(GraphEdge::new(
                                ref_memory.id.clone(),
                                current_id.clone(),
                                edge_type,
                            ));
                        }
                        queue.push_back((ref_memory.id.clone(), current_depth + 1));
                    }
                }
            }

            all_memories.push(memory);
        }

        let memory_ids: Vec<String> = all_memories.iter().map(|m| m.id.clone()).collect();
        let (documents, doc_edges) = Self::get_source_documents(conn, &memory_ids).await?;
        all_edges.extend(doc_edges);

        Ok(GraphData {
            memories: all_memories,
            edges: all_edges,
            documents,
        })
    }

    pub async fn get_container_graph(
        conn: &Connection,
        container_tag: &str,
        max_nodes: u32,
    ) -> Result<GraphData> {
        let mut rows = conn
            .query(
                r#"SELECT id, memory, space_id, container_tag, version, is_latest,
                          parent_memory_id, root_memory_id, memory_relations, source_count,
                          is_inference, is_forgotten, is_static, forget_after, forget_reason,
                          memory_type, last_accessed, confidence, metadata, created_at, updated_at
                   FROM memories
                   WHERE container_tag = ?1 AND is_latest = 1 AND is_forgotten = 0
                   ORDER BY created_at DESC
                   LIMIT ?2"#,
                params![container_tag, max_nodes],
            )
            .await?;

        let mut memories = Vec::new();
        while let Some(row) = rows.next().await? {
            memories.push(Self::row_to_memory(&row)?);
        }

        let memory_id_set: HashSet<String> = memories.iter().map(|m| m.id.clone()).collect();
        let mut edges = Vec::new();

        for memory in &memories {
            for (related_id, relation_type) in &memory.memory_relations {
                if memory_id_set.contains(related_id) {
                    edges.push(GraphEdge::new(
                        memory.id.clone(),
                        related_id.clone(),
                        Self::relation_to_edge_type(relation_type),
                    ));
                }
            }
        }

        let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        let (documents, doc_edges) = Self::get_source_documents(conn, &memory_ids).await?;
        edges.extend(doc_edges);

        Ok(GraphData {
            memories,
            edges,
            documents,
        })
    }

    /// Fetch a cached user profile from the `user_profiles` table.
    /// Returns `None` if no cache entry exists for this container_tag.
    pub async fn get_cached_profile(
        conn: &Connection,
        container_tag: &str,
    ) -> Result<Option<CachedProfile>> {
        let mut rows = conn
            .query(
                "SELECT container_tag, narrative, summary, cached_at FROM user_profiles WHERE container_tag = ?1",
                params![container_tag],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(CachedProfile {
                container_tag: row.get(0)?,
                narrative: row.get(1)?,
                summary: row.get(2)?,
                cached_at: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Insert or update a cached user profile in the `user_profiles` table.
    pub async fn upsert_cached_profile(
        conn: &Connection,
        container_tag: &str,
        narrative: Option<&str>,
        summary: Option<&str>,
    ) -> Result<()> {
        let cached_at = Utc::now().to_rfc3339();
        conn.execute(
            r#"
            INSERT INTO user_profiles (container_tag, narrative, summary, cached_at)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(container_tag) DO UPDATE SET
                narrative = ?2,
                summary = ?3,
                cached_at = ?4
            "#,
            params![container_tag, narrative, summary, cached_at],
        )
        .await?;

        Ok(())
    }

    pub fn row_to_memory(row: &libsql::Row) -> Result<Memory> {
        Ok(Memory {
            id: row.get(0)?,
            memory: row.get(1)?,
            space_id: row.get(2)?,
            container_tag: row.get(3)?,
            version: row.get(4)?,
            is_latest: row.get::<i32>(5)? != 0,
            parent_memory_id: row.get(6)?,
            root_memory_id: row.get(7)?,
            memory_relations: serde_json::from_str(&row.get::<String>(8)?).unwrap_or_default(),
            source_count: row.get(9)?,
            is_inference: row.get::<i32>(10)? != 0,
            is_forgotten: row.get::<i32>(11)? != 0,
            is_static: row.get::<i32>(12)? != 0,
            forget_after: row
                .get::<Option<String>>(13)?
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            forget_reason: row.get(14)?,
            memory_type: row.get::<String>(15)?.parse().unwrap_or_default(),
            last_accessed: row
                .get::<Option<String>>(16)?
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            confidence: row.get(17)?,
            metadata: serde_json::from_str(&row.get::<String>(18)?).unwrap_or_default(),
            created_at: DateTime::parse_from_rfc3339(&row.get::<String>(19)?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&row.get::<String>(20)?)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MemoryRelationType;
    use std::collections::HashMap;

    async fn setup_test_db() -> Connection {
        let conn = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .unwrap()
            .connect()
            .unwrap();

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

        conn
    }

    #[tokio::test]
    async fn test_update_relations_adds_new_relations() {
        let conn = setup_test_db().await;

        let memory = Memory::new(
            "mem1".to_string(),
            "Test memory".to_string(),
            "space1".to_string(),
        );
        MemoryRepository::create(&conn, &memory).await.unwrap();

        let mut new_relations = HashMap::new();
        new_relations.insert("mem2".to_string(), MemoryRelationType::Updates);
        new_relations.insert("mem3".to_string(), MemoryRelationType::Extends);

        MemoryRepository::update_relations(&conn, "mem1", new_relations)
            .await
            .unwrap();

        let updated = MemoryRepository::get_by_id(&conn, "mem1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.memory_relations.len(), 2);
        assert_eq!(
            updated.memory_relations.get("mem2"),
            Some(&MemoryRelationType::Updates)
        );
        assert_eq!(
            updated.memory_relations.get("mem3"),
            Some(&MemoryRelationType::Extends)
        );
    }

    #[tokio::test]
    async fn test_update_relations_first_write_wins() {
        let conn = setup_test_db().await;

        let mut memory = Memory::new(
            "mem1".to_string(),
            "Test memory".to_string(),
            "space1".to_string(),
        );
        memory
            .memory_relations
            .insert("mem2".to_string(), MemoryRelationType::Updates);
        MemoryRepository::create(&conn, &memory).await.unwrap();

        let mut new_relations = HashMap::new();
        new_relations.insert("mem2".to_string(), MemoryRelationType::Extends);
        new_relations.insert("mem3".to_string(), MemoryRelationType::Derives);

        MemoryRepository::update_relations(&conn, "mem1", new_relations)
            .await
            .unwrap();

        let updated = MemoryRepository::get_by_id(&conn, "mem1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.memory_relations.len(), 2);
        assert_eq!(
            updated.memory_relations.get("mem2"),
            Some(&MemoryRelationType::Updates)
        );
        assert_eq!(
            updated.memory_relations.get("mem3"),
            Some(&MemoryRelationType::Derives)
        );
    }

    #[tokio::test]
    async fn test_add_relation_single() {
        let conn = setup_test_db().await;

        let memory = Memory::new(
            "mem1".to_string(),
            "Test memory".to_string(),
            "space1".to_string(),
        );
        MemoryRepository::create(&conn, &memory).await.unwrap();

        MemoryRepository::add_relation(&conn, "mem1", "mem2", MemoryRelationType::Updates)
            .await
            .unwrap();

        let updated = MemoryRepository::get_by_id(&conn, "mem1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.memory_relations.len(), 1);
        assert_eq!(
            updated.memory_relations.get("mem2"),
            Some(&MemoryRelationType::Updates)
        );
    }

    #[tokio::test]
    async fn test_update_relations_nonexistent_memory() {
        let conn = setup_test_db().await;

        let mut new_relations = HashMap::new();
        new_relations.insert("mem2".to_string(), MemoryRelationType::Updates);

        let result = MemoryRepository::update_relations(&conn, "nonexistent", new_relations).await;
        assert!(result.is_err());
    }

    async fn create_memory_with_embedding(
        conn: &Connection,
        id: &str,
        forget_after: Option<DateTime<Utc>>,
    ) -> Memory {
        let mut memory = Memory::new(
            id.to_string(),
            format!("Test memory {}", id),
            "space1".to_string(),
        );
        memory.forget_after = forget_after;

        MemoryRepository::create(conn, &memory).await.unwrap();

        // Create a simple embedding (all zeros with one 1.0 for matching)
        let mut embedding = vec![0.0f32; 384];
        embedding[0] = 1.0;

        MemoryRepository::update_embedding(conn, id, &embedding)
            .await
            .unwrap();

        memory
    }

    #[tokio::test]
    async fn test_update_last_accessed_batch_empty() {
        let conn = setup_test_db().await;
        let updated = MemoryRepository::update_last_accessed_batch(&conn, &[])
            .await
            .unwrap();
        assert_eq!(updated, 0);
    }

    #[tokio::test]
    async fn test_update_last_accessed_batch_single() {
        let conn = setup_test_db().await;

        let mut memory = Memory::new(
            "e1".to_string(),
            "Episode 1".to_string(),
            "space1".to_string(),
        );
        memory.memory_type = crate::models::MemoryType::Episode;
        MemoryRepository::create(&conn, &memory).await.unwrap();

        let updated = MemoryRepository::update_last_accessed_batch(&conn, &["e1"])
            .await
            .unwrap();
        assert_eq!(updated, 1);

        let fetched = MemoryRepository::get_by_id(&conn, "e1")
            .await
            .unwrap()
            .unwrap();
        assert!(fetched.last_accessed.is_some());
    }

    #[tokio::test]
    async fn test_update_last_accessed_batch_multiple() {
        let conn = setup_test_db().await;

        for i in 1..=3 {
            let id = format!("e{}", i);
            let mut memory =
                Memory::new(id.clone(), format!("Episode {}", i), "space1".to_string());
            memory.memory_type = crate::models::MemoryType::Episode;
            MemoryRepository::create(&conn, &memory).await.unwrap();
        }

        let ids = vec!["e1", "e2", "e3"];
        let updated = MemoryRepository::update_last_accessed_batch(&conn, &ids)
            .await
            .unwrap();
        assert_eq!(updated, 3);
    }

    #[tokio::test]
    async fn test_update_last_accessed_batch_excludes_non_episode() {
        let conn = setup_test_db().await;

        let mut episode = Memory::new(
            "ep".to_string(),
            "Episode".to_string(),
            "space1".to_string(),
        );
        episode.memory_type = crate::models::MemoryType::Episode;
        MemoryRepository::create(&conn, &episode).await.unwrap();

        let mut fact = Memory::new("f1".to_string(), "Fact".to_string(), "space1".to_string());
        fact.memory_type = crate::models::MemoryType::Fact;
        MemoryRepository::create(&conn, &fact).await.unwrap();

        let updated = MemoryRepository::update_last_accessed_batch(&conn, &["ep", "f1"])
            .await
            .unwrap();
        assert_eq!(updated, 1);

        let fetched_fact = MemoryRepository::get_by_id(&conn, "f1")
            .await
            .unwrap()
            .unwrap();
        assert!(fetched_fact.last_accessed.is_none());
    }

    #[tokio::test]
    async fn test_search_similar_excludes_expired_memories() {
        let conn = setup_test_db().await;

        // Create memory with forget_after in the past
        let past_date = Utc::now() - chrono::Duration::days(1);
        create_memory_with_embedding(&conn, "expired", Some(past_date)).await;

        // Create memory with forget_after in the future
        let future_date = Utc::now() + chrono::Duration::days(1);
        create_memory_with_embedding(&conn, "valid", Some(future_date)).await;

        // Search with query embedding that matches our test embeddings
        let query_embedding = vec![1.0f32; 384];
        let results = MemoryRepository::search_similar(
            &conn,
            &query_embedding,
            10,
            0.0,
            None,
            false, // exclude forgotten
        )
        .await
        .unwrap();

        // Should only find the valid memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, "valid");
    }

    #[tokio::test]
    async fn test_search_similar_includes_null_forget_after() {
        let conn = setup_test_db().await;

        // Create memory without forget_after (None)
        create_memory_with_embedding(&conn, "no_expiry", None).await;

        // Search
        let query_embedding = vec![1.0f32; 384];
        let results =
            MemoryRepository::search_similar(&conn, &query_embedding, 10, 0.0, None, false)
                .await
                .unwrap();

        // Should include memory without forget_after
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, "no_expiry");
    }

    #[tokio::test]
    async fn test_search_similar_include_forgotten_true_returns_expired() {
        let conn = setup_test_db().await;

        // Create memory with forget_after in the past
        let past_date = Utc::now() - chrono::Duration::days(1);
        create_memory_with_embedding(&conn, "expired", Some(past_date)).await;

        // Search with include_forgotten = true
        let query_embedding = vec![1.0f32; 384];
        let results = MemoryRepository::search_similar(
            &conn,
            &query_embedding,
            10,
            0.0,
            None,
            true, // include forgotten
        )
        .await
        .unwrap();

        // Should find the expired memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, "expired");
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_returns_expired_memories() {
        let conn = setup_test_db().await;

        // Create memory with forget_after in the past
        let past_date = Utc::now() - chrono::Duration::hours(2);
        let mut memory = Memory::new(
            "expired1".to_string(),
            "Should be forgotten".to_string(),
            "space1".to_string(),
        );
        memory.forget_after = Some(past_date);
        MemoryRepository::create(&conn, &memory).await.unwrap();

        // Query for candidates
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "expired1");
        assert_eq!(candidates[0].is_forgotten, false);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_excludes_already_forgotten() {
        let conn = setup_test_db().await;

        // Create expired memory and mark it as forgotten
        let past_date = Utc::now() - chrono::Duration::hours(2);
        let mut memory = Memory::new(
            "already_forgotten".to_string(),
            "Already forgotten".to_string(),
            "space1".to_string(),
        );
        memory.forget_after = Some(past_date);
        MemoryRepository::create(&conn, &memory).await.unwrap();
        MemoryRepository::forget(&conn, "already_forgotten", Some("test reason"))
            .await
            .unwrap();

        // Query for candidates
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        // Should not return the already-forgotten memory
        assert_eq!(candidates.len(), 0);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_excludes_null_forget_after() {
        let conn = setup_test_db().await;

        // Create memory without forget_after (None)
        let memory = Memory::new(
            "no_expiry".to_string(),
            "Never expires".to_string(),
            "space1".to_string(),
        );
        MemoryRepository::create(&conn, &memory).await.unwrap();

        // Query for candidates
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        // Should not return memories without forget_after
        assert_eq!(candidates.len(), 0);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_excludes_future_expiry() {
        let conn = setup_test_db().await;

        // Create memory with forget_after in the future
        let future_date = Utc::now() + chrono::Duration::hours(2);
        let mut memory = Memory::new(
            "future_expiry".to_string(),
            "Expires in the future".to_string(),
            "space1".to_string(),
        );
        memory.forget_after = Some(future_date);
        MemoryRepository::create(&conn, &memory).await.unwrap();

        // Query for candidates
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        // Should not return memories that expire in the future
        assert_eq!(candidates.len(), 0);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_handles_empty_results() {
        let conn = setup_test_db().await;

        // Don't create any memories
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        // Should return empty vector
        assert_eq!(candidates.len(), 0);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_returns_multiple() {
        let conn = setup_test_db().await;

        // Create multiple expired memories
        let past_date = Utc::now() - chrono::Duration::hours(2);
        for i in 1..=3 {
            let mut memory = Memory::new(
                format!("expired{}", i),
                format!("Memory {}", i),
                "space1".to_string(),
            );
            memory.forget_after = Some(past_date);
            MemoryRepository::create(&conn, &memory).await.unwrap();
        }

        // Query for candidates
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, Utc::now())
            .await
            .unwrap();

        // Should return all three
        assert_eq!(candidates.len(), 3);
    }

    #[tokio::test]
    async fn test_get_forgetting_candidates_boundary_condition() {
        let conn = setup_test_db().await;

        // Create memory with forget_after exactly at the boundary
        let boundary_time = Utc::now();
        let mut memory = Memory::new(
            "boundary".to_string(),
            "Boundary test".to_string(),
            "space1".to_string(),
        );
        memory.forget_after = Some(boundary_time);
        MemoryRepository::create(&conn, &memory).await.unwrap();

        // Query with same time - should NOT include (< not <=)
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, boundary_time)
            .await
            .unwrap();

        assert_eq!(candidates.len(), 0);

        // Query with time 1 second later - should include
        let one_second_later = boundary_time + chrono::Duration::seconds(1);
        let candidates = MemoryRepository::get_forgetting_candidates(&conn, one_second_later)
            .await
            .unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "boundary");
    }

    #[tokio::test]
    async fn test_get_seed_memories_filters_and_orders() {
        let conn = setup_test_db().await;

        // Create various memories
        let mut m1 = Memory::new("s1".to_string(), "Seed 1".to_string(), "space1".to_string());
        m1.memory_type = crate::models::MemoryType::Fact;
        m1.is_inference = false;
        m1.is_forgotten = false;
        m1.created_at = Utc::now() - chrono::Duration::hours(3);
        MemoryRepository::create(&conn, &m1).await.unwrap();

        let mut m2 = Memory::new("s2".to_string(), "Seed 2".to_string(), "space1".to_string());
        m2.memory_type = crate::models::MemoryType::Preference;
        m2.is_inference = false;
        m2.is_forgotten = false;
        m2.created_at = Utc::now() - chrono::Duration::hours(1);
        MemoryRepository::create(&conn, &m2).await.unwrap();

        // Episodes are now included (filtered at application layer if exclude_episodes is set)
        let mut e = Memory::new(
            "e1".to_string(),
            "Episode".to_string(),
            "space1".to_string(),
        );
        e.memory_type = crate::models::MemoryType::Episode;
        e.created_at = Utc::now() - chrono::Duration::hours(2);
        MemoryRepository::create(&conn, &e).await.unwrap();

        // Should be excluded: inference
        let mut inf = Memory::new(
            "i1".to_string(),
            "Inference".to_string(),
            "space1".to_string(),
        );
        inf.memory_type = crate::models::MemoryType::Fact;
        inf.is_inference = true;
        MemoryRepository::create(&conn, &inf).await.unwrap();

        // Should be excluded: forgotten
        let mut f = Memory::new(
            "f1".to_string(),
            "Forgotten".to_string(),
            "space1".to_string(),
        );
        f.memory_type = crate::models::MemoryType::Fact;
        f.is_forgotten = true;
        MemoryRepository::create(&conn, &f).await.unwrap();

        let seeds = MemoryRepository::get_seed_memories(&conn, 10)
            .await
            .unwrap();
        // Should include s2, e1, and s1 ordered by created_at DESC => s2 first, then e1, then s1
        assert_eq!(seeds.len(), 3);
        assert_eq!(seeds[0].id, "s2");
        assert_eq!(seeds[1].id, "e1");
        assert_eq!(seeds[2].id, "s1");
    }

    #[tokio::test]
    async fn test_check_inference_exists_matches_all_sources() {
        let conn = setup_test_db().await;

        // Create an inference memory that derives from two sources
        let mut inf = Memory::new(
            "inf1".to_string(),
            "Infer".to_string(),
            "space1".to_string(),
        );
        inf.is_inference = true;
        inf.memory_relations
            .insert("s1".to_string(), MemoryRelationType::Derives);
        inf.memory_relations
            .insert("s2".to_string(), MemoryRelationType::Derives);
        MemoryRepository::create(&conn, &inf).await.unwrap();

        // Create a non-matching inference
        let mut inf2 = Memory::new(
            "inf2".to_string(),
            "Infer2".to_string(),
            "space1".to_string(),
        );
        inf2.is_inference = true;
        inf2.memory_relations
            .insert("s1".to_string(), MemoryRelationType::Derives);
        MemoryRepository::create(&conn, &inf2).await.unwrap();

        let exists = MemoryRepository::check_inference_exists(
            &conn,
            &vec!["s1".to_string(), "s2".to_string()],
        )
        .await
        .unwrap();
        assert!(exists);

        let not_exists = MemoryRepository::check_inference_exists(&conn, &vec!["s3".to_string()])
            .await
            .unwrap();
        assert!(!not_exists);
    }

    #[tokio::test]
    async fn test_check_inference_exists_rejects_superset() {
        let conn = setup_test_db().await;

        let mut inf = Memory::new(
            "inf_super".to_string(),
            "Superset inference".to_string(),
            "space1".to_string(),
        );
        inf.is_inference = true;
        inf.memory_relations
            .insert("s1".to_string(), MemoryRelationType::Derives);
        inf.memory_relations
            .insert("s2".to_string(), MemoryRelationType::Derives);
        inf.memory_relations
            .insert("s3".to_string(), MemoryRelationType::Derives);
        MemoryRepository::create(&conn, &inf).await.unwrap();

        let subset_match =
            MemoryRepository::check_inference_exists(&conn, &["s1".to_string(), "s2".to_string()])
                .await
                .unwrap();
        assert!(
            !subset_match,
            "subset [s1,s2] must NOT match inference with [s1,s2,s3]"
        );

        let exact_match = MemoryRepository::check_inference_exists(
            &conn,
            &["s1".to_string(), "s2".to_string(), "s3".to_string()],
        )
        .await
        .unwrap();
        assert!(
            exact_match,
            "exact [s1,s2,s3] must match inference with [s1,s2,s3]"
        );
    }

    #[tokio::test]
    async fn test_update_version_chain_sets_fields() {
        let conn = setup_test_db().await;

        let old = Memory::new("old1".to_string(), "Old memory".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &old).await.unwrap();

        let new_mem = Memory::new("new1".to_string(), "New memory".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &new_mem).await.unwrap();

        MemoryRepository::update_version_chain(&conn, "new1", "old1", "old1", 2)
            .await
            .unwrap();

        let updated = MemoryRepository::get_by_id(&conn, "new1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.parent_memory_id.as_deref(), Some("old1"));
        assert_eq!(updated.root_memory_id.as_deref(), Some("old1"));
        assert_eq!(updated.version, 2);
    }

    #[tokio::test]
    async fn test_update_version_chain_preserves_root_across_generations() {
        let conn = setup_test_db().await;

        let v1 = Memory::new("v1".to_string(), "Version 1".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &v1).await.unwrap();

        let v2 = Memory::new("v2".to_string(), "Version 2".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &v2).await.unwrap();

        MemoryRepository::update_version_chain(&conn, "v2", "v1", "v1", 2)
            .await
            .unwrap();

        let v3 = Memory::new("v3".to_string(), "Version 3".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &v3).await.unwrap();

        let v2_mem = MemoryRepository::get_by_id(&conn, "v2").await.unwrap().unwrap();
        let root = v2_mem.root_memory_id.unwrap_or_else(|| v2_mem.id.clone());
        MemoryRepository::update_version_chain(&conn, "v3", "v2", &root, v2_mem.version + 1)
            .await
            .unwrap();

        let v3_mem = MemoryRepository::get_by_id(&conn, "v3").await.unwrap().unwrap();
        assert_eq!(v3_mem.parent_memory_id.as_deref(), Some("v2"));
        assert_eq!(v3_mem.root_memory_id.as_deref(), Some("v1"));
        assert_eq!(v3_mem.version, 3);
    }

    #[tokio::test]
    async fn test_get_by_ids_returns_matching_memories() {
        let conn = setup_test_db().await;

        for id in &["m1", "m2", "m3"] {
            let mem = Memory::new(id.to_string(), format!("Memory {}", id), "space1".to_string());
            MemoryRepository::create(&conn, &mem).await.unwrap();
        }

        let ids = vec!["m1".to_string(), "m3".to_string()];
        let results = MemoryRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert_eq!(results.len(), 2);

        let result_ids: Vec<&str> = results.iter().map(|m| m.id.as_str()).collect();
        assert!(result_ids.contains(&"m1"));
        assert!(result_ids.contains(&"m3"));
    }

    #[tokio::test]
    async fn test_get_by_ids_empty_input() {
        let conn = setup_test_db().await;

        let mem = Memory::new("m1".to_string(), "Memory 1".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &mem).await.unwrap();

        let results = MemoryRepository::get_by_ids(&conn, &[]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_ids_excludes_forgotten() {
        let conn = setup_test_db().await;

        let mem = Memory::new("m1".to_string(), "Memory 1".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &mem).await.unwrap();
        MemoryRepository::forget(&conn, "m1", Some("test")).await.unwrap();

        let mem2 = Memory::new("m2".to_string(), "Memory 2".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &mem2).await.unwrap();

        let ids = vec!["m1".to_string(), "m2".to_string()];
        let results = MemoryRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn test_get_by_ids_nonexistent_ids() {
        let conn = setup_test_db().await;

        let ids = vec!["no_such".to_string(), "missing".to_string()];
        let results = MemoryRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_ids_partial_match() {
        let conn = setup_test_db().await;

        let mem = Memory::new("m1".to_string(), "Memory 1".to_string(), "space1".to_string());
        MemoryRepository::create(&conn, &mem).await.unwrap();

        let ids = vec!["m1".to_string(), "missing".to_string()];
        let results = MemoryRepository::get_by_ids(&conn, &ids).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }
}
