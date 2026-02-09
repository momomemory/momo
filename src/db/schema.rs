use libsql::Connection;

use crate::error::Result;

pub async fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
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
        );

        CREATE INDEX IF NOT EXISTS idx_documents_custom_id ON documents(custom_id);
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

        -- Chunks table with vector embedding
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedded_content TEXT,
            position INTEGER NOT NULL,
            token_count INTEGER,
            embedding F32_BLOB(384),
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

        -- Memories table
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory TEXT NOT NULL,
            space_id TEXT NOT NULL,
            container_tag TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            is_latest INTEGER NOT NULL DEFAULT 1,
            parent_memory_id TEXT,
            root_memory_id TEXT,
            memory_relations TEXT DEFAULT '{}',
            source_count INTEGER DEFAULT 0,
            is_inference INTEGER NOT NULL DEFAULT 0,
            is_forgotten INTEGER NOT NULL DEFAULT 0,
            is_static INTEGER NOT NULL DEFAULT 0,
            forget_after TEXT,
            forget_reason TEXT,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            last_accessed TEXT,
            confidence REAL,
            metadata TEXT DEFAULT '{}',
            embedding F32_BLOB(384),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (parent_memory_id) REFERENCES memories(id),
            FOREIGN KEY (root_memory_id) REFERENCES memories(id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_space_id ON memories(space_id);
        CREATE INDEX IF NOT EXISTS idx_memories_container_tag ON memories(container_tag);
        CREATE INDEX IF NOT EXISTS idx_memories_is_latest ON memories(is_latest);
        CREATE INDEX IF NOT EXISTS idx_memories_is_forgotten ON memories(is_forgotten);
        -- Compound index for contradiction detection queries that filter by
        -- container_tag + active memories (is_latest=1, is_forgotten=0)
        CREATE INDEX IF NOT EXISTS idx_memories_container_latest_forgotten
            ON memories(container_tag, is_latest, is_forgotten);
        -- Partial index for forget_after to exclude NULL values
        CREATE INDEX IF NOT EXISTS idx_memories_forget_after ON memories(forget_after) WHERE forget_after IS NOT NULL;

        -- Memory sources linking memories to documents/chunks
        CREATE TABLE IF NOT EXISTS memory_sources (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            document_id TEXT,
            chunk_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_memory_sources_memory_id ON memory_sources(memory_id);

        -- Container tags metadata
        CREATE TABLE IF NOT EXISTS container_tags (
            tag TEXT PRIMARY KEY,
            metadata TEXT DEFAULT '{}',
            document_count INTEGER DEFAULT 0,
            memory_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- User profiles cache
        CREATE TABLE IF NOT EXISTS user_profiles (
            container_tag TEXT PRIMARY KEY,
            narrative TEXT,
            summary TEXT,
            cached_at TEXT
        );

        -- API keys for authentication
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            name TEXT,
            permissions TEXT DEFAULT '[]',
            rate_limit INTEGER DEFAULT 1000,
            is_active INTEGER NOT NULL DEFAULT 1,
            last_used_at TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);

        -- Metadata key-value store
        CREATE TABLE IF NOT EXISTS momo_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .await?;

    create_vector_indexes(conn).await?;
    migrate_memory_type_column(conn).await?;
    migrate_container_tags_llm_filter(conn).await?;

    Ok(())
}

async fn migrate_memory_type_column(conn: &Connection) -> Result<()> {
    // Check if memory_type column exists
    let column_exists: bool = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='memory_type'",
            (),
        )
        .await?
        .next()
        .await?
        .map(|row| row.get::<i64>(0).unwrap_or(0) > 0)
        .unwrap_or(false);

    if !column_exists {
        tracing::info!("Migrating memories table: adding memory_type column");
        conn.execute(
            "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact'",
            (),
        )
        .await?;
        tracing::info!("Migration complete: memory_type column added");
    }

    // Check if last_accessed column exists
    let last_accessed_exists: bool = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='last_accessed'",
            (),
        )
        .await?
        .next()
        .await?
        .map(|row| row.get::<i64>(0).unwrap_or(0) > 0)
        .unwrap_or(false);

    if !last_accessed_exists {
        tracing::info!("Migrating memories table: adding last_accessed column");
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT", ())
            .await?;
        tracing::info!("Migration complete: last_accessed column added");
    }

    // Check if confidence column exists
    let confidence_exists: bool = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='confidence'",
            (),
        )
        .await?
        .next()
        .await?
        .map(|row| row.get::<i64>(0).unwrap_or(0) > 0)
        .unwrap_or(false);

    if !confidence_exists {
        tracing::info!("Migrating memories table: adding confidence column");
        conn.execute("ALTER TABLE memories ADD COLUMN confidence REAL", ())
            .await?;
        tracing::info!("Migration complete: confidence column added");
    }

    // Ensure user_profiles table exists (for databases created before it was in the schema)
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS user_profiles (
            container_tag TEXT PRIMARY KEY,
            narrative TEXT,
            summary TEXT,
            cached_at TEXT
        )
        "#,
        (),
    )
    .await?;

    Ok(())
}

async fn migrate_container_tags_llm_filter(conn: &Connection) -> Result<()> {
    let should_llm_filter_exists: bool = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('container_tags') WHERE name='should_llm_filter'",
            (),
        )
        .await?
        .next()
        .await?
        .map(|row| row.get::<i64>(0).unwrap_or(0) > 0)
        .unwrap_or(false);

    if !should_llm_filter_exists {
        tracing::info!("Migrating container_tags table: adding should_llm_filter column");
        conn.execute(
            "ALTER TABLE container_tags ADD COLUMN should_llm_filter INTEGER DEFAULT 0",
            (),
        )
        .await?;
        tracing::info!("Migration complete: should_llm_filter column added");
    }

    let filter_prompt_exists: bool = conn
        .query(
            "SELECT COUNT(*) FROM pragma_table_info('container_tags') WHERE name='filter_prompt'",
            (),
        )
        .await?
        .next()
        .await?
        .map(|row| row.get::<i64>(0).unwrap_or(0) > 0)
        .unwrap_or(false);

    if !filter_prompt_exists {
        tracing::info!("Migrating container_tags table: adding filter_prompt column");
        conn.execute(
            "ALTER TABLE container_tags ADD COLUMN filter_prompt TEXT",
            (),
        )
        .await?;
        tracing::info!("Migration complete: filter_prompt column added");
    }

    Ok(())
}

async fn create_vector_indexes(conn: &Connection) -> Result<()> {
    let chunk_index_exists: bool = conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='chunks_embedding_idx'",
            (),
        )
        .await?
        .next()
        .await?
        .is_some();

    if !chunk_index_exists {
        if let Err(e) = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks(libsql_vector_idx(embedding))",
                (),
            )
            .await
        {
            tracing::warn!("Vector index creation failed for chunks (may already exist): {e}");
        }
    }

    let memory_index_exists: bool = conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='memories_embedding_idx'",
            (),
        )
        .await?
        .next()
        .await?
        .is_some();

    if !memory_index_exists {
        if let Err(e) = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories(libsql_vector_idx(embedding))",
                (),
            )
            .await
        {
            tracing::warn!("Vector index creation failed for memories (may already exist): {e}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsql::Builder;

    #[tokio::test]
    async fn test_container_tags_schema_with_llm_filter_fields() {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        let conn = db.connect().unwrap();

        init_schema(&conn).await.unwrap();

        let result = conn
            .query(
                "SELECT name, type, dflt_value FROM pragma_table_info('container_tags') WHERE name IN ('should_llm_filter', 'filter_prompt')",
                (),
            )
            .await
            .unwrap();

        let mut rows = Vec::new();
        let mut result_set = result;
        while let Some(row) = result_set.next().await.unwrap() {
            let name: String = row.get(0).unwrap();
            let col_type: String = row.get(1).unwrap();
            let default: String = row
                .get::<Option<String>>(2)
                .unwrap()
                .unwrap_or_else(|| "NULL".to_string());
            rows.push((name, col_type, default));
        }

        assert_eq!(
            rows.len(),
            2,
            "Expected 2 columns (should_llm_filter, filter_prompt)"
        );

        let should_llm_filter = rows.iter().find(|(name, _, _)| name == "should_llm_filter");
        assert!(
            should_llm_filter.is_some(),
            "should_llm_filter column should exist"
        );
        let (_, col_type, default) = should_llm_filter.unwrap();
        assert_eq!(
            col_type, "INTEGER",
            "should_llm_filter should be INTEGER type"
        );
        assert_eq!(default, "0", "should_llm_filter should default to 0");

        let filter_prompt = rows.iter().find(|(name, _, _)| name == "filter_prompt");
        assert!(filter_prompt.is_some(), "filter_prompt column should exist");
        let (_, col_type, default) = filter_prompt.unwrap();
        assert_eq!(col_type, "TEXT", "filter_prompt should be TEXT type");
        assert_eq!(default, "NULL", "filter_prompt should default to NULL");
    }
}
