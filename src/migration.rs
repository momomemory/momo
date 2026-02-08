use std::io::{self, Write};

use crate::db::traits::DatabaseBackend;
use crate::embeddings::EmbeddingProvider;
use crate::error::Result;

pub enum MigrationDecision {
    NotNeeded,
    Approved,
    Rejected,
}

/// Check if embedding dimensions are compatible with the database.
///
/// If dimensions mismatch, either prompt the user or check force_rebuild flag.
pub async fn check_dimension_compatibility(
    db: &dyn DatabaseBackend,
    provider: &EmbeddingProvider,
    force_rebuild: bool,
) -> Result<MigrationDecision> {
    let model_dimensions = provider.dimensions();
    let stored_dimensions = db.get_embedding_dimensions().await?;

    match stored_dimensions {
        None => {
            tracing::info!(
                "Fresh database, storing embedding dimensions: {}",
                model_dimensions
            );
            db.set_embedding_dimensions(model_dimensions).await?;
            Ok(MigrationDecision::NotNeeded)
        }
        Some(db_dims) if db_dims == model_dimensions => {
            tracing::info!("Embedding dimensions match: {}", model_dimensions);
            Ok(MigrationDecision::NotNeeded)
        }
        Some(db_dims) => {
            tracing::warn!(
                "Dimension mismatch: database has {} dimensions, model produces {}",
                db_dims,
                model_dimensions
            );

            if force_rebuild {
                tracing::info!("Force rebuild flag set, proceeding with migration");
                return Ok(MigrationDecision::Approved);
            }

            print!(
                "\nEmbedding dimension mismatch detected!\n\
                 Database: {} dimensions\n\
                 Model: {} dimensions\n\n\
                 This requires re-embedding all documents.\n\
                 Proceed with migration? [y/N]: ",
                db_dims, model_dimensions
            );
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
                Ok(MigrationDecision::Approved)
            } else {
                Ok(MigrationDecision::Rejected)
            }
        }
    }
}

/// Trigger re-embedding of all documents.
///
/// This marks all documents as 'queued' and updates the stored dimensions.
/// The background pipeline will then re-embed them.
pub async fn trigger_reembedding(db: &dyn DatabaseBackend, new_dimensions: usize) -> Result<()> {
    tracing::info!(
        "Starting re-embedding migration to {} dimensions",
        new_dimensions
    );

    db.queue_all_documents_for_reprocessing().await?;

    db.delete_all_chunks().await?;

    db.set_embedding_dimensions(new_dimensions).await?;

    tracing::info!("Migration prepared: documents queued for re-embedding");

    Ok(())
}
