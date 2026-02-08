use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, debug};
use whisper_rs::{WhisperContext as WhisperRsContext, WhisperContextParameters, FullParams, SamplingStrategy};

use crate::config::TranscriptionConfig;
use crate::error::{MomoError, Result};

/// Wrapper around whisper-rs WhisperContext with thread-safe access
pub struct WhisperContext {
    context: Arc<Mutex<WhisperRsContext>>,
    config: TranscriptionConfig,
}

impl WhisperContext {
    /// Create a new WhisperContext from configuration
    ///
    /// # Arguments
    /// * `config` - TranscriptionConfig with model_path and other settings
    ///
    /// # Returns
    /// * `Ok(WhisperContext)` - Successfully initialized context
    /// * `Err(MomoError::Transcription)` - Failed to load model or invalid config
    pub fn new(config: &TranscriptionConfig) -> Result<Self> {
        let model_path = config
            .model_path
            .as_ref()
            .ok_or_else(|| {
                MomoError::Transcription(
                    "model_path is required for local Whisper backend".to_string(),
                )
            })?;

        info!(
            model_path = %model_path,
            "Initializing Whisper context"
        );

        let params = WhisperContextParameters::default();

        let ctx = WhisperRsContext::new_with_params(model_path, params).map_err(|e| {
            MomoError::Transcription(format!("Failed to load Whisper model: {}", e))
        })?;

        info!("Whisper context initialized successfully");

        Ok(Self {
            context: Arc::new(Mutex::new(ctx)),
            config: config.clone(),
        })
    }

    /// Transcribe audio samples to text
    ///
    /// # Arguments
    /// * `audio_samples` - PCM audio samples (f32, 16kHz mono) normalized to [-1.0, 1.0]
    ///
    /// # Returns
    /// * `Ok(String)` - Transcribed text
    /// * `Err(MomoError::Transcription)` - Transcription failed
    ///
    /// # Notes
    /// This method uses spawn_blocking because Whisper is CPU-intensive.
    /// The audio_samples are expected to be 16kHz mono PCM in f32 format.
    pub async fn transcribe(&self, audio_samples: &[f32]) -> Result<String> {
        let samples = audio_samples.to_vec();
        let context = Arc::clone(&self.context);

        debug!(
            sample_count = samples.len(),
            duration_secs = samples.len() as f32 / 16000.0,
            "Starting transcription"
        );

        let result = tokio::task::spawn_blocking(move || {
            let mut ctx = context.blocking_lock();

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_print_progress(false);
            params.set_print_realtime(false);

            let mut state = ctx
                .create_state()
                .map_err(|e| MomoError::Transcription(format!("Failed to create Whisper state: {}", e)))?;

            state
                .full(params, &samples)
                .map_err(|e| MomoError::Transcription(format!("Transcription failed: {}", e)))?;

            let num_segments = state.full_n_segments();
            if num_segments < 0 {
                return Err(MomoError::Transcription("Invalid segment count".to_string()));
            }

            let mut transcript = String::new();
            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    match segment.to_str() {
                        Ok(segment_text) => {
                            if !transcript.is_empty() && !transcript.ends_with(' ') {
                                transcript.push(' ');
                            }
                            transcript.push_str(segment_text);
                        }
                        Err(e) => {
                            return Err(MomoError::Transcription(format!(
                                "Failed to get text for segment {}: {}",
                                i, e
                            )));
                        }
                    }
                } else {
                    return Err(MomoError::Transcription(format!(
                        "Failed to get segment {}",
                        i
                    )));
                }
            }

            Ok::<String, MomoError>(transcript.trim().to_string())
        })
        .await
        .map_err(|e| MomoError::Transcription(format!("Transcription task panicked: {}", e)))??;

        info!(
            text_length = result.len(),
            segment_count = result.split_whitespace().count(),
            "Transcription completed"
        );

        Ok(result)
    }

    /// Get the configuration
    pub fn config(&self) -> &TranscriptionConfig {
        &self.config
    }
}

impl Clone for WhisperContext {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config(model_path: Option<&str>) -> TranscriptionConfig {
        TranscriptionConfig {
            model: "local/whisper-small".to_string(),
            api_key: None,
            base_url: None,
            model_path: model_path.map(|s| s.to_string()),
            timeout_secs: 300,
            max_file_size: 104857600,
            max_duration_secs: 7200,
        }
    }

    #[test]
    fn test_whisper_context_requires_model_path() {
        let config = create_test_config(None);
        let result = WhisperContext::new(&config);
        assert!(result.is_err());
        assert!(matches!(result, Err(MomoError::Transcription(_))));
    }

    #[test]
    fn test_whisper_context_invalid_model_path() {
        let config = create_test_config(Some("/nonexistent/model.bin"));
        let result = WhisperContext::new(&config);
        assert!(result.is_err());
        assert!(matches!(result, Err(MomoError::Transcription(_))));
    }

    #[tokio::test]
    async fn test_transcribe_silence() {
        let silence = vec![0.0f32; 16000];

        let config = create_test_config(Some("/nonexistent/model.bin"));
        let result = WhisperContext::new(&config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_transcribe_error_handling() {
        let config = create_test_config(Some("/nonexistent/model.bin"));
        let result = WhisperContext::new(&config);
        
        match result {
            Err(MomoError::Transcription(msg)) => {
                assert!(msg.contains("Failed to load Whisper model"));
            }
            _ => panic!("Expected Transcription error"),
        }
    }

    #[test]
    fn test_whisper_context_clone() {
        let config = create_test_config(None);
        let mock_config = create_test_config(Some("/tmp/mock"));
        assert_eq!(mock_config.model, "local/whisper-small");
    }

    #[tokio::test]
    #[ignore]
    async fn test_local_whisper_e2e() {
        
        let model_path = std::env::var("WHISPER_MODEL_PATH")
            .unwrap_or_else(|_| "./models/ggml-small.en.bin".to_string());

        let config = create_test_config(Some(&model_path));
        let context = WhisperContext::new(&config);

        if context.is_err() {
            println!("Skipping e2e test - no Whisper model available");
            return;
        }

        let context = context.unwrap();

        let sample_rate = 16000;
        let duration = 1.0;
        let frequency = 440.0;
        let mut samples = Vec::new();

        for i in 0..(sample_rate as f32 * duration) as usize {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            samples.push(sample);
        }

        let result = context.transcribe(&samples).await;
        assert!(result.is_ok());

        let text = result.unwrap();
        println!("Transcribed text: {}", text);
    }
}
