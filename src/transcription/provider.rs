use std::time::Duration;

use tracing::{info, warn};

use crate::config::{parse_provider_model, TranscriptionConfig};
use crate::error::{MomoError, Result};

use super::api::TranscriptionApiClient;

enum TranscriptionBackend {
    Local {
        whisper: super::whisper::WhisperContext,
    },
    Api {
        client: TranscriptionApiClient,
    },
    Unavailable {
        reason: String,
    },
}

pub struct TranscriptionProvider {
    backend: TranscriptionBackend,
    config: TranscriptionConfig,
}

impl TranscriptionProvider {
    pub fn new(config: &TranscriptionConfig) -> Result<Self> {
        let (provider, _model_name) = parse_provider_model(&config.model);

        let backend = if provider.eq_ignore_ascii_case("local") {
            // Local Whisper backend
            match super::whisper::WhisperContext::new(config) {
                Ok(whisper) => {
                    info!("Local Whisper backend initialized");
                    TranscriptionBackend::Local { whisper }
                }
                Err(e) => {
                    let reason = format!("Whisper backend unavailable: {e}");
                    warn!("{}", reason);
                    TranscriptionBackend::Unavailable { reason }
                }
            }
        } else {
            // API backend (openai, openrouter, etc.)
            match TranscriptionApiClient::new(config) {
                Ok(client) => {
                    info!(provider = %provider, "Transcription API backend initialized");
                    TranscriptionBackend::Api { client }
                }
                Err(e) => {
                    let reason = format!("Transcription API backend unavailable: {e}");
                    warn!("{}", reason);
                    TranscriptionBackend::Unavailable { reason }
                }
            }
        };

        Ok(Self {
            backend,
            config: config.clone(),
        })
    }

    #[cfg(test)]
    pub fn unavailable(reason: &str) -> Self {
        Self {
            backend: TranscriptionBackend::Unavailable {
                reason: reason.to_string(),
            },
            config: TranscriptionConfig::default(),
        }
    }

    pub fn is_available(&self) -> bool {
        !matches!(self.backend, TranscriptionBackend::Unavailable { .. })
    }

    pub async fn transcribe(&self, audio_bytes: &[u8]) -> Result<String> {
        let timeout_duration = Duration::from_secs(self.config.timeout_secs);

        let result =
            tokio::time::timeout(timeout_duration, self.transcribe_internal(audio_bytes)).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => Err(MomoError::Transcription(format!(
                "Transcription timed out after {} seconds",
                self.config.timeout_secs
            ))),
        }
    }

    async fn transcribe_internal(&self, audio_bytes: &[u8]) -> Result<String> {
        match &self.backend {
            TranscriptionBackend::Local { whisper } => {
                use super::preprocessing::AudioPreprocessor;

                let (samples, sample_rate, channels) =
                    AudioPreprocessor::decode(audio_bytes, None)?;

                let pcm_samples =
                    AudioPreprocessor::resample_to_16khz_mono(samples, sample_rate, channels)?;

                whisper.transcribe(&pcm_samples).await
            }
            TranscriptionBackend::Api { client } => client.transcribe(audio_bytes, None).await,
            TranscriptionBackend::Unavailable { reason } => {
                Err(MomoError::TranscriptionUnavailable(reason.clone()))
            }
        }
    }
}

impl Clone for TranscriptionProvider {
    fn clone(&self) -> Self {
        match &self.backend {
            TranscriptionBackend::Local { whisper } => Self {
                backend: TranscriptionBackend::Local {
                    whisper: whisper.clone(),
                },
                config: self.config.clone(),
            },
            TranscriptionBackend::Api { client } => Self {
                backend: TranscriptionBackend::Api {
                    client: client.clone(),
                },
                config: self.config.clone(),
            },
            TranscriptionBackend::Unavailable { reason } => Self {
                backend: TranscriptionBackend::Unavailable {
                    reason: reason.clone(),
                },
                config: self.config.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_provider_graceful_degradation() {
        let config = TranscriptionConfig::default();
        let result = TranscriptionProvider::new(&config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transcription_unavailable_returns_error() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: None,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        let result = provider.transcribe(&[]).await;
        assert!(matches!(
            result,
            Err(MomoError::TranscriptionUnavailable(_))
        ));
    }

    #[test]
    fn test_transcription_provider_is_available() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: None,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        assert!(!provider.is_available());
    }

    #[test]
    fn test_transcription_provider_clone() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: None,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        let cloned = provider.clone();
        assert!(!cloned.is_available());
    }

    #[test]
    fn test_transcription_api_backend_no_api_key_falls_back_to_unavailable() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: None,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        assert!(!provider.is_available());
    }

    #[test]
    fn test_transcription_api_backend_with_api_key_is_available() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: Some("test-key".to_string()),
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        assert!(provider.is_available());
    }

    #[test]
    fn test_transcription_api_backend_clone() {
        let config = TranscriptionConfig {
            model: "openai/whisper-1".to_string(),
            api_key: Some("test-key".to_string()),
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::new(&config).unwrap();
        let cloned = provider.clone();
        assert!(cloned.is_available());
    }
}
