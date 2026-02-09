use std::sync::Arc;
use std::time::Duration;

use leptess::LepTess;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::OcrConfig;
use crate::error::{MomoError, Result};

use super::api::{DeepSeekOcrClient, MistralOcrClient, OpenAiVisionClient};

#[derive(Clone)]
enum OcrApiClient {
    Mistral(MistralOcrClient),
    DeepSeek(DeepSeekOcrClient),
    OpenAi(OpenAiVisionClient),
}

impl OcrApiClient {
    async fn ocr(&self, image_bytes: &[u8]) -> Result<String> {
        match self {
            OcrApiClient::Mistral(c) => c.ocr(image_bytes).await,
            OcrApiClient::DeepSeek(c) => c.ocr(image_bytes).await,
            OcrApiClient::OpenAi(c) => c.ocr(image_bytes).await,
        }
    }
}

enum OcrBackend {
    Local { tesseract: Arc<Mutex<LepTess>> },
    Api { client: OcrApiClient },
    Unavailable { reason: String },
}

pub struct OcrProvider {
    backend: OcrBackend,
    config: OcrConfig,
}

fn create_tesseract(languages: &str) -> std::result::Result<LepTess, String> {
    LepTess::new(None, languages).map_err(|e| e.to_string())
}

impl OcrProvider {
    pub fn new(config: &OcrConfig) -> Result<Self> {
        let model_lower = config.model.to_lowercase();
        let provider_prefix = model_lower.split('/').next().unwrap_or("local");

        let backend = match provider_prefix {
            "mistral" => match MistralOcrClient::new(config) {
                Ok(client) => {
                    info!("Mistral OCR API backend initialized");
                    OcrBackend::Api {
                        client: OcrApiClient::Mistral(client),
                    }
                }
                Err(e) => {
                    let reason = format!("Mistral OCR backend unavailable: {e}");
                    warn!("{}", reason);
                    OcrBackend::Unavailable { reason }
                }
            },
            "deepseek" => match DeepSeekOcrClient::new(config) {
                Ok(client) => {
                    info!("DeepSeek OCR API backend initialized");
                    OcrBackend::Api {
                        client: OcrApiClient::DeepSeek(client),
                    }
                }
                Err(e) => {
                    let reason = format!("DeepSeek OCR backend unavailable: {e}");
                    warn!("{}", reason);
                    OcrBackend::Unavailable { reason }
                }
            },
            "openai" => match OpenAiVisionClient::new(config) {
                Ok(client) => {
                    info!("OpenAI Vision OCR API backend initialized");
                    OcrBackend::Api {
                        client: OcrApiClient::OpenAi(client),
                    }
                }
                Err(e) => {
                    let reason = format!("OpenAI Vision OCR backend unavailable: {e}");
                    warn!("{}", reason);
                    OcrBackend::Unavailable { reason }
                }
            },
            _ => match create_tesseract(&config.languages) {
                Ok(lt) => {
                    info!(languages = %config.languages, "Tesseract OCR initialized");
                    OcrBackend::Local {
                        tesseract: Arc::new(Mutex::new(lt)),
                    }
                }
                Err(e) => {
                    let reason = format!("Tesseract not available: {e}");
                    warn!("{}", reason);
                    OcrBackend::Unavailable { reason }
                }
            },
        };

        Ok(Self {
            backend,
            config: config.clone(),
        })
    }

    pub fn is_available(&self) -> bool {
        !matches!(self.backend, OcrBackend::Unavailable { .. })
    }

    pub async fn ocr(&self, image_bytes: &[u8]) -> Result<String> {
        let timeout_duration = Duration::from_secs(self.config.timeout_secs);

        let result = tokio::time::timeout(timeout_duration, self.ocr_internal(image_bytes)).await;

        match result {
            Ok(inner_result) => inner_result,
            Err(_) => Err(MomoError::Ocr(format!(
                "OCR operation timed out after {} seconds",
                self.config.timeout_secs
            ))),
        }
    }

    async fn ocr_internal(&self, image_bytes: &[u8]) -> Result<String> {
        match &self.backend {
            OcrBackend::Local { tesseract } => {
                let bytes = image_bytes.to_vec();
                let tesseract = Arc::clone(tesseract);

                let text = tokio::task::spawn_blocking(move || {
                    let mut lt = tesseract.blocking_lock();
                    lt.set_image_from_mem(&bytes)
                        .map_err(|e| MomoError::Ocr(format!("Failed to set image: {e}")))?;
                    lt.get_utf8_text()
                        .map_err(|e| MomoError::Ocr(format!("Failed to extract text: {e}")))
                })
                .await
                .map_err(|e| MomoError::Ocr(format!("OCR task panicked: {e}")))??;

                Ok(text.trim().to_string())
            }
            OcrBackend::Api { client } => client.ocr(image_bytes).await,
            OcrBackend::Unavailable { reason } => Err(MomoError::OcrUnavailable(reason.clone())),
        }
    }
}

impl Clone for OcrProvider {
    fn clone(&self) -> Self {
        match &self.backend {
            OcrBackend::Local { tesseract } => Self {
                backend: OcrBackend::Local {
                    tesseract: Arc::clone(tesseract),
                },
                config: self.config.clone(),
            },
            OcrBackend::Api { client } => Self {
                backend: OcrBackend::Api {
                    client: client.clone(),
                },
                config: self.config.clone(),
            },
            OcrBackend::Unavailable { reason } => Self {
                backend: OcrBackend::Unavailable {
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
    fn test_ocr_provider_graceful_degradation() {
        let config = OcrConfig {
            model: "local/tesseract".to_string(),
            api_key: None,
            base_url: None,
            languages: "eng".to_string(),
            timeout_secs: 60,
            max_image_dimension: 4096,
            min_image_dimension: 50,
        };

        let result = OcrProvider::new(&config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ocr_unavailable_returns_error() {
        let provider = OcrProvider {
            backend: OcrBackend::Unavailable {
                reason: "Test unavailable".to_string(),
            },
            config: OcrConfig {
                model: "local/tesseract".to_string(),
                api_key: None,
                base_url: None,
                languages: "eng".to_string(),
                timeout_secs: 60,
                max_image_dimension: 4096,
                min_image_dimension: 50,
            },
        };

        let result = provider.ocr(&[]).await;
        assert!(matches!(result, Err(MomoError::OcrUnavailable(_))));
    }

    fn make_config(model: &str, api_key: Option<&str>) -> OcrConfig {
        OcrConfig {
            model: model.to_string(),
            api_key: api_key.map(String::from),
            base_url: None,
            languages: "eng".to_string(),
            timeout_secs: 60,
            max_image_dimension: 4096,
            min_image_dimension: 50,
        }
    }

    #[test]
    fn test_mistral_model_without_api_key_falls_back_to_unavailable() {
        let config = make_config("mistral/pixtral-12b", None);
        let provider = OcrProvider::new(&config).unwrap();
        assert!(!provider.is_available());
    }

    #[test]
    fn test_deepseek_model_without_api_key_falls_back_to_unavailable() {
        let config = make_config("deepseek/deepseek-vl", None);
        let provider = OcrProvider::new(&config).unwrap();
        assert!(!provider.is_available());
    }

    #[test]
    fn test_openai_model_without_api_key_falls_back_to_unavailable() {
        let config = make_config("openai/gpt-4o", None);
        let provider = OcrProvider::new(&config).unwrap();
        assert!(!provider.is_available());
    }

    #[test]
    fn test_api_backed_ocr_provider_clone() {
        let config = make_config("mistral/pixtral-12b", None);
        let provider = OcrProvider::new(&config).unwrap();
        let cloned = provider.clone();
        assert_eq!(provider.is_available(), cloned.is_available());
    }

    #[test]
    fn test_local_model_routes_to_tesseract() {
        let config = make_config("local/tesseract", None);
        let provider = OcrProvider::new(&config).unwrap();
        let _ = provider.is_available();
    }
}
