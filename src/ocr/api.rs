use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::OcrConfig;
use crate::error::{MomoError, Result};

#[derive(Clone, Debug)]
pub struct MistralOcrClient {
    client: Client,
    api_key: String,
    base_url: String,
    timeout_secs: u64,
}

#[derive(Clone, Debug)]
pub struct DeepSeekOcrClient {
    client: Client,
    api_key: String,
    base_url: String,
    timeout_secs: u64,
}

#[derive(Clone, Debug)]
pub struct OpenAiVisionClient {
    client: Client,
    api_key: String,
    base_url: String,
    timeout_secs: u64,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: Vec<ContentPart>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

impl MistralOcrClient {
    pub fn new(config: &OcrConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| MomoError::Ocr("API key required for Mistral OCR".to_string()))?;

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.mistral.ai/v1".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MomoError::Ocr(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            base_url,
            timeout_secs: config.timeout_secs,
        })
    }

    pub async fn ocr(&self, image_bytes: &[u8]) -> Result<String> {
        let base64_image = STANDARD.encode(image_bytes);
        let data_url = format!("data:image/png;base64,{}", base64_image);

        let request = ChatRequest {
            model: "pixtral-12b-2409".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: vec![
                    ContentPart::Text {
                        text: "Extract all text from this image. Return only the extracted text without any explanations or formatting.".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl { url: data_url },
                    },
                ],
            }],
            max_tokens: 4096,
        };

        self.make_request(&request).await
    }

    async fn make_request(&self, request: &ChatRequest) -> Result<String> {
        let mut retries = 0;
        let max_retries = 3;

        loop {
            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let chat_response: ChatResponse = resp.json().await.map_err(|e| {
                            MomoError::Ocr(format!("Failed to parse response: {}", e))
                        })?;

                        return chat_response
                            .choices
                            .first()
                            .map(|c| c.message.content.clone())
                            .ok_or_else(|| MomoError::Ocr("No response from API".to_string()));
                    } else if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(MomoError::Ocr(format!(
                                "API request failed after {} retries: {}",
                                max_retries,
                                resp.status()
                            )));
                        }
                        let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                        tokio::time::sleep(delay).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(MomoError::Ocr(format!(
                            "API request failed: {} - {}",
                            status, body
                        )));
                    }
                }
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(MomoError::Ocr(format!(
                            "API request failed after {} retries: {}",
                            max_retries, e
                        )));
                    }
                    let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

impl DeepSeekOcrClient {
    pub fn new(config: &OcrConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| MomoError::Ocr("API key required for DeepSeek OCR".to_string()))?;

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.deepseek.com/v1".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MomoError::Ocr(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            base_url,
            timeout_secs: config.timeout_secs,
        })
    }

    pub async fn ocr(&self, image_bytes: &[u8]) -> Result<String> {
        let base64_image = STANDARD.encode(image_bytes);
        let data_url = format!("data:image/png;base64,{}", base64_image);

        let request = ChatRequest {
            model: "deepseek-vl".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: vec![
                    ContentPart::Text {
                        text: "Extract all text from this image. Return only the extracted text without any explanations or formatting.".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl { url: data_url },
                    },
                ],
            }],
            max_tokens: 4096,
        };

        self.make_request(&request).await
    }

    async fn make_request(&self, request: &ChatRequest) -> Result<String> {
        let mut retries = 0;
        let max_retries = 3;

        loop {
            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let chat_response: ChatResponse = resp.json().await.map_err(|e| {
                            MomoError::Ocr(format!("Failed to parse response: {}", e))
                        })?;

                        return chat_response
                            .choices
                            .first()
                            .map(|c| c.message.content.clone())
                            .ok_or_else(|| MomoError::Ocr("No response from API".to_string()));
                    } else if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(MomoError::Ocr(format!(
                                "API request failed after {} retries: {}",
                                max_retries,
                                resp.status()
                            )));
                        }
                        let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                        tokio::time::sleep(delay).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(MomoError::Ocr(format!(
                            "API request failed: {} - {}",
                            status, body
                        )));
                    }
                }
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(MomoError::Ocr(format!(
                            "API request failed after {} retries: {}",
                            max_retries, e
                        )));
                    }
                    let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

impl OpenAiVisionClient {
    pub fn new(config: &OcrConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| MomoError::Ocr("API key required for OpenAI Vision".to_string()))?;

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MomoError::Ocr(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            base_url,
            timeout_secs: config.timeout_secs,
        })
    }

    pub async fn ocr(&self, image_bytes: &[u8]) -> Result<String> {
        let base64_image = STANDARD.encode(image_bytes);
        let data_url = format!("data:image/png;base64,{}", base64_image);

        let request = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: vec![
                    ContentPart::Text {
                        text: "Extract all text from this image. Return only the extracted text without any explanations or formatting.".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl { url: data_url },
                    },
                ],
            }],
            max_tokens: 4096,
        };

        self.make_request(&request).await
    }

    async fn make_request(&self, request: &ChatRequest) -> Result<String> {
        let mut retries = 0;
        let max_retries = 3;

        loop {
            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let chat_response: ChatResponse = resp.json().await.map_err(|e| {
                            MomoError::Ocr(format!("Failed to parse response: {}", e))
                        })?;

                        return chat_response
                            .choices
                            .first()
                            .map(|c| c.message.content.clone())
                            .ok_or_else(|| MomoError::Ocr("No response from API".to_string()));
                    } else if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(MomoError::Ocr(format!(
                                "API request failed after {} retries: {}",
                                max_retries,
                                resp.status()
                            )));
                        }
                        let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                        tokio::time::sleep(delay).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(MomoError::Ocr(format!(
                            "API request failed: {} - {}",
                            status, body
                        )));
                    }
                }
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(MomoError::Ocr(format!(
                            "API request failed after {} retries: {}",
                            max_retries, e
                        )));
                    }
                    let delay = Duration::from_millis(100 * (2_u64.pow(retries)));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> OcrConfig {
        OcrConfig {
            model: "local/tesseract".to_string(),
            api_key: None,
            base_url: None,
            languages: "eng".to_string(),
            timeout_secs: 60,
            max_image_dimension: 4096,
            min_image_dimension: 50,
        }
    }

    #[test]
    fn test_mistral_client_requires_api_key() {
        let config = create_test_config();
        let result = MistralOcrClient::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key required"));
    }

    #[test]
    fn test_deepseek_client_requires_api_key() {
        let config = create_test_config();
        let result = DeepSeekOcrClient::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key required"));
    }

    #[test]
    fn test_openai_vision_client_requires_api_key() {
        let config = create_test_config();
        let result = OpenAiVisionClient::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key required"));
    }

    #[test]
    fn test_mistral_client_with_api_key() {
        let mut config = create_test_config();
        config.api_key = Some("test-key".to_string());
        let result = MistralOcrClient::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deepseek_client_with_api_key() {
        let mut config = create_test_config();
        config.api_key = Some("test-key".to_string());
        let result = DeepSeekOcrClient::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_openai_vision_client_with_api_key() {
        let mut config = create_test_config();
        config.api_key = Some("test-key".to_string());
        let result = OpenAiVisionClient::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_base64_encoding() {
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let encoded = STANDARD.encode(&bytes);
        assert!(!encoded.is_empty());
        assert_eq!(encoded, "/9j/4A==");
    }

    #[test]
    fn test_custom_base_url() {
        let mut config = create_test_config();
        config.api_key = Some("test-key".to_string());
        config.base_url = Some("https://custom.api.com/v1".to_string());

        let client = MistralOcrClient::new(&config).unwrap();
        assert_eq!(client.base_url, "https://custom.api.com/v1");
    }

    #[test]
    fn test_default_base_urls() {
        let mut config = create_test_config();
        config.api_key = Some("test-key".to_string());

        let mistral = MistralOcrClient::new(&config.clone()).unwrap();
        assert!(mistral.base_url.contains("mistral"));

        let deepseek = DeepSeekOcrClient::new(&config.clone()).unwrap();
        assert!(deepseek.base_url.contains("deepseek"));

        let openai = OpenAiVisionClient::new(&config).unwrap();
        assert!(openai.base_url.contains("openai"));
    }
}
