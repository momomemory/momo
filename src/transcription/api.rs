use std::time::Duration;

use reqwest::{multipart, Client, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::{
    config::TranscriptionConfig,
    error::{MomoError, Result},
};

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(default)]
    segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TranscriptionSegment {
    #[serde(default)]
    id: u32,
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptionApiClient {
    client: Client,
    config: TranscriptionConfig,
}

impl TranscriptionApiClient {
    pub fn new(config: &TranscriptionConfig) -> Result<Self> {
        // Validate config
        if config.api_key.is_none() {
            return Err(MomoError::Transcription(
                "API key required for transcription API".to_string(),
            ));
        }

        let timeout = Duration::from_secs(config.timeout_secs);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| MomoError::Transcription(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            config: config.clone(),
        })
    }

    pub async fn transcribe(
        &self,
        audio_bytes: &[u8],
        file_extension: Option<&str>,
    ) -> Result<String> {
        let mut last_error: Option<MomoError> = None;
        let max_retries = 3; // Default max retries

        for attempt in 0..=max_retries {
            if attempt > 0 {
                // Exponential backoff: 100ms, 200ms, 400ms
                let delay_ms = 100 * 2_u64.pow(attempt - 1);
                debug!("Retry attempt {} after {}ms", attempt, delay_ms);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            match self.transcribe_internal(audio_bytes, file_extension).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    // Check if error is retryable
                    let retryable = matches!(
                        &e,
                        MomoError::Transcription(msg) if msg.contains("500") || msg.contains("timeout")
                    );

                    if !retryable {
                        // Non-retryable error (401, 429, etc.) - return immediately
                        return Err(e);
                    }

                    if attempt < max_retries {
                        warn!(
                            "Transcription attempt {} failed (retryable): {}",
                            attempt + 1,
                            e
                        );
                        last_error = Some(e);
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MomoError::Transcription("Transcription failed after retries".to_string())
        }))
    }

    async fn transcribe_internal(
        &self,
        audio_bytes: &[u8],
        file_extension: Option<&str>,
    ) -> Result<String> {
        // Build multipart form
        let file_name = format!("audio.{}", file_extension.unwrap_or("mp3"));
        let mime_type = self.infer_mime_type(file_extension);

        let file_part = multipart::Part::bytes(audio_bytes.to_vec())
            .file_name(file_name)
            .mime_str(&mime_type)
            .map_err(|e| MomoError::Transcription(format!("Invalid MIME type: {e}")))?;

        let form = multipart::Form::new()
            .part("file", file_part)
            .text("model", self.config.model.clone())
            .text("response_format", "json");

        // Build request
        let base_url = self.config.base_url.as_deref().unwrap_or(OPENAI_BASE_URL);
        let url = format!("{base_url}/audio/transcriptions");

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| MomoError::Transcription("API key not configured".to_string()))?;

        debug!("Sending transcription request to {}", url);

        // Send request
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    MomoError::Transcription("Request timeout".to_string())
                } else {
                    MomoError::Transcription(format!("Request failed: {e}"))
                }
            })?;

        let status = response.status();
        debug!("Transcription response status: {}", status);

        // Handle HTTP errors
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error response".to_string());

            return Err(self.map_http_error(status, &error_body));
        }

        // Parse response
        let transcription_response: TranscriptionResponse = response.json().await.map_err(|e| {
            MomoError::Transcription(format!("Failed to parse transcription response: {e}"))
        })?;

        if transcription_response.text.trim().is_empty() {
            return Err(MomoError::Transcription(
                "Transcription response contained empty text".to_string(),
            ));
        }

        Ok(transcription_response.text)
    }

    fn infer_mime_type(&self, file_extension: Option<&str>) -> String {
        match file_extension {
            Some("mp3") => "audio/mpeg",
            Some("wav") => "audio/wav",
            Some("m4a") => "audio/mp4",
            Some("ogg") => "audio/ogg",
            Some("webm") => "audio/webm",
            Some("flac") => "audio/flac",
            _ => "audio/mpeg", // Default to MP3
        }
        .to_string()
    }

    fn map_http_error(&self, status: StatusCode, error_body: &str) -> MomoError {
        match status {
            StatusCode::UNAUTHORIZED => MomoError::Transcription(format!(
                "Authentication failed (401): Invalid API key. Error: {error_body}"
            )),
            StatusCode::TOO_MANY_REQUESTS => MomoError::Transcription(format!(
                "Rate limit exceeded (429): Too many requests. Error: {error_body}"
            )),
            StatusCode::INTERNAL_SERVER_ERROR => MomoError::Transcription(format!(
                "Server error (500): The transcription service encountered an error. Error: {error_body}"
            )),
            _ => MomoError::Transcription(format!(
                "Transcription API error ({status}): {error_body}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        matchers::{header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    fn test_config() -> TranscriptionConfig {
        TranscriptionConfig {
            model: "whisper-1".to_string(),
            api_key: Some("test-api-key".to_string()),
            base_url: None,
            model_path: None,
            timeout_secs: 10,
            max_file_size: 25 * 1024 * 1024,
            max_duration_secs: 600,
        }
    }

    #[tokio::test]
    async fn test_api_client_creation() {
        let config = test_config();
        let result = TranscriptionApiClient::new(&config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_api_client_creation_no_api_key() {
        let mut config = test_config();
        config.api_key = None;

        let result = TranscriptionApiClient::new(&config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MomoError::Transcription(_)));
    }

    #[tokio::test]
    async fn test_api_multipart_encoding() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Test transcription"
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let audio_bytes = b"fake audio data";

        let result = client.transcribe(audio_bytes, Some("mp3")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Test transcription");
    }

    #[tokio::test]
    async fn test_api_auth_header() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Auth successful"
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let result = client.transcribe(b"audio", Some("mp3")).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_api_response_parsing() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Parsed response text",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Parsed"
                    },
                    {
                        "id": 1,
                        "start": 1.5,
                        "end": 3.0,
                        "text": "response text"
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let result = client.transcribe(b"audio", None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Parsed response text");
    }

    #[tokio::test]
    async fn test_api_error_401() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            })))
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let result = client.transcribe(b"audio", None).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, MomoError::Transcription(_)));
        let error_msg = format!("{error:?}");
        assert!(error_msg.contains("401"));
    }

    #[tokio::test]
    async fn test_api_error_429() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let result = client.transcribe(b"audio", None).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, MomoError::Transcription(_)));
        let error_msg = format!("{error:?}");
        assert!(error_msg.contains("429"));
    }

    #[tokio::test]
    async fn test_api_error_500() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            })))
            .mount(&mock_server)
            .await;

        let mut config = test_config();
        config.base_url = Some(mock_server.uri());

        let client = TranscriptionApiClient::new(&config).unwrap();
        let result = client.transcribe(b"audio", None).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, MomoError::Transcription(_)));
        let error_msg = format!("{error:?}");
        assert!(error_msg.contains("500"));
    }

    #[test]
    fn test_infer_mime_type() {
        let config = test_config();
        let client = TranscriptionApiClient::new(&config).unwrap();

        assert_eq!(client.infer_mime_type(Some("mp3")), "audio/mpeg");
        assert_eq!(client.infer_mime_type(Some("wav")), "audio/wav");
        assert_eq!(client.infer_mime_type(Some("m4a")), "audio/mp4");
        assert_eq!(client.infer_mime_type(None), "audio/mpeg");
    }
}
