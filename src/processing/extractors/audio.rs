use crate::config::TranscriptionConfig;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;
use crate::transcription::{AudioPreprocessor, TranscriptionProvider};

use super::ExtractedContent;

pub struct AudioExtractor;

impl AudioExtractor {
    /// Extract text from audio using transcription
    ///
    /// # Arguments
    /// * `bytes` - Raw audio bytes (MP3, WAV, M4A)
    /// * `provider` - Transcription provider instance
    /// * `config` - Transcription configuration
    ///
    /// # Returns
    /// ExtractedContent with transcribed text and metadata
    pub async fn extract(
        bytes: &[u8],
        provider: &TranscriptionProvider,
        config: &TranscriptionConfig,
    ) -> Result<ExtractedContent> {
        // Validate input
        if bytes.is_empty() {
            return Err(MomoError::Transcription("Empty audio data".to_string()));
        }

        // Check file size limit
        if bytes.len() as u64 > config.max_file_size {
            return Err(MomoError::Transcription(format!(
                "Audio file size {} exceeds limit {}",
                bytes.len(),
                config.max_file_size
            )));
        }

        // Check duration limit (before provider check so oversized audio is always rejected)
        if config.max_duration_secs > 0 {
            if let Ok((samples, sample_rate, channels)) = AudioPreprocessor::decode(bytes, None) {
                let total_frames = samples.len() / channels.max(1);
                let duration_secs = total_frames as u64 / sample_rate.max(1) as u64;
                if duration_secs > config.max_duration_secs {
                    return Err(MomoError::Transcription(format!(
                        "Audio duration {}s exceeds limit of {}s",
                        duration_secs, config.max_duration_secs
                    )));
                }
            }
        }

        // Check provider availability
        if !provider.is_available() {
            return Err(MomoError::TranscriptionUnavailable(
                "Transcription provider not available".to_string(),
            ));
        }

        // Provider handles preprocessing: Local backend decodes/resamples for Whisper,
        // API backend uploads original bytes to cloud provider.
        let text = provider.transcribe(bytes).await?;
        let word_count = text.split_whitespace().count() as i32;

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Audio,
            url: None,
            word_count,
            source_path: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TranscriptionConfig {
        TranscriptionConfig::default()
    }

    #[test]
    fn test_audio_extractor_struct_exists() {
        let _ = AudioExtractor;
    }

    #[tokio::test]
    async fn test_extract_returns_error_for_empty_audio() {
        let empty_data: Vec<u8> = vec![];
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test");

        let result = AudioExtractor::extract(&empty_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Empty"), "Error should mention empty: {err}");
    }

    #[tokio::test]
    async fn test_extract_returns_error_when_provider_unavailable() {
        let audio_data = vec![0u8; 100];
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test unavailable");

        let result = AudioExtractor::extract(&audio_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unavailable") || err.contains("not available"),
            "Error should mention unavailable: {err}"
        );
    }

    #[tokio::test]
    async fn test_extract_rejects_oversized_file() {
        let config = TranscriptionConfig {
            max_file_size: 100,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::unavailable("test");
        let large_data = vec![0u8; 200];

        let result = AudioExtractor::extract(&large_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds") || err.contains("size"),
            "Error should mention size limit: {err}"
        );
    }

    fn create_wav_bytes(duration_secs: u32, sample_rate: u32) -> Vec<u8> {
        let num_samples = duration_secs * sample_rate;
        let data_size = num_samples * 2; // 16-bit mono
        let file_size = 36 + data_size;
        let mut buf = Vec::with_capacity(file_size as usize + 8);

        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        buf.resize(buf.len() + data_size as usize, 0); // silent samples

        buf
    }

    #[tokio::test]
    async fn test_extract_rejects_audio_exceeding_duration_limit() {
        let config = TranscriptionConfig {
            max_duration_secs: 1,
            max_file_size: 100_000_000,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::unavailable("test");
        let wav_data = create_wav_bytes(5, 16000);

        let result = AudioExtractor::extract(&wav_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("duration") || err.contains("exceeds"),
            "Error should mention duration limit: {err}"
        );
    }

    #[tokio::test]
    async fn test_extract_allows_audio_within_duration_limit() {
        let config = TranscriptionConfig {
            max_duration_secs: 10,
            max_file_size: 100_000_000,
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::unavailable("test");
        let wav_data = create_wav_bytes(2, 16000);

        let result = AudioExtractor::extract(&wav_data, &provider, &config).await;

        let err = result.unwrap_err().to_string();
        assert!(
            !err.contains("duration"),
            "Should not fail on duration: {err}"
        );
    }
}
