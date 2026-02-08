use crate::config::TranscriptionConfig;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;
use crate::transcription::{AudioPreprocessor, TranscriptionProvider};

use super::ExtractedContent;

pub struct VideoExtractor;

impl VideoExtractor {
    /// Extract text from video by extracting audio track and transcribing
    ///
    /// # Arguments
    /// * `bytes` - Raw video bytes (MP4, WebM, AVI, MKV)
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
        // Quick detection for YouTube URLs passed as "bytes" (e.g., when the
        // caller supplied a URL string instead of raw video bytes). Return a
        // clear, user-facing error until full YouTube handling is implemented.
        //
        // TODO: Implement YouTube transcription
        // Requirements:
        // - Extract video ID from URL
        // - Download video using reqwest or yt-dlp
        // - Extract transcript from YouTube API if available
        // - Fallback to download + transcribe approach
        // - Handle rate limiting and quota
        // - Add YOUTUBE_API_KEY env var support
        if let Ok(s) = std::str::from_utf8(bytes) {
            let s_lower = s.to_lowercase();
            if s_lower.contains("youtube.com") || s_lower.contains("youtu.be") || s_lower.contains("m.youtube.com") {
                return Err(MomoError::Transcription(
                    "YouTube URLs are not yet supported. Video transcription requires uploaded files (MP4, WebM, AVI, MKV). See TODO in code for implementation.".to_string()
                ));
            }
        }

        // Validate input
        if bytes.is_empty() {
            return Err(MomoError::Transcription("Empty video data".to_string()));
        }

        // Check file size limit
        if bytes.len() as u64 > config.max_file_size {
            return Err(MomoError::Transcription(format!(
                "Video file size {} exceeds limit {}",
                bytes.len(),
                config.max_file_size
            )));
        }

        // Check provider availability
        if !provider.is_available() {
            return Err(MomoError::TranscriptionUnavailable(
                "Transcription provider not available".to_string(),
            ));
        }

        tracing::debug!("Extracting audio from video file ({} bytes)", bytes.len());

        // Extract audio from video using symphonia
        let audio_samples = Self::extract_audio_track(bytes)?;

        tracing::debug!(
            "Extracted {} audio samples from video",
            audio_samples.len()
        );

        // Encode the extracted PCM samples as WAV bytes so the transcription
        // provider receives valid audio (not the raw video container bytes).
        let wav_bytes = Self::encode_pcm_to_wav(&audio_samples)?;

        tracing::debug!(
            "Encoded {} PCM samples to {} WAV bytes",
            audio_samples.len(),
            wav_bytes.len()
        );

        let text = provider.transcribe(&wav_bytes).await?;
        let word_count = text.split_whitespace().count() as i32;

        tracing::info!(
            "Video transcription complete: {} words from {} samples",
            word_count,
            audio_samples.len()
        );

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Video,
            url: None,
            word_count,
            source_path: None,
        })
    }

    /// Encode f32 PCM samples (16kHz mono) into a WAV byte buffer.
    fn encode_pcm_to_wav(samples: &[f32]) -> Result<Vec<u8>> {
        const SAMPLE_RATE: u32 = 16000;
        const NUM_CHANNELS: u16 = 1;
        const BITS_PER_SAMPLE: u16 = 16;
        const BYTE_RATE: u32 = SAMPLE_RATE * NUM_CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
        const BLOCK_ALIGN: u16 = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);

        let pcm_i16: Vec<i16> = samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect();

        let data_size = (pcm_i16.len() * 2) as u32;
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(44 + data_size as usize);

        // RIFF header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        // fmt sub-chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // sub-chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        buf.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
        buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        buf.extend_from_slice(&BYTE_RATE.to_le_bytes());
        buf.extend_from_slice(&BLOCK_ALIGN.to_le_bytes());
        buf.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

        // data sub-chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for sample in &pcm_i16 {
            buf.extend_from_slice(&sample.to_le_bytes());
        }

        Ok(buf)
    }

    /// Extract audio track from video file and preprocess to 16kHz mono PCM
    ///
    /// # Arguments
    /// * `bytes` - Raw video file bytes
    ///
    /// # Returns
    /// Vec<f32> of preprocessed PCM samples (16kHz mono)
    fn extract_audio_track(bytes: &[u8]) -> Result<Vec<f32>> {
        // Use AudioPreprocessor to decode video audio track
        // Symphonia supports video containers and will extract the first audio track
        let (samples, sample_rate, channels) = AudioPreprocessor::decode(bytes, None)?;

        tracing::debug!(
            "Decoded audio: {} samples at {}Hz, {} channels",
            samples.len(),
            sample_rate,
            channels
        );

        // Resample to 16kHz mono (Whisper requirement)
        let preprocessed = AudioPreprocessor::resample_to_16khz_mono(samples, sample_rate, channels)?;

        tracing::debug!(
            "Preprocessed to {} samples at 16kHz mono",
            preprocessed.len()
        );

        Ok(preprocessed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TranscriptionConfig {
        TranscriptionConfig::default()
    }

    #[test]
    fn test_video_extractor_struct_exists() {
        let _ = VideoExtractor;
    }

    #[tokio::test]
    async fn test_extract_returns_error_for_empty_video() {
        let empty_data: Vec<u8> = vec![];
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test");

        let result = VideoExtractor::extract(&empty_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Empty"), "Error should mention empty: {}", err);
    }

    #[tokio::test]
    async fn test_extract_returns_error_when_provider_unavailable() {
        let video_data = vec![0u8; 100];
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test unavailable");

        let result = VideoExtractor::extract(&video_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unavailable") || err.contains("not available"),
            "Error should mention unavailable: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_extract_rejects_oversized_file() {
        let config = TranscriptionConfig {
            max_file_size: 100, // Very small limit for testing
            ..TranscriptionConfig::default()
        };
        let provider = TranscriptionProvider::unavailable("test");
        let large_data = vec![0u8; 200]; // Larger than limit

        let result = VideoExtractor::extract(&large_data, &provider, &config).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds") || err.contains("size"),
            "Error should mention size limit: {}",
            err
        );
    }

    #[test]
    fn test_extract_audio_track_empty_data() {
        let empty_data: Vec<u8> = vec![];
        let result = VideoExtractor::extract_audio_track(&empty_data);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_youtube_url_returns_error() {
        // Simulate passing a URL string as bytes
        let url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
        let bytes = url.as_bytes();
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test");

        let result = VideoExtractor::extract(bytes, &provider, &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_youtube_error_message_correct() {
        let url = "https://youtu.be/dQw4w9WgXcQ";
        let bytes = url.as_bytes();
        let config = create_test_config();
        let provider = TranscriptionProvider::unavailable("test");

        let result = VideoExtractor::extract(bytes, &provider, &config).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        let expected = "YouTube URLs are not yet supported. Video transcription requires uploaded files (MP4, WebM, AVI, MKV). See TODO in code for implementation.";
        assert!(err.contains(expected), "Error should contain expected message: {}", err);
    }

    #[test]
    fn test_encode_pcm_to_wav_produces_valid_header() {
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let wav = VideoExtractor::encode_pcm_to_wav(&samples).unwrap();

        // WAV header: RIFF....WAVE
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");

        // Data size = 16000 samples * 2 bytes each = 32000
        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_size, 32000);

        // Total file should be 44 header + 32000 data
        assert_eq!(wav.len(), 44 + 32000);
    }

    #[test]
    fn test_encode_pcm_to_wav_clamps_values() {
        let samples = vec![-2.0, 2.0, 0.5, -0.5];
        let wav = VideoExtractor::encode_pcm_to_wav(&samples).unwrap();

        // Skip 44-byte header, read i16 samples
        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);

        assert_eq!(s0, i16::MIN + 1); // -1.0 clamped → -32767
        assert_eq!(s1, i16::MAX);     // 1.0 clamped → 32767
    }

    #[test]
    fn test_encode_pcm_to_wav_empty() {
        let wav = VideoExtractor::encode_pcm_to_wav(&[]).unwrap();
        assert_eq!(wav.len(), 44); // Header only, no data
    }
}
