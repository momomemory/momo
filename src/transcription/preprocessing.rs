use std::io::Cursor;

use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::debug;

use crate::error::{MomoError, Result};

const TARGET_SAMPLE_RATE: u32 = 16000;
const TARGET_CHANNELS: usize = 1; // Mono

/// Audio format detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    M4a,
    Unknown,
}

impl AudioFormat {
    /// Detect format from file extension or content type
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "mp3" => Self::Mp3,
            "wav" => Self::Wav,
            "m4a" | "aac" => Self::M4a,
            _ => Self::Unknown,
        }
    }

    /// Detect format from content type (MIME type)
    pub fn from_content_type(content_type: &str) -> Self {
        match content_type.to_lowercase().as_str() {
            "audio/mpeg" | "audio/mp3" => Self::Mp3,
            "audio/wav" | "audio/wave" | "audio/x-wav" => Self::Wav,
            "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Self::M4a,
            _ => Self::Unknown,
        }
    }

    /// Detect format from file bytes (magic number detection)
    pub fn from_bytes(bytes: &[u8]) -> Self {
        // MP3 ID3v2 header is 10 bytes long. Check for ID3 tag early so
        // short ID3-only buffers are detected correctly.
        if bytes.len() >= 10 && bytes.starts_with(b"ID3") {
            return Self::Mp3;
        }

        // For other checks we require at least 12 bytes (RIFF/WAVE and ftyp checks)
        if bytes.len() < 12 {
            // Also allow MPEG frame sync detection which only needs 2 bytes
            if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
                return Self::Mp3;
            }
            return Self::Unknown;
        }

        // MP3: MPEG frame sync (0xFFEx) - checked here for buffers >= 12 bytes
        if bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
            return Self::Mp3;
        }

        // WAV: Check for RIFF header
        if bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
            return Self::Wav;
        }

        // M4A: Check for ftyp box
        if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
            // Check for M4A type indicators
            if bytes.len() >= 16 {
                let brand = &bytes[8..12];
                if brand == b"M4A " || brand == b"mp42" || brand == b"isom" {
                    return Self::M4a;
                }
            }
        }

        Self::Unknown
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::M4a => "m4a",
            Self::Unknown => "bin",
        }
    }
}

/// Audio preprocessing module for transcription
pub struct AudioPreprocessor;

impl AudioPreprocessor {
    /// Decode audio bytes and convert to f32 PCM samples
    ///
    /// Supports MP3, WAV, and M4A formats via symphonia
    ///
    /// # Arguments
    /// * `bytes` - Raw audio file bytes
    /// * `format_hint` - Optional format hint (extension or MIME type)
    ///
    /// # Returns
    /// Tuple of (samples, sample_rate, channels)
    pub fn decode(bytes: &[u8], format_hint: Option<&str>) -> Result<(Vec<f32>, u32, usize)> {
        if bytes.is_empty() {
            return Err(MomoError::Transcription("Empty audio data".to_string()));
        }

        // Create media source stream from bytes (need owned Vec for 'static lifetime)
        let bytes_owned = bytes.to_vec();
        let cursor = Cursor::new(bytes_owned);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        // Create probe hint
        let mut hint = Hint::new();
        if let Some(hint_str) = format_hint {
            // Try as extension first
            hint.with_extension(hint_str);
        }

        // Probe format
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| {
                MomoError::Transcription(format!("Failed to probe audio format: {}", e))
            })?;

        let mut format = probed.format;

        // Find the first audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| MomoError::Transcription("No audio tracks found".to_string()))?;

        // Get sample rate and channels from track
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| MomoError::Transcription("Sample rate not available".to_string()))?;

        let channels = track
            .codec_params
            .channels
            .ok_or_else(|| MomoError::Transcription("Channel info not available".to_string()))?
            .count();

        debug!(
            "Decoding audio: {} Hz, {} channels, codec: {:?}",
            sample_rate, channels, track.codec_params.codec
        );

        // Create decoder
        let decoder_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| MomoError::Transcription(format!("Failed to create decoder: {}", e)))?;

        // Decode all packets
        let mut samples = Vec::new();

        loop {
            // Get next packet
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // End of stream
                    break;
                }
                Err(e) => {
                    return Err(MomoError::Transcription(format!(
                        "Failed to read packet: {}",
                        e
                    )));
                }
            };

            // Decode packet
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Convert samples to f32
                    Self::convert_samples(&mut samples, decoded)?;
                }
                Err(symphonia::core::errors::Error::DecodeError(e)) => {
                    // Skip decode errors but continue processing
                    debug!("Decode error (skipping): {}", e);
                    continue;
                }
                Err(e) => {
                    return Err(MomoError::Transcription(format!(
                        "Failed to decode audio: {}",
                        e
                    )));
                }
            }
        }

        if samples.is_empty() {
            return Err(MomoError::Transcription(
                "No audio samples decoded".to_string(),
            ));
        }

        debug!("Decoded {} samples at {} Hz", samples.len(), sample_rate);

        Ok((samples, sample_rate, channels))
    }

    /// Convert audio buffer to f32 samples (handles all sample formats)
    fn convert_samples(samples: &mut Vec<f32>, buffer: AudioBufferRef) -> Result<()> {
        match buffer {
            AudioBufferRef::U8(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::U16(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::U24(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::U32(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::S8(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::S16(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::S24(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::S32(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
            AudioBufferRef::F32(buf) => {
                samples.extend_from_slice(buf.chan(0));
            }
            AudioBufferRef::F64(buf) => {
                for &sample in buf.chan(0) {
                    samples.push(f32::from_sample(sample));
                }
            }
        }
        Ok(())
    }

    /// Convert stereo (or multi-channel) to mono by averaging channels
    fn to_mono(samples: Vec<f32>, channels: usize) -> Vec<f32> {
        if channels == 1 {
            return samples;
        }

        let frame_count = samples.len() / channels;
        let mut mono = Vec::with_capacity(frame_count);

        for frame_idx in 0..frame_count {
            let mut sum = 0.0;
            for ch in 0..channels {
                sum += samples[frame_idx * channels + ch];
            }
            mono.push(sum / channels as f32);
        }

        debug!(
            "Converted {} channels to mono: {} frames",
            channels, frame_count
        );
        mono
    }

    /// Resample audio to 16kHz mono PCM
    ///
    /// # Arguments
    /// * `samples` - Input PCM samples
    /// * `sample_rate` - Input sample rate
    /// * `channels` - Number of input channels
    ///
    /// # Returns
    /// 16kHz mono f32 PCM samples
    pub fn resample_to_16khz_mono(
        samples: Vec<f32>,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(MomoError::Transcription(
                "Empty samples for resampling".to_string(),
            ));
        }

        // Convert to mono first
        let mono_samples = Self::to_mono(samples, channels);

        // If already at target sample rate, return as-is
        if sample_rate == TARGET_SAMPLE_RATE {
            debug!("Audio already at target sample rate (16kHz)");
            return Ok(mono_samples);
        }

        debug!(
            "Resampling from {} Hz to {} Hz",
            sample_rate, TARGET_SAMPLE_RATE
        );

        // Calculate chunk size (use 1024 frames for efficiency)
        let chunk_size = 1024.min(mono_samples.len());

        // Create FFT resampler
        let mut resampler = FftFixedIn::<f32>::new(
            sample_rate as usize,
            TARGET_SAMPLE_RATE as usize,
            chunk_size,
            2, // sub-chunks for overlap-add
            TARGET_CHANNELS,
        )
        .map_err(|e| MomoError::Transcription(format!("Failed to create resampler: {}", e)))?;

        let mut output_samples = Vec::new();
        let mut pos = 0;

        // Process in chunks
        while pos < mono_samples.len() {
            let end = (pos + chunk_size).min(mono_samples.len());
            let chunk = &mono_samples[pos..end];

            // Prepare input as Vec<Vec<f32>> (one channel)
            let input = vec![chunk.to_vec()];

            // Process chunk
            let output = if end - pos == chunk_size {
                // Full chunk
                resampler
                    .process(&input, None)
                    .map_err(|e| MomoError::Transcription(format!("Resampling failed: {}", e)))?
            } else {
                // Partial chunk at end
                let mut padded = vec![0.0; chunk_size];
                padded[..chunk.len()].copy_from_slice(chunk);
                let input_padded = vec![padded];
                let mut result = resampler
                    .process(&input_padded, None)
                    .map_err(|e| MomoError::Transcription(format!("Resampling failed: {}", e)))?;

                // Trim padding from output
                let expected_out = ((chunk.len() as f32 / sample_rate as f32)
                    * TARGET_SAMPLE_RATE as f32) as usize;
                if result[0].len() > expected_out {
                    result[0].truncate(expected_out);
                }
                result
            };

            output_samples.extend_from_slice(&output[0]);
            pos = end;
        }

        debug!(
            "Resampled {} samples to {} samples at {} Hz",
            mono_samples.len(),
            output_samples.len(),
            TARGET_SAMPLE_RATE
        );

        Ok(output_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_from_extension() {
        assert_eq!(AudioFormat::from_extension("mp3"), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_extension("MP3"), AudioFormat::Mp3);
        assert_eq!(AudioFormat::from_extension("wav"), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("WAV"), AudioFormat::Wav);
        assert_eq!(AudioFormat::from_extension("m4a"), AudioFormat::M4a);
        assert_eq!(AudioFormat::from_extension("M4A"), AudioFormat::M4a);
        assert_eq!(AudioFormat::from_extension("unknown"), AudioFormat::Unknown);
    }

    #[test]
    fn test_audio_format_from_content_type() {
        assert_eq!(
            AudioFormat::from_content_type("audio/mpeg"),
            AudioFormat::Mp3
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/mp3"),
            AudioFormat::Mp3
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/wav"),
            AudioFormat::Wav
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/wave"),
            AudioFormat::Wav
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/m4a"),
            AudioFormat::M4a
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/mp4"),
            AudioFormat::M4a
        );
        assert_eq!(
            AudioFormat::from_content_type("audio/unknown"),
            AudioFormat::Unknown
        );
    }

    #[test]
    fn test_audio_format_from_bytes_mp3() {
        // ID3 tag header
        let mp3_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00";
        eprintln!("MP3 ID3 bytes: {:02X?}", mp3_id3);
        eprintln!("Length: {}", mp3_id3.len());
        eprintln!("Starts with ID3: {}", mp3_id3.starts_with(b"ID3"));
        let result = AudioFormat::from_bytes(mp3_id3);
        eprintln!("Detected: {:?}", result);
        assert_eq!(result, AudioFormat::Mp3);

        // MPEG frame sync
        let mp3_sync = b"\xFF\xFB\x90\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(AudioFormat::from_bytes(mp3_sync), AudioFormat::Mp3);
    }

    #[test]
    fn test_audio_format_from_bytes_wav() {
        // RIFF WAVE header
        let wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        assert_eq!(AudioFormat::from_bytes(wav_header), AudioFormat::Wav);
    }

    #[test]
    fn test_audio_format_from_bytes_empty() {
        assert_eq!(AudioFormat::from_bytes(&[]), AudioFormat::Unknown);
    }

    #[test]
    fn test_audio_format_extension() {
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::M4a.extension(), "m4a");
        assert_eq!(AudioFormat::Unknown.extension(), "bin");
    }

    #[test]
    fn test_decode_empty_audio() {
        let result = AudioPreprocessor::decode(&[], None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Empty"));
    }

    #[test]
    fn test_decode_mp3() {
        // This test requires a real MP3 file. Skip if not available.
        // In practice, decode() would be tested with integration tests using real audio files.
        // For unit tests, we verify the error handling works correctly.
        let invalid_mp3 = b"\xFF\xFB\x00\x00"; // MPEG sync but truncated
        let result = AudioPreprocessor::decode(invalid_mp3, Some("mp3"));

        // Should fail gracefully (either probe failure or no samples decoded)
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_wav() {
        // Test with a minimal valid WAV file (manually constructed)
        // WAV structure: RIFF header (12) + fmt chunk (24) + data chunk header (8) + samples
        let mut wav_data = Vec::new();

        // RIFF header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(36u32 + 200).to_le_bytes()); // file size - 8
        wav_data.extend_from_slice(b"WAVE");

        // fmt chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // 1 channel (mono)
        wav_data.extend_from_slice(&16000u32.to_le_bytes()); // 16kHz sample rate
        wav_data.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&200u32.to_le_bytes()); // data size (100 samples * 2 bytes)

        // 100 samples of silence (16-bit PCM)
        for _ in 0..100 {
            wav_data.extend_from_slice(&0i16.to_le_bytes());
        }

        let result = AudioPreprocessor::decode(&wav_data, Some("wav"));
        assert!(result.is_ok(), "WAV decode failed: {:?}", result.err());

        let (samples, sample_rate, channels) = result.unwrap();
        assert!(!samples.is_empty(), "No samples decoded");
        assert_eq!(sample_rate, 16000, "Expected 16kHz sample rate");
        assert_eq!(channels, 1, "Expected mono");
        assert_eq!(samples.len(), 100, "Expected 100 samples");
    }

    #[test]
    fn test_decode_unsupported() {
        // Random bytes that don't represent any audio format
        let invalid_data = vec![0xFF; 100];
        let result = AudioPreprocessor::decode(&invalid_data, None);

        assert!(result.is_err(), "Should fail on unsupported format");
    }

    #[test]
    fn test_resample_empty() {
        let result = AudioPreprocessor::resample_to_16khz_mono(vec![], 44100, 1);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Empty"));
    }

    #[test]
    fn test_resample_to_16khz() {
        // Create test samples at 44100Hz (100ms = 4410 samples)
        let sample_rate = 44100;
        let duration_sec = 0.1;
        let num_samples = (sample_rate as f32 * duration_sec) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let result = AudioPreprocessor::resample_to_16khz_mono(samples, sample_rate, 1);

        assert!(result.is_ok(), "Resampling failed: {:?}", result.err());
        let resampled = result.unwrap();

        // Expected: duration * 16000 = 0.1 * 16000 = 1600 samples
        // Rubato FFT resampler may produce slightly different output due to:
        // 1. Chunk-based processing with overlap
        // 2. Filter delay compensation
        // 3. Edge effects at boundaries
        // Accept 15% tolerance to account for these algorithmic differences
        let expected_samples = (duration_sec * 16000.0) as usize;
        let tolerance = (expected_samples as f32 * 0.15) as usize;

        assert!(
            resampled.len() >= expected_samples.saturating_sub(tolerance)
                && resampled.len() <= expected_samples + tolerance,
            "Expected ~{} samples (±{}), got {}. Ratio: {}",
            expected_samples,
            tolerance,
            resampled.len(),
            resampled.len() as f32 / expected_samples as f32
        );
    }

    #[test]
    fn test_resample_already_16khz() {
        let samples: Vec<f32> = vec![0.0; 1000];
        let result = AudioPreprocessor::resample_to_16khz_mono(samples.clone(), 16000, 1);

        assert!(result.is_ok());
        let resampled = result.unwrap();
        assert_eq!(
            resampled.len(),
            samples.len(),
            "Should not change length when already at 16kHz"
        );
    }

    #[test]
    fn test_to_mono_already_mono() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let mono = AudioPreprocessor::to_mono(samples.clone(), 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn test_to_mono_stereo() {
        // Stereo: [L1, R1, L2, R2, L3, R3]
        let stereo = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mono = AudioPreprocessor::to_mono(stereo, 2);

        // Expected: [(1+2)/2, (3+4)/2, (5+6)/2] = [1.5, 3.5, 5.5]
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 1.5).abs() < 0.001);
        assert!((mono[1] - 3.5).abs() < 0.001);
        assert!((mono[2] - 5.5).abs() < 0.001);
    }

    #[test]
    fn test_preprocessing_module_exists() {
        let _ = AudioPreprocessor;
    }
}
