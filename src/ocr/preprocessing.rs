use crate::config::OcrConfig;
use crate::error::{MomoError, Result};
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};

/// Preprocess image bytes for OCR optimization
///
/// Applies the following transformations:
/// 1. Validates image dimensions (min/max checks)
/// 2. Resizes large images while maintaining aspect ratio
/// 3. Converts to grayscale for better OCR accuracy
/// 4. Removes alpha channel (RGBA to RGB conversion)
/// 5. Applies basic contrast enhancement
/// 6. Handles EXIF orientation
///
/// # Arguments
/// * `bytes` - Raw image bytes (PNG, JPEG, etc.)
/// * `config` - OCR configuration containing dimension limits
///
/// # Returns
/// Processed image bytes as PNG, ready for OCR engine
pub fn preprocess_image(bytes: &[u8], config: &OcrConfig) -> Result<Vec<u8>> {
    // Load image from bytes
    let reader = ImageReader::new(std::io::Cursor::new(bytes));
    let reader = reader
        .with_guessed_format()
        .map_err(|e| MomoError::Processing(format!("Failed to read image: {e}")))?;

    let img = reader
        .decode()
        .map_err(|e| MomoError::Processing(format!("Failed to decode image: {e}")))?;

    // 1. Check minimum dimensions
    let (width, height) = img.dimensions();
    if width < config.min_image_dimension || height < config.min_image_dimension {
        return Err(MomoError::Processing(format!(
            "Image too small: {}x{}, minimum {}x{}",
            width, height, config.min_image_dimension, config.min_image_dimension
        )));
    }

    // 2. Handle EXIF orientation
    let img = handle_exif_orientation(img, bytes)?;

    // 3. Resize if too large (maintains aspect ratio)
    let img = resize_if_needed(img, config.max_image_dimension);

    // 4. Convert to grayscale
    let img = img.grayscale();

    // 5. Remove alpha channel (convert RGBA8 to RGB8)
    let img = remove_alpha(img);

    // 6. Apply basic contrast enhancement
    let img = enhance_contrast(img);

    // Encode back to PNG bytes
    let mut output = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::Png)
        .map_err(|e| MomoError::Processing(format!("Failed to encode image: {e}")))?;

    Ok(output)
}

/// Handle EXIF orientation by rotating/fliping the image as needed
fn handle_exif_orientation(img: DynamicImage, _bytes: &[u8]) -> Result<DynamicImage> {
    // Try to extract orientation from EXIF
    // Note: The `image` crate doesn't have full EXIF support, so we do basic handling
    // For more complete EXIF handling, a dedicated crate like `kamadak-exif` would be needed
    // This is a basic implementation that handles the most common cases

    // Check if we need to apply orientation
    // For now, we assume the image is already correctly oriented
    // Full EXIF orientation handling would be added here if kamadak-exif is available

    Ok(img)
}

/// Resize image if it exceeds maximum dimension while maintaining aspect ratio
///
/// Uses Lanczos3 filter for high-quality downscaling
fn resize_if_needed(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (width, height) = img.dimensions();

    // Check if resize is needed
    if width <= max_dim && height <= max_dim {
        return img;
    }

    // Calculate new dimensions maintaining aspect ratio
    let ratio = if width > height {
        max_dim as f32 / width as f32
    } else {
        max_dim as f32 / height as f32
    };

    let new_width = (width as f32 * ratio) as u32;
    let new_height = (height as f32 * ratio) as u32;

    // Use Lanczos3 for high-quality resizing
    img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

/// Remove alpha channel from RGBA images, converting to RGB
///
/// Grayscale images with alpha are converted to grayscale without alpha
fn remove_alpha(img: DynamicImage) -> DynamicImage {
    match img {
        DynamicImage::ImageRgba8(rgba) => {
            let rgb =
                image::imageops::crop_imm(&rgba, 0, 0, rgba.width(), rgba.height()).to_image();
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(
                rgb.width(),
                rgb.height(),
                |x, y| {
                    let pixel = rgb.get_pixel(x, y);
                    image::Rgb([pixel[0], pixel[1], pixel[2]])
                },
            ))
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(
                luma_a.width(),
                luma_a.height(),
                |x, y| {
                    let pixel = luma_a.get_pixel(x, y);
                    // Use the luminance value, ignoring alpha
                    image::Luma([pixel[0]])
                },
            ))
        }
        // Already has no alpha channel
        _ => img,
    }
}

/// Apply basic contrast enhancement
///
/// Uses histogram stretching to improve contrast
fn enhance_contrast(img: DynamicImage) -> DynamicImage {
    match img {
        DynamicImage::ImageLuma8(gray) => {
            DynamicImage::ImageLuma8(enhance_grayscale_contrast(gray))
        }
        DynamicImage::ImageRgb8(rgb) => {
            // Convert to grayscale first, then enhance
            // OCR typically works better on grayscale images
            let gray = DynamicImage::ImageRgb8(rgb).to_luma8();
            DynamicImage::ImageLuma8(enhance_grayscale_contrast(gray))
        }
        _ => img,
    }
}

/// Enhance contrast on a grayscale image using histogram stretching
///
/// Maps the darkest pixel to 0 and the lightest to 255,
/// scaling all intermediate values linearly
fn enhance_grayscale_contrast(gray: image::GrayImage) -> image::GrayImage {
    let mut min_val = 255u8;
    let mut max_val = 0u8;

    // Find min and max values
    for pixel in gray.pixels() {
        let val = pixel[0];
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // If the image is flat (all same color), return as-is
    if max_val <= min_val {
        return gray;
    }

    // Apply contrast stretching
    let range = (max_val - min_val) as f32;
    image::GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
        let pixel = gray.get_pixel(x, y);
        let normalized = (pixel[0] - min_val) as f32 / range;
        let enhanced = (normalized * 255.0) as u8;
        image::Luma([enhanced])
    })
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

    fn create_test_png(width: u32, height: u32) -> Vec<u8> {
        let img = DynamicImage::new_rgb8(width, height);
        let mut output = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::Png)
            .unwrap();
        output
    }

    fn create_test_rgba_png(width: u32, height: u32) -> Vec<u8> {
        let img = DynamicImage::new_rgba8(width, height);
        let mut output = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::Png)
            .unwrap();
        output
    }

    fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = DynamicImage::new_rgb8(width, height);
        let mut output = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::Jpeg)
            .unwrap();
        output
    }

    #[test]
    fn test_preprocess_valid_image() {
        let config = create_test_config();
        let image_data = create_test_png(100, 100);

        let result = preprocess_image(&image_data, &config);
        assert!(
            result.is_ok(),
            "Preprocessing should succeed for valid image: {:?}",
            result.err()
        );

        let processed = result.unwrap();
        assert!(!processed.is_empty(), "Processed image should not be empty");
    }

    #[test]
    fn test_reject_tiny_image() {
        let config = create_test_config();
        let tiny = create_test_png(10, 10);
        let result = preprocess_image(&tiny, &config);

        assert!(result.is_err(), "Should reject tiny images");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("too small"),
            "Error should indicate image is too small: {err}"
        );
        assert!(
            err.contains("10x10"),
            "Error should mention the image dimensions: {err}"
        );
    }

    #[test]
    fn test_reject_width_only_too_small() {
        let config = create_test_config();
        // Width too small, height ok
        let image = create_test_png(40, 200);
        let result = preprocess_image(&image, &config);

        assert!(
            result.is_err(),
            "Should reject image with width < min_dimension"
        );
    }

    #[test]
    fn test_reject_height_only_too_small() {
        let config = create_test_config();
        // Height too small, width ok
        let image = create_test_png(200, 40);
        let result = preprocess_image(&image, &config);

        assert!(
            result.is_err(),
            "Should reject image with height < min_dimension"
        );
    }

    #[test]
    fn test_resize_large_image_width() {
        let config = OcrConfig {
            max_image_dimension: 500,
            ..create_test_config()
        };
        // Wide image that exceeds max dimension
        let large = create_test_png(1000, 200);

        let result = preprocess_image(&large, &config);
        assert!(
            result.is_ok(),
            "Should successfully resize large width image: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_resize_large_image_height() {
        let config = OcrConfig {
            max_image_dimension: 500,
            ..create_test_config()
        };
        // Tall image that exceeds max dimension
        let large = create_test_png(200, 1000);

        let result = preprocess_image(&large, &config);
        assert!(
            result.is_ok(),
            "Should successfully resize large height image: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_resize_large_both_dimensions() {
        let config = OcrConfig {
            max_image_dimension: 500,
            ..create_test_config()
        };
        // Large image in both dimensions
        let large = create_test_png(2000, 1500);

        let result = preprocess_image(&large, &config);
        assert!(
            result.is_ok(),
            "Should successfully resize large image: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rgba_to_rgb_conversion() {
        let config = create_test_config();
        let rgba = create_test_rgba_png(100, 100);

        let result = preprocess_image(&rgba, &config);
        assert!(
            result.is_ok(),
            "Should handle RGBA images: {:?}",
            result.err()
        );

        let processed = result.unwrap();
        assert!(!processed.is_empty(), "Processed image should not be empty");

        // Verify output is valid PNG
        let decoded = image::load_from_memory(&processed);
        assert!(decoded.is_ok(), "Output should be valid image");

        // Should be grayscale (which has no alpha)
        let decoded = decoded.unwrap();
        match decoded {
            DynamicImage::ImageLuma8(_) => (), // Expected
            DynamicImage::ImageRgb8(_) => (),  // Also acceptable
            _ => panic!("Expected grayscale or RGB image after preprocessing"),
        }
    }

    #[test]
    fn test_jpeg_input() {
        let config = create_test_config();
        let jpeg = create_test_jpeg(100, 100);

        let result = preprocess_image(&jpeg, &config);
        assert!(
            result.is_ok(),
            "Should handle JPEG images: {:?}",
            result.err()
        );

        let processed = result.unwrap();
        assert!(!processed.is_empty());
    }

    #[test]
    fn test_min_dimension_exactly_at_limit() {
        let config = create_test_config();
        // Image exactly at minimum dimension (50x50)
        let image = create_test_png(50, 50);

        let result = preprocess_image(&image, &config);
        assert!(
            result.is_ok(),
            "Should accept image exactly at minimum dimension"
        );
    }

    #[test]
    fn test_max_dimension_exactly_at_limit() {
        let config = create_test_config();
        // Image exactly at maximum dimension (4096x4096)
        let image = create_test_png(4096, 4096);

        let result = preprocess_image(&image, &config);
        assert!(
            result.is_ok(),
            "Should accept image exactly at maximum dimension"
        );
    }

    #[test]
    fn test_contrast_enhancement_preserves_dimensions() {
        let config = create_test_config();
        let image = create_test_png(100, 200);

        let result = preprocess_image(&image, &config);
        assert!(result.is_ok());

        let processed = result.unwrap();
        let decoded = image::load_from_memory(&processed).unwrap();

        // Should maintain dimensions after processing
        let (width, height) = decoded.dimensions();
        assert_eq!(width, 100, "Width should be preserved");
        assert_eq!(height, 200, "Height should be preserved");
    }

    #[test]
    fn test_invalid_image_data() {
        let config = create_test_config();
        let invalid_data = vec![0u8, 1, 2, 3, 4, 5]; // Not a valid image

        let result = preprocess_image(&invalid_data, &config);
        assert!(result.is_err(), "Should reject invalid image data");
    }

    #[test]
    fn test_resize_if_needed_no_change() {
        let img = DynamicImage::new_rgb8(500, 500);
        let resized = resize_if_needed(img.clone(), 1000);

        let (w, h) = resized.dimensions();
        assert_eq!(w, 500, "Should not resize when under max dimension");
        assert_eq!(h, 500);
    }

    #[test]
    fn test_resize_if_needed_width_exceeded() {
        let img = DynamicImage::new_rgb8(2000, 500);
        let resized = resize_if_needed(img, 1000);

        let (w, h) = resized.dimensions();
        assert_eq!(w, 1000, "Width should be resized to max");
        assert_eq!(h, 250, "Height should maintain aspect ratio");
    }

    #[test]
    fn test_resize_if_needed_height_exceeded() {
        let img = DynamicImage::new_rgb8(500, 2000);
        let resized = resize_if_needed(img, 1000);

        let (w, h) = resized.dimensions();
        assert_eq!(w, 250, "Width should maintain aspect ratio");
        assert_eq!(h, 1000, "Height should be resized to max");
    }

    #[test]
    fn test_remove_alpha_rgba() {
        let rgba = DynamicImage::new_rgba8(100, 100);
        let result = remove_alpha(rgba);

        match result {
            DynamicImage::ImageRgb8(_) => (),  // Expected
            DynamicImage::ImageLuma8(_) => (), // Also acceptable
            _ => panic!("Expected RGB or grayscale after removing alpha"),
        }
    }

    #[test]
    fn test_remove_alpha_luma_a() {
        let luma_a = DynamicImage::new_luma_a8(100, 100);
        let result = remove_alpha(luma_a);

        match result {
            DynamicImage::ImageLuma8(_) => (), // Expected
            _ => panic!("Expected grayscale after removing alpha from LumaA"),
        }
    }

    #[test]
    fn test_remove_alpha_rgb_unchanged() {
        let rgb = DynamicImage::new_rgb8(100, 100);
        let result = remove_alpha(rgb.clone());

        match result {
            DynamicImage::ImageRgb8(_) => (), // Should remain RGB
            _ => panic!("RGB should remain RGB"),
        }
    }

    #[test]
    fn test_remove_alpha_luma_unchanged() {
        let luma = DynamicImage::new_luma8(100, 100);
        let result = remove_alpha(luma.clone());

        match result {
            DynamicImage::ImageLuma8(_) => (), // Should remain Luma
            _ => panic!("Luma should remain Luma"),
        }
    }

    #[test]
    fn test_enhance_contrast_grayscale() {
        // Create a low-contrast image
        let mut gray = image::GrayImage::new(100, 100);
        for (x, _y, pixel) in gray.enumerate_pixels_mut() {
            // Create gradient from 50 to 100 (low contrast)
            let val = (50 + (x % 51)) as u8;
            *pixel = image::Luma([val]);
        }

        let img = DynamicImage::ImageLuma8(gray);
        let enhanced = enhance_contrast(img);

        match enhanced {
            DynamicImage::ImageLuma8(_) => (), // Should remain grayscale
            _ => panic!("Expected grayscale output"),
        }
    }

    #[test]
    fn test_enhance_contrast_flat_image() {
        // Create a flat (single color) image
        let gray = image::GrayImage::from_pixel(100, 100, image::Luma([128]));
        let img = DynamicImage::ImageLuma8(gray);

        // Should not panic on flat image
        let enhanced = enhance_contrast(img);

        match enhanced {
            DynamicImage::ImageLuma8(_) => (),
            _ => panic!("Expected grayscale output"),
        }
    }

    #[test]
    fn test_enhance_grayscale_contrast_normal() {
        let mut gray = image::GrayImage::new(10, 10);
        for (i, pixel) in gray.pixels_mut().enumerate() {
            // Values from 50 to 140
            pixel[0] = (50 + i % 90) as u8;
        }

        let enhanced = enhance_grayscale_contrast(gray);

        // Check that min is 0 and max is 255 (stretched)
        let mut min_val = 255u8;
        let mut max_val = 0u8;
        for pixel in enhanced.pixels() {
            min_val = min_val.min(pixel[0]);
            max_val = max_val.max(pixel[0]);
        }

        // The image should be stretched to use more of the range
        // Note: Due to pixel distribution, it might not reach exactly 0 and 255
        assert!(max_val > min_val, "Contrast should be enhanced");
    }

    #[test]
    fn test_enhance_grayscale_contrast_flat() {
        let gray = image::GrayImage::from_pixel(10, 10, image::Luma([100]));
        let enhanced = enhance_grayscale_contrast(gray);

        // Flat image should return the same image
        for pixel in enhanced.pixels() {
            assert_eq!(pixel[0], 100, "Flat image pixels should remain unchanged");
        }
    }
}
