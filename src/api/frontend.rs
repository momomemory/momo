use axum::body::Body;
use axum::extract::Path;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "frontend/dist"]
struct FrontendAssets;

pub async fn serve_root() -> Response {
    serve_asset_path("").await
}

pub async fn serve_path(Path(path): Path<String>) -> Response {
    serve_asset_path(&path).await
}

async fn serve_asset_path(path: &str) -> Response {
    let requested = path.trim_start_matches('/');
    let target = if requested.is_empty() {
        "index.html"
    } else {
        requested
    };

    if target.contains("..") {
        return StatusCode::BAD_REQUEST.into_response();
    }

    if let Some(response) = response_for_file(target) {
        return response;
    }

    // SPA fallback for non-file routes (e.g. "/graph" deep links)
    if !target.contains('.') {
        if let Some(response) = response_for_file("index.html") {
            return response;
        }
    }

    StatusCode::NOT_FOUND.into_response()
}

fn response_for_file(path: &str) -> Option<Response> {
    let file = FrontendAssets::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();

    let mut response = Response::new(Body::from(file.data.into_owned()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime.as_ref()).ok()?,
    );
    Some(response)
}
