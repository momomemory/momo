//! # V1 API Response Envelope & Error Contract
//!
//! Defines the canonical wire format for all v1 API responses. Every endpoint
//! returns an [`ApiResponse<T>`] envelope with three optional top-level fields:
//!
//! ```json
//! {
//!   "data": { ... },       // present on success, absent on error
//!   "meta": { "nextCursor": "...", "total": 42 },  // optional pagination
//!   "error": { "code": "not_found", "message": "..." }  // present on error, absent on success
//! }
//! ```
//!
//! ## ID Formats
//!
//! - **documentId**: nanoid, 21 characters (e.g. `"V1StGXR8_Z5jdHi6B-myT"`)
//! - **ingestionId**: UUID v4 (e.g. `"550e8400-e29b-41d4-a716-446655440000"`)
//! - **memoryId**: nanoid, 21 characters
//!
//! ## Cursor Pagination
//!
//! Cursors are opaque base64-encoded strings. Clients must not parse or
//! construct them. An invalid cursor returns `400 invalid_request`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::error::MomoError;

/// Machine-readable error code included in every error response.
///
/// Serialized as a snake_case string on the wire (e.g. `"invalid_request"`).
/// Each variant maps to a fixed HTTP status code via [`ErrorCode::status`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The request was malformed, had invalid parameters, or failed validation.
    /// HTTP 400.
    InvalidRequest,
    /// Authentication is required or the provided credentials are invalid.
    /// HTTP 401.
    Unauthorized,
    /// The requested resource does not exist. HTTP 404.
    NotFound,
    /// The request conflicts with the current state of the resource. HTTP 409.
    Conflict,
    /// An unexpected server-side error occurred. Internal details are never
    /// leaked to the client. HTTP 500.
    InternalError,
    /// The requested feature or endpoint is not implemented. HTTP 501.
    NotImplemented,
}

impl ErrorCode {
    /// Returns the HTTP status code corresponding to this error code.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidRequest => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Conflict => StatusCode::CONFLICT,
            Self::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
            Self::NotImplemented => StatusCode::NOT_IMPLEMENTED,
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest => write!(f, "invalid_request"),
            Self::Unauthorized => write!(f, "unauthorized"),
            Self::NotFound => write!(f, "not_found"),
            Self::Conflict => write!(f, "conflict"),
            Self::InternalError => write!(f, "internal_error"),
            Self::NotImplemented => write!(f, "not_implemented"),
        }
    }
}

/// Structured error payload within the API envelope.
///
/// ```json
/// { "code": "not_found", "message": "Memory mem_abc123 not found" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ApiError {
    /// Machine-readable error classification.
    pub code: ErrorCode,
    /// Human-readable description safe to display to end users.
    /// Internal implementation details are never included.
    pub message: String,
}

/// Pagination metadata included in list responses.
///
/// Field names serialize as camelCase on the wire (`nextCursor`, `total`).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResponseMeta {
    /// Opaque cursor to pass as `cursor` in the next request. `None` means
    /// there are no more results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Total number of matching items (when cheaply available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

/// Cursor-based pagination parameters accepted by list endpoints.
///
/// - `limit` defaults to 20 and is clamped to `1..=100`.
/// - `cursor` is an opaque base64 string from a previous `ResponseMeta.nextCursor`.
///   Passing an invalid cursor returns `400 invalid_request`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CursorPagination {
    /// Maximum number of items to return. Clamped to `1..=100`, defaults to 20.
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Opaque pagination cursor from a previous response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

fn default_limit() -> u32 {
    20
}

impl Default for CursorPagination {
    fn default() -> Self {
        Self {
            limit: default_limit(),
            cursor: None,
        }
    }
}

impl CursorPagination {
    /// Validate and normalize pagination parameters.
    ///
    /// - Clamps `limit` to `1..=100`.
    /// - Returns the validated struct (cursor validity is checked by the
    ///   repository layer at query time).
    #[allow(dead_code)]
    pub fn validate(mut self) -> Self {
        self.limit = self.limit.clamp(1, 100);
        self
    }
}

/// Canonical v1 API response envelope.
///
/// Every v1 endpoint returns this shape. On success, `data` is present and
/// `error` is absent. On error, `error` is present and `data` is absent.
/// `meta` is optionally present for paginated or enriched responses.
///
/// The HTTP status code is derived from the error code (on error) or
/// from the explicit status set via constructors like [`ApiResponse::created`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T: Serialize> {
    /// The response payload. Present on success, absent on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    /// Pagination or enrichment metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResponseMeta>,
    /// Error details. Present on error, absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ApiError>,

    /// HTTP status to use in the response. Not serialized on the wire.
    #[serde(skip)]
    status: StatusCode,
}

impl<T: Serialize> ApiResponse<T> {
    /// Success response with data (HTTP 200).
    pub fn success(data: T) -> Self {
        Self {
            data: Some(data),
            meta: None,
            error: None,
            status: StatusCode::OK,
        }
    }

    /// Success response with data and pagination metadata (HTTP 200).
    pub fn success_with_meta(data: T, meta: ResponseMeta) -> Self {
        Self {
            data: Some(data),
            meta: Some(meta),
            error: None,
            status: StatusCode::OK,
        }
    }

    /// Resource created response (HTTP 201).
    pub fn created(data: T) -> Self {
        Self {
            data: Some(data),
            meta: None,
            error: None,
            status: StatusCode::CREATED,
        }
    }

    /// Accepted for processing response (HTTP 202).
    ///
    /// Used when the server has accepted the request but processing is not
    /// yet complete (e.g. document ingestion queued).
    pub fn accepted(data: T) -> Self {
        Self {
            data: Some(data),
            meta: None,
            error: None,
            status: StatusCode::ACCEPTED,
        }
    }

    /// Error response. HTTP status is derived from the [`ErrorCode`].
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        let status = code.status();
        Self {
            data: None,
            meta: None,
            error: Some(ApiError {
                code,
                message: message.into(),
            }),
            status,
        }
    }
}

impl<T: Serialize> IntoResponse for ApiResponse<T> {
    fn into_response(self) -> Response {
        let status = self.status;
        match serde_json::to_value(&self) {
            Ok(body) => (status, Json(body)).into_response(),
            Err(_) => {
                let fallback = ApiResponse::<()>::error(
                    ErrorCode::InternalError,
                    "An internal error occurred",
                );
                let body = serde_json::json!({
                    "error": {
                        "code": "internal_error",
                        "message": "An internal error occurred"
                    }
                });
                (fallback.status, Json(body)).into_response()
            }
        }
    }
}

impl<T: Serialize> From<MomoError> for ApiResponse<T> {
    /// Convert a [`MomoError`] into a v1 [`ApiResponse`].
    ///
    /// Internal error details are **never** leaked to the client. For
    /// `internal_error` responses, a generic message is returned and the
    /// real error is logged via `tracing::error!`.
    fn from(err: MomoError) -> Self {
        match err {
            MomoError::NotFound(ref msg) => ApiResponse::error(ErrorCode::NotFound, msg.clone()),

            MomoError::Validation(ref msg) => {
                ApiResponse::error(ErrorCode::InvalidRequest, msg.clone())
            }

            MomoError::ApiAuth(_) => {
                ApiResponse::error(ErrorCode::Unauthorized, "Authentication required")
            }

            MomoError::Json(ref e) => {
                ApiResponse::error(ErrorCode::InvalidRequest, format!("Invalid JSON: {e}"))
            }

            MomoError::UrlParse(ref e) => {
                ApiResponse::error(ErrorCode::InvalidRequest, format!("Invalid URL: {e}"))
            }

            MomoError::ApiRateLimit { retry_after } => {
                let msg = match retry_after {
                    Some(secs) => format!("Rate limit exceeded, retry after {secs} seconds"),
                    None => "Rate limit exceeded".to_string(),
                };
                ApiResponse::error(ErrorCode::InvalidRequest, msg)
            }

            MomoError::LlmRateLimit { retry_after } => {
                let msg = match retry_after {
                    Some(secs) => format!("Rate limit exceeded, retry after {secs} seconds"),
                    None => "Rate limit exceeded".to_string(),
                };
                ApiResponse::error(ErrorCode::InvalidRequest, msg)
            }

            MomoError::LlmUnavailable(ref msg) => {
                ApiResponse::error(ErrorCode::NotImplemented, msg.clone())
            }

            MomoError::OcrUnavailable(ref msg) => {
                ApiResponse::error(ErrorCode::NotImplemented, msg.clone())
            }

            MomoError::TranscriptionUnavailable(ref msg) => {
                ApiResponse::error(ErrorCode::NotImplemented, msg.clone())
            }

            ref internal @ (MomoError::Database(_)
            | MomoError::Processing(_)
            | MomoError::Embedding(_)
            | MomoError::Http(_)
            | MomoError::Io(_)
            | MomoError::Internal(_)
            | MomoError::Ocr(_)
            | MomoError::Transcription(_)
            | MomoError::Llm(_)
            | MomoError::Reranker(_)) => {
                tracing::error!(error = %internal, "Internal error mapped to v1 response");
                ApiResponse::error(ErrorCode::InternalError, "An internal error occurred")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_response_serializes_without_error() {
        let resp = ApiResponse::success("hello");
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["data"], "hello");
        assert!(json.get("error").is_none());
        assert!(json.get("meta").is_none());
    }

    #[test]
    fn error_response_serializes_without_data() {
        let resp = ApiResponse::<()>::error(ErrorCode::NotFound, "gone");
        let json = serde_json::to_value(&resp).expect("serialize");
        assert!(json.get("data").is_none());
        assert_eq!(json["error"]["code"], "not_found");
        assert_eq!(json["error"]["message"], "gone");
    }

    #[test]
    fn success_with_meta_serializes_all_fields() {
        let meta = ResponseMeta {
            next_cursor: Some("abc123".into()),
            total: Some(42),
        };
        let resp = ApiResponse::success_with_meta(vec![1, 2, 3], meta);
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["data"], serde_json::json!([1, 2, 3]));
        assert_eq!(json["meta"]["nextCursor"], "abc123");
        assert_eq!(json["meta"]["total"], 42);
    }

    #[test]
    fn meta_without_optional_fields_omits_them() {
        let meta = ResponseMeta {
            next_cursor: None,
            total: Some(10),
        };
        let json = serde_json::to_value(&meta).expect("serialize");
        assert!(json.get("nextCursor").is_none());
        assert_eq!(json["total"], 10);
    }

    #[test]
    fn error_code_status_mapping() {
        assert_eq!(ErrorCode::InvalidRequest.status(), StatusCode::BAD_REQUEST);
        assert_eq!(ErrorCode::Unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(ErrorCode::NotFound.status(), StatusCode::NOT_FOUND);
        assert_eq!(ErrorCode::Conflict.status(), StatusCode::CONFLICT);
        assert_eq!(
            ErrorCode::InternalError.status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            ErrorCode::NotImplemented.status(),
            StatusCode::NOT_IMPLEMENTED
        );
    }

    #[test]
    fn error_code_serializes_snake_case() {
        let json = serde_json::to_value(&ErrorCode::InvalidRequest).expect("serialize");
        assert_eq!(json, "invalid_request");

        let json = serde_json::to_value(&ErrorCode::InternalError).expect("serialize");
        assert_eq!(json, "internal_error");

        let json = serde_json::to_value(&ErrorCode::NotImplemented).expect("serialize");
        assert_eq!(json, "not_implemented");
    }

    #[test]
    fn error_code_deserializes_snake_case() {
        let code: ErrorCode = serde_json::from_str("\"not_found\"").expect("deserialize");
        assert_eq!(code, ErrorCode::NotFound);
    }

    #[test]
    fn cursor_pagination_defaults() {
        let pag = CursorPagination::default();
        assert_eq!(pag.limit, 20);
        assert!(pag.cursor.is_none());
    }

    #[test]
    fn cursor_pagination_clamps_high() {
        let pag = CursorPagination {
            limit: 999,
            cursor: None,
        }
        .validate();
        assert_eq!(pag.limit, 100);
    }

    #[test]
    fn cursor_pagination_clamps_low() {
        let pag = CursorPagination {
            limit: 0,
            cursor: None,
        }
        .validate();
        assert_eq!(pag.limit, 1);
    }

    #[test]
    fn created_response_has_201_status() {
        let resp = ApiResponse::created("new-resource");
        assert_eq!(resp.status, StatusCode::CREATED);
    }

    #[test]
    fn accepted_response_has_202_status() {
        let resp = ApiResponse::accepted("queued");
        assert_eq!(resp.status, StatusCode::ACCEPTED);
    }

    #[test]
    fn momo_error_not_found_maps_correctly() {
        let resp: ApiResponse<()> = MomoError::NotFound("gone".into()).into();
        assert_eq!(
            resp.error.as_ref().expect("error").code,
            ErrorCode::NotFound
        );
    }

    #[test]
    fn momo_error_internal_does_not_leak() {
        let resp: ApiResponse<()> = MomoError::Internal("secret debug info".into()).into();
        let err = resp.error.as_ref().expect("error");
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(err.message, "An internal error occurred");
    }

    #[test]
    fn momo_error_unavailable_maps_to_not_implemented() {
        let resp: ApiResponse<()> = MomoError::LlmUnavailable("no LLM".into()).into();
        assert_eq!(
            resp.error.as_ref().expect("error").code,
            ErrorCode::NotImplemented
        );
    }

    #[test]
    fn momo_error_rate_limit_maps_to_invalid_request() {
        let resp: ApiResponse<()> = MomoError::ApiRateLimit {
            retry_after: Some(30),
        }
        .into();
        let err = resp.error.as_ref().expect("error");
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert!(err.message.contains("30"));
    }
}
