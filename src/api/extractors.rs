use axum::extract::rejection::JsonRejection;
use axum::extract::FromRequest;

use crate::error::MomoError;

#[derive(FromRequest)]
#[from_request(via(axum::Json), rejection(MomoError))]
#[allow(dead_code)]
pub struct AppJson<T>(pub T);

impl From<JsonRejection> for MomoError {
    fn from(rejection: JsonRejection) -> Self {
        map_json_rejection(rejection)
    }
}

fn map_json_rejection(rejection: JsonRejection) -> MomoError {
    match rejection {
        JsonRejection::JsonDataError(err) => {
            let message = err.to_string();
            if let Some(field) = extract_missing_field(&message) {
                MomoError::Validation(format!("Missing required field: {field}"))
            } else {
                MomoError::Validation(format!("Invalid JSON: {message}"))
            }
        }
        JsonRejection::JsonSyntaxError(err) => {
            MomoError::Validation(format!("JSON syntax error: {err}"))
        }
        JsonRejection::MissingJsonContentType(_) => {
            MomoError::Validation("Missing `Content-Type: application/json` header".to_string())
        }
        JsonRejection::BytesRejection(_) => {
            MomoError::Internal("Failed to read request body".to_string())
        }
        _ => MomoError::Validation(rejection.to_string()),
    }
}

fn extract_missing_field(message: &str) -> Option<&str> {
    let prefix = "missing field `";
    let start = message.find(prefix)? + prefix.len();
    let remaining = message.get(start..)?;
    let end = remaining.find('`')?;
    remaining.get(..end)
}
