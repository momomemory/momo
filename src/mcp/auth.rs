use axum::{
    body::Body,
    extract::State,
    http::{
        header::{AUTHORIZATION, WWW_AUTHENTICATE},
        HeaderMap, HeaderName, HeaderValue, Request, StatusCode,
    },
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::api::AppState;

#[derive(Debug, Clone)]
pub struct McpAuthContext {
    pub user_id: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub container_tag: Option<String>,
}

pub async fn mcp_auth_middleware(
    State(state): State<AppState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let project_header = state.config.mcp.project_header.clone();
    let container_tag = project_tag_from_headers(request.headers(), &project_header)
        .or_else(|| Some(state.config.mcp.default_container_tag.clone()));

    if !state.config.mcp.require_auth {
        request.extensions_mut().insert(McpAuthContext {
            user_id: "anonymous".to_string(),
            email: None,
            name: None,
            container_tag,
        });
        return next.run(request).await;
    }

    if state.config.server.api_keys.is_empty() {
        return unauthorized_json_rpc(
            "Unauthorized: API keys not configured. Set MOMO_API_KEYS to enable MCP access.",
        );
    }

    let Some(auth_header) = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return unauthorized_challenge("Unauthorized");
    };

    let Some(token) = auth_header.strip_prefix("Bearer ") else {
        return unauthorized_json_rpc(
            "Unauthorized: Invalid authorization header format. Expected: Bearer <token>",
        );
    };
    let token = token.to_string();

    if !state.config.server.api_keys.iter().any(|key| key == &token) {
        return unauthorized_json_rpc("Unauthorized: Invalid or expired API key");
    }

    request.extensions_mut().insert(McpAuthContext {
        user_id: user_id_from_api_key(&token),
        email: None,
        name: None,
        container_tag,
    });

    next.run(request).await
}

pub fn auth_context_from_parts(parts: &axum::http::request::Parts) -> Option<McpAuthContext> {
    parts.extensions.get::<McpAuthContext>().cloned()
}

fn user_id_from_api_key(api_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let digest = hasher.finalize();
    let hash = format!("{digest:x}");
    format!("api_key_{}", &hash[..16])
}

fn project_tag_from_headers(headers: &HeaderMap, configured_header: &str) -> Option<String> {
    let configured = HeaderName::from_bytes(configured_header.as_bytes())
        .map_err(|error| {
            tracing::warn!(
                error = %error,
                header = configured_header,
                "Invalid MOMO_MCP_PROJECT_HEADER value; falling back to x-sm-project"
            );
            error
        })
        .ok();

    let value = if let Some(header_name) = configured {
        headers.get(header_name)
    } else {
        headers.get("x-sm-project")
    };

    value
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_string)
}

fn unauthorized_challenge(message: &str) -> Response {
    let mut response = (StatusCode::UNAUTHORIZED, message.to_string()).into_response();

    response.headers_mut().insert(
        WWW_AUTHENTICATE,
        HeaderValue::from_static(
            "Bearer resource_metadata=\"/.well-known/oauth-protected-resource\"",
        ),
    );
    response.headers_mut().insert(
        "Access-Control-Expose-Headers",
        HeaderValue::from_static("WWW-Authenticate"),
    );

    response
}

fn unauthorized_json_rpc(message: &str) -> Response {
    let payload = json!({
        "jsonrpc": "2.0",
        "error": {
            "code": -32000,
            "message": message,
        },
        "id": serde_json::Value::Null,
    });

    let mut response = (StatusCode::UNAUTHORIZED, axum::Json(payload)).into_response();

    response.headers_mut().insert(
        WWW_AUTHENTICATE,
        HeaderValue::from_static(
            "Bearer error=\"invalid_token\", resource_metadata=\"/.well-known/oauth-protected-resource\"",
        ),
    );
    response.headers_mut().insert(
        "Access-Control-Expose-Headers",
        HeaderValue::from_static("WWW-Authenticate"),
    );

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_id_is_stable_and_hides_key() {
        let user_id = user_id_from_api_key("super-secret");
        assert!(user_id.starts_with("api_key_"));
        assert_ne!(user_id, "super-secret");
        assert_eq!(user_id.len(), "api_key_".len() + 16);
    }

    #[test]
    fn project_tag_reads_custom_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-sm-project", HeaderValue::from_static("project-a"));

        let tag = project_tag_from_headers(&headers, "x-sm-project");
        assert_eq!(tag.as_deref(), Some("project-a"));
    }
}
