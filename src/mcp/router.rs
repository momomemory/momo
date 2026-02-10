use axum::{
    extract::State, http::HeaderMap, middleware, response::IntoResponse, routing::get, Router,
};
use serde_json::json;

use crate::api::AppState;
use crate::config::Config;

use super::{auth::mcp_auth_middleware, server::streamable_http_service};

pub fn mcp_router(state: AppState) -> Router<AppState> {
    if !state.config.mcp.enabled {
        return Router::new();
    }

    let mcp_path = state.config.mcp.path.clone();
    let mcp_service = streamable_http_service(state.clone());

    let mcp_routes = Router::new()
        .nest_service(&mcp_path, mcp_service)
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            mcp_auth_middleware,
        ));

    Router::new()
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_protected_resource),
        )
        .route(
            "/.well-known/oauth-authorization-server",
            get(oauth_authorization_server),
        )
        .merge(mcp_routes)
}

async fn oauth_protected_resource(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let base_url = discover_base_url(&state.config, &headers);
    let resource = format!("{}{}", base_url, state.config.mcp.path);

    let authorization_server = state
        .config
        .mcp
        .oauth_authorization_server
        .clone()
        .unwrap_or_else(|| base_url.clone());

    axum::Json(json!({
        "resource": resource,
        "authorization_servers": [authorization_server],
        "scopes_supported": ["openid", "profile", "email", "offline_access"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": "https://github.com/momomemory/momo",
    }))
}

async fn oauth_authorization_server(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let base_url = discover_base_url(&state.config, &headers);

    let issuer = state
        .config
        .mcp
        .oauth_authorization_server
        .clone()
        .unwrap_or(base_url);

    axum::Json(json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{issuer}/oauth/authorize"),
        "token_endpoint": format!("{issuer}/oauth/token"),
        "registration_endpoint": format!("{issuer}/oauth/register"),
        "jwks_uri": format!("{issuer}/oauth/jwks"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["openid", "profile", "email", "offline_access"],
        "code_challenge_methods_supported": ["S256"],
    }))
}

fn discover_base_url(config: &Config, headers: &HeaderMap) -> String {
    if let Some(public_url) = &config.mcp.public_url {
        return public_url.trim_end_matches('/').to_string();
    }

    let forwarded_proto = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "http".to_string());

    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get("host"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}:{}", config.server.host, config.server.port));

    format!("{forwarded_proto}://{host}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_prefers_public_url_when_configured() {
        let mut config = crate::config::Config::default();
        config.mcp.public_url = Some("https://mcp.example.com/".to_string());
        let headers = HeaderMap::new();

        let base_url = discover_base_url(&config, &headers);
        assert_eq!(base_url, "https://mcp.example.com");
    }

    #[test]
    fn base_url_uses_forwarded_headers() {
        let config = crate::config::Config::default();

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-proto", "https".parse().unwrap());
        headers.insert("x-forwarded-host", "memory.example.com".parse().unwrap());

        let base_url = discover_base_url(&config, &headers);
        assert_eq!(base_url, "https://memory.example.com");
    }
}
