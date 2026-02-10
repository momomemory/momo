# Momo MCP Guide

[← Back to README](./README.md)

Momo includes a built-in MCP server (streamable HTTP) designed to mirror Supermemory-style workflows while running in the same binary as the REST API and web console.

## What You Get

- Endpoint: `POST /mcp` (configurable with `MOMO_MCP_PATH`)
- OAuth discovery metadata:
  - `GET /.well-known/oauth-protected-resource`
  - `GET /.well-known/oauth-authorization-server`
- Tools:
  - `memory` (`save` and `forget`)
  - `recall`
  - `listProjects`
  - `whoAmI`
- Resources:
  - `supermemory://profile`
  - `supermemory://projects`
- Prompt:
  - `context`

## Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `MOMO_MCP_ENABLED` | Enable built-in MCP routes | `true` |
| `MOMO_MCP_PATH` | HTTP path for MCP endpoint | `/mcp` |
| `MOMO_MCP_REQUIRE_AUTH` | Require Bearer auth on MCP requests | `true` |
| `MOMO_MCP_DEFAULT_CONTAINER_TAG` | Fallback project/container tag | `default` |
| `MOMO_MCP_PROJECT_HEADER` | Header used for project scoping | `x-sm-project` |
| `MOMO_MCP_PUBLIC_URL` | Optional public base URL for discovery responses | (unset) |
| `MOMO_MCP_AUTHORIZATION_SERVER` | Optional OAuth issuer URL for discovery responses | (unset) |

Authentication keys are configured with `MOMO_API_KEYS` (comma-separated). If MCP auth is required and no keys are configured, MCP requests are rejected.

## Client Configuration Example

```json
{
  "mcpServers": {
    "momo": {
      "url": "http://localhost:3000/mcp",
      "headers": {
        "Authorization": "Bearer your_momo_api_key",
        "x-sm-project": "default"
      }
    }
  }
}
```

## Manual MCP Handshake (curl)

MCP streamable HTTP requires a 3-step handshake:

1. `initialize`
2. `notifications/initialized`
3. Normal MCP requests (`tools/list`, `tools/call`, `resources/list`, etc.)

```bash
API_KEY="your_momo_api_key"

# 1) initialize
curl -sS -D /tmp/mcp-init.headers \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "x-sm-project: default" \
  --data '{"jsonrpc":"2.0","id":"init-1","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"manual-curl","version":"0.1.0"}}}' \
  http://localhost:3000/mcp

SESSION_ID="$(awk -F': ' 'tolower($1)=="mcp-session-id" {gsub(/\r/,"",$2); print $2}' /tmp/mcp-init.headers)"

# 2) notifications/initialized
curl -sS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: ${SESSION_ID}" \
  -H "x-sm-project: default" \
  --data '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  http://localhost:3000/mcp

# 3) tools/list
curl -sS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: ${SESSION_ID}" \
  -H "x-sm-project: default" \
  --data '{"jsonrpc":"2.0","id":"tools-1","method":"tools/list","params":{}}' \
  http://localhost:3000/mcp
```

## Project Scoping

- `MOMO_MCP_PROJECT_HEADER` controls which header supplies the project/container tag.
- Default header is `x-sm-project` for Supermemory compatibility.
- You can also pass `containerTag` in tool/prompt arguments.
- If neither is provided, Momo falls back to `MOMO_MCP_DEFAULT_CONTAINER_TAG`.

## Troubleshooting

- `401 Unauthorized`: missing/invalid bearer key, or auth required without configured `MOMO_API_KEYS`.
- `Unauthorized: Session not found`: missing or stale `mcp-session-id`.
- `expect initialized notification`: call `notifications/initialized` before `tools/*` or `resources/*`.
- MCP path mismatch: if `MOMO_MCP_PATH` is customized, use that exact URL in your MCP client.
