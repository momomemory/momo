# Momo Documentation

> _"You, my friend, are just a few plumbs short of a fruit pie."_ — Momo

Momo is a self-hostable AI memory system designed to give agents long-term persistent memory. It is a single Rust binary that uses LibSQL's native vector search, meaning no external vector database like Pinecone or Weaviate is required.

The system handles document ingestion, memory extraction, and hybrid search with a focus on ease of use and local-first performance. It supports a wide range of content types including URLs, PDFs, Office documents, and even audio/video via local Whisper integration.

## Documentation

| Guide                                   | Description                                                         |
| --------------------------------------- | ------------------------------------------------------------------- |
| [API Reference](./api.md)               | REST API v1 reference, all endpoints, and request/response formats. |
| [MCP Guide](./mcp.md)                   | Built-in MCP server setup, auth, tools/resources, and manual usage. |
| [Self-Hosting Guide](./self-hosting.md) | Installation, configuration, and deployment instructions.           |
| [Release Strategy](./release-strategy.md) | Versioning and release process for server and SDKs.               |

## Quick Links

- **GitHub Repository**: [https://github.com/momomemory/momo](https://github.com/momomemory/momo)
- **TypeScript SDK**: [https://github.com/momomemory/sdk-typescript](https://github.com/momomemory/sdk-typescript)
- **Container Image (GHCR)**: [ghcr.io/momomemory/momo:latest](https://ghcr.io/momomemory/momo:latest)

## Quick Start

Run Momo with Docker:

```bash
docker run --name momo -d --restart unless-stopped -p 3000:3000 -v momo-data:/data ghcr.io/momomemory/momo:latest
```

MCP is available at `http://localhost:3000/mcp`. See [MCP Guide](./mcp.md).

Add a memory:

```bash
curl -X POST http://localhost:3000/api/v1/conversations:ingest \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I prefer dark mode"}], "containerTag": "user_1"}'
```

[← Back to Main README](../README.md)
