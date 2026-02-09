# Momo Documentation

> *"You, my friend, are just a few plumbs short of a fruit pie."* — Momo

Momo is a self-hostable AI memory system designed to give agents long-term persistent memory. It is a single Rust binary that uses LibSQL's native vector search, meaning no external vector database like Pinecone or Weaviate is required.

The system handles document ingestion, memory extraction, and hybrid search with a focus on ease of use and local-first performance. It supports a wide range of content types including URLs, PDFs, Office documents, and even audio/video via local Whisper integration.

## Documentation

| Guide | Description |
|-------|-------------|
| [API Reference](./api.md) | REST API v1 reference, all endpoints, and request/response formats. |
| [Self-Hosting Guide](./self-hosting.md) | Installation, configuration, and deployment instructions. |

## Quick Links

- **GitHub Repository**: [https://github.com/momomemory/momo](https://github.com/momomemory/momo)
- **TypeScript SDK**: [https://github.com/momomemory/sdk-typescript](https://github.com/momomemory/sdk-typescript)
- **Docker Hub**: [https://hub.docker.com/r/momomemory/momo](https://hub.docker.com/r/momomemory/momo)

## Quick Start

Run Momo with Docker:

```bash
docker run -p 3000:3000 -v ./data:/data momomemory/momo
```

Add a memory:

```bash
curl -X POST http://localhost:3000/api/v1/conversations:ingest \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I prefer dark mode"}], "containerTag": "user_1"}'
```

[← Back to Main README](../README.md)
