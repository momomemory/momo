# Momo - Self-Hostable AI Memory System

> *"You, my friend, are just a few plumbs short of a fruit pie."* — Momo

Momo is a self-hostable AI memory system written in Rust — inspired by SuperMemory. It provides long-term memory for AI agents using LibSQL's native vector search capabilities — no external vector database required. Single binary, runs anywhere.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [SDKs](#sdks)
- [Docker](#docker)
- [Development](#development)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Quick Start

The fastest way to get Momo running is via Docker:

```bash
docker run --name momo -d --restart unless-stopped -p 3000:3000 -v momo-data:/data ghcr.io/momomemory/momo:latest
```

Then open:

- Web console: `http://localhost:3000/`
- API docs: `http://localhost:3000/api/v1/docs`

### Add a Memory
```bash
curl -X POST http://localhost:3000/api/v1/conversations:ingest \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I prefer dark mode"}], "containerTag": "user_1"}'
```

### Search Everything
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"q": "What are the user preferences?", "containerTags": ["user_1"], "scope": "hybrid"}'
```

## Features

- **Vector Search**: Native LibSQL vector embeddings (no external vector DB needed)
- **Embedded Web Console**: Bun + Preact UI embedded in the binary and served from `/`
- **Local AI Pipeline**: Built-in support for local embeddings (FastEmbed) and transcription (Whisper)
- **External Embeddings**: Support for OpenAI, OpenRouter, Ollama, and LM Studio APIs
- **Universal Ingestion**: Extract from URLs, PDFs, HTML, DOCX, Images (OCR), and Audio/Video
- **Memory Versioning**: Track memory updates with parent/child relationships
- **Contradiction Management**: Automatically detect and resolve conflicting memories
- **Knowledge Graph**: Memory relationships, graph traversal, container-level graphs
- **User Profiling**: Auto-generated user profiles from accumulated memories
- **Autonomous Synthesis**: Optional background engine that derives new insights from existing data
- **Intelligent Decay**: Relevance scoring that automatically prunes stale or irrelevant memories
- **AST-Aware Code Chunking**: Tree-sitter based chunking for multiple programming languages
- **Multi-Tenant by Design**: Scalable container-based isolation for multi-user applications
- **Reranking**: Improved search relevance using cross-encoder models

## Documentation

For detailed documentation, see the [docs](./docs/README.md) directory.

- [Full Documentation](./docs/README.md)
- [API Reference](./docs/api.md)
- [Self-Hosting Guide](./docs/self-hosting.md)

## SDKs

| Language | Package | Status |
|----------|---------|--------|
| **TypeScript** | [`@momomemory/sdk`](https://github.com/momomemory/sdk-typescript) | Stable |
| **Python** | `momomemory-sdk` | Coming Soon |
| **Rust** | `momo-sdk` | Coming Soon |
| **Go** | `momo-go` | Coming Soon |

## Docker

Use the published container image:

```bash
docker run --name momo -d --restart unless-stopped -p 3000:3000 -v momo-data:/data ghcr.io/momomemory/momo:latest
```

To follow logs:

```bash
docker logs -f momo
```

## Development

```bash
# From monorepo root: run backend + frontend with auto-reload
just dev

# Build frontend bundle (embedded in binary)
just build-frontend

# Build server (includes frontend bundle)
just build

# Run tests
cargo test

# Check for issues
cargo clippy

# Format code
cargo fmt
```

Note: when frontend assets are missing, Rust build uses `momo/build.rs` to run `bun install` and `bun run build`.

## Maintainers

[@watzon](https://github.com/watzon)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get involved. PRs accepted.

## Credits

- Inspired by [Supermemory](https://supermemory.ai)
- Named after Momo, Aang's loyal flying lemur companion 🦇
- Built with [LibSQL](https://libsql.org), [FastEmbed](https://github.com/Anush008/fastembed-rs), and [Axum](https://github.com/tokio-rs/axum)

## License

[MIT](LICENSE) © Momo Contributors
