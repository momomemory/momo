# Momo V1 REST API Reference

Welcome to the Momo API reference. Momo is a self-hostable AI memory system that provides a unified interface for document ingestion, memory extraction, and hybrid search.

[← Back to README](./README.md)

## Table of Contents
- [Response Envelope](#response-envelope)
- [Authentication](#authentication)
- [Pagination](#pagination)
- [Error Codes](#error-codes)
- [ID Formats](#id-formats)
- [Enums](#enums)
- [Health & System](#health--system)
- [Documents](#documents)
- [Ingestions](#ingestions)
- [Search](#search)
- [Memories](#memories)
- [Graph](#graph)
- [Profile](#profile)
- [Conversations](#conversations)
- [Admin](#admin)

---

## Response Envelope

All API responses follow a consistent envelope format.

```json
{
  "data": { ... },
  "meta": {
    "nextCursor": "string",
    "total": 42
  },
  "error": {
    "code": "snake_case_error_code",
    "message": "Human readable message"
  }
}
```

- `data`: Present on success, containing the requested resource or result.
- `error`: Present on failure, containing an error code and message.
- `meta`: Optional, used for pagination (e.g., in list endpoints).

---

## Authentication

Authentication is handled via Bearer tokens in the `Authorization` header.

```http
Authorization: Bearer <your_api_key>
```

- API keys are configured via the `MOMO_API_KEYS` environment variable.
- If no keys are configured, authentication is disabled.
- Failed authentication returns a `401 Unauthorized` response with `{"error": {"code": "unauthorized", "message": "Authentication required"}}`.

---

## Pagination

List endpoints use cursor-based pagination.

- **`limit`**: (Optional) Number of items to return. Default: 20, Max: 100. Clamped to 1..100.
- **`cursor`**: (Optional) Opaque base64 string to fetch the next page.

Next page cursors are provided in the `meta.nextCursor` field of the response.

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | The request parameters or body are invalid. |
| `unauthorized` | 401 | Authentication is required or the provided token is invalid. |
| `not_found` | 404 | The requested resource was not found. |
| `conflict` | 409 | A conflict occurred (e.g., duplicate custom ID). |
| `internal_error` | 500 | An unexpected server error occurred. |
| `not_implemented` | 501 | The requested feature is not yet implemented. |

---

## ID Formats

- **`documentId`**: 21-character NanoID (e.g., `V1StGXR8_Z5jdHi6B-myT`).
- **`memoryId`**: 21-character NanoID (e.g., `V1StGXR8_Z5jdHi6B-myT`).
- **`ingestionId`**: UUID v4 (e.g., `550e8400-e29b-41d4-a716-446655440000`).

---

## Enums

### IngestionStatus
`"queued"`, `"processing"`, `"completed"`, `"failed"`

### V1DocumentType
`"text"`, `"pdf"`, `"webpage"`, `"image"`, `"video"`, `"audio"`, `"markdown"`, `"code"`, `"csv"`, `"docx"`, `"pptx"`, `"xlsx"`, `"unknown"`

### V1MemoryType
`"fact"`, `"preference"`, `"episode"`

### SearchScope
`"documents"`, `"memories"`, `"hybrid"` (default)

### GraphNodeType
`"memory"`, `"document"`

### GraphEdgeType
`"updates"`, `"relatesTo"`, `"conflictsWith"`, `"derivedFrom"`, `"sources"`

---

## Health & System

### Health Check
`GET /api/v1/health`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/health
```

**Example Response:**
```json
{
  "data": {
    "status": "ok",
    "version": "0.1.0"
  }
}
```

### OpenAPI Spec
`GET /api/v1/openapi.json`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/openapi.json
```

### API Documentation (UI)
`GET /api/v1/docs`

Renders the ReDoc UI for API exploration.

---

## Documents

### Create Document
`POST /api/v1/documents`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/documents \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Sample document content...",
    "containerTag": "user_123",
    "customId": "unique_ref_001",
    "metadata": { "source": "manual" },
    "contentType": "text/plain",
    "extractMemories": true
  }'
```

**Example Response (202 Accepted):**
```json
{
  "data": {
    "documentId": "V1StGXR8_Z5jdHi6B-myT",
    "ingestionId": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### List Documents
`GET /api/v1/documents`

**Example Request:**
```bash
curl "http://localhost:3000/api/v1/documents?containerTags=user_123&limit=10" \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "data": {
    "documents": [
      {
        "documentId": "V1StGXR8_Z5jdHi6B-myT",
        "customId": "unique_ref_001",
        "title": "Welcome Note",
        "docType": "text",
        "ingestionStatus": "completed",
        "metadata": {},
        "containerTags": ["user_123"],
        "createdAt": "2024-02-08T12:00:00Z",
        "updatedAt": "2024-02-08T12:00:00Z"
      }
    ]
  },
  "meta": {
    "nextCursor": "eyJsYXN0X2lkIjogIjEifQ=="
  }
}
```

### Get Document
`GET /api/v1/documents/{documentId}`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/documents/V1StGXR8_Z5jdHi6B-myT \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "data": {
    "documentId": "V1StGXR8_Z5jdHi6B-myT",
    "customId": "unique_ref_001",
    "title": "Welcome Note",
    "content": "Full text content here...",
    "summary": "Short summary of the text",
    "url": "https://example.com/doc",
    "docType": "text",
    "ingestionStatus": "completed",
    "metadata": {},
    "containerTags": ["user_123"],
    "chunkCount": 5,
    "createdAt": "2024-02-08T12:00:00Z",
    "updatedAt": "2024-02-08T12:00:00Z"
  }
}
```

### Update Document
`PATCH /api/v1/documents/{documentId}`

**Example Request:**
```bash
curl -X PATCH http://localhost:3000/api/v1/documents/V1StGXR8_Z5jdHi6B-myT \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated Title",
    "metadata": { "category": "archive" }
  }'
```

### Delete Document
`DELETE /api/v1/documents/{documentId}`

**Example Request:**
```bash
curl -X DELETE http://localhost:3000/api/v1/documents/V1StGXR8_Z5jdHi6B-myT \
  -H "Authorization: Bearer <token>"
```

### Batch Create Documents
`POST /api/v1/documents:batch`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/documents:batch \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      { "content": "Doc 1", "customId": "ref_1" },
      { "content": "Doc 2", "customId": "ref_2" }
    ],
    "containerTag": "user_123"
  }'
```

### Upload File
`POST /api/v1/documents:upload`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/documents:upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/document.pdf" \
  -F "containerTag=user_123"
```

---

## Ingestions

### Get Ingestion Status
`GET /api/v1/ingestions/{ingestionId}`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/ingestions/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "data": {
    "documentId": "V1StGXR8_Z5jdHi6B-myT",
    "status": "completed",
    "title": "Uploaded Document",
    "createdAt": "2024-02-08T12:00:00Z"
  }
}
```

---

## Search

### Unified Search
`POST /api/v1/search`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is my favorite color?",
    "containerTags": ["user_123"],
    "limit": 5,
    "rerank": true
  }'
```

**Example Response:**
```json
{
  "data": {
    "results": [
      {
        "type": "memory",
        "memoryId": "mem_abc123",
        "content": "User prefers the color blue.",
        "similarity": 0.92,
        "metadata": {},
        "updatedAt": "2024-02-08T12:00:00Z"
      },
      {
        "type": "document",
        "documentId": "doc_xyz789",
        "title": "Bio",
        "docType": "text",
        "score": 0.85,
        "chunks": [
          { "content": "...favorite color is blue...", "score": 0.88 }
        ],
        "metadata": {},
        "createdAt": "2024-02-08T12:00:00Z",
        "updatedAt": "2024-02-08T12:00:00Z"
      }
    ],
    "total": 2,
    "timingMs": 145
  }
}
```

---

## Memories

### Create Memory
`POST /api/v1/memories`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/memories \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User lives in Berlin.",
    "containerTag": "user_123",
    "memoryType": "fact"
  }'
```

**Example Response (201 Created):**
```json
{
  "data": {
    "memoryId": "mem_berlin_001",
    "content": "User lives in Berlin.",
    "containerTag": "user_123",
    "memoryType": "fact",
    "version": 1,
    "isLatest": true,
    "isInference": false,
    "isForgotten": false,
    "isStatic": false,
    "metadata": {},
    "createdAt": "2024-02-08T12:00:00Z",
    "updatedAt": "2024-02-08T12:00:00Z"
  }
}
```

### List Memories
`GET /api/v1/memories`

**Example Request:**
```bash
curl "http://localhost:3000/api/v1/memories?containerTag=user_123&limit=20" \
  -H "Authorization: Bearer <token>"
```

### Get Memory
`GET /api/v1/memories/{memoryId}`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/memories/mem_abc123 \
  -H "Authorization: Bearer <token>"
```

### Update Memory
`PATCH /api/v1/memories/{memoryId}`

**Example Request:**
```bash
curl -X PATCH http://localhost:3000/api/v1/memories/mem_abc123 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User lives in Berlin, Germany.",
    "isStatic": true
  }'
```

**Example Response:**
```json
{
  "data": {
    "memoryId": "mem_abc123",
    "content": "User lives in Berlin, Germany.",
    "version": 2,
    "createdAt": "2024-02-08T12:05:00Z"
  }
}
```

### Delete Memory (Forget by ID)
`DELETE /api/v1/memories/{memoryId}`

**Example Request:**
```bash
curl -X DELETE http://localhost:3000/api/v1/memories/mem_abc123 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{ "reason": "obsolete" }'
```

**Example Response:**
```json
{
  "data": {
    "memoryId": "mem_abc123",
    "forgotten": true
  }
}
```

### Forget Memory by Content
`POST /api/v1/memories:forget`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/memories:forget \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Where the user lives",
    "containerTag": "user_123",
    "reason": "User requested deletion"
  }'
```

**Example Response:**
```json
{
  "data": {
    "memoryId": "mem_abc123",
    "forgotten": true
  }
}
```

---

## Graph

### Memory Graph
`GET /api/v1/memories/{memoryId}/graph`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/memories/mem_abc123/graph \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "data": {
    "nodes": [
      { "id": "mem_abc123", "type": "memory", "metadata": {} },
      { "id": "doc_xyz789", "type": "document", "metadata": {} }
    ],
    "links": [
      { "source": "mem_abc123", "target": "doc_xyz789", "type": "sources" }
    ]
  }
}
```

### Container Graph
`GET /api/v1/containers/{tag}/graph`

**Example Request:**
```bash
curl http://localhost:3000/api/v1/containers/user_123/graph \
  -H "Authorization: Bearer <token>"
```

---

## Profile

### Compute User Profile
`POST /api/v1/profile:compute`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/profile:compute \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "containerTag": "user_123",
    "generateNarrative": true
  }'
```

**Example Response:**
```json
{
  "data": {
    "containerTag": "user_123",
    "narrative": "The user lives in Berlin and likes pizza.",
    "staticFacts": [
      { "content": "Lives in Berlin", "confidence": 1.0, "createdAt": "2024-02-08T12:00:00Z" }
    ],
    "dynamicFacts": [],
    "totalMemories": 2,
    "lastUpdated": "2024-02-08T12:00:00Z"
  }
}
```

---

## Conversations

### Ingest Conversation
`POST /api/v1/conversations:ingest`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/conversations:ingest \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "I like pizza." }
    ],
    "containerTag": "user_123",
    "memoryType": "preference"
  }'
```

**Example Response (201 Created):**
```json
{
  "data": {
    "memoriesExtracted": 1,
    "memoryIds": ["mem_pizza_001"],
    "sessionId": "chat_001"
  }
}
```

---

## Admin

### Run Forgetting Cycle
`POST /api/v1/admin/forgetting:run`

**Example Request:**
```bash
curl -X POST http://localhost:3000/api/v1/admin/forgetting:run \
  -H "Authorization: Bearer <token>"
```

**Example Response:**
```json
{
  "data": {
    "memoriesForgotten": 2,
    "memoriesEvaluated": 150
  }
}
```

