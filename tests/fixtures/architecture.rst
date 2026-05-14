System Architecture
===================

The ccmcp system consists of three main components that work together
to provide an AI-accessible knowledge base over local and remote documents.

Ingestion Pipeline
------------------

Documents are ingested from filesystem paths, web URLs, and Google Drive
folders. Each source is fingerprinted with a SHA-256 content hash. Changed
documents are re-chunked and re-embedded atomically using a versioned swap:
new points are upserted before old points are deleted, ensuring the collection
is never left incomplete.

The structural chunker splits documents at heading and paragraph boundaries
without requiring an LLM. Chunks range from 50 to 512 tokens. Headings are
preserved as a prefix in every sub-chunk so that retrieval context is
self-contained.

Vector Storage
--------------

Qdrant stores both dense and sparse vectors in a hybrid collection. Dense
vectors use INT8 scalar quantisation with a random orthogonal preconditioning
matrix that redistributes variance uniformly across dimensions. Sparse vectors
use BM25 term frequencies with IDF weighting.

Hybrid search fuses dense and sparse ranked lists using Reciprocal Rank Fusion.
The collection lives at localhost:6333 by default and persists to disk.

MCP Interface
-------------

Two MCP server transports expose the knowledge base to AI coding assistants.
The stdio transport is launched as a subprocess by Claude Code CLI. The SSE
transport runs as a persistent HTTP server on port 7700 and is consumed by
Cursor IDE and other HTTP-capable clients.

Both transports expose two tools: qdrant_find for retrieval and qdrant_store
for writing agent artifacts back into the knowledge base.
