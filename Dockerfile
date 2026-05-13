FROM python:3.12-slim

# lxml and readability need libxml2/libxslt
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 libxslt1.1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY ccmcp/ ccmcp/

# Install runtime dependencies (no dev extras)
RUN uv pip install --system --no-cache -e .

# Pre-download embedding models so first startup is instant
RUN python - <<'EOF'
from fastembed import TextEmbedding, SparseTextEmbedding
TextEmbedding("BAAI/bge-small-en-v1.5")
SparseTextEmbedding("Qdrant/bm25")
EOF

# /data  — rotation matrix + SQLite state DB (named volume)
# /repos — source files (bind-mounted by user, read-only)
VOLUME ["/data", "/repos"]

EXPOSE 7700

ENV QDRANT_URL=http://qdrant:6333 \
    CCMCP_SOURCE_PATH=/repos

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
