# ccmcp — Complete Agent Reference

Claude Code MCP Knowledge Base. Self-hosted, AI-first retrieval system.
Technical docs and source code are indexed in a local Qdrant vector database
and exposed to Claude Code CLI and Cursor via MCP servers.

Local project root: `~/code/ccmcp`
GitHub repo:        `github.com/<user>/ccmcp`  (private)

---

## 1. What this system does

Technical documentation, source code, and wiki articles are ingested from
three source types (filesystem, web URLs, Google Drive), chunked by document
structure, embedded with a hybrid dense+sparse model, and stored in Qdrant.

Two MCP servers sit on top of Qdrant:
- stdio transport → Claude Code CLI reads it automatically during coding sessions
- SSE transport   → Cursor connects to it at http://localhost:7700/sse

When either agent needs context, it calls `qdrant-find` and gets back the
most relevant passages from the indexed corpus. It can also call `qdrant-store`
to write knowledge back (agent artifacts, session notes).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INGESTION SOURCES                         │
│   Filesystem (inotify/FSEvents)  │  Web URLs  │  Google Drive   │
└──────────────┬───────────────────┴────────────┴─────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DOCUMENT MANAGEMENT LAYER                      │
│  SHA-256 dedup · versioned atomic swap · orphan cleanup         │
│  SQLite state store: ~/.local/share/ccmcp/state.db              │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              STRUCTURAL CHUNKER  (Phase 1 — active)             │
│  Split at heading / paragraph / function boundaries             │
│  No LLM required. 50–512 token chunks, self-contained.          │
│                                                                 │
│  Phase 2 (future): swap in proposition extraction agent         │
│  backend: "none" | "ollama" | "claude_api"  (config switch)     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PIPELINE                           │
│  fastembed ONNX (CoreML on macOS, CPU on Linux/WSL2)            │
│    Dense:  BAAI/bge-small-en-v1.5  (384-dim)                   │
│    Sparse: Qdrant/bm25                                          │
│  Random orthogonal rotation applied to dense before upsert      │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         QDRANT                                  │
│  Hybrid search: dense (cosine) + sparse (BM25) fused via RRF   │
│  Scalar INT8 quantisation (float16 on macOS M-series)           │
│  Payload: text, source_uri, section, hash, version, timestamps  │
└──────────┬──────────────────────┬───────────────────────────────┘
           │                      │
    MCP stdio (Claude Code)  MCP SSE :7700 (Cursor)
```

---

## 3. Tech Stack — decisions and rationale

### Qdrant
Chosen for: native hybrid dense+sparse search, scalar/binary quantisation,
official MCP server (`mcp-server-qdrant`), Rust binary with low RAM footprint,
runs without Docker. On WSL2 and Linux: native binary. On macOS: Homebrew.

### fastembed
Chosen for: ONNX runtime (no PyTorch, no CUDA), auto-selects CoreML on Apple
Silicon, INT8-quantised models ship by default, handles both dense and BM25
sparse vectors in one library.

Dense model:  `BAAI/bge-small-en-v1.5`  — 384-dim, ~60 MB, strong MTEB score,
              fast on CPU. Good balance of quality and speed without GPU.
Sparse model: `Qdrant/bm25`             — BM25 term frequencies. Catches exact
              technical terms (hostnames, CLI flags, model names, error codes)
              that semantic embeddings miss.

### Hybrid search (RRF)
Dense vectors cover semantic similarity. Sparse (BM25) covers exact vocabulary.
Reciprocal Rank Fusion merges both result sets. For technical documentation the
combination is significantly better than either alone.

### Random orthogonal rotation
Applied to all dense vectors before upsert. Redistributes variance uniformly
across dimensions, improving INT8 quantisation effectiveness. The rotation
matrix is generated once (`python -m ccmcp setup`), stored as a .npy file,
and applied identically to query vectors at search time. Inner product
structure is preserved exactly.

### Structural chunking (Phase 1)
No LLM dependency for ingestion. Split at natural document boundaries
(headings, paragraphs, function definitions). Each chunk is 50–512 tokens.
Heading is preserved as a prefix in the chunk text. Quality is lower than
proposition-level indexing but sufficient for a first deployment. Upgrade to
proposition extraction in Phase 2 if retrieval quality is insufficient.

### MCP transport split
- stdio: Claude Code launches the MCP server as a subprocess. No persistent
  daemon needed. Configured in ~/.claude.json.
- SSE: Cursor needs a persistent HTTP endpoint. Runs as a system service
  (systemd on Linux/WSL2, launchd on macOS).

---

## 4. Deployment Targets

Three targets share the same codebase. Platform differences are in `deploy/`.

### 4.1 WSL2 on Windows 11 (primary dev target)

**Environment isolation via Nix devShell.**
The repo lives on the Dev Drive: `D:\code\ccmcp` (Windows) = `/mnt/d/code/ccmcp` (WSL2).
All runtime state (.qdrant/, .venv/, config.yaml, rotation_matrix.npy, state.db)
stays inside the repo directory on D: — nothing touches the WSL2 system volume.

**Prerequisites:**
```bash
# 1. Confirm systemd is enabled
cat /etc/wsl.conf
# Must contain:
# [boot]
# systemd=true
# If missing: run deploy/wsl2/setup.sh

# 2. Confirm Nix is installed
nix --version   # if missing: sh <(curl -L https://nixos.org/nix/install) --daemon
```

**Enter the dev environment:**
```bash
cd /mnt/d/code/ccmcp
nix develop          # first run downloads deps (~2 min), subsequent runs: instant
```

**inotify caveat:** inotify events do NOT fire for paths under /mnt/c/ or
/mnt/d/ (Windows filesystem, 9P layer). Keep source repos inside WSL2 (~/)
for live watch mode. For Windows-side paths, use scheduled `ccmcp scan`
instead. The filesystem source detects /mnt/ paths automatically and falls
back to PollingObserver.

**Qdrant:** runs as a native Linux binary inside the Nix shell. Data stored
at `$PWD/.qdrant/` (inside the repo on D:). Started automatically by the
shellHook when entering `nix develop`.

**Service management:** systemd user units in `deploy/wsl2/`.

---

### 4.2 Linux server (always-on, primary production target)

No Docker required. Qdrant runs as a native binary via systemd service.

**Prerequisites:**
```bash
sudo apt install python3.12 python3.12-pip
pip install uv
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz -C ~/.local/bin/
```

**Service management:** systemd user units in `deploy/linux/`.

**Embedding:** CPU-only, ONNX runtime via fastembed, AVX2 if available.
Expect ~80–200 chunks/sec. Ingestion is a background service; latency is irrelevant.

**Quantisation:** INT8 scalar (default). 4× smaller than float32. Random
preconditioning applied to improve INT8 effectiveness.

---

### 4.3 macOS — MacBook Air M5 15"

**Installation:**
```bash
brew install qdrant uv
pip3 install -e ".[dev]"

# Optional: Ollama for Phase 2 proposition extraction
brew install ollama
ollama pull qwen2.5:14b   # fits in 16 GB alongside Qdrant + embeddings
```

**Service management:** launchd agents in `deploy/macos/`.

**Embedding:** ONNX CoreML execution provider selected automatically on Apple
Silicon. Delegates to Neural Engine + Metal GPU. Expect ~800–1500 chunks/sec.

**Quantisation:** float16 (Half Scalar Quantisation) by default. Ample unified
memory on M5. Rotation preconditioning still applied.

**Power management:** Full scans skip if on battery
(`power.skip_full_scan_on_battery: true`). Incremental updates always run.
`caffeinate -i` wraps ingestion to prevent sleep.

**Secrets storage:** API keys and Drive credentials in macOS Keychain:
```bash
security add-generic-password -a ccmcp -s ccmcp.anthropic.api_key -w "sk-ant-..."
```
```python
import subprocess
key = subprocess.check_output(
    ["security", "find-generic-password",
     "-a", "ccmcp", "-s", "ccmcp.anthropic.api_key", "-w"]
).decode().strip()
```

**Phase 2 extraction backend (configurable):**
```yaml
extraction:
  backend: "ollama"       # or "claude_api" or "none"
  ollama:
    model: "qwen2.5:14b"  # 16 GB model; qwen2.5:32b for 32 GB
    base_url: "http://localhost:11434"
  claude_api:
    model: "claude-sonnet-4-20250514"
```

---

## 5. Repository Layout

```
~/code/ccmcp/
├── CLAUDE.md                    ← this file (complete agent reference)
├── pyproject.toml
├── flake.nix                    ← Nix devShell (WSL2 + Linux)
├── flake.lock
├── config.yaml                  ← gitignored, copy from config.example.yaml
├── config.example.yaml
├── rotation_matrix.npy          ← gitignored, generated by: ccmcp setup
├── .gitignore
│
├── ccmcp/
│   ├── __init__.py
│   ├── __main__.py              ← CLI entrypoint (click + rich)
│   ├── config.py                ← load/validate config.yaml
│   ├── chunker.py               ← structural chunking
│   ├── embedder.py              ← fastembed dense+sparse, rotation
│   ├── store.py                 ← Qdrant client wrapper
│   ├── state.py                 ← SQLite ingestion ledger
│   ├── controller.py            ← orchestration: scan, watch, dedup, swap
│   └── sources/
│       ├── __init__.py
│       ├── filesystem.py        ← inotify/FSEvents/polling watcher
│       ├── web.py               ← HTML fetch, readability, ETag tracking
│       └── drive.py             ← Google Drive API, service account auth
│
├── deploy/
│   ├── linux/
│   │   ├── qdrant.service
│   │   ├── ccmcp-controller.service
│   │   ├── ccmcp-mcp-sse.service
│   │   └── install.sh
│   ├── macos/
│   │   ├── ai.qdrant.server.plist
│   │   ├── ai.ccmcp.controller.plist
│   │   ├── ai.ccmcp.mcp.sse.plist
│   │   └── install.sh
│   └── wsl2/
│       ├── setup.sh
│       ├── ccmcp-controller.service
│       └── ccmcp-mcp-sse.service
│
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_store.py            ← @pytest.mark.integration
    ├── test_state.py
    ├── test_filesystem.py
    ├── test_web.py
    └── test_drive.py
```

---

## 6. Configuration Reference

```yaml
# ~/code/ccmcp/config.yaml  (gitignored — copy from config.example.yaml)

qdrant:
  url: "http://localhost:6333"
  collection: "techdocs"
  # api_key: ""              # Qdrant Cloud only

embedding:
  dense_model: "BAAI/bge-small-en-v1.5"
  sparse_model: "Qdrant/bm25"
  rotation_matrix: "~/code/ccmcp/rotation_matrix.npy"
  # Generated by: python -m ccmcp setup
  # Never regenerate after first ingestion

extraction:
  backend: "none"             # "none" | "ollama" | "claude_api"
  ollama:
    model: "qwen2.5:14b"
    base_url: "http://localhost:11434"
    context_window: 32768
  claude_api:
    model: "claude-sonnet-4-20250514"
    # Set env: ANTHROPIC_API_KEY — never put key here

sources:
  filesystem:
    enabled: true
    roots: []
      # - "~/projects/my-repo"
    watch: true
    extensions: [".md",".rst",".txt",".py",".go",".rs",".js",".ts",
                 ".yaml",".yml",".json",".c",".cpp",".h"]
    ignore: ["node_modules",".git","__pycache__","*.pyc",
             "build","dist",".venv","*.egg-info"]

  web:
    enabled: false
    urls: []
    sitemaps: []
    crawl_depth: 0
    rate_limit_ms: 500
    user_agent: "ccmcp-ingestion/1.0"

  google_drive:
    enabled: false
    credentials_file: ""     # abs path to service account JSON
                             # macOS: leave empty, read from Keychain
    folders: []
      # - id: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
      #   name: "SR Linux Docs"
    poll_interval_min: 15

power:
  skip_full_scan_on_battery: true   # macOS only

state:
  db_path: "~/.local/share/ccmcp/state.db"
  artifact_ttl_days: 30
```

**Environment variables:**
```
ANTHROPIC_API_KEY   Claude API key (extraction.backend = claude_api)
QDRANT_API_KEY      Qdrant Cloud only
CCMCP_CONFIG        Config file path override (default: ~/code/ccmcp/config.yaml)
```

---

## 7. Qdrant Schema

### Collection setup

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="techdocs",
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            quantization_config=models.ScalarQuantizationConfig(
                scalar=models.ScalarQuantization(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)
```

### Point schema

```python
{
    "id":     "<uuid5(source_uri + str(chunk_index))>",  # deterministic
    "vector": {
        "dense":  [float, ...],       # 384 floats, rotation applied, INT8 stored
        "sparse": {"indices": [...], "values": [...]},
    },
    "payload": {
        "text":          "...",       # chunk text — what the LLM reads
        "source_uri":    "...",       # file://abs/path | https://... | drive://id
        "source_type":   "fs",        # "fs" | "web" | "drive" | "artifact"
        "doc_id":        "...",       # sha256(source_uri) — groups chunks per doc
        "chunk_index":   0,           # 0-based position within document
        "section":       "...",       # nearest heading, or ""
        "content_hash":  "sha256:...",
        "version":       1,           # incremented on update
        "ingested_at":   "2026-05-12T10:00:00Z",
    }
}
```

### Hybrid search

```python
results = client.query_points(
    collection_name="techdocs",
    prefetch=[
        models.Prefetch(query=dense_vector,  using="dense",  limit=20),
        models.Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
)
# Return payload["text"] to the LLM — never the vectors
```

---

## 8. Ingestion Pipeline

### Chunking rules

**Markdown / RST:**
- Split at heading boundaries (H1–H4)
- Each chunk = heading + body up to next heading
- Body > 512 tokens: split further at blank-line paragraph boundaries
- Preserve heading as prefix in every sub-chunk: `"## Section\n\n<body>"`
- Minimum chunk: 50 tokens. Merge smaller chunks with the next one.

**Source code (.py, .go, .rs, .js, .ts, .c, .cpp, .h):**
- Split at top-level function/class definitions using regex (no tree-sitter)
- Python: `^(def |class )`, Go: `^func `, Rust: `^(fn |impl |pub fn )`
- Each chunk = signature + body, max 512 tokens
- File-level docstring/module comment = chunk_index 0

**Plain text / YAML / JSON:**
- Split at blank-line boundaries, max 512 tokens

**Chunk dataclass:**
```python
@dataclass
class Chunk:
    text:        str
    section:     str   # nearest heading or ""
    chunk_index: int   # 0-based
    source_uri:  str
```

### Deduplication and versioned update

1. `content_hash = "sha256:" + sha256(file_content)`
2. Look up `source_uri` in state DB
3. Hash matches → skip
4. Hash differs → atomic swap:
   - Embed new chunks
   - Upsert with `version = old_version + 1`
   - Verify upsert count
   - Delete points where `doc_id = X AND version = old_version`
   - Update state DB
5. No record → insert as version 1

Never delete before inserting. Insert → verify → delete.

### Orphan cleanup

After every full scan:
1. Collect all `source_uri` values seen this scan
2. Query state DB for records with `last_seen < scan_start_time`
3. Delete Qdrant points by `doc_id` + delete state DB row for each orphan
4. Log count

Run after scan completes, not during.

---

## 9. Embedding Pipeline

```python
import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model  = TextEmbedding("BAAI/bge-small-en-v1.5")   # auto CoreML on macOS
sparse_model = SparseTextEmbedding("Qdrant/bm25")

R = np.load("~/code/ccmcp/rotation_matrix.npy")           # shape (384, 384)

def embed(texts: list[str]):
    dense  = np.array(list(dense_model.embed(texts))) @ R
    sparse = list(sparse_model.embed(texts))
    return dense, sparse

def generate_rotation_matrix(dim: int = 384, path: str = "rotation_matrix.npy"):
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists. Regenerating invalidates all indexed vectors.")
    R, _ = np.linalg.qr(np.random.randn(dim, dim))
    np.save(path, R)
```

The rotation matrix must never be regenerated after the first ingestion.
Changing R invalidates all vectors in Qdrant and requires a full re-ingest.

---

## 10. Ingestion Sources

### Filesystem

```python
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

def get_observer(root: str):
    # inotify does not work on WSL2 /mnt/ paths
    return PollingObserver(timeout=5) if root.startswith("/mnt/") else Observer()
```

Scan: `os.walk`, filter by extensions and ignore patterns, yield `SourceFile`.
Watch: `watchdog.Observer`, call callback on CREATE and MODIFY events.

### Web URLs

```python
import httpx
from readability import Document

def fetch(url: str, config, state) -> SourceFile | None:
    record  = state.get(url)
    headers = {}
    if record and record.etag:      headers["If-None-Match"]     = record.etag
    if record and record.last_mod:  headers["If-Modified-Since"] = record.last_mod

    resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    if resp.status_code == 304:
        return None   # unchanged

    text = html_to_text(Document(resp.text).summary())
    return SourceFile(source_uri=url, content=text,
                      etag=resp.headers.get("ETag"),
                      last_modified=resp.headers.get("Last-Modified"))
```

### Google Drive

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_service(credentials_file: str):
    creds = service_account.Credentials.from_service_account_file(
        credentials_file, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)
```

Supported MIME types: `application/vnd.google-apps.document` (export as text),
`text/plain`, `text/markdown`. Change detection via Drive `modifiedTime` + `version`.

On macOS: load credentials JSON from Keychain (see Section 4.3).

---

## 11. Document & Artifact Management

### SQLite state store

```sql
CREATE TABLE IF NOT EXISTS sources (
    source_uri    TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    last_seen     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ok',
    etag          TEXT,
    last_modified TEXT,
    drive_version TEXT
);
```

### Agent artifacts

Written by `qdrant-store` MCP tool → stored in `ccmcp-artifacts` collection
with `source_type: "artifact"` and `session_id`. TTL cleanup at startup:

```python
cutoff = datetime.utcnow() - timedelta(days=config.state.artifact_ttl_days)
client.delete("ccmcp-artifacts",
    points_selector=models.FilterSelector(filter=models.Filter(must=[
        models.FieldCondition(key="ingested_at",
            range=models.DatetimeRange(lt=cutoff.isoformat()))
    ])))
```

---

## 12. MCP Integration

### Claude Code CLI — add to `~/.claude.json`

```json
{
  "mcpServers": {
    "ccmcp": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "techdocs",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}
```

### Cursor — SSE, persistent service

```bash
uvx mcp-server-qdrant --transport sse --port 7700
# Point Cursor at: http://localhost:7700/sse
```

Managed by systemd (`ccmcp-mcp-sse.service`) on Linux/WSL2,
launchd (`ai.ccmcp.mcp.sse.plist`) on macOS.

---

## 13. CLI Reference

```
ccmcp setup
  Create Qdrant collection (idempotent).
  Generate rotation_matrix.npy (refuses to overwrite).
  Create state DB directory.
  Print MCP config snippet for ~/.claude.json.

ccmcp scan
  One-shot full ingestion of all enabled sources.
  Skips on battery if power.skip_full_scan_on_battery (macOS).

ccmcp watch
  Full scan once, then continuous filesystem watch.
  Web and Drive re-checked on their poll interval.

ccmcp ingest <path-or-url>
  Ingest a single file or URL immediately.

ccmcp status
  Qdrant collection stats + source summary from state DB.

ccmcp reset
  Drop collection, clear state DB.
  Prompts: "Type RESET to confirm".
  Does not delete rotation_matrix.npy.
```

---

## 14. Nix DevShell (WSL2 + Linux)

```nix
# ~/code/ccmcp/flake.nix
{
  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs   = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.qdrant
            pkgs.uv
            (python.withPackages (ps: [ ps.pip ps.virtualenv ]))
          ];

          shellHook = ''
            export CCMCP_CONFIG="$PWD/config.yaml"
            export QDRANT_DATA="$PWD/.qdrant"

            if [ ! -d .venv ]; then
              python -m venv .venv
              .venv/bin/pip install -e ".[dev]" -q
            fi
            source .venv/bin/activate

            if ! curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
              mkdir -p "$QDRANT_DATA"
              qdrant --storage-path "$QDRANT_DATA" \
                     --log-level warn >/tmp/qdrant.log 2>&1 &
              echo "Qdrant starting (log: /tmp/qdrant.log)"
              sleep 1
            fi

            echo "ccmcp devShell ready — $(python --version)"
            echo "First time? Run: ccmcp setup"
          '';
        };
      }
    );
}
```

Usage:
```bash
cd /mnt/d/code/ccmcp
nix develop       # enter isolated shell, Qdrant starts automatically
ccmcp setup       # first time only
ccmcp scan
ccmcp watch
```

---

## 15. Deploy Configs

### Linux / WSL2 — systemd units

**deploy/linux/qdrant.service**
```ini
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/qdrant --storage-path %h/.local/share/ccmcp/qdrant
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

**deploy/linux/ccmcp-controller.service**
```ini
[Unit]
Description=ccmcp ingestion controller
After=qdrant.service

[Service]
Type=simple
WorkingDirectory=%h/code/ccmcp
ExecStart=%h/code/ccmcp/.venv/bin/python -m ccmcp watch
EnvironmentFile=%h/.config/ccmcp/env
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

**deploy/linux/ccmcp-mcp-sse.service**
```ini
[Unit]
Description=ccmcp MCP SSE server (Cursor)
After=qdrant.service

[Service]
Type=simple
ExecStart=/usr/local/bin/uvx mcp-server-qdrant --transport sse --port 7700
Environment=QDRANT_URL=http://localhost:6333
Environment=COLLECTION_NAME=techdocs
Restart=on-failure

[Install]
WantedBy=default.target
```

**deploy/linux/install.sh**
```bash
#!/usr/bin/env bash
set -e
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for unit in qdrant.service ccmcp-controller.service ccmcp-mcp-sse.service; do
    ln -sf "$SCRIPT_DIR/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload
systemctl --user enable --now qdrant.service ccmcp-controller.service ccmcp-mcp-sse.service
echo "Done. Logs: journalctl --user -u ccmcp-controller -f"
```

### macOS — launchd plists

**deploy/macos/ai.qdrant.server.plist**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.qdrant.server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/qdrant</string>
    <string>--storage-path</string>
    <string>/Users/YOU/Library/Application Support/ccmcp/qdrant</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/qdrant.log</string>
  <key>StandardErrorPath</key><string>/tmp/qdrant.log</string>
</dict></plist>
```

**deploy/macos/ai.ccmcp.controller.plist**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.ccmcp.controller</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/python3</string>
    <string>-m</string><string>ccmcp</string><string>watch</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/code/ccmcp</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CCMCP_CONFIG</key><string>/Users/YOU/code/ccmcp/config.yaml</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/ccmcp-controller.log</string>
  <key>StandardErrorPath</key><string>/tmp/ccmcp-controller.log</string>
</dict></plist>
```

**deploy/macos/ai.ccmcp.mcp.sse.plist**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.ccmcp.mcp.sse</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uvx</string>
    <string>mcp-server-qdrant</string>
    <string>--transport</string><string>sse</string>
    <string>--port</string><string>7700</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>QDRANT_URL</key><string>http://localhost:6333</string>
    <key>COLLECTION_NAME</key><string>techdocs</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/ccmcp-mcp-sse.log</string>
  <key>StandardErrorPath</key><string>/tmp/ccmcp-mcp-sse.log</string>
</dict></plist>
```

**deploy/macos/install.sh**
```bash
#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS"
for plist in ai.qdrant.server.plist ai.ccmcp.controller.plist ai.ccmcp.mcp.sse.plist; do
    cp "$SCRIPT_DIR/$plist" "$AGENTS/$plist"
    launchctl load "$AGENTS/$plist"
done
echo "LaunchAgents loaded."
```

### WSL2 setup script

**deploy/wsl2/setup.sh**
```bash
#!/usr/bin/env bash
set -e
echo "=== ccmcp WSL2 setup ==="

if ! grep -q "systemd=true" /etc/wsl.conf 2>/dev/null; then
    echo "Adding systemd=true to /etc/wsl.conf (requires sudo)"
    printf "[boot]\nsystemd=true\n" | sudo tee -a /etc/wsl.conf
    echo "Run 'wsl --shutdown' from PowerShell, then reopen WSL2."
    exit 1
fi
echo "✓ systemd enabled"

mkdir -p ~/.local/share/ccmcp ~/.config/ccmcp

ENV="$HOME/.config/ccmcp/env"
if [ ! -f "$ENV" ]; then
    printf "# ANTHROPIC_API_KEY=sk-ant-...\n# QDRANT_API_KEY=\n" > "$ENV"
    echo "Created $ENV — add API keys if needed"
fi

echo ""
echo "Next steps:"
echo "  cd /mnt/d/code/ccmcp && nix develop"
echo "  ccmcp setup"
echo "  edit config.yaml, add filesystem roots"
echo "  ccmcp scan"
echo "  deploy/linux/install.sh   (to run as services)"
```

---

## 16. Testing

```bash
pytest tests/ -v                           # unit tests only (CI)
pytest tests/ -v -m integration            # integration tests (needs Qdrant)
pytest tests/test_chunker.py -v            # single module
```

**Markers:** `@pytest.mark.integration` requires Qdrant on localhost:6333.
CI (GitHub Actions) runs unit tests only. Integration tests run inside `nix develop`.

**Coverage per module:**
- `test_config.py`      load from temp file, missing fields, path expansion
- `test_chunker.py`     each file type, heading preserved, max size enforced
- `test_embedder.py`    correct dims, rotation applied, sparse non-zero, no overwrite
- `test_store.py`       upsert → search → versioned swap → delete  [integration]
- `test_state.py`       all CRUD ops, unseen query, temp DB
- `test_filesystem.py`  scan temp dir tree, watch callback fires
- `test_web.py`         mock httpx: 200+extract, 304 skip, sitemap parse
- `test_drive.py`       mock Drive API: list, fetch, change detection

---

## 17. Build Order

One PR per item. Implement in dependency order — each module only imports
those above it in this list.

```
 1.  ccmcp/config.py
 2.  ccmcp/state.py
 3.  ccmcp/chunker.py
 4.  ccmcp/embedder.py
 5.  ccmcp/store.py
 6.  ccmcp/sources/filesystem.py
 7.  ccmcp/sources/web.py
 8.  ccmcp/sources/drive.py
 9.  ccmcp/controller.py
10.  ccmcp/__main__.py
11.  deploy/linux/
12.  deploy/macos/
13.  deploy/wsl2/
14.  flake.nix
```

---

## 18. Phase 2 — Proposition Extraction (do not implement yet)

When `extraction.backend != "none"`, structural chunks are passed through an
extraction agent before embedding. The agent returns atomic propositions —
self-contained factual statements — plus optional synthetic questions.

Each proposition becomes a Qdrant point with:
- `payload.text`               = original chunk text (what the LLM reads)
- `payload.proposition`        = proposition text (what was embedded)
- `payload.extraction_backend` = "ollama" | "claude_api"

Phase 1 and Phase 2 points coexist in the same collection, differentiated
by the `extraction_backend` payload field (absent on Phase 1 points).

**Ollama:**
```python
resp = httpx.post("http://localhost:11434/api/generate", json={
    "model": config.extraction.ollama.model,
    "prompt": PROPOSITION_PROMPT.format(text=chunk.text),
    "stream": False,
})
```

**Claude API with prompt caching:**
```python
client.messages.create(
    model=config.extraction.claude_api.model,
    max_tokens=1024,
    system=[{"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": chunk.text}],
)
```

Switching backends does not require re-indexing Phase 1 points.
Only newly ingested or changed documents use the active backend.
