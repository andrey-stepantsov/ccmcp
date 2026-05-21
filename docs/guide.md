# ccmcp — User Guide

## 1. What ccmcp does

ccmcp is a self-hosted knowledge base that ingests technical documentation, source code, and wiki articles from local filesystems, web URLs, and Google Drive; chunks and embeds the content using a hybrid dense+sparse model; stores vectors in a local Qdrant database; and exposes the index to AI coding assistants via an MCP server and to humans via a browser search UI — all without sending your content to any external service.

---

## 2. The problem it solves

**Context window limits.** Large codebases, documentation sets, and runbook libraries exceed what fits in a single prompt. Claude Code and Cursor can only use what you explicitly paste in, or what the model already knows from training.

**Ephemeral sessions.** Notes, decisions, and architectural context written during one session are gone in the next. There is no persistent memory by default.

**Proprietary content.** Vendor SDKs, internal APIs, private runbooks, and anything under NDA are not in any model's training data. The model will hallucinate or refuse when asked about them.

**Exact term misses.** Semantic vector search is poor at matching CLI flags, hostnames, error codes, and version strings. A query for `--storage-path` should find `--storage-path`, not just documents about "configuration options."

---

## 3. How it works

You point ccmcp at one or more sources — directories on disk, a list of URLs, or Google Drive folders. ccmcp reads each document, splits it into structured chunks (at heading and function boundaries), and embeds each chunk with both a dense semantic vector and a sparse BM25 vector. Both vectors are written into a local Qdrant collection.

When Claude Code or Cursor needs context, the MCP bridge intercepts the query, embeds it the same way, runs a hybrid search against Qdrant, and returns the top-ranked text chunks as tool results. The AI reads those chunks and answers using your actual content.

The same Qdrant collection also powers a browser UI at `http://localhost:7700`. You can run the same search the AI runs, see scores, and inspect chunk boundaries.

File changes are tracked via SHA-256 hashes. Updating a file re-embeds only that document's chunks and atomically swaps the old vectors out.

---

## 4. Core concepts

**Sources.** A source is a configured input: a filesystem root (with extension filters and ignore patterns), a list of URLs (with optional sitemap crawling), or a Google Drive folder (polled via service account). Sources can be enabled or disabled independently.

**Hybrid search.** Each query runs two parallel searches — dense cosine similarity over 384-dimensional vectors for semantic matching, and BM25 sparse matching for exact vocabulary. Results are merged via Reciprocal Rank Fusion. BM25 matters for technical content: if your docs mention `--storage-path` or `grpc.StatusCode.UNAVAILABLE`, a semantic-only search will miss them unless the query uses nearly identical wording. BM25 catches exact terms regardless.

**Scopes.** The `qdrant_list_scopes` MCP tool returns the distinct `source_uri` prefixes in the index. This lets an AI (or a human) understand what is indexed and narrow a search to a specific directory or domain.

**The MCP bridge.** ccmcp exposes an SSE MCP server on port 7700. Any MCP client that supports SSE transport — Claude Code CLI, Cursor, or any other — connects to `http://localhost:7700/sse` and gets three tools: `qdrant_find` (hybrid search), `qdrant_store` (write an artifact or note back to the index), and `qdrant_list_scopes` (enumerate indexed sources). The AI decides when to call these; you do not have to instruct it to look things up.

---

## 5. Scenarios

### A — Personal codebase assistant (solo dev with large repo)

You have a monorepo with 200k lines of Go, a docs directory, and a changelog. You add the repo root as a filesystem source. ccmcp indexes everything overnight. Now when you ask Claude Code "where is the retry logic for the gRPC client, and what backoff strategy does it use?", it calls `qdrant_find`, retrieves the relevant function chunks, and answers from your actual implementation rather than guessing from Go idioms.

Without ccmcp: you paste in files manually, or accept answers based on training data that knows nothing about your specific retry wrapper.

### B — Team-shared knowledge base (server deployment)

You deploy ccmcp on a shared Linux server. Every developer on the team points their Claude Code and Cursor at `http://server:7700/sse`. The index includes the internal API spec, architecture decision records, runbooks, and incident post-mortems. A new team member asking "how do we handle database failover?" gets the actual runbook, not a generic answer.

Changes to docs in the indexed directories are picked up within minutes by the filesystem watcher. No one needs to manually update a wiki or paste context into every conversation.

### C — Vendor/API documentation (avoiding stale training data)

You use a vendor SDK that ships monthly. The model's training data may be months behind. You add the vendor's docs site to the `web.urls` list. ccmcp fetches, indexes, and re-fetches on change (via ETag/Last-Modified). Your AI assistant answers questions about the SDK from the current version you actually have installed, not from training data that predates the breaking change in the last release.

### D — Proprietary or air-gapped content

You have technical documentation that cannot leave your hardware — NDAs, unreleased product specs, customer data. ccmcp runs entirely on your machine or your server. The Qdrant database is local. The MCP server is local. No content is sent anywhere. The embedding models run via ONNX locally. Nothing touches an external API at ingestion time.

The `qdrant_store` tool lets the AI write artifacts back to the same local index — session notes, architectural summaries, research outputs — which persist across sessions and are searchable in future ones.

### E — Claude Code and Cursor on the same index

Both clients connect to `http://localhost:7700/sse` simultaneously. The same Qdrant collection serves both. Work done in one tool is immediately visible to the other. If you store a note via `qdrant_store` in Claude Code, a subsequent search in Cursor finds it. If you index a new directory, both clients see it on their next query.

### F — Human search (browser UI)

You want to find something in the index yourself, without involving an AI. Open `http://localhost:7700` in a browser, type your query, and see the same ranked chunks the AI sees — with source URIs, section headings, and scores. Useful for verifying what the AI has access to, debugging chunk boundaries, or quickly locating a specific runbook without opening a new AI session.

---

## 6. What the AI sees

A concrete example. Your index contains the ccmcp source code and its deployment docs.

**Query:** "how does the filesystem watcher handle WSL2 paths?"

ccmcp runs hybrid search and returns three chunks:

**Chunk 1** — `file:///home/you/code/ccmcp/ccmcp/sources/filesystem.py`, section: `get_observer`
```
def get_observer(root: str):
    # inotify does not work on WSL2 /mnt/ paths
    return PollingObserver(timeout=5) if root.startswith("/mnt/") else Observer()
```

**Chunk 2** — `file:///home/you/code/ccmcp/CLAUDE.md`, section: `inotify caveat`
```
inotify events do NOT fire for paths under /mnt/c/ or /mnt/d/ (Windows
filesystem, 9P layer). Keep source repos inside WSL2 (~/) for live watch
mode. For Windows-side paths, use scheduled `ccmcp scan` instead. The
filesystem source detects /mnt/ paths automatically and falls back to
PollingObserver.
```

**Chunk 3** — `file:///home/you/code/ccmcp/ccmcp/sources/filesystem.py`, section: `module docstring`
```
Filesystem source: inotify-based watcher (Linux) with automatic fallback
to PollingObserver for paths that do not support inotify (WSL2 /mnt/,
network mounts).
```

The AI now has the implementation, the documented rationale, and the module context. It answers: "For paths starting with `/mnt/`, ccmcp automatically uses `PollingObserver` with a 5-second timeout instead of inotify, because the Windows 9P filesystem layer does not propagate inotify events. For repos that live inside the WSL2 filesystem (`~/`), the native inotify-based `Observer` is used."

That answer came from your codebase, not from the model's training data about watchdog or WSL2 in general.

---

## 7. The search UI

The browser UI at `http://localhost:7700` provides a single search box over the full index. Each result shows:

- The matched chunk text
- The source URI (file path, URL, or Drive ID)
- The section heading (nearest heading in the document)
- The RRF fusion score and individual dense/sparse scores

It is read-only. There is no editing UI. The intent is inspection and verification, not content management. If you want to understand why the AI gave a particular answer, run the same query here and see exactly what it retrieved.

---

## 8. Deployment paths

| Method | Time | Notes |
|---|---|---|
| Docker Compose | ~5 min | `docker compose up -d`. No prerequisites except Docker. Qdrant and the MCP server start together. Data persists in a named volume. Best for quick evaluation or team server. |
| Linux / WSL2 native | ~15 min | Qdrant binary + Python venv. Systemd user units for automatic startup. WSL2 requires systemd enabled in `/etc/wsl.conf`. Uses inotify on native paths, polling on `/mnt/` paths. |
| macOS M-series | ~15 min | Qdrant via Homebrew, Python via uv. ONNX CoreML execution provider selected automatically — embedding runs on the Neural Engine. Launchd agents for startup. ~800–1500 chunks/sec ingestion vs. ~80–200 on CPU-only Linux. |

For a team server, Docker Compose is the straightforward choice. For a personal machine where you care about startup speed and resource overhead, the native deployment is lighter.

---

## 9. Limitations and honest trade-offs

**Index freshness lag.** Filesystem changes are picked up by inotify within seconds on native paths, but by polling (every 5 seconds) on WSL2 `/mnt/` paths. Web URLs and Google Drive are polled on a configurable interval (default 15 minutes). The index is never fully real-time.

**Chunk quality is structural, not propositional.** ccmcp splits documents at heading and function boundaries. This produces reasonable chunks but not optimal ones. A 400-token function body that contains three distinct facts will surface as one chunk. Proposition-level extraction (where an LLM rewrites each chunk into atomic factual statements) would improve retrieval precision but requires an additional inference step and is not implemented in the current release.

**No whole-file context.** Retrieval returns chunks, not full files. If answering a question requires reading an entire 2000-line file as a unit, ccmcp will return fragments. For that case, you should provide the file directly. ccmcp is best when the answer is localized to a section.

**When you don't need it.** If your entire codebase fits comfortably in a context window and you rarely ask about it across sessions, the overhead of running ccmcp is not justified. It adds value at scale — large repos, large doc sets, or content the model has never seen.

---

## 10. Getting started

**Step 1: Start the stack**

```bash
docker compose up -d
```

This starts Qdrant and the ccmcp MCP SSE server. The browser UI is available at `http://localhost:7700`. Add your sources to `config.yaml` (copy from `config.example.yaml`), then trigger the first ingestion:

```bash
docker compose exec ccmcp ccmcp scan
```

**Step 2: Connect Claude Code**

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "ccmcp": {
      "type": "sse",
      "url": "http://localhost:7700/sse"
    }
  }
}
```

Restart Claude Code. The `qdrant_find`, `qdrant_store`, and `qdrant_list_scopes` tools will appear automatically.

**Step 3: Connect Cursor**

In Cursor settings, under MCP servers, add:

```
http://localhost:7700/sse
```

Both clients connect to the same index simultaneously. No additional configuration is required.
