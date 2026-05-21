"""Prometheus metrics and web app for ccmcp observability.

make_observable_app wraps the MCP SSE app with:
  GET  /metrics  — Prometheus scrape endpoint
  GET  /         — browser search UI (when store/embedder provided)
  GET  /scopes   — list indexed project scopes as JSON
  POST /search   — hybrid search, returns JSON results
"""
from __future__ import annotations

import time
from threading import Thread

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ── Collection / store gauges (updated by background poller) ─────────────────

COLLECTION_POINTS = Gauge(
    "ccmcp_collection_points",
    "Current number of points in the main document collection",
)
ARTIFACT_POINTS = Gauge(
    "ccmcp_artifact_points",
    "Current number of points in the artifacts collection",
)
SOURCES_INDEXED = Gauge(
    "ccmcp_sources_indexed",
    "Number of unique source URIs tracked in the state database",
)

# ── Ingestion counters ────────────────────────────────────────────────────────

CHUNKS_INGESTED = Counter(
    "ccmcp_chunks_ingested_total",
    "Chunks successfully written to Qdrant",
    ["source_type"],
)
DOCUMENTS_INGESTED = Counter(
    "ccmcp_documents_ingested_total",
    "Documents fully ingested (new or changed content)",
    ["source_type"],
)
INGEST_ERRORS = Counter(
    "ccmcp_ingest_errors_total",
    "Documents that raised an exception during ingestion",
    ["source_type"],
)

# ── Timing histograms ─────────────────────────────────────────────────────────

INGEST_SECONDS = Histogram(
    "ccmcp_ingest_seconds",
    "Wall time to ingest one document end-to-end (chunk → embed → upsert)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
EMBED_SECONDS = Histogram(
    "ccmcp_embed_seconds",
    "Wall time for one embed() call covering a batch of texts",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
SEARCH_SECONDS = Histogram(
    "ccmcp_search_seconds",
    "Wall time for one hybrid search() call",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
)

# ── Size distributions ────────────────────────────────────────────────────────

CHUNK_CHARS = Histogram(
    "ccmcp_chunk_chars",
    "Character length of each text chunk written to Qdrant",
    buckets=[100, 250, 500, 1_000, 2_000, 4_000, 8_000],
)
EMBED_BATCH_SIZE = Histogram(
    "ccmcp_embed_batch_size",
    "Number of texts passed to each embed() call",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)
SEARCH_RESULTS_RETURNED = Histogram(
    "ccmcp_search_results_returned",
    "Number of results returned by each search() call",
    buckets=[0, 1, 2, 5, 10, 20],
)


# ── Background collection-size poller ─────────────────────────────────────────

def start_collection_poller(store, state, interval: int = 30) -> None:
    """Refresh collection-size gauges every *interval* seconds in a daemon thread."""

    def _poll() -> None:
        while True:
            time.sleep(interval)
            try:
                info = store.collection_info()
                COLLECTION_POINTS.set(info.get("points_count") or 0)
                SOURCES_INDEXED.set(len(state.all()))
                try:
                    ainfo = store.artifact_collection_info()
                    ARTIFACT_POINTS.set(ainfo.get("points_count") or 0)
                except Exception:
                    pass
            except Exception:
                pass

    Thread(target=_poll, daemon=True, name="ccmcp-metrics-poller").start()


# ── Browser search UI (embedded, no static-file serving needed) ───────────────

_SEARCH_UI_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ccmcp · search</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }
    :root {
      --bg:      #0d1117; --surface: #161b22; --border: #30363d;
      --text:    #e6edf3; --muted:   #8b949e; --accent:  #58a6ff;
      --green:   #3fb950; --purple:  #d2a8ff; --yellow:  #d29922;
    }
    body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; line-height: 1.5; min-height: 100vh;
    }
    header {
      border-bottom: 1px solid var(--border);
      padding: 12px 24px; display: flex; align-items: center; gap: 12px;
    }
    header h1 { font-size: 16px; font-weight: 600; color: var(--accent); font-family: monospace; }
    header .tagline { color: var(--muted); font-size: 12px; }
    .pill {
      margin-left: auto; font-size: 11px; padding: 2px 8px;
      border-radius: 10px; border: 1px solid var(--border); color: var(--muted);
    }
    .pill.ok { color: var(--green); border-color: var(--green); }
    main { max-width: 860px; margin: 0 auto; padding: 32px 24px; }
    .search-row { display: flex; gap: 8px; margin-bottom: 16px; }
    #query {
      flex: 1; padding: 10px 14px; background: var(--surface);
      border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); font-size: 15px; outline: none; transition: border-color .15s;
    }
    #query:focus { border-color: var(--accent); }
    #scope {
      padding: 10px 12px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; color: var(--text); font-size: 13px;
      outline: none; cursor: pointer; min-width: 150px;
    }
    #scope:focus { border-color: var(--accent); }
    button[type=submit] {
      padding: 10px 20px; background: var(--accent); border: none;
      border-radius: 6px; color: #0d1117; font-size: 14px; font-weight: 600;
      cursor: pointer; transition: opacity .15s; white-space: nowrap;
    }
    button[type=submit]:hover { opacity: .85; }
    button[type=submit]:disabled { opacity: .4; cursor: not-allowed; }
    #status { color: var(--muted); font-size: 12px; margin-bottom: 16px; min-height: 18px; }
    .result {
      background: var(--surface); border: 1px solid var(--border);
      border-left: 3px solid var(--border);
      border-radius: 8px; padding: 16px 18px; margin-bottom: 12px;
    }
    .result.fs      { border-left-color: var(--accent); }
    .result.web     { border-left-color: var(--green); }
    .result.drive   { border-left-color: var(--purple); }
    .result.artifact{ border-left-color: var(--yellow); }
    .result-meta {
      display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
    }
    .badge {
      font-size: 10px; padding: 1px 6px; border-radius: 10px;
      border: 1px solid var(--border); color: var(--muted);
      text-transform: uppercase; letter-spacing: .04em;
    }
    .result-source {
      font-family: monospace; font-size: 12px; color: var(--accent); word-break: break-all;
    }
    .result-section { color: var(--muted); font-size: 12px; }
    .result-section::before { content: "\\00a7 "; }
    .result-text {
      color: var(--text); font-size: 13px; line-height: 1.6;
      white-space: pre-wrap; word-break: break-word;
    }
    .result-text.truncated {
      display: -webkit-box; -webkit-line-clamp: 4;
      -webkit-box-orient: vertical; overflow: hidden;
    }
    .expand-btn {
      background: none; border: none; color: var(--accent);
      font-size: 12px; cursor: pointer; padding: 4px 0; margin-top: 4px; display: block;
    }
    .no-results { text-align: center; color: var(--muted); padding: 48px 0; }
    .hint { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
    kbd {
      font-family: monospace; font-size: 11px; padding: 1px 5px;
      border: 1px solid var(--border); border-radius: 4px; background: var(--surface);
    }
  </style>
</head>
<body>
  <header>
    <h1>ccmcp</h1>
    <span class="tagline">knowledge base search</span>
    <span class="pill" id="status-pill">connecting…</span>
  </header>
  <main>
    <form id="search-form">
      <div class="search-row">
        <input id="query" type="search" placeholder="Search your knowledge base…"
               autofocus autocomplete="off" spellcheck="false">
        <select id="scope"><option value="">All sources</option></select>
        <button type="submit" id="search-btn">Search</button>
      </div>
    </form>
    <p class="hint">Press <kbd>Enter</kbd> to search · <kbd>Ctrl K</kbd> to focus</p>
    <div id="status"></div>
    <div id="results"></div>
  </main>
  <script>
    const $ = id => document.getElementById(id);

    function stype(r) {
      if (r.source_type) return r.source_type;
      const u = r.source_uri || '';
      if (u.startsWith('file://')) return 'fs';
      if (u.startsWith('http')) return 'web';
      if (u.startsWith('drive://')) return 'drive';
      return 'artifact';
    }

    function fmtUri(uri) {
      return uri.startsWith('file://') ? uri.slice(7) : uri;
    }

    function renderResult(r) {
      const st = stype(r);
      const el = document.createElement('div');
      el.className = 'result ' + st;

      const meta = document.createElement('div');
      meta.className = 'result-meta';

      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = st;
      meta.appendChild(badge);

      const src = document.createElement('span');
      src.className = 'result-source';
      src.textContent = fmtUri(r.source_uri || '');
      meta.appendChild(src);

      if (r.section) {
        const sec = document.createElement('span');
        sec.className = 'result-section';
        sec.textContent = r.section;
        meta.appendChild(sec);
      }
      el.appendChild(meta);

      const text = document.createElement('pre');
      text.className = 'result-text truncated';
      text.textContent = r.text || '';
      el.appendChild(text);

      const btn = document.createElement('button');
      btn.className = 'expand-btn';
      btn.textContent = 'Show more';
      btn.onclick = () => {
        const t = text.classList.toggle('truncated');
        btn.textContent = t ? 'Show more' : 'Show less';
      };
      el.appendChild(btn);
      return el;
    }

    async function loadScopes() {
      try {
        const r = await fetch('/scopes');
        const scopes = await r.json();
        const sel = $('scope');
        scopes.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.name || s.source_root;
          opt.textContent = s.name || s.source_root.split('/').filter(Boolean).at(-1);
          sel.appendChild(opt);
        });
        const pill = $('status-pill');
        pill.textContent = scopes.length + ' source' + (scopes.length !== 1 ? 's' : '');
        pill.className = 'pill ok';
      } catch(e) {
        $('status-pill').textContent = 'offline';
      }
    }

    $('search-form').addEventListener('submit', async e => {
      e.preventDefault();
      const query = $('query').value.trim();
      if (!query) return;

      const scope = $('scope').value;
      const btn = $('search-btn');
      const statusEl = $('status');
      const resultsEl = $('results');

      btn.disabled = true;
      statusEl.textContent = 'Searching…';
      resultsEl.innerHTML = '';

      try {
        const body = { query, limit: 15 };
        if (scope) body.scope = [scope];

        const resp = await fetch('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (!resp.ok) {
          statusEl.textContent = 'Error: ' + (data.detail || resp.statusText);
          return;
        }

        const n = data.count;
        statusEl.textContent = n > 0
          ? n + ' result' + (n !== 1 ? 's' : '') + ' for \\u201c' + data.query + '\\u201d'
          : '';

        if (n === 0) {
          resultsEl.innerHTML = '<div class="no-results">No results found.</div>';
        } else {
          data.results.forEach(r => resultsEl.appendChild(renderResult(r)));
        }
      } catch(err) {
        statusEl.textContent = 'Network error: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    });

    document.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        $('query').focus();
        $('query').select();
      }
    });

    loadScopes();
  </script>
</body>
</html>
"""


# ── ASGI app wrapper ──────────────────────────────────────────────────────────

def make_observable_app(mcp_sse_app, *, store=None, embedder=None):
    """Wrap the MCP SSE app with observability and human search routes.

    Always adds:
      GET /metrics  — Prometheus scrape endpoint

    When store and embedder are provided, also adds:
      GET /         — browser search UI
      GET /scopes   — list indexed project scopes as JSON
      POST /search  — hybrid search, returns JSON results
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from starlette.routing import Mount, Route

    async def _metrics(request: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    routes: list = [Route("/metrics", _metrics)]

    if store is not None and embedder is not None:
        async def _ui(request: Request) -> HTMLResponse:
            return HTMLResponse(_SEARCH_UI_HTML)

        async def _scopes(request: Request) -> JSONResponse:
            try:
                return JSONResponse(store.list_scopes())
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)

        async def _search(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"detail": "invalid JSON"}, status_code=400)

            query = (body.get("query") or "").strip()
            if not query:
                return JSONResponse({"detail": "query is required"}, status_code=400)

            limit = min(max(1, int(body.get("limit", 10))), 50)
            scope: list[str] | None = body.get("scope")  # list or null

            search_filter = None
            if scope and scope != ["*"]:
                from qdrant_client import models as qm
                search_filter = qm.Filter(should=[
                    qm.FieldCondition(key="project_name", match=qm.MatchAny(any=scope)),
                    qm.FieldCondition(key="tags", match=qm.MatchAny(any=scope)),
                ])

            dense, sparse_list = embedder.embed([query])
            results = store.search(dense[0], sparse_list[0], limit=limit, filter=search_filter)
            return JSONResponse({"query": query, "count": len(results), "results": results})

        routes += [
            Route("/", _ui, methods=["GET"]),
            Route("/scopes", _scopes, methods=["GET"]),
            Route("/search", _search, methods=["POST"]),
        ]

    routes.append(Mount("/", app=mcp_sse_app))
    return Starlette(routes=routes)
