"""Prometheus metrics for ccmcp observability.

Metric naming follows the Prometheus convention: snake_case, unit suffix
(_seconds, _bytes, _total, _chars).

prometheus_client automatically exposes process_cpu_seconds_total,
process_resident_memory_bytes, process_open_fds, and friends via its
built-in ProcessCollector — CPU and memory tracking are therefore free.
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


# ── ASGI app wrapper ──────────────────────────────────────────────────────────

def make_observable_app(mcp_sse_app):
    """Wrap the MCP SSE app with a /metrics endpoint for Prometheus scraping."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    async def _metrics(request: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return Starlette(routes=[
        Route("/metrics", _metrics),
        Mount("/", app=mcp_sse_app),
    ])
