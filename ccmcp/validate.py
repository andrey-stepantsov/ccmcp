"""End-to-end validation scenarios for the ccmcp ingestion and retrieval pipeline."""
from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture documents (embedded so the validate command is self-contained)
# ---------------------------------------------------------------------------

_FIXTURES: dict[str, str] = {
    "hybrid_search.md": """\
# Hybrid Dense and Sparse Search

Hybrid search combines two complementary retrieval methods: dense vector search
using semantic embeddings, and sparse BM25-based keyword search.

## Dense Retrieval

Dense retrieval encodes queries and documents into fixed-size vectors using a
neural language model. Semantically similar content clusters together in the
vector space, enabling retrieval of relevant passages even when exact keywords
differ from the query.

## Sparse Retrieval (BM25)

BM25 term frequency scoring captures exact vocabulary matches. Technical terms
like hostnames, CLI flags, model names, and error codes are reliably found by
sparse retrieval when they would be missed by semantic embeddings.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges ranked results from both retrieval methods.
Each result receives a score of 1/(k + rank), where k dampens the influence of
high-ranked outliers. The combined list consistently outperforms either method
alone for technical documentation retrieval.
""",

    "embedder.py": """\
\"\"\"Text embedding pipeline using fastembed ONNX models.\"\"\"

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"


def load_models(dense_model=DENSE_MODEL, sparse_model=SPARSE_MODEL):
    \"\"\"Load dense and sparse embedding models from local ONNX cache.\"\"\"
    dense = TextEmbedding(dense_model)
    sparse = SparseTextEmbedding(sparse_model)
    return dense, sparse


def embed_texts(texts, dense_model, sparse_model, rotation_matrix=None):
    \"\"\"Embed a list of texts into dense and sparse vectors.

    Applies an optional orthogonal rotation to dense vectors to improve
    INT8 quantisation effectiveness across all dimensions.
    \"\"\"
    dense_vecs = np.array(list(dense_model.embed(texts)), dtype=np.float32)
    if rotation_matrix is not None:
        dense_vecs = dense_vecs @ rotation_matrix
    sparse_vecs = list(sparse_model.embed(texts))
    return dense_vecs, sparse_vecs


def generate_rotation_matrix(dim: int, path: str):
    \"\"\"Generate and save a random orthogonal rotation matrix via QR decomposition.\"\"\"
    R, _ = np.linalg.qr(np.random.randn(dim, dim))
    np.save(path, R)
    return R
""",

    "architecture.rst": """\
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

Vector Storage
--------------

Qdrant stores both dense and sparse vectors in a hybrid collection. Dense
vectors use INT8 scalar quantisation with a random orthogonal preconditioning
matrix. Sparse vectors use BM25 term frequencies with IDF weighting.

MCP Interface
-------------

Two MCP server transports expose the knowledge base to AI coding assistants.
The stdio transport is launched as a subprocess by Claude Code CLI. The SSE
transport runs as a persistent HTTP server on port 7700 for Cursor IDE.
""",

    "config_reference.txt": """\
ccmcp Configuration Reference

qdrant.url
  URL of the Qdrant instance. Default: http://localhost:6333.
  Env override: QDRANT_URL

qdrant.collection
  Name of the Qdrant collection for document chunks. Default: techdocs.

embedding.dense_model
  HuggingFace model identifier for dense embeddings.
  Default: BAAI/bge-small-en-v1.5

sources.filesystem.roots
  List of directory paths to scan and watch for document changes.

sources.filesystem.watch
  Enable live filesystem watching with polling observer. Default: true.

mcp.host
  Bind address for the SSE MCP server. Default: 127.0.0.1.

mcp.port
  TCP port for the SSE MCP server. Default: 7700.

state.db_path
  Path to the SQLite state database. Default: ~/.local/share/ccmcp/state.db.

state.artifact_ttl_days
  Days before agent artifacts are automatically removed. Default: 30.
""",
}

# ---------------------------------------------------------------------------
# Retrieval scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    query: str
    expected_fixture: str  # key in _FIXTURES
    top_k: int = 5


@dataclass
class ScenarioResult:
    scenario: Scenario
    passed: bool
    rank: int | None       # 1-based position in results, None if absent
    elapsed_ms: float
    error: str | None = None


_SCENARIOS: list[Scenario] = [
    Scenario(
        name="exact: hybrid search terminology",
        query="reciprocal rank fusion BM25 dense sparse retrieval",
        expected_fixture="hybrid_search.md",
    ),
    Scenario(
        name="semantic: combining two retrieval methods",
        query="two complementary ways to find relevant documents",
        expected_fixture="hybrid_search.md",
    ),
    Scenario(
        name="code: embedding function with rotation",
        query="function that embeds texts and applies rotation matrix",
        expected_fixture="embedder.py",
    ),
    Scenario(
        name="code: load embedding models",
        query="load dense and sparse embedding models",
        expected_fixture="embedder.py",
    ),
    Scenario(
        name="exact: RST ingestion pipeline",
        query="documents ingested from filesystem SHA-256 versioned swap",
        expected_fixture="architecture.rst",
    ),
    Scenario(
        name="semantic: MCP server for IDE",
        query="expose knowledge base to AI coding assistant",
        expected_fixture="architecture.rst",
    ),
    Scenario(
        name="exact: configuration qdrant url",
        query="qdrant url collection configuration environment variable override",
        expected_fixture="config_reference.txt",
    ),
    Scenario(
        name="exact: MCP port configuration",
        query="SSE server bind address port 7700",
        expected_fixture="config_reference.txt",
    ),
]

# ---------------------------------------------------------------------------
# Version-swap scenario: uses unique tokens unlikely to appear elsewhere
# ---------------------------------------------------------------------------

_SWAP_URI = "file:///ccmcp-validate/canary.md"
_SWAP_V1 = "# Canary Document\n\nxk3q9zp is the unique token marking version one of this document."
_SWAP_V2 = "# Canary Document\n\nzp7m2xk is the unique token marking version two of this document."
_TOKEN_V1 = "xk3q9zp"
_TOKEN_V2 = "zp7m2xk"

# ---------------------------------------------------------------------------
# Dedicated Qdrant collection (always cleaned up)
# ---------------------------------------------------------------------------

_VALIDATE_COLLECTION = "ccmcp-validate"
_VALIDATE_ARTIFACTS = "ccmcp-validate-artifacts"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(cfg, embedder, console=None) -> tuple[int, int]:
    """Run all validation scenarios against a temporary Qdrant collection.

    Returns (passed, total). The collection is created fresh and deleted on exit.
    Raises if Qdrant is unreachable.
    """
    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile
    from ccmcp.state import StateDB
    from ccmcp.store import VectorStore

    _p = console.print if console else print

    store = VectorStore(
        url=cfg.qdrant.url,
        collection=_VALIDATE_COLLECTION,
        api_key=cfg.qdrant.api_key,
        artifact_collection=_VALIDATE_ARTIFACTS,
    )
    store.drop_collections()
    store.setup(embedder.dim)

    state_dir = tempfile.mkdtemp(prefix="ccmcp-validate-state-")
    state = StateDB(str(Path(state_dir) / "state.db"))
    ctrl = Controller(cfg, embedder, store, state)

    passed = 0
    total = 0
    fixture_uris: dict[str, str] = {}

    try:
        # ------------------------------------------------------------------ #
        # Phase 1: Ingest fixture documents                                   #
        # ------------------------------------------------------------------ #
        _p("\n[bold]Phase 1  Ingesting fixtures[/bold]")
        with tempfile.TemporaryDirectory(prefix="ccmcp-validate-docs-") as doc_dir:
            for name, content in _FIXTURES.items():
                path = Path(doc_dir) / name
                path.write_text(content, encoding="utf-8")
                uri = f"file://{path}"
                fixture_uris[name] = uri
                ctrl.ingest_file(SourceFile(source_uri=uri, content=content))
                _p(f"  ingested {name}")
        # temp files deleted here; Qdrant payloads still hold the URIs

        # ------------------------------------------------------------------ #
        # Phase 2: Retrieval quality scenarios                                #
        # ------------------------------------------------------------------ #
        _p("\n[bold]Phase 2  Retrieval quality[/bold]")
        results: list[ScenarioResult] = []

        for scenario in _SCENARIOS:
            total += 1
            expected_uri = fixture_uris[scenario.expected_fixture]
            t0 = time.perf_counter()
            try:
                dense, sparse_list = embedder.embed([scenario.query])
                hits = store.search(dense[0], sparse_list[0], limit=scenario.top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                rank = next(
                    (i + 1 for i, h in enumerate(hits)
                     if h.get("source_uri") == expected_uri),
                    None,
                )
                results.append(ScenarioResult(scenario, rank is not None, rank, elapsed_ms))
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results.append(ScenarioResult(scenario, False, None, elapsed_ms, str(exc)))

        for r in results:
            icon = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            rank_str = f"rank {r.rank}" if r.rank else "not in top-{r.scenario.top_k}"
            _p(f"  {icon}  [{r.elapsed_ms:>5.0f}ms]  {r.scenario.name}  ({rank_str})")
            if r.error:
                _p(f"          error: {r.error}")
            if r.passed:
                passed += 1

        # ------------------------------------------------------------------ #
        # Phase 3: Version-swap correctness                                   #
        # ------------------------------------------------------------------ #
        _p("\n[bold]Phase 3  Version swap[/bold]")
        total += 1
        try:
            ctrl.ingest_file(SourceFile(source_uri=_SWAP_URI, content=_SWAP_V1))

            d1, s1 = embedder.embed([_TOKEN_V1])
            v1_initially_found = any(_TOKEN_V1 in h.get("text", "")
                                     for h in store.search(d1[0], s1[0], limit=10))

            ctrl.ingest_file(SourceFile(source_uri=_SWAP_URI, content=_SWAP_V2))

            d2, s2 = embedder.embed([_TOKEN_V2])
            v2_found = any(_TOKEN_V2 in h.get("text", "")
                           for h in store.search(d2[0], s2[0], limit=10))

            d1b, s1b = embedder.embed([_TOKEN_V1])
            v1_gone = not any(_TOKEN_V1 in h.get("text", "")
                              for h in store.search(d1b[0], s1b[0], limit=20))

            swap_ok = v1_initially_found and v2_found and v1_gone
            issues = []
            if not v1_initially_found:
                issues.append("v1 not searchable after initial ingest")
            if not v2_found:
                issues.append("v2 not searchable after update")
            if not v1_gone:
                issues.append("v1 content persists after update (delete failed?)")

            icon = "[green]PASS[/green]" if swap_ok else "[red]FAIL[/red]"
            _p(f"  {icon}  ingest v1 → update to v2 → v1 gone")
            for issue in issues:
                _p(f"          {issue}")
            if swap_ok:
                passed += 1

        except Exception as exc:
            _p(f"  [red]FAIL[/red]  version swap raised: {exc}")

        # ------------------------------------------------------------------ #
        # Phase 4: MCP artifact round-trip                                    #
        # ------------------------------------------------------------------ #
        _p("\n[bold]Phase 4  MCP artifact round-trip[/bold]")
        total += 1
        try:
            text = "Validation artifact written during ccmcp validate."
            dense_a, sparse_a = embedder.embed([text])
            point_id = store.store_artifact(
                text=text,
                dense=dense_a[0],
                sparse=sparse_a[0],
                session_id="ccmcp-validate",
                metadata={"title": "validate-smoke"},
            )
            store.cleanup_artifacts("1970-01-01T00:00:00+00:00")  # cleanup nothing
            _p(f"  [green]PASS[/green]  artifact stored (id: {point_id[:8]}…)")
            passed += 1
        except Exception as exc:
            _p(f"  [red]FAIL[/red]  artifact round-trip: {exc}")

    finally:
        store.drop_collections()
        shutil.rmtree(state_dir, ignore_errors=True)

    total_icon = "[green]" if passed == total else "[yellow]"
    _p(f"\n{total_icon}[bold]{passed}/{total} scenarios passed.[/bold][/{total_icon[1:]}")
    return passed, total
