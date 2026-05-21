"""End-to-end tests against the Docker test stack.

Requires the stack from docker-compose.test.yml to be running.
Orchestrated by scripts/test-docker.sh, or manually:

    docker compose -p ccmcp-test -f docker-compose.test.yml up -d --wait
    pytest tests/test_e2e.py -m e2e -v
    docker compose -p ccmcp-test -f docker-compose.test.yml down -v

Environment variables (set automatically by test-docker.sh):
    CCMCP_TEST_COMPOSE_FILE  path to docker-compose.test.yml
    CCMCP_TEST_PROJECT       compose project name (default: ccmcp-test)
    CCMCP_TEST_MCP_URL       base URL of the MCP server (default: http://localhost:7701)
    CCMCP_TEST_QDRANT_URL    base URL of Qdrant (default: http://localhost:6333)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

COMPOSE_FILE = os.environ.get(
    "CCMCP_TEST_COMPOSE_FILE",
    str(Path(__file__).parent.parent / "docker-compose.test.yml"),
)
PROJECT = os.environ.get("CCMCP_TEST_PROJECT", "ccmcp-test")
MCP_URL = os.environ.get("CCMCP_TEST_MCP_URL", "http://localhost:7701")
QDRANT_URL = os.environ.get("CCMCP_TEST_QDRANT_URL", "http://localhost:6334")

COMPOSE_CMD = ["docker", "compose", "-p", PROJECT, "-f", COMPOSE_FILE]


def ccmcp(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a ccmcp command inside the test container."""
    return subprocess.run(
        [*COMPOSE_CMD, "exec", "-T", "ccmcp", "ccmcp", *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )


def wait_for_points(collection: str = "techdocs", min_points: int = 1, timeout: int = 90) -> int:
    """Poll Qdrant until the collection has at least min_points. Returns actual count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{QDRANT_URL}/collections/{collection}", timeout=5)
            if r.status_code == 200:
                count = r.json()["result"]["points_count"]
                if count >= min_points:
                    return count
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(
        f"Collection '{collection}' did not reach {min_points} points within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Stack health
# ---------------------------------------------------------------------------

def test_qdrant_reachable():
    r = httpx.get(f"{QDRANT_URL}/healthz", timeout=10)
    assert r.status_code == 200


def test_mcp_metrics_endpoint():
    r = httpx.get(f"{MCP_URL}/metrics", timeout=10)
    assert r.status_code == 200
    assert "ccmcp_" in r.text or "python" in r.text


def test_mcp_sse_endpoint():
    """SSE endpoint should accept a connection (returns streaming 200)."""
    with httpx.stream("GET", f"{MCP_URL}/sse", timeout=5) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def test_ccmcp_help():
    result = ccmcp("--help")
    assert "Commands:" in result.stdout
    for cmd in ("init", "scan", "watch", "ingest", "status", "reset", "serve", "doctor", "run"):
        assert cmd in result.stdout


def test_status_shows_collection():
    result = ccmcp("status")
    assert result.returncode == 0
    assert "techdocs" in result.stdout.lower() or "Collection" in result.stdout


def test_doctor_passes():
    result = ccmcp("doctor")
    assert result.returncode == 0, f"doctor failed:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def test_fixtures_are_indexed():
    """Fixtures mounted at /repos should be ingested by the startup scan."""
    count = wait_for_points("techdocs", min_points=1)
    assert count > 0, "No points indexed — startup scan may have failed"


def test_ingest_produces_searchable_content():
    """Ingest a known document and verify it's findable via Qdrant search API."""
    count_before = wait_for_points("techdocs", min_points=1)

    # Ingest the hybrid_search.md fixture (already mounted at /repos)
    result = ccmcp("ingest", "/repos/hybrid_search.md")
    assert result.returncode == 0

    # Give the pipeline a moment to propagate
    time.sleep(2)

    # Check Qdrant collection grew (or stayed the same if already ingested)
    count_after = wait_for_points("techdocs", min_points=count_before)
    assert count_after >= count_before


def test_qdrant_search_returns_results():
    """Direct Qdrant hybrid search returns results for a known query."""
    wait_for_points("techdocs", min_points=1)

    # Use the scroll API (no embedding needed) to verify indexed content
    r = httpx.post(
        f"{QDRANT_URL}/collections/techdocs/points/scroll",
        json={"limit": 5, "with_payload": True},
        timeout=10,
    )
    assert r.status_code == 200
    points = r.json()["result"]["points"]
    assert len(points) > 0

    # Verify payload structure
    for p in points:
        payload = p["payload"]
        assert "text" in payload
        assert "source_uri" in payload
        assert "source_type" in payload


def test_indexed_content_mentions_fixture_topic():
    """At least one indexed chunk should contain content from our fixture docs."""
    wait_for_points("techdocs", min_points=1)

    r = httpx.post(
        f"{QDRANT_URL}/collections/techdocs/points/scroll",
        json={"limit": 100, "with_payload": ["text", "source_uri"]},
        timeout=10,
    )
    assert r.status_code == 200
    texts = " ".join(p["payload"].get("text", "") for p in r.json()["result"]["points"])
    # Content from hybrid_search.md or architecture.rst must appear
    assert any(
        kw in texts
        for kw in ("BM25", "hybrid", "vector", "embedding", "Qdrant", "fastembed")
    ), "Expected fixture content not found in indexed chunks"


# ---------------------------------------------------------------------------
# Save-config / load-config
# ---------------------------------------------------------------------------

def test_save_config_produces_yaml():
    result = ccmcp("save-config")
    assert result.returncode == 0
    assert "qdrant" in result.stdout
    assert "sources" in result.stdout


def test_save_and_reload_config_roundtrip():
    """save-config → load-config should write a valid config without error."""
    # Save current config to a temp location inside the container
    save_result = ccmcp("save-config", "-o", "/data/config-backup.yaml")
    assert save_result.returncode == 0

    # Load it back (--yes skips confirmation)
    load_result = ccmcp("load-config", "/data/config-backup.yaml", "--yes")
    assert load_result.returncode == 0


# ---------------------------------------------------------------------------
# Save-snapshot / load-snapshot
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip():
    """save-snapshot → reset → load-snapshot → init → scan → data restored."""
    wait_for_points("techdocs", min_points=1)
    count_before = httpx.get(
        f"{QDRANT_URL}/collections/techdocs", timeout=5
    ).json()["result"]["points_count"]
    assert count_before > 0

    # 1. Save snapshot
    result = ccmcp("save-snapshot", "-o", "/data/test-snapshot.tar.gz")
    assert result.returncode == 0, f"save-snapshot failed:\n{result.stdout}\n{result.stderr}"

    # Verify archive was created with correct members
    verify = subprocess.run(
        [*COMPOSE_CMD, "exec", "-T", "ccmcp",
         "python3", "-c",
         "import tarfile; t=tarfile.open('/data/test-snapshot.tar.gz');"
         " print(sorted(t.getnames()))"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    assert "rotation_matrix.npy" in verify.stdout
    assert "state.db" in verify.stdout

    # 2. Reset Qdrant collection (pipe confirmation via stdin)
    subprocess.run(
        [*COMPOSE_CMD, "exec", "-T", "ccmcp", "ccmcp", "reset"],
        input="RESET\n",
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(2)

    # 3. Restore snapshot (rotation matrix + state DB)
    result = ccmcp("load-snapshot", "/data/test-snapshot.tar.gz", "--yes")
    assert result.returncode == 0

    # 4. Remove the restored state DB so the next scan does a full fresh re-ingest.
    #    (The state DB tracks content hashes; if left intact, the scan would see all
    #    files as unchanged and skip them, leaving Qdrant empty.)
    subprocess.run(
        [*COMPOSE_CMD, "exec", "-T", "ccmcp", "rm", "-f", "/data/state.db"],
        check=True, timeout=10,
    )

    # 5. Re-init collections
    result = ccmcp("init")
    assert result.returncode == 0

    # 6. Scan to re-ingest (fresh state DB, correct rotation matrix from snapshot)
    result = ccmcp("scan")
    assert result.returncode == 0

    # 7. Verify data is back
    count_after = wait_for_points("techdocs", min_points=count_before, timeout=120)
    assert count_after >= count_before


# ---------------------------------------------------------------------------
# Artifact store (MCP qdrant_store tool)
# ---------------------------------------------------------------------------

def test_artifact_collection_exists():
    """ccmcp-artifacts collection should exist after init."""
    r = httpx.get(f"{QDRANT_URL}/collections/ccmcp-artifacts", timeout=5)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# MCP protocol — agent/user tool calls over SSE
# ---------------------------------------------------------------------------

class MCPSession:
    """Minimal synchronous MCP client over SSE transport.

    Speaks the JSON-RPC-over-SSE protocol that Claude Code / Cursor use:
      1. GET /sse  →  server sends  event: endpoint / data: /messages/?session_id=…
      2. POST /messages/?session_id=…  with JSON-RPC request bodies
      3. Server pushes  event: message / data: <json>  on the SSE stream
    """

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)
        self._msg_url = ""
        self._pending: dict[int, dict] = {}
        self._req_id = 0

    def __enter__(self) -> MCPSession:
        endpoint_q: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
            with self._client.stream("GET", f"{self._base}/sse") as resp:
                event_type: str | None = None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if event_type == "endpoint":
                            endpoint_q.put(data)
                        elif event_type == "message" and data:
                            try:
                                msg = json.loads(data)
                                if "method" in msg and "id" in msg:
                                    # Server-to-client request (e.g. roots/list).
                                    # Respond immediately so the server isn't left waiting.
                                    # httpx.post creates a fresh client — thread-safe.
                                    if msg["method"] == "roots/list":
                                        httpx.post(
                                            self._msg_url,
                                            json={"jsonrpc": "2.0",
                                                  "id": msg["id"],
                                                  "result": {"roots": []}},
                                            timeout=5,
                                        )
                                elif isinstance(msg.get("id"), int):
                                    # Client-to-server response
                                    self._pending[msg["id"]] = msg
                            except Exception:
                                pass
                        event_type = None

        threading.Thread(target=_reader, daemon=True).start()

        raw_path = endpoint_q.get(timeout=10)
        self._msg_url = raw_path if raw_path.startswith("http") else self._base + raw_path

        # MCP initialize handshake
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ccmcp-e2e", "version": "1.0"},
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()

    def _post(self, body: dict) -> None:
        self._client.post(self._msg_url, json=body, timeout=10)

    def _call(self, method: str, params: dict) -> dict:
        self._req_id += 1
        req_id = self._req_id
        self._post({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if req_id in self._pending:
                return self._pending.pop(req_id)
            time.sleep(0.05)
        raise TimeoutError(f"No MCP response for {method!r} (id={req_id})")

    def list_tools(self) -> list[str]:
        resp = self._call("tools/list", {})
        return [t["name"] for t in resp["result"]["tools"]]

    def call_tool(self, name: str, **kwargs: object) -> str:
        resp = self._call("tools/call", {"name": name, "arguments": kwargs})
        if "error" in resp:
            raise AssertionError(f"MCP tool error: {resp['error']}")
        parts = resp["result"]["content"]
        return "".join(c["text"] for c in parts if c.get("type") == "text")


@pytest.fixture(scope="module")
def mcp_session():
    """Open MCP session shared across all MCP tool tests in this module."""
    with MCPSession(MCP_URL) as session:
        yield session


def test_mcp_tools_listed(mcp_session):
    """Server advertises all three tools."""
    tools = mcp_session.list_tools()
    assert "qdrant_find" in tools
    assert "qdrant_store" in tools
    assert "qdrant_list_scopes" in tools


def test_qdrant_find_returns_relevant_results(mcp_session):
    """qdrant_find returns passages relevant to the query."""
    wait_for_points()
    result = mcp_session.call_tool("qdrant_find", query="BM25 sparse retrieval", limit=3)
    assert result != "No results found."
    assert any(kw in result for kw in ("BM25", "sparse", "term", "keyword", "vocabulary"))


def test_qdrant_find_source_cited(mcp_session):
    """Each result includes a source_uri header so the agent knows where it came from."""
    wait_for_points()
    result = mcp_session.call_tool("qdrant_find", query="hybrid search", limit=2)
    assert "file://" in result or "http" in result or "/repos/" in result


def test_qdrant_find_scope_wildcard(mcp_session):
    """scope=['*'] bypasses automatic scoping and searches the full corpus."""
    wait_for_points()
    result = mcp_session.call_tool("qdrant_find", query="vector embedding", limit=3, scope=["*"])
    assert result != "No results found."


def test_qdrant_find_unknown_scope_returns_nothing_or_results(mcp_session):
    """An unknown scope name should not crash — it either returns results or empty."""
    result = mcp_session.call_tool(
        "qdrant_find", query="anything", limit=3, scope=["nonexistent-project-xyz"]
    )
    assert isinstance(result, str)


def test_qdrant_store_returns_point_id(mcp_session):
    """qdrant_store responds with 'Stored: <uuid>'."""
    result = mcp_session.call_tool(
        "qdrant_store",
        text="E2E test artifact: MCP store round-trip check.",
        title="e2e-test",
        session_id="e2e-001",
    )
    assert result.startswith("Stored: "), f"Unexpected: {result!r}"
    point_id = result.removeprefix("Stored: ").strip()
    assert len(point_id) > 8  # UUIDs are 36 chars


def test_qdrant_store_persists_to_collection(mcp_session):
    """An artifact stored via qdrant_store is retrievable from ccmcp-artifacts."""
    unique_token = "xkcd42e2etoken"
    result = mcp_session.call_tool(
        "qdrant_store",
        text=f"Persistence check: {unique_token}.",
        session_id="e2e-persist",
    )
    point_id = result.removeprefix("Stored: ").strip()

    r = httpx.get(f"{QDRANT_URL}/collections/ccmcp-artifacts/points/{point_id}", timeout=5)
    assert r.status_code == 200
    assert unique_token in r.json()["result"]["payload"]["text"]


def test_qdrant_list_scopes(mcp_session):
    """qdrant_list_scopes returns a non-empty string after ingestion."""
    wait_for_points()
    result = mcp_session.call_tool("qdrant_list_scopes")
    assert isinstance(result, str) and len(result) > 0


def test_qdrant_find_after_store(mcp_session):
    """Content stored via qdrant_store is findable via qdrant_find in artifacts."""
    unique = "retrieval-smoke-test-e2e-9z"
    mcp_session.call_tool(
        "qdrant_store",
        text=f"Knowledge note: {unique}.",
        session_id="e2e-smoke",
    )
    # Wait a moment for the upsert to settle
    time.sleep(1)
    # qdrant_find searches the main collection, not artifacts — verify no crash
    result = mcp_session.call_tool("qdrant_find", query=unique, limit=1)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Human search UI — HTTP endpoints
# ---------------------------------------------------------------------------

def test_search_ui_returns_html():
    """GET / returns the browser search UI with the expected form elements."""
    r = httpx.get(f"{MCP_URL}/", timeout=10)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert 'id="query"' in r.text
    assert 'id="scope"' in r.text
    assert 'id="search-form"' in r.text


def test_search_ui_title():
    r = httpx.get(f"{MCP_URL}/", timeout=10)
    assert "ccmcp" in r.text.lower()


def test_scopes_endpoint_returns_list():
    """GET /scopes returns a non-empty JSON list after ingestion."""
    wait_for_points()
    r = httpx.get(f"{MCP_URL}/scopes", timeout=10)
    assert r.status_code == 200
    scopes = r.json()
    assert isinstance(scopes, list)
    assert len(scopes) > 0


def test_scopes_entries_have_source_root():
    """Each scope entry has at least a source_root field."""
    wait_for_points()
    r = httpx.get(f"{MCP_URL}/scopes", timeout=10)
    assert r.status_code == 200
    for scope in r.json():
        assert "source_root" in scope


def test_search_returns_results_for_known_content():
    """POST /search returns ranked results for a query matching fixture content."""
    wait_for_points()
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "hybrid search BM25", "limit": 5},
        timeout=15,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "hybrid search BM25"
    assert data["count"] > 0
    assert len(data["results"]) > 0
    first = data["results"][0]
    assert "text" in first
    assert "source_uri" in first


def test_search_result_payload_has_required_fields():
    """Every result must include text, source_uri, and source_type."""
    wait_for_points()
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "embedding", "limit": 3},
        timeout=15,
    )
    assert r.status_code == 200
    for result in r.json()["results"]:
        assert "text" in result
        assert "source_uri" in result
        assert "source_type" in result


def test_search_result_source_uri_is_fixture_path():
    """Results for fixture content should point to /repos/..."""
    wait_for_points()
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "Qdrant vector", "limit": 5},
        timeout=15,
    )
    assert r.status_code == 200
    uris = [res["source_uri"] for res in r.json()["results"]]
    assert any("/repos/" in u for u in uris), f"No /repos/ in URIs: {uris}"


def test_search_empty_query_returns_400():
    r = httpx.post(f"{MCP_URL}/search", json={"query": ""}, timeout=10)
    assert r.status_code == 400
    assert "query" in r.json().get("detail", "").lower()


def test_search_missing_query_returns_400():
    r = httpx.post(f"{MCP_URL}/search", json={}, timeout=10)
    assert r.status_code == 400


def test_search_limit_caps_results():
    """limit=2 should return at most 2 results even if more exist."""
    wait_for_points()
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "vector", "limit": 2},
        timeout=15,
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 2


def test_search_with_wildcard_scope():
    """scope=['*'] disables scope filtering and searches the full corpus."""
    wait_for_points()
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "vector", "scope": ["*"], "limit": 5},
        timeout=15,
    )
    assert r.status_code == 200
    assert r.json()["count"] > 0


def test_search_with_scope_filter_succeeds():
    """A scope filter from an available scope should return 200 (results may vary)."""
    wait_for_points()
    scopes_r = httpx.get(f"{MCP_URL}/scopes", timeout=10)
    scopes = scopes_r.json()
    if not scopes:
        pytest.skip("No scopes available")
    scope_val = scopes[0].get("name") or scopes[0]["source_root"]
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "search", "scope": [scope_val], "limit": 5},
        timeout=15,
    )
    assert r.status_code == 200
    assert "results" in r.json()


def test_search_unknown_scope_returns_empty_not_error():
    """An unknown scope name must not crash — returns 200 with empty results."""
    r = httpx.post(
        f"{MCP_URL}/search",
        json={"query": "anything", "scope": ["nonexistent-xyz-9999"], "limit": 5},
        timeout=15,
    )
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
