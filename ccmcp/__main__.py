from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread

import click
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from qdrant_client import models as qdrant_models
from rich.console import Console
from rich.table import Table

from ccmcp.config import load_config
from ccmcp.embedder import Embedder
from ccmcp.state import StateDB
from ccmcp.store import VectorStore

console = Console()


def _components(config_path: str | None):
    cfg = load_config(config_path)
    embedder = Embedder(
        dense_model=cfg.embedding.dense_model,
        sparse_model=cfg.embedding.sparse_model,
        rotation_matrix_path=cfg.embedding.rotation_matrix,
    )
    store = VectorStore(
        url=cfg.qdrant.url,
        collection=cfg.qdrant.collection,
        api_key=cfg.qdrant.api_key,
    )
    state = StateDB(cfg.state.db_path)
    return cfg, embedder, store, state


async def _roots_filter(ctx: Context) -> qdrant_models.Filter | None:
    """Build a Qdrant filter scoped to the session's MCP roots.

    Returns None (unfiltered) when:
    - the client didn't send roots, or
    - no active root contains a .ccmcp marker file.
    """
    from pathlib import Path as _Path

    from ccmcp.marker import load as load_marker

    try:
        result = await ctx.request_context.session.list_roots()
        roots = result.roots
    except Exception:
        return None

    if not roots:
        return None

    root_paths: list[str] = []
    include_tags: list[str] = []

    for root in roots:
        raw_uri = str(root.uri)
        # Root URIs are always file:// per the MCP spec
        path = raw_uri.removeprefix("file://")
        try:
            resolved = str(_Path(path).resolve())
        except Exception:
            resolved = path

        marker = load_marker(resolved)
        if marker:
            root_paths.append(marker.source_root)
            include_tags.extend(marker.include)

    if not root_paths:
        return None  # no .ccmcp files → unscoped search

    # Match any chunk from an active root, OR carrying an included tag.
    conditions: list[qdrant_models.Condition] = [
        qdrant_models.FieldCondition(
            key="source_root",
            match=qdrant_models.MatchAny(any=root_paths),
        )
    ]
    if include_tags:
        conditions.append(
            qdrant_models.FieldCondition(
                key="tags",
                match=qdrant_models.MatchAny(any=include_tags),
            )
        )
    return qdrant_models.Filter(should=conditions)


def _build_mcp(cfg, embedder, store) -> FastMCP:
    mcp = FastMCP("ccmcp", description="Hybrid vector search over your codebase")

    @mcp.tool(description=(
        "Search the knowledge base for passages relevant to a query. "
        "Returns up to `limit` results ranked by hybrid dense+sparse relevance."
    ))
    async def qdrant_find(query: str, limit: int = 10, ctx: Context = None) -> str:
        query = query[:8192]
        limit = max(1, min(limit, cfg.mcp.result_limit))
        search_filter = await _roots_filter(ctx) if ctx is not None else None
        dense, sparse_list = embedder.embed([query])
        results = store.search(dense[0], sparse_list[0], limit=limit, filter=search_filter)
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            src = r.get("source_uri", "unknown")
            section = r.get("section", "")
            text = r.get("text", "")
            header = f"[{i}] {src}" + (f" § {section}" if section else "")
            parts.append(f"{header}\n\n{text}")
        return "\n\n---\n\n".join(parts)

    @mcp.tool(description=(
        "Store a note or artifact in the knowledge base so it can be retrieved later."
    ))
    def qdrant_store(text: str, title: str = "", session_id: str = "") -> str:
        dense, sparse_list = embedder.embed([text])
        point_id = store.store_artifact(
            text=text,
            dense=dense[0],
            sparse=sparse_list[0],
            session_id=session_id,
            metadata={"title": title},
        )
        # TTL cleanup on each store call
        cutoff = (datetime.now(UTC) - timedelta(days=cfg.state.artifact_ttl_days)).isoformat()
        try:
            store.cleanup_artifacts(cutoff)
        except Exception:
            pass
        return f"Stored: {point_id}"

    return mcp


@click.group()
@click.option("--config", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
@click.pass_context
def setup(ctx):
    """Initialize Qdrant collections and generate rotation matrix."""
    cfg, embedder, store, _ = _components(ctx.obj["config_path"])
    console.print("[bold]ccmcp setup[/bold]")

    try:
        embedder.setup()
        console.print(f"[green]✓[/green] Rotation matrix: {cfg.embedding.rotation_matrix}")
    except FileExistsError as e:
        console.print(f"[yellow]⚠[/yellow] {e}")

    store.setup(embedder.dim)
    console.print(
        f"[green]✓[/green] Qdrant collections ready ({cfg.qdrant.collection}, ccmcp-artifacts)"
    )

    console.print("\n[bold]Add to ~/.claude.json:[/bold]")
    console.print(f'''\
{{
  "mcpServers": {{
    "ccmcp": {{
      "type": "sse",
      "url": "http://localhost:{cfg.mcp.port}/sse"
    }}
  }}
}}''')


@cli.command()
@click.pass_context
def scan(ctx):
    """One-shot full ingestion of all enabled sources."""
    from ccmcp.controller import Controller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    console.print("[bold]Scanning…[/bold]")
    Controller(cfg, embedder, store, state).scan()
    console.print("[green]Scan complete.[/green]")


@cli.command()
@click.pass_context
def watch(ctx):
    """Full scan then continuous watch (polling + hourly rescan)."""
    from ccmcp.controller import Controller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    console.print("[bold]Watch mode started.[/bold]")
    Controller(cfg, embedder, store, state).watch()


@cli.command()
@click.argument("path_or_url")
@click.pass_context
def ingest(ctx, path_or_url: str):
    """Ingest a single file or URL immediately."""
    from ccmcp.controller import Controller
    from ccmcp.sources import SourceFile
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    ctrl = Controller(cfg, embedder, store, state)

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        from ccmcp.sources import web as web_mod
        sf = web_mod.fetch(path_or_url, cfg.sources.web.user_agent)
    else:
        p = Path(path_or_url).expanduser().resolve()
        with open(p, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        sf = SourceFile(source_uri=f"file://{p}", content=content)

    if sf:
        ctrl.ingest_file(sf)
        console.print(f"[green]Ingested:[/green] {path_or_url}")
    else:
        console.print(
            f"[yellow]Nothing to ingest (unchanged or unavailable):[/yellow] {path_or_url}"
        )


@cli.command()
@click.pass_context
def status(ctx):
    """Show collection stats, store sizes, and indexed source count."""
    cfg, _, store, state = _components(ctx.obj["config_path"])

    # ── Main collection ──────────────────────────────────────────────────────
    info = store.collection_info()
    t = Table(title=f"Collection: {cfg.qdrant.collection}", show_header=True)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in info.items():
        t.add_row(k, str(v))
    console.print(t)

    # ── Artifact collection ──────────────────────────────────────────────────
    try:
        ainfo = store.artifact_collection_info()
        at = Table(title="Collection: ccmcp-artifacts", show_header=True)
        at.add_column("Metric")
        at.add_column("Value", justify="right")
        for k, v in ainfo.items():
            at.add_row(k, str(v))
        console.print(at)
    except Exception:
        console.print("[yellow]Artifact collection not available.[/yellow]")

    # ── State DB ─────────────────────────────────────────────────────────────
    records = state.all()
    by_type: dict[str, int] = {}
    for r in records:
        prefix = (
            "fs" if r.source_uri.startswith("file://")
            else "web" if r.source_uri.startswith("http")
            else "drive" if r.source_uri.startswith("drive://")
            else "other"
        )
        by_type[prefix] = by_type.get(prefix, 0) + 1

    console.print(f"\n[bold]{len(records)}[/bold] sources in state DB")
    for stype, count in sorted(by_type.items()):
        console.print(f"  {stype}: {count}")

    console.print(
        f"\nMetrics endpoint (when server is running): "
        f"http://{cfg.mcp.host}:{cfg.mcp.port}/metrics"
    )


@cli.command()
@click.pass_context
def reset(ctx):
    """Drop Qdrant collection and clear state DB. Requires confirmation."""
    cfg, _, store, _ = _components(ctx.obj["config_path"])
    confirm = click.prompt("Type RESET to confirm")
    if confirm != "RESET":
        console.print("Aborted.")
        return
    store.drop_collections()
    if Path(cfg.state.db_path).exists():
        Path(cfg.state.db_path).unlink()
    console.print("[red]Reset complete. Run 'ccmcp setup' before next use.[/red]")


@cli.command()
@click.option("--host", default=None, help="Bind host (default from config)")
@click.option("--port", default=None, type=int, help="Bind port (default from config)")
@click.pass_context
def serve(ctx, host: str | None, port: int | None):
    """Start the MCP SSE server (for Claude Code and Cursor)."""
    from ccmcp.metrics import make_observable_app, start_collection_poller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    h = host or cfg.mcp.host
    p = port or cfg.mcp.port
    mcp = _build_mcp(cfg, embedder, store)
    start_collection_poller(store, state)
    app = make_observable_app(mcp.sse_app())
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    console.print(f"[dim]Prometheus metrics → http://{h}:{p}/metrics[/dim]")
    uvicorn.run(app, host=h, port=p, log_level="warning")


@cli.command()
@click.pass_context
def validate(ctx):
    """Run end-to-end validation scenarios against a temporary Qdrant collection."""
    from ccmcp.validate import run_validation
    cfg, embedder, _, _ = _components(ctx.obj["config_path"])
    console.print("[bold]ccmcp validate[/bold]")
    passed, total = run_validation(cfg, embedder, console=console)
    raise SystemExit(0 if passed == total else 1)


@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.pass_context
def start(ctx, host: str | None, port: int | None):
    """Start ingestion controller and MCP SSE server together (Docker entrypoint)."""
    from ccmcp.controller import Controller
    from ccmcp.metrics import make_observable_app, start_collection_poller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    h = host or cfg.mcp.host
    p = port or cfg.mcp.port

    t = Thread(target=Controller(cfg, embedder, store, state).watch, daemon=True, name="controller")
    t.start()
    console.print("[bold]Controller started in background.[/bold]")

    start_collection_poller(store, state)
    mcp = _build_mcp(cfg, embedder, store)
    app = make_observable_app(mcp.sse_app())
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    console.print(f"[dim]Prometheus metrics → http://{h}:{p}/metrics[/dim]")
    uvicorn.run(app, host=h, port=p, log_level="warning")


if __name__ == "__main__":
    cli()
