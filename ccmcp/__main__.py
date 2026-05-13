from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread

import click
import uvicorn
from mcp.server.fastmcp import FastMCP
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


def _build_mcp(cfg, embedder, store) -> FastMCP:
    mcp = FastMCP("ccmcp", description="Hybrid vector search over your codebase")

    @mcp.tool(description=(
        "Search the knowledge base for passages relevant to a query. "
        "Returns up to `limit` results ranked by hybrid dense+sparse relevance."
    ))
    def qdrant_find(query: str, limit: int = 10) -> str:
        dense, sparse_list = embedder.embed([query])
        results = store.search(dense[0], sparse_list[0], limit=min(limit, cfg.mcp.result_limit))
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

    store.setup()
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
    """Show collection stats and indexed source count."""
    cfg, _, store, state = _components(ctx.obj["config_path"])

    info = store.collection_info()
    t = Table(title=f"Collection: {cfg.qdrant.collection}")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in info.items():
        t.add_row(k, str(v))
    console.print(t)

    records = state.all()
    console.print(f"\n[bold]{len(records)}[/bold] sources in state DB")


@cli.command()
@click.pass_context
def reset(ctx):
    """Drop Qdrant collection and clear state DB. Requires confirmation."""
    cfg, _, store, _ = _components(ctx.obj["config_path"])
    confirm = click.prompt("Type RESET to confirm")
    if confirm != "RESET":
        console.print("Aborted.")
        return
    store._client.delete_collection(cfg.qdrant.collection)
    store._client.delete_collection("ccmcp-artifacts")
    if Path(cfg.state.db_path).exists():
        Path(cfg.state.db_path).unlink()
    console.print("[red]Reset complete. Run 'ccmcp setup' before next use.[/red]")


@cli.command()
@click.option("--host", default=None, help="Bind host (default from config)")
@click.option("--port", default=None, type=int, help="Bind port (default from config)")
@click.pass_context
def serve(ctx, host: str | None, port: int | None):
    """Start the MCP SSE server (for Claude Code and Cursor)."""
    cfg, embedder, store, _ = _components(ctx.obj["config_path"])
    h = host or cfg.mcp.host
    p = port or cfg.mcp.port
    mcp = _build_mcp(cfg, embedder, store)
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    uvicorn.run(mcp.sse_app(), host=h, port=p, log_level="warning")


@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.pass_context
def start(ctx, host: str | None, port: int | None):
    """Start ingestion controller and MCP SSE server together (Docker entrypoint)."""
    from ccmcp.controller import Controller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    h = host or cfg.mcp.host
    p = port or cfg.mcp.port

    t = Thread(target=Controller(cfg, embedder, store, state).watch, daemon=True, name="controller")
    t.start()
    console.print("[bold]Controller started in background.[/bold]")

    mcp = _build_mcp(cfg, embedder, store)
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    uvicorn.run(mcp.sse_app(), host=h, port=p, log_level="warning")


if __name__ == "__main__":
    cli()
