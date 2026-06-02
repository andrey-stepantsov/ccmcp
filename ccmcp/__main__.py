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

from ccmcp.config import SourcePath, config_to_dict, load_config
from ccmcp.controller import _resolved
from ccmcp.embedder import Embedder
from ccmcp.state import StateDB
from ccmcp.store import VectorStore

console = Console()
log = logging.getLogger(__name__)


def _host_to_container(path: str) -> str:
    """Rewrite a host-side path to its container-side equivalent.

    Set CCMCP_HOST_MOUNT (e.g. /Users/alice/code) and CCMCP_CONTAINER_MOUNT
    (e.g. /workspace) when running under Docker so that MCP root URIs sent
    by the host-resident agent resolve against container-side index paths.
    """
    import os
    host = os.environ.get("CCMCP_HOST_MOUNT", "").rstrip("/")
    container = os.environ.get("CCMCP_CONTAINER_MOUNT", "").rstrip("/")
    if not host or not container:
        return path
    if path == host:
        return container
    if path.startswith(host + "/"):
        return container + path[len(host):]
    return path


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


def _build_path_index(cfg) -> dict[str, SourcePath]:
    """Build a resolved-path → SourcePath index from config for MCP root resolution."""
    return {_resolved(sp.path): sp for sp in cfg.sources.filesystem.paths}


def _match_root(
    resolved: str, path_index: dict[str, SourcePath]
) -> tuple[str, SourcePath] | None:
    """Match a resolved path against the configured sources by longest ancestor prefix.

    Allows the agent to open a subdirectory of an indexed project and still
    get scoped retrieval. Returns the (configured_root, SourcePath) pair
    whose root is the deepest ancestor of `resolved`, or None on miss.
    """
    import os
    best: tuple[str, SourcePath] | None = None
    for cfg_root, sp in path_index.items():
        if resolved == cfg_root or resolved.startswith(cfg_root + os.sep):
            if best is None or len(cfg_root) > len(best[0]):
                best = (cfg_root, sp)
    return best


async def _roots_filter(
    ctx: Context, path_index: dict[str, SourcePath]
) -> qdrant_models.Filter | None:
    """Build a Qdrant filter scoped to the session's MCP roots.

    Resolves incoming MCP root URIs against the config path index, accepting
    either the configured root itself or any subdirectory of it. Honours
    CCMCP_HOST_MOUNT / CCMCP_CONTAINER_MOUNT for Docker host→container
    path remapping. Returns None (unscoped) only when no root matches —
    in which case we log so the silent fallback is visible.
    """
    try:
        result = await ctx.request_context.session.list_roots()
        roots = result.roots
    except Exception:
        return None

    if not roots:
        return None

    root_paths: list[str] = []
    include_tags: list[str] = []
    incoming: list[str] = []

    for root in roots:
        raw_uri = str(root.uri)
        path = raw_uri.removeprefix("file://")
        path = _host_to_container(path)
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = path
        incoming.append(resolved)

        match = _match_root(resolved, path_index)
        if match is None:
            continue
        cfg_root, sp = match
        if cfg_root not in root_paths:
            root_paths.append(cfg_root)
        include_tags.extend(sp.include)

    if not root_paths:
        log.info(
            "no configured source matches MCP roots %s — search will be unscoped "
            "(set CCMCP_HOST_MOUNT/CCMCP_CONTAINER_MOUNT for Docker, "
            "or add the path to sources.filesystem.paths)",
            incoming,
        )
        return None

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


def _scope_filter(scope: list[str]) -> qdrant_models.Filter:
    """Build a Qdrant filter that matches chunks by project_name OR tags."""
    return qdrant_models.Filter(should=[
        qdrant_models.FieldCondition(
            key="project_name",
            match=qdrant_models.MatchAny(any=scope),
        ),
        qdrant_models.FieldCondition(
            key="tags",
            match=qdrant_models.MatchAny(any=scope),
        ),
    ])


def _build_mcp(cfg, embedder, store) -> FastMCP:
    mcp = FastMCP("ccmcp", instructions="Hybrid vector search over your codebase")
    path_index = _build_path_index(cfg)

    @mcp.tool(description=(
        "Search the knowledge base for passages relevant to a query. "
        "Returns up to `limit` results ranked by hybrid dense+sparse relevance. "
        "Pass `scope` as a list of project names or tags (from qdrant_list_scopes) "
        "to restrict results to specific projects. "
        "Omit `scope` to use automatic project scoping based on the active workspace. "
        'Pass `scope=[\"*\"]` to search the full corpus regardless of active project.'
    ))
    async def qdrant_find(
        query: str,
        limit: int = 10,
        scope: list[str] | None = None,
        ctx: Context | None = None,
    ) -> str:
        query = query[:8192]
        limit = max(1, min(limit, cfg.mcp.result_limit))

        if scope is not None:
            # Explicit scope overrides automatic root-based filtering.
            search_filter = None if (not scope or scope == ["*"]) else _scope_filter(scope)
        else:
            search_filter = await _roots_filter(ctx, path_index) if ctx is not None else None

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
        "List all project scopes available in the knowledge base. "
        "Returns project names, tags, and root paths for every project "
        "that was indexed with a .ccmcp marker file. "
        "Use the names or tags returned here in the `scope` parameter of qdrant_find."
    ))
    def qdrant_list_scopes() -> str:
        scopes = store.list_scopes()
        if not scopes:
            return (
                "No project scopes found. "
                "Add a .ccmcp file to a project root and re-index to enable scoping."
            )
        lines = [f"{len(scopes)} project scope(s) in the knowledge base:\n"]
        for s in scopes:
            tags = ", ".join(s["tags"]) if s["tags"] else "(none)"
            lines.append(f"• {s['name']}")
            lines.append(f"  tags: {tags}")
            lines.append(f"  root: {s['source_root']}")
            lines.append("")
        lines.append(
            'Use names or tags in qdrant_find: scope=["my-project", "shared-lib"]'
        )
        lines.append('To search everything: scope=["*"]')
        return "\n".join(lines)

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
def init(ctx):
    """Initialize Qdrant collections and generate rotation matrix.

    Creates the Qdrant collection with hybrid dense+sparse vectors and
    generates rotation_matrix.npy (required before first ingestion).
    Safe to run multiple times — skips existing collections and refuses
    to overwrite the rotation matrix.

    Prints the MCP config snippet to add to ~/.claude.json when done.
    """
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
    """Index all configured sources once and exit.

    Walks filesystem roots, fetches web URLs, and polls Google Drive.
    Skips unchanged files (SHA-256 dedup). Removes vectors for deleted
    files. Safe to run repeatedly — re-run any time to pick up changes
    without using watch mode.
    """
    from ccmcp.controller import Controller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    console.print("[bold]Scanning…[/bold]")
    Controller(cfg, embedder, store, state).scan()
    console.print("[green]Scan complete.[/green]")


@cli.command()
@click.pass_context
def watch(ctx):
    """Scan once then watch filesystem for changes continuously.

    Runs a full scan on startup, then watches configured roots for
    file changes (inotify on Linux; polling on WSL2 /mnt/ paths).
    Re-scans all sources every hour. Runs until interrupted (Ctrl-C).
    """
    from ccmcp.controller import Controller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    console.print("[bold]Watch mode started.[/bold]")
    Controller(cfg, embedder, store, state).watch()


@cli.command()
@click.argument("path_or_url")
@click.pass_context
def ingest(ctx, path_or_url: str):
    """Ingest a single file or URL immediately.

    \b
    Examples:
      ccmcp ingest ~/notes/architecture.md
      ccmcp ingest https://docs.example.com/api/overview
    """
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
    """Show collection stats and indexed source counts.

    Prints Qdrant point/vector counts for the main and artifact
    collections, and a per-type breakdown of indexed sources from
    the state DB (fs / web / drive). Also shows the Prometheus
    metrics endpoint URL.
    """
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
    """Drop Qdrant collections and clear the state DB.

    Prompts "Type RESET to confirm" before proceeding. Does not delete
    the rotation matrix. Run 'ccmcp setup' to recreate collections
    before next use.
    """
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
    """Start the MCP SSE server only (no ingestion controller).

    Serves the MCP protocol over SSE at http://HOST:PORT/sse.
    Prometheus metrics available at http://HOST:PORT/metrics.
    Use 'run' to start ingestion and the server together.
    """
    from ccmcp.metrics import make_observable_app, start_collection_poller
    cfg, embedder, store, state = _components(ctx.obj["config_path"])
    h = host or cfg.mcp.host
    p = port or cfg.mcp.port
    mcp = _build_mcp(cfg, embedder, store)
    start_collection_poller(store, state)
    app = make_observable_app(mcp.sse_app(), store=store, embedder=embedder)
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    console.print(f"[dim]Prometheus metrics → http://{h}:{p}/metrics[/dim]")
    console.print(f"[dim]Search UI          → http://{h}:{p}/[/dim]")
    uvicorn.run(app, host=h, port=p, log_level="warning")


@cli.command()
@click.pass_context
def doctor(ctx):
    """Run end-to-end validation against a live Qdrant instance.

    Indexes sample documents into a temporary collection, runs hybrid
    searches, and verifies results meet expected relevance thresholds.
    The temporary collection is cleaned up after the run.
    Exits 0 if all scenarios pass, 1 otherwise.
    """
    from ccmcp.validate import run_validation
    cfg, embedder, _, _ = _components(ctx.obj["config_path"])
    console.print("[bold]ccmcp validate[/bold]")
    passed, total = run_validation(cfg, embedder, console=console)
    raise SystemExit(0 if passed == total else 1)


@cli.command("save-config")
@click.option("--output", "-o", default="-", help="Output file path (default: stdout)")
@click.pass_context
def save_config(ctx, output: str):
    """Export the current resolved config to a YAML file.

    Merges config file, environment variables, and defaults into a single
    portable YAML file. Use with load-config to restore on a new deployment.

    \b
    Examples:
      ccmcp save-config                        # print to stdout
      ccmcp save-config -o backup.yaml         # write to file
      docker compose exec ccmcp ccmcp save-config -o /data/config-backup.yaml
    """
    import yaml as _yaml

    cfg = load_config(ctx.obj["config_path"])
    data = config_to_dict(cfg)
    text = _yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if output == "-":
        console.print(text, highlight=False, markup=False)
    else:
        Path(output).write_text(text, encoding="utf-8")
        console.print(f"[green]Config saved to:[/green] {output}")


@cli.command("load-config")
@click.argument("input_file")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def load_config_cmd(ctx, input_file: str, yes: bool):
    """Load a config file and write it to the active config path.

    Validates the config, then writes it to the path that ccmcp reads on
    startup (CCMCP_CONFIG env var, or config.yaml in the working directory).
    After loading, run 'ccmcp init' and 'ccmcp scan' to rebuild the dataset.

    \b
    Examples:
      ccmcp load-config backup.yaml
      docker compose exec ccmcp ccmcp load-config /data/config-backup.yaml
    """
    src = Path(input_file)
    if not src.exists():
        raise click.ClickException(f"File not found: {input_file}")

    # Validate by loading it
    try:
        candidate = load_config(str(src))
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from None

    import os
    dest = Path(os.environ.get("CCMCP_CONFIG", "config.yaml"))

    if not yes:
        click.echo(f"Write config to: {dest}")
        click.echo(f"  sources: {len(candidate.sources.filesystem.paths)} filesystem path(s)")
        click.echo(f"  qdrant:  {candidate.qdrant.url} / {candidate.qdrant.collection}")
        click.confirm("Proceed?", abort=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"[green]Config loaded to:[/green] {dest}")
    console.print(
        "Run [bold]ccmcp init[/bold] then [bold]ccmcp scan[/bold] to rebuild the dataset."
    )


@cli.command("save-snapshot")
@click.option("--output", "-o", default="ccmcp-snapshot.tar.gz",
              help="Output archive path (default: ccmcp-snapshot.tar.gz)")
@click.pass_context
def save_snapshot(ctx, output: str):
    """Save rotation matrix and state DB to a portable archive.

    The snapshot lets you redeploy without a full re-index: restore it
    with load-snapshot, run 'ccmcp init', then 'ccmcp scan' — unchanged
    files are skipped thanks to the restored state DB.

    \b
    Examples:
      ccmcp save-snapshot -o /tmp/ccmcp-snapshot.tar.gz
      docker compose exec ccmcp ccmcp save-snapshot -o /data/snapshot.tar.gz
    """
    import tarfile

    cfg = load_config(ctx.obj["config_path"])
    rot = Path(cfg.embedding.rotation_matrix)
    db = Path(cfg.state.db_path)

    missing = [str(p) for p in (rot, db) if not p.exists()]
    if missing:
        raise click.ClickException(f"Cannot snapshot — missing files: {', '.join(missing)}")

    with tarfile.open(output, "w:gz") as tar:
        tar.add(rot, arcname="rotation_matrix.npy")
        tar.add(db, arcname="state.db")

    console.print(f"[green]Snapshot saved to:[/green] {output}")
    console.print(f"  rotation_matrix.npy  ({rot.stat().st_size // 1024} KB)")
    console.print(f"  state.db             ({db.stat().st_size // 1024} KB)")


@cli.command("load-snapshot")
@click.argument("archive")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def load_snapshot(ctx, archive: str, yes: bool):
    """Restore rotation matrix and state DB from a snapshot archive.

    After loading, run 'ccmcp init' then 'ccmcp scan' to rebuild the
    Qdrant index. Unchanged files will be skipped automatically.

    \b
    Examples:
      ccmcp load-snapshot ccmcp-snapshot.tar.gz
      docker compose exec ccmcp ccmcp load-snapshot /data/snapshot.tar.gz
    """
    import tarfile

    src = Path(archive)
    if not src.exists():
        raise click.ClickException(f"File not found: {archive}")

    cfg = load_config(ctx.obj["config_path"])
    rot_dest = Path(cfg.embedding.rotation_matrix)
    db_dest = Path(cfg.state.db_path)

    with tarfile.open(src, "r:gz") as tar:
        members = {m.name for m in tar.getmembers()}
        if "rotation_matrix.npy" not in members or "state.db" not in members:
            raise click.ClickException("Archive is missing rotation_matrix.npy or state.db")

        if not yes:
            click.echo("Restore to:")
            click.echo(f"  rotation_matrix → {rot_dest}")
            click.echo(f"  state.db        → {db_dest}")
            if rot_dest.exists():
                click.echo("[warning] rotation_matrix.npy already exists and will be overwritten")
            click.confirm("Proceed?", abort=True)

        rot_dest.parent.mkdir(parents=True, exist_ok=True)
        db_dest.parent.mkdir(parents=True, exist_ok=True)

        member = tar.getmember("rotation_matrix.npy")
        member.name = rot_dest.name
        tar.extract(member, path=rot_dest.parent, filter="data")

        member = tar.getmember("state.db")
        member.name = db_dest.name
        tar.extract(member, path=db_dest.parent, filter="data")

    console.print("[green]Snapshot restored.[/green]")
    console.print("Run [bold]ccmcp init[/bold] then [bold]ccmcp scan[/bold] to rebuild the index.")


@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.pass_context
def run(ctx, host: str | None, port: int | None):
    """Start ingestion controller and MCP server together.

    Runs the ingestion controller (watch mode) in a background thread
    and the MCP SSE server in the foreground. This is the default
    Docker entrypoint — use 'serve' if you manage ingestion separately.
    """
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
    app = make_observable_app(mcp.sse_app(), store=store, embedder=embedder)
    console.print(f"[bold]ccmcp MCP server[/bold] → http://{h}:{p}/sse")
    console.print(f"[dim]Prometheus metrics → http://{h}:{p}/metrics[/dim]")
    console.print(f"[dim]Search UI          → http://{h}:{p}/[/dim]")
    uvicorn.run(app, host=h, port=p, log_level="warning")


if __name__ == "__main__":
    cli()
