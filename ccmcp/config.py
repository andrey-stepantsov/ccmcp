from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection: str = "techdocs"
    api_key: str = ""


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    context_window: int = 32768


@dataclass
class ClaudeApiConfig:
    model: str = "claude-sonnet-4-20250514"


@dataclass
class ExtractionConfig:
    backend: str = "none"  # "none" | "ollama" | "claude_api"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    claude_api: ClaudeApiConfig = field(default_factory=ClaudeApiConfig)


@dataclass
class EmbeddingConfig:
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rotation_matrix: str = "/data/rotation_matrix.npy"


@dataclass
class SourcePath:
    path: str
    name: str = ""
    tags: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)


@dataclass
class FilesystemConfig:
    enabled: bool = True
    paths: list[SourcePath] = field(default_factory=list)
    watch: bool = True
    extensions: list[str] = field(default_factory=lambda: [
        ".md", ".rst", ".txt", ".py", ".go", ".rs", ".js", ".ts",
        ".yaml", ".yml",
        ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh",
    ])
    ignore: list[str] = field(default_factory=lambda: [
        "node_modules", ".git", "__pycache__", "*.pyc",
        "build", "dist", ".venv", "*.egg-info",
    ])
    respect_gitignore: bool = True
    poll_interval: int = 30


@dataclass
class WebConfig:
    enabled: bool = False
    urls: list[str] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)
    crawl_depth: int = 0
    rate_limit_ms: int = 500
    user_agent: str = "ccmcp-ingestion/1.0"


@dataclass
class DriveFolder:
    id: str
    name: str


@dataclass
class GoogleDriveConfig:
    enabled: bool = False
    credentials_file: str = ""
    folders: list[DriveFolder] = field(default_factory=list)
    poll_interval_min: int = 15


@dataclass
class SourcesConfig:
    filesystem: FilesystemConfig = field(default_factory=FilesystemConfig)
    web: WebConfig = field(default_factory=WebConfig)
    google_drive: GoogleDriveConfig = field(default_factory=GoogleDriveConfig)


@dataclass
class StateConfig:
    db_path: str = "/data/state.db"
    artifact_ttl_days: int = 30


@dataclass
class McpConfig:
    port: int = 7700
    host: str = "127.0.0.1"
    result_limit: int = 10


@dataclass
class Config:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    state: StateConfig = field(default_factory=StateConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


def config_to_dict(cfg: Config) -> dict:
    """Serialise a resolved Config to a plain dict suitable for YAML export."""
    return {
        "qdrant": {
            "url": cfg.qdrant.url,
            "collection": cfg.qdrant.collection,
            **({"api_key": cfg.qdrant.api_key} if cfg.qdrant.api_key else {}),
        },
        "embedding": {
            "dense_model": cfg.embedding.dense_model,
            "sparse_model": cfg.embedding.sparse_model,
            "rotation_matrix": cfg.embedding.rotation_matrix,
        },
        "extraction": {
            "backend": cfg.extraction.backend,
        },
        "sources": {
            "filesystem": {
                "enabled": cfg.sources.filesystem.enabled,
                "paths": [
                    ({"path": sp.path, "name": sp.name, "tags": sp.tags, "include": sp.include}
                     if sp.name or sp.tags or sp.include else sp.path)
                    for sp in cfg.sources.filesystem.paths
                ],
                "watch": cfg.sources.filesystem.watch,
                "extensions": cfg.sources.filesystem.extensions,
                "ignore": cfg.sources.filesystem.ignore,
                "respect_gitignore": cfg.sources.filesystem.respect_gitignore,
                "poll_interval": cfg.sources.filesystem.poll_interval,
            },
            "web": {
                "enabled": cfg.sources.web.enabled,
                "urls": cfg.sources.web.urls,
                "sitemaps": cfg.sources.web.sitemaps,
                "rate_limit_ms": cfg.sources.web.rate_limit_ms,
                "user_agent": cfg.sources.web.user_agent,
            },
            "google_drive": {
                "enabled": cfg.sources.google_drive.enabled,
                "credentials_file": cfg.sources.google_drive.credentials_file,
                "folders": [{"id": f.id, "name": f.name} for f in cfg.sources.google_drive.folders],
                "poll_interval_min": cfg.sources.google_drive.poll_interval_min,
            },
        },
        "mcp": {
            "host": cfg.mcp.host,
            "port": cfg.mcp.port,
            "result_limit": cfg.mcp.result_limit,
        },
        "state": {
            "db_path": cfg.state.db_path,
            "artifact_ttl_days": cfg.state.artifact_ttl_days,
        },
    }


def _parse_source_paths(raw: list) -> list[SourcePath]:
    """Parse path entries that are either plain strings or dicts with metadata."""
    result = []
    for entry in raw:
        if isinstance(entry, str):
            result.append(SourcePath(path=str(Path(entry).expanduser())))
        elif isinstance(entry, dict):
            result.append(SourcePath(
                path=str(Path(entry["path"]).expanduser()),
                name=str(entry.get("name", "")),
                tags=[str(t) for t in entry.get("tags", [])],
                include=[str(t) for t in entry.get("include", [])],
            ))
    return result


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("CCMCP_CONFIG", "config.yaml")
    raw: dict = {}
    if Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()

    if q := raw.get("qdrant"):
        cfg.qdrant = QdrantConfig(
            url=q.get("url", cfg.qdrant.url),
            collection=q.get("collection", cfg.qdrant.collection),
            api_key=q.get("api_key", cfg.qdrant.api_key),
        )
    # env vars take precedence over file
    if v := os.environ.get("QDRANT_URL"):
        cfg.qdrant.url = v
    if v := os.environ.get("QDRANT_API_KEY"):
        cfg.qdrant.api_key = v
    if v := os.environ.get("QDRANT_COLLECTION"):
        cfg.qdrant.collection = v

    if e := raw.get("embedding"):
        cfg.embedding = EmbeddingConfig(
            dense_model=e.get("dense_model", cfg.embedding.dense_model),
            sparse_model=e.get("sparse_model", cfg.embedding.sparse_model),
            rotation_matrix=e.get("rotation_matrix", cfg.embedding.rotation_matrix),
        )

    if ex := raw.get("extraction"):
        cfg.extraction.backend = ex.get("backend", "none")
        if ol := ex.get("ollama"):
            cfg.extraction.ollama = OllamaConfig(
                model=ol.get("model", cfg.extraction.ollama.model),
                base_url=ol.get("base_url", cfg.extraction.ollama.base_url),
                context_window=ol.get("context_window", cfg.extraction.ollama.context_window),
            )
        if ca := ex.get("claude_api"):
            cfg.extraction.claude_api = ClaudeApiConfig(
                model=ca.get("model", cfg.extraction.claude_api.model),
            )

    if s := raw.get("sources"):
        if fs := s.get("filesystem"):
            cfg.sources.filesystem = FilesystemConfig(
                enabled=fs.get("enabled", True),
                paths=_parse_source_paths(fs.get("paths", fs.get("roots", []))),
                watch=fs.get("watch", True),
                extensions=fs.get("extensions", cfg.sources.filesystem.extensions),
                ignore=fs.get("ignore", cfg.sources.filesystem.ignore),
                respect_gitignore=fs.get("respect_gitignore", True),
                poll_interval=fs.get("poll_interval", 30),
            )
        if wb := s.get("web"):
            cfg.sources.web = WebConfig(
                enabled=wb.get("enabled", False),
                urls=wb.get("urls", []),
                sitemaps=wb.get("sitemaps", []),
                crawl_depth=wb.get("crawl_depth", 0),
                rate_limit_ms=wb.get("rate_limit_ms", 500),
                user_agent=wb.get("user_agent", "ccmcp-ingestion/1.0"),
            )
        if dr := s.get("google_drive"):
            cfg.sources.google_drive = GoogleDriveConfig(
                enabled=dr.get("enabled", False),
                credentials_file=dr.get("credentials_file", ""),
                folders=[DriveFolder(**f) for f in dr.get("folders", [])],
                poll_interval_min=dr.get("poll_interval_min", 15),
            )

    # CCMCP_SOURCE_PATH overrides filesystem paths (Docker use case).
    # Stamp a project name derived from the path basename so scoping works
    # even without an explicit config entry.
    source_path = os.environ.get("CCMCP_SOURCE_PATH")
    if source_path and not cfg.sources.filesystem.paths:
        derived_name = Path(source_path.rstrip("/")).name or source_path
        cfg.sources.filesystem.paths = [
            SourcePath(path=source_path, name=derived_name)
        ]
        cfg.sources.filesystem.enabled = True

    if st := raw.get("state"):
        cfg.state = StateConfig(
            db_path=str(Path(st.get("db_path", cfg.state.db_path)).expanduser()),
            artifact_ttl_days=st.get("artifact_ttl_days", 30),
        )

    if m := raw.get("mcp"):
        cfg.mcp = McpConfig(
            port=m.get("port", cfg.mcp.port),
            host=m.get("host", cfg.mcp.host),
            result_limit=m.get("result_limit", cfg.mcp.result_limit),
        )

    return cfg
