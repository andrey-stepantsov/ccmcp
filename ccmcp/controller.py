from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from ccmcp.chunker import chunk_file
from ccmcp.config import Config, SourcePath
from ccmcp.embedder import Embedder
from ccmcp.metrics import (
    CHUNK_CHARS,
    CHUNKS_INGESTED,
    DOCUMENTS_INGESTED,
    INGEST_ERRORS,
    INGEST_SECONDS,
)
from ccmcp.sources import SourceFile
from ccmcp.state import SourceRecord, StateDB, _now
from ccmcp.store import VectorStore

log = logging.getLogger(__name__)

_RESCAN_INTERVAL = 3600  # full re-scan every hour while watching


def _content_hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _source_type(uri: str) -> str:
    if uri.startswith("file://"):
        return "fs"
    if uri.startswith("drive://"):
        return "drive"
    if uri.startswith("http://") or uri.startswith("https://"):
        return "web"
    raise ValueError(f"Unknown URI scheme: {uri!r}")


def _source_type_safe(uri: str) -> str:
    try:
        return _source_type(uri)
    except ValueError:
        return "unknown"


def _resolved(path: str) -> str:
    return os.path.realpath(str(Path(path).expanduser()))


class Controller:
    def __init__(self, config: Config, embedder: Embedder, store: VectorStore, state: StateDB):
        self._cfg = config
        self._embedder = embedder
        self._store = store
        self._state = state

    def ingest_file(
        self,
        sf: SourceFile,
        source_root: str = "",
        project_name: str = "",
        tags: list[str] | None = None,
    ):
        h = _content_hash(sf.content)
        record = self._state.get(sf.source_uri)
        now = _now()

        if record and record.content_hash == h:
            record.last_seen = now
            self._state.upsert(record)
            return

        stype = _source_type(sf.source_uri)
        t0 = time.perf_counter()
        version = (record.version + 1) if record else 1
        # Non-file:// URIs (web, drive) use text chunking; pass empty path to fall through
        path = sf.source_uri.removeprefix("file://") if sf.source_uri.startswith("file://") else ""
        chunks = chunk_file(path, sf.content, sf.source_uri)
        if not chunks:
            return

        for c in chunks:
            CHUNK_CHARS.observe(len(c.text))

        dense, sparse = self._embedder.embed([c.text for c in chunks])
        count = self._store.upsert(
            chunks, dense, sparse, version, stype,
            source_root=source_root, project_name=project_name, tags=tags,
        )
        if count != len(chunks):
            raise RuntimeError(
                f"Upsert failed: expected {len(chunks)} points, got {count} for {sf.source_uri}"
            )
        INGEST_SECONDS.observe(time.perf_counter() - t0)
        CHUNKS_INGESTED.labels(source_type=stype).inc(count)
        DOCUMENTS_INGESTED.labels(source_type=stype).inc()
        log.info("upserted %d chunks for %s (v%d)", count, sf.source_uri, version)

        if record:
            self._store.delete_old_version(sf.source_uri, record.version)

        self._state.upsert(SourceRecord(
            source_uri=sf.source_uri,
            doc_id=self._store.doc_id(sf.source_uri),
            content_hash=h,
            version=version,
            last_seen=now,
            etag=sf.etag,
            last_modified=sf.last_modified,
            drive_version=sf.drive_version,
        ))

    def scan(self):
        scan_start = _now()
        cfg = self._cfg

        if cfg.sources.filesystem.enabled and cfg.sources.filesystem.paths:
            from ccmcp.sources import filesystem as fs_mod

            for sp in cfg.sources.filesystem.paths:
                source_root = _resolved(sp.path)
                project_name = sp.name or Path(sp.path).name
                tags = sp.tags or None
                files = fs_mod.scan(
                    roots=[sp.path],
                    extensions=cfg.sources.filesystem.extensions,
                    ignore=cfg.sources.filesystem.ignore,
                    respect_gitignore=cfg.sources.filesystem.respect_gitignore,
                )
                log.info("filesystem scan %s: %d files found", sp.path, len(files))
                for sf in files:
                    try:
                        self.ingest_file(
                            sf, source_root=source_root,
                            project_name=project_name, tags=tags,
                        )
                    except Exception as exc:
                        log.warning("ingest failed %s: %s", sf.source_uri, exc)
                        INGEST_ERRORS.labels(source_type=_source_type_safe(sf.source_uri)).inc()

        if cfg.sources.web.enabled and cfg.sources.web.urls:
            from ccmcp.sources import web as web_mod
            files = web_mod.fetch_all(
                urls=cfg.sources.web.urls,
                sitemaps=cfg.sources.web.sitemaps,
                user_agent=cfg.sources.web.user_agent,
                rate_limit_ms=cfg.sources.web.rate_limit_ms,
                state=self._state,
            )
            for sf in files:
                try:
                    self.ingest_file(sf)
                except Exception as exc:
                    log.warning("ingest failed %s: %s", sf.source_uri, exc)
                    INGEST_ERRORS.labels(source_type=_source_type_safe(sf.source_uri)).inc()

        if cfg.sources.google_drive.enabled and cfg.sources.google_drive.credentials_file:
            from ccmcp.sources import drive as drive_mod
            files = drive_mod.fetch_all(
                credentials_file=cfg.sources.google_drive.credentials_file,
                folders=cfg.sources.google_drive.folders,
                state=self._state,
            )
            for sf in files:
                try:
                    self.ingest_file(sf)
                except Exception as exc:
                    log.warning("ingest failed %s: %s", sf.source_uri, exc)
                    INGEST_ERRORS.labels(source_type=_source_type_safe(sf.source_uri)).inc()

        self._cleanup_orphans(scan_start)

    def _cleanup_orphans(self, scan_start: str):
        for rec in self._state.unseen_since(scan_start):
            self._store.delete_doc(rec.source_uri)
            self._state.delete(rec.source_uri)
            log.info("removed orphan: %s", rec.source_uri)

    def watch(self):
        self.scan()

        if not (self._cfg.sources.filesystem.enabled and self._cfg.sources.filesystem.paths):
            return

        from ccmcp.sources import filesystem as fs_mod

        fscfg = self._cfg.sources.filesystem

        # Build resolved path → SourcePath index for per-file event lookup.
        path_index: dict[str, SourcePath] = {
            _resolved(sp.path): sp for sp in fscfg.paths
        }

        def _sp_for_path(file_path: str) -> SourcePath | None:
            abs_path = os.path.realpath(file_path)
            for abs_root, sp in path_index.items():
                if abs_path.startswith(abs_root + os.sep) or abs_path == abs_root:
                    return sp
            return None

        def on_change(sf: SourceFile):
            try:
                file_path = sf.source_uri.removeprefix("file://")
                sp = _sp_for_path(file_path)
                self.ingest_file(
                    sf,
                    source_root=_resolved(sp.path) if sp else "",
                    project_name=(sp.name or Path(sp.path).name) if sp else "",
                    tags=sp.tags or None if sp else None,
                )
            except Exception as exc:
                log.warning("watch callback failed %s: %s", sf.source_uri, exc)

        observer = fs_mod.watch(
            roots=[sp.path for sp in fscfg.paths],
            extensions=fscfg.extensions,
            ignore=fscfg.ignore,
            callback=on_change,
            poll_interval=fscfg.poll_interval,
            respect_gitignore=fscfg.respect_gitignore,
        )
        try:
            while True:
                time.sleep(_RESCAN_INTERVAL)
                self.scan()
        finally:
            observer.stop()
            observer.join()
