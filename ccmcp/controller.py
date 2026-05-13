from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime

from ccmcp.chunker import chunk_file
from ccmcp.config import Config
from ccmcp.embedder import Embedder
from ccmcp.sources import SourceFile
from ccmcp.state import SourceRecord, StateDB
from ccmcp.store import VectorStore

log = logging.getLogger(__name__)

_RESCAN_INTERVAL = 3600  # full re-scan every hour while watching


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _content_hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _source_type(uri: str) -> str:
    if uri.startswith("file://"):
        return "fs"
    if uri.startswith("drive://"):
        return "drive"
    return "web"


class Controller:
    def __init__(self, config: Config, embedder: Embedder, store: VectorStore, state: StateDB):
        self._cfg = config
        self._embedder = embedder
        self._store = store
        self._state = state

    def ingest_file(self, sf: SourceFile):
        h = _content_hash(sf.content)
        record = self._state.get(sf.source_uri)
        now = _now()

        if record and record.content_hash == h:
            record.last_seen = now
            self._state.upsert(record)
            return

        version = (record.version + 1) if record else 1
        path = sf.source_uri.removeprefix("file://")
        chunks = chunk_file(path, sf.content, sf.source_uri)
        if not chunks:
            return

        dense, sparse = self._embedder.embed([c.text for c in chunks])
        count = self._store.upsert(chunks, dense, sparse, version, _source_type(sf.source_uri))
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

        if cfg.sources.filesystem.enabled and cfg.sources.filesystem.roots:
            from ccmcp.sources import filesystem as fs_mod
            files = fs_mod.scan(
                roots=cfg.sources.filesystem.roots,
                extensions=cfg.sources.filesystem.extensions,
                ignore=cfg.sources.filesystem.ignore,
            )
            log.info("filesystem scan: %d files found", len(files))
            for sf in files:
                try:
                    self.ingest_file(sf)
                except Exception as exc:
                    log.warning("ingest failed %s: %s", sf.source_uri, exc)

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

        self._cleanup_orphans(scan_start)

    def _cleanup_orphans(self, scan_start: str):
        for rec in self._state.unseen_since(scan_start):
            self._store.delete_doc(rec.source_uri)
            self._state.delete(rec.source_uri)
            log.info("removed orphan: %s", rec.source_uri)

    def watch(self):
        self.scan()

        if not (self._cfg.sources.filesystem.enabled and self._cfg.sources.filesystem.roots):
            return

        from ccmcp.sources import filesystem as fs_mod

        def on_change(sf: SourceFile):
            try:
                self.ingest_file(sf)
            except Exception as exc:
                log.warning("watch callback failed %s: %s", sf.source_uri, exc)

        fscfg = self._cfg.sources.filesystem
        observer = fs_mod.watch(
            roots=fscfg.roots,
            extensions=fscfg.extensions,
            ignore=fscfg.ignore,
            callback=on_change,
            poll_interval=fscfg.poll_interval,
        )
        try:
            while True:
                time.sleep(_RESCAN_INTERVAL)
                self.scan()
        finally:
            observer.stop()
            observer.join()
