from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceRecord:
    source_uri: str
    doc_id: str
    content_hash: str
    version: int
    last_seen: str
    status: str = "ok"
    etag: str | None = None
    last_modified: str | None = None
    drive_version: str | None = None


class StateDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_uri    TEXT PRIMARY KEY,
                    doc_id        TEXT NOT NULL,
                    content_hash  TEXT NOT NULL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    last_seen     TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'ok',
                    etag          TEXT,
                    last_modified TEXT,
                    drive_version TEXT
                )
            """)

    def get(self, source_uri: str) -> SourceRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE source_uri = ?", (source_uri,)
            ).fetchone()
        return SourceRecord(**dict(row)) if row else None

    def upsert(self, record: SourceRecord):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sources
                    (source_uri, doc_id, content_hash, version, last_seen, status,
                     etag, last_modified, drive_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uri) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    content_hash=excluded.content_hash,
                    version=excluded.version,
                    last_seen=excluded.last_seen,
                    status=excluded.status,
                    etag=excluded.etag,
                    last_modified=excluded.last_modified,
                    drive_version=excluded.drive_version
            """, (
                record.source_uri, record.doc_id, record.content_hash,
                record.version, record.last_seen, record.status,
                record.etag, record.last_modified, record.drive_version,
            ))

    def delete(self, source_uri: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM sources WHERE source_uri = ?", (source_uri,))

    def unseen_since(self, cutoff: str) -> list[SourceRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE last_seen < ?", (cutoff,)
            ).fetchall()
        return [SourceRecord(**dict(r)) for r in rows]

    def all(self) -> list[SourceRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM sources").fetchall()
        return [SourceRecord(**dict(r)) for r in rows]
