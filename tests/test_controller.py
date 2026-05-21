from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from ccmcp.controller import Controller, _source_type
from ccmcp.sources import SourceFile
from ccmcp.state import SourceRecord


def _now():
    return datetime.now(UTC).isoformat()


def _rec(uri="file:///a.md", content_hash="sha256:old", version=1):
    return SourceRecord(
        source_uri=uri,
        doc_id="docid",
        content_hash=content_hash,
        version=version,
        last_seen=_now(),
    )


@pytest.fixture
def ctrl():
    cfg = MagicMock()
    cfg.sources.filesystem.enabled = False
    cfg.sources.web.enabled = False
    cfg.sources.google_drive.enabled = False

    embedder = MagicMock()

    def embed_side_effect(texts):
        n = len(texts)
        return (np.zeros((n, 384), dtype=np.float32), [MagicMock() for _ in range(n)])

    embedder.embed.side_effect = embed_side_effect

    store = MagicMock()
    store.upsert.side_effect = (
        lambda chunks, dense, sparse, version, source_type, **_kw: len(chunks)
    )
    store.doc_id.return_value = "abc123"

    state = MagicMock()
    state.get.return_value = None
    state.unseen_since.return_value = []

    return Controller(cfg, embedder, store, state)


def test_ingest_file_skips_when_hash_unchanged(ctrl):
    content = "# Hello\n\nContent."
    h = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    ctrl._state.get.return_value = _rec(content_hash=h)
    ctrl.ingest_file(SourceFile(source_uri="file:///a.md", content=content))
    ctrl._store.upsert.assert_not_called()


def test_ingest_file_upserts_new_content(ctrl):
    ctrl.ingest_file(SourceFile(source_uri="file:///new.md", content="# Hello\n\nContent."))
    ctrl._store.upsert.assert_called_once()
    ctrl._store.delete_old_version.assert_not_called()


def test_ingest_file_raises_on_upsert_count_mismatch(ctrl):
    ctrl._store.upsert.side_effect = None
    ctrl._store.upsert.return_value = 0
    with pytest.raises(RuntimeError, match="Upsert failed"):
        ctrl.ingest_file(SourceFile(source_uri="file:///fail.md", content="# Hello\n\nContent."))


def test_ingest_file_deletes_old_version_after_upsert(ctrl):
    ctrl._state.get.return_value = _rec(uri="file:///update.md", version=2)
    ctrl.ingest_file(SourceFile(source_uri="file:///update.md", content="# New\n\nNew content."))
    ctrl._store.delete_old_version.assert_called_once_with("file:///update.md", 2)


def test_ingest_file_increments_version(ctrl):
    ctrl._state.get.return_value = _rec(uri="file:///ver.md", version=3)
    ctrl.ingest_file(SourceFile(source_uri="file:///ver.md", content="# Changed\n\nNew body."))
    call_args = ctrl._store.upsert.call_args
    assert call_args.args[3] == 4  # version = old + 1


def test_ingest_web_source_succeeds(ctrl):
    ctrl.ingest_file(SourceFile(
        source_uri="https://example.com/page",
        content="Some plain content without headings.",
    ))
    ctrl._store.upsert.assert_called_once()


def test_ingest_drive_source_succeeds(ctrl):
    ctrl.ingest_file(SourceFile(
        source_uri="drive://abc123",
        content="Drive document content.",
    ))
    ctrl._store.upsert.assert_called_once()


def test_cleanup_orphans_removes_unseen(ctrl):
    orphan = _rec(uri="file:///gone.md")
    ctrl._state.unseen_since.return_value = [orphan]
    ctrl._cleanup_orphans("2026-06-01T00:00:00+00:00")
    ctrl._store.delete_doc.assert_called_once_with("file:///gone.md")
    ctrl._state.delete.assert_called_once_with("file:///gone.md")


def test_cleanup_orphans_no_op_when_all_seen(ctrl):
    ctrl._state.unseen_since.return_value = []
    ctrl._cleanup_orphans("2026-06-01T00:00:00+00:00")
    ctrl._store.delete_doc.assert_not_called()


# ---------------------------------------------------------------------------
# _source_type
# ---------------------------------------------------------------------------

def test_source_type_file():
    assert _source_type("file:///path/to/file.md") == "fs"


def test_source_type_drive():
    assert _source_type("drive://abc123") == "drive"


def test_source_type_http():
    assert _source_type("http://example.com/page") == "web"


def test_source_type_https():
    assert _source_type("https://example.com/page") == "web"


def test_source_type_unknown_raises():
    with pytest.raises(ValueError, match="Unknown URI scheme"):
        _source_type("/bare/path/without/scheme")
