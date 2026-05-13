
import pytest

from ccmcp.state import SourceRecord, StateDB


@pytest.fixture
def db(tmp_path):
    return StateDB(str(tmp_path / "state.db"))


def _rec(uri="file:///a.md", version=1, last_seen="2026-01-01T00:00:00+00:00"):
    return SourceRecord(
        source_uri=uri,
        doc_id="docid",
        content_hash="sha256:abc",
        version=version,
        last_seen=last_seen,
    )


def test_get_missing(db):
    assert db.get("file:///nonexistent") is None


def test_upsert_and_get(db):
    rec = _rec()
    db.upsert(rec)
    got = db.get(rec.source_uri)
    assert got is not None
    assert got.content_hash == "sha256:abc"
    assert got.version == 1


def test_upsert_updates_existing(db):
    db.upsert(_rec(version=1))
    db.upsert(_rec(version=2, last_seen="2026-06-01T00:00:00+00:00"))
    got = db.get("file:///a.md")
    assert got.version == 2


def test_delete(db):
    db.upsert(_rec())
    db.delete("file:///a.md")
    assert db.get("file:///a.md") is None


def test_unseen_since(db):
    db.upsert(_rec("file:///old.md", last_seen="2026-01-01T00:00:00+00:00"))
    db.upsert(_rec("file:///new.md", last_seen="2026-06-01T00:00:00+00:00"))
    unseen = db.unseen_since("2026-03-01T00:00:00+00:00")
    uris = {r.source_uri for r in unseen}
    assert "file:///old.md" in uris
    assert "file:///new.md" not in uris


def test_all(db):
    db.upsert(_rec("file:///a.md"))
    db.upsert(_rec("file:///b.md"))
    assert len(db.all()) == 2
