from __future__ import annotations

from unittest.mock import MagicMock, patch

from ccmcp.sources.drive import _fetch_item


def _item(mime="application/vnd.google-apps.document", version="3", file_id="abc123"):
    return {"id": file_id, "name": "Test File", "mimeType": mime, "version": version}


def _state(drive_version=None):
    state = MagicMock()
    if drive_version is not None:
        state.get.return_value = MagicMock(drive_version=drive_version)
    else:
        state.get.return_value = None
    return state


def test_fetch_item_skips_unchanged_version():
    svc = MagicMock()
    result = _fetch_item(svc, _item(version="5"), _state(drive_version="5"))
    assert result is None
    svc.files().export.assert_not_called()


def test_fetch_item_fetches_when_version_changed():
    svc = MagicMock()
    svc.files().export().execute.return_value = b"Updated content"
    result = _fetch_item(svc, _item(version="6"), _state(drive_version="5"))
    assert result is not None
    assert "Updated content" in result.content


def test_fetch_item_exports_google_doc():
    svc = MagicMock()
    svc.files().export().execute.return_value = b"Hello from Google Docs"
    item = _item(mime="application/vnd.google-apps.document", version="3")
    result = _fetch_item(svc, item, _state())
    assert result is not None
    assert result.source_uri == "drive://abc123"
    assert "Hello from Google Docs" in result.content
    assert result.drive_version == "3"


def test_fetch_item_exports_as_string():
    svc = MagicMock()
    svc.files().export().execute.return_value = "String content"
    result = _fetch_item(svc, _item(), _state())
    assert result is not None
    assert result.content == "String content"


@patch("ccmcp.sources.drive.MediaIoBaseDownload")
def test_fetch_item_downloads_plain_text(mock_dl_class):
    def fake_init(buf, _req):
        buf.write(b"plain text content")
        m = MagicMock()
        m.next_chunk.return_value = (None, True)
        return m

    mock_dl_class.side_effect = fake_init

    svc = MagicMock()
    item = _item(mime="text/plain", version="1", file_id="txt1")
    result = _fetch_item(svc, item, _state())
    assert result is not None
    assert "plain text content" in result.content
    assert result.source_uri == "drive://txt1"


@patch("ccmcp.sources.drive.MediaIoBaseDownload")
def test_fetch_item_downloads_markdown(mock_dl_class):
    def fake_init(buf, _req):
        buf.write(b"# Title\n\nMarkdown body.")
        m = MagicMock()
        m.next_chunk.return_value = (None, True)
        return m

    mock_dl_class.side_effect = fake_init

    svc = MagicMock()
    item = _item(mime="text/markdown", version="2", file_id="md1")
    result = _fetch_item(svc, item, _state())
    assert result is not None
    assert "Title" in result.content


def test_fetch_item_returns_none_for_unsupported_mime():
    svc = MagicMock()
    item = _item(mime="video/mp4", version="1")
    result = _fetch_item(svc, item, _state())
    assert result is None


def test_fetch_item_returns_none_on_api_exception():
    svc = MagicMock()
    svc.files().export().execute.side_effect = Exception("API error")
    result = _fetch_item(svc, _item(), _state())
    assert result is None


def test_fetch_item_new_source_with_no_state_record():
    svc = MagicMock()
    svc.files().export().execute.return_value = b"New document"
    state = MagicMock()
    state.get.return_value = None
    result = _fetch_item(svc, _item(version="1"), state)
    assert result is not None
    assert result.drive_version == "1"
