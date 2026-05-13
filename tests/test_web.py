import httpx
import respx

from ccmcp.sources.web import _fetch_sitemap, fetch

SAMPLE_HTML = """
<html><body>
<h1>Doc Title</h1>
<p>This is the main content of the document.</p>
</body></html>
"""

SAMPLE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
</urlset>
"""


@respx.mock
def test_fetch_200_returns_source_file():
    respx.get("https://example.com/doc").mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML, headers={"ETag": '"abc"'})
    )
    sf = fetch("https://example.com/doc", "test-agent")
    assert sf is not None
    assert "content" in sf.content.lower() or "doc" in sf.content.lower()
    assert sf.source_uri == "https://example.com/doc"
    assert sf.etag == '"abc"'


@respx.mock
def test_fetch_304_returns_none():
    respx.get("https://example.com/doc").mock(
        return_value=httpx.Response(304)
    )

    class FakeRecord:
        etag = '"abc"'
        last_modified = None

    sf = fetch("https://example.com/doc", "test-agent", FakeRecord())
    assert sf is None


@respx.mock
def test_fetch_404_returns_none():
    respx.get("https://example.com/missing").mock(
        return_value=httpx.Response(404)
    )
    sf = fetch("https://example.com/missing", "test-agent")
    assert sf is None


@respx.mock
def test_fetch_sends_conditional_headers():
    class FakeRecord:
        etag = '"xyz"'
        last_modified = "Mon, 01 Jan 2026 00:00:00 GMT"

    req = respx.get("https://example.com/doc").mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )
    fetch("https://example.com/doc", "test-agent", FakeRecord())
    assert req.called
    sent = req.calls[0].request
    assert sent.headers.get("If-None-Match") == '"xyz"'
    assert "2026" in sent.headers.get("If-Modified-Since", "")


@respx.mock
def test_fetch_network_error_returns_none():
    respx.get("https://example.com/broken").mock(side_effect=httpx.ConnectError("fail"))
    sf = fetch("https://example.com/broken", "test-agent")
    assert sf is None


@respx.mock
def test_fetch_sitemap_returns_urls():
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=httpx.Response(200, text=SAMPLE_SITEMAP)
    )
    urls = _fetch_sitemap("https://example.com/sitemap.xml", "test-agent")
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls


@respx.mock
def test_fetch_sitemap_empty_on_error():
    respx.get("https://example.com/sitemap.xml").mock(
        return_value=httpx.Response(500)
    )
    urls = _fetch_sitemap("https://example.com/sitemap.xml", "test-agent")
    assert urls == []
