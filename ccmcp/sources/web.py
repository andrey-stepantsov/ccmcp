from __future__ import annotations

import time

import httpx
from bs4 import BeautifulSoup
from readability import Document

from ccmcp.sources import SourceFile


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def fetch(url: str, user_agent: str, state_record=None) -> SourceFile | None:
    headers: dict[str, str] = {"User-Agent": user_agent}
    if state_record:
        if state_record.etag:
            headers["If-None-Match"] = state_record.etag
        if state_record.last_modified:
            headers["If-Modified-Since"] = state_record.last_modified

    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    except httpx.RequestError:
        return None

    if resp.status_code == 304:
        return None
    if resp.status_code != 200:
        return None

    text = _html_to_text(Document(resp.text).summary())
    if not text.strip():
        return None

    return SourceFile(
        source_uri=url,
        content=text,
        etag=resp.headers.get("ETag"),
        last_modified=resp.headers.get("Last-Modified"),
    )


def _fetch_sitemap(url: str, user_agent: str) -> list[str]:
    try:
        resp = httpx.get(url, headers={"User-Agent": user_agent}, follow_redirects=True, timeout=30)
    except httpx.RequestError:
        return []
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "xml")
    return [loc.get_text() for loc in soup.find_all("loc")]


def fetch_all(
    urls: list[str],
    sitemaps: list[str],
    user_agent: str,
    rate_limit_ms: int,
    state,
) -> list[SourceFile]:
    all_urls = list(urls)
    for sitemap in sitemaps:
        all_urls.extend(_fetch_sitemap(sitemap, user_agent))
    # deduplicate while preserving order
    seen: set[str] = set()
    deduped = [u for u in all_urls if not (u in seen or seen.add(u))]  # type: ignore[func-returns-value]

    results: list[SourceFile] = []
    for url in deduped:
        sf = fetch(url, user_agent, state.get(url))
        if sf:
            results.append(sf)
        if rate_limit_ms > 0:
            time.sleep(rate_limit_ms / 1000)
    return results
