import asyncio
import concurrent.futures
import logging
import os
import httpx
from ddgs import DDGS
from .fetcher import skip

_log = logging.getLogger(__name__)

_DDG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)
_FANDOM_ALLOW = frozenset({"fandom.com"})

async def _searxng_search(query: str, max_results: int = 10) -> list[dict]:
    """Primary search via self-hosted SearXNG."""
    try:
        # Strict timeout so a cold SearXNG container doesn't hang the UI for 30s.
        async with httpx.AsyncClient(timeout=7.5) as client:
            resp = await asyncio.wait_for(
                client.get(
                    "http://localhost:8888/search",
                    params={"q": query, "format": "json"}
                ),
                timeout=7.5
            )
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                        "title": r.get("title", "")
                    })
                return results[:max_results]
    except Exception as e:
        _log.warning("SearXNG failed: %s", e)
    return []

async def _ddg_search(
    query: str, site: str | None = None, max_results: int = 10,
    allowed: "frozenset[str]" = frozenset(), max_attempts: int = 3,
) -> list[dict]:
    """Secondary search using DuckDuckGo library with multi-backend."""
    search_query = f"site:{site} {query}" if site else query
    last_exc = None
    for attempt in range(max_attempts):
        try:
            # Instantiate DDGS per request to guarantee a fresh VQD token and avoid cross-thread async loop closures
            results = await asyncio.get_running_loop().run_in_executor(
                _DDG_EXECUTOR,
                lambda: list(DDGS(timeout=7.5).text(search_query, max_results=max_results, backend="duckduckgo,google,bing,brave,startpage"))
            )
            mapped = [
                {"url": r["href"], "snippet": r.get("body", ""), "title": r.get("title", "")}
                for r in (results or [])
                if r.get("href") and not skip(r["href"], allowed)
            ]
            if mapped:
                return mapped
        except Exception as e:
            last_exc = e
        await asyncio.sleep(0.5 * (attempt + 1))
    _log.warning("DDGS multi-backend failed for %r: %s", search_query, last_exc)
    return []

async def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Tertiary search using Tavily API (basic)."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                        "title": r.get("title", "")
                    })
                return results
            else:
                _log.warning("Tavily API returned %d: %s", resp.status_code, resp.text)
    except Exception as e:
        _log.warning("Tavily API failed: %s", e)
    return []
