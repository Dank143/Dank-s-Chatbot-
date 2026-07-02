import asyncio
import logging
import sys
from urllib.parse import urlparse

import httpx

try:
    import trafilatura as _trafilatura
except ImportError:
    _trafilatura = None

try:
    from patchright.async_api import async_playwright as _patchright
except ImportError:
    _patchright = None

_log = logging.getLogger(__name__)

_playwright_instance = None
_browser_instance = None

async def warmup_browser():
    global _playwright_instance, _browser_instance
    if _patchright is None:
        return
    if _browser_instance is None:
        try:
            # Uvicorn on Windows uses SelectorEventLoop, which lacks subprocess support.
            if sys.platform == "win32" and isinstance(asyncio.get_running_loop(), getattr(asyncio, "SelectorEventLoop", type(None))):
                _log.debug("Playwright is disabled (Windows SelectorEventLoop lacks subprocess support).")
                return
                
            _playwright_instance = await _patchright().start()
            _browser_instance = await _playwright_instance.chromium.launch(headless=True)
            _log.debug("Playwright browser warmup ok")
        except Exception:
            _log.debug("Playwright browser warmup failed", exc_info=True)

async def shutdown_browser():
    global _playwright_instance, _browser_instance
    if _browser_instance:
        await _browser_instance.close()
        _browser_instance = None
    if _playwright_instance:
        await _playwright_instance.stop()
        _playwright_instance = None

_JINA_BASE = "https://r.jina.ai/"
_MAX_CHARS = 20000
_MIN_CHARS = 2000
# Min snippet length to count as usable fallback context.
_MIN_SNIPPET = 80
_JINA_TIMEOUT = 5.0

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
_jina_client = httpx.AsyncClient(timeout=_JINA_TIMEOUT, follow_redirects=True, limits=_limits)
_fetch_client = httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=_BROWSER_HEADERS, limits=_limits)

# MediaWiki APIs need a descriptive UA per Wikimedia API policy.
_MW_HEADERS = {"User-Agent": "NIMChatbot/1.0 (web-search; contact vibecodersunity@gmail.com)"}
_mw_client = httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=_MW_HEADERS, limits=_limits)

_SKIP_DOMAINS = {
    "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "facebook.com",
    "fandom.com",
}

_CLOUDFLARE_DOMAINS = {"reddit.com"}

# MediaWiki article-path prefixes.
_MW_PAGE_PREFIXES = ("/wiki/", "/w/")
_MW_API_CANDIDATES = ("/w/api.php", "/api.php")
# host -> working api.php url (or None); lazily probed.
_mw_endpoint_cache: dict[str, "str | None"] = {}


def skip(url: str, allowed: "frozenset[str]" = frozenset()) -> bool:
    try:
        host = urlparse(url).hostname or ""
        if any(host == d or host.endswith("." + d) for d in allowed):
            return False
        return any(host == d or host.endswith("." + d) for d in _SKIP_DOMAINS)
    except Exception:
        return False


def truncate(text: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    cut = text[:_MAX_CHARS].rsplit(". ", 1)
    return (cut[0] + ".") if len(cut) > 1 else text[:_MAX_CHARS]


async def _extract(html: str) -> str:
    """Run trafilatura off-thread; truncate, and drop content under the floor."""
    if not html or _trafilatura is None:
        return ""
    text = await asyncio.to_thread(
        _trafilatura.extract, html,
        include_comments=False, include_tables=True, no_fallback=False,
        favor_recall=True,
    ) or ""
    return truncate(text) if len(text) >= _MIN_CHARS else ""


async def _jina_fetch(url: str) -> str:
    # Jina handles JS/table-heavy wiki pages
    try:
        res = await _jina_client.get(f"{_JINA_BASE}{url}", headers={"Accept": "text/plain"})
        res.raise_for_status()
        text = res.text
        if len(text) < _MIN_CHARS or "Just a moment" in text or "Ray ID:" in text:
            return ""
        return truncate(text)
    except Exception:
        _log.debug("Jina fetch failed for %r", url, exc_info=True)
        return ""


async def _trafilatura_fetch(url: str) -> str:
    if _trafilatura is None:
        return ""
    try:
        resp = await _fetch_client.get(url)
        resp.raise_for_status()
        return await _extract(resp.text)
    except Exception:
        _log.debug("Trafilatura fetch failed for %r", url, exc_info=True)
        return ""


async def _detect_mediawiki(host: str) -> "str | None":
    """Probe a host's api.php candidates concurrently; cache the working endpoint."""
    if host in _mw_endpoint_cache:
        return _mw_endpoint_cache[host]

    async def _probe(api_url: str) -> "str | None":
        try:
            r = await _mw_client.get(
                api_url, params={"action": "query", "meta": "siteinfo", "format": "json"}
            )
            if (
                r.status_code == 200
                and "application/json" in r.headers.get("content-type", "")
                and "query" in r.json()
            ):
                return api_url
        except Exception:
            pass
        return None

    results = await asyncio.gather(
        *[_probe(f"https://{host}{p}") for p in _MW_API_CANDIDATES]
    )
    endpoint = next((u for u in results if u), None)
    _mw_endpoint_cache[host] = endpoint
    return endpoint





def _mediawiki_title(path: str) -> "str | None":
    for prefix in _MW_PAGE_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):] or None
    return None


async def _mediawiki_api(api_url: str, title: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(api_url, params={
        "action": "query",
        "titles": title,
        "redirects": 1,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "format": "json",
    })
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return next(iter(pages.values()), {}).get("extract", "")


async def _mediawiki_fetch(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or ""
        if not host:
            return ""
        page_title = _mediawiki_title(parsed.path)
        if page_title is None:
            return ""
        api_url = await _detect_mediawiki(host)
        if api_url is None:
            return ""
        text = await _mediawiki_api(api_url, page_title, _mw_client)
        # Subpage miss (e.g. "Foo/Bar") — retry against parent titles.
        segments = page_title.split("/")
        while len(text) < 10 and len(segments) > 1:
            segments = segments[:-1]
            text = await _mediawiki_api(api_url, "/".join(segments), _mw_client)
        return truncate(text) if len(text) >= _MIN_CHARS else ""
    except Exception:
        _log.debug("MediaWiki fetch failed for %r", url, exc_info=True)
        return ""


def _is_cloudflare_domain(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return any(host == d or host.endswith("." + d) for d in _CLOUDFLARE_DOMAINS)


async def _playwright_fetch(url: str) -> str:
    if _patchright is None or _trafilatura is None or _browser_instance is None:
        return ""
    try:
        ctx = await _browser_instance.new_context(
            viewport={"width": 1280, "height": 800},
        )
        try:
            page = await ctx.new_page()
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
            html = await page.content()
        finally:
            await ctx.close()
        return await _extract(html)
    except Exception:
        _log.debug("Playwright fetch failed for %r", url, exc_info=True)
        return ""


# Fetcher coroutines by label, in default race order.
_FETCHERS = {
    "mediawiki": _mediawiki_fetch,
    "jina": _jina_fetch,
    "trafilatura": _trafilatura_fetch,
}
# host -> last-successful fetcher. Skips the 2 that always fail there.
_host_fetcher: dict[str, str] = {}


async def _race(url: str, labels: list[str]) -> tuple[str, str]:
    """Race the given fetchers; return (text, label) of the first non-empty."""
    async def _labeled(label):
        return label, await _FETCHERS[label](url)

    futs = [asyncio.ensure_future(_labeled(l)) for l in labels]
    try:
        for coro in asyncio.as_completed(futs):
            label, text = await coro
            if text:
                return text, label
    except Exception:
        pass
    finally:
        for f in futs:
            if not f.done():
                f.cancel()
    return "", ""


async def fetch_content(url: str, snippet: str = "") -> tuple[str, str]:
    """Fetch page text, preferring the host's known-good fetcher.
    Falls back to playwright (cloudflare) then the snippet."""
    host = urlparse(url).hostname or ""
    known = _host_fetcher.get(host)

    # Fast path: try the host's proven fetcher alone.
    if known:
        k_task = asyncio.create_task(_FETCHERS[known](url))
        tasks = [k_task]
        t_task = None
        
        if known != "trafilatura":
            t_task = asyncio.create_task(_trafilatura_fetch(url))
            tasks.append(t_task)
            
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                text = task.result()
                if text:
                    for p in pending: p.cancel()
                    winning_label = known if task == k_task else "trafilatura"
                    if winning_label != known:
                        _host_fetcher[host] = winning_label
                    return text, winning_label
            except Exception:
                pass
                
        if pending:
            done, pending = await asyncio.wait(pending)
            for task in done:
                try:
                    text = task.result()
                    if text:
                        winning_label = known if task == k_task else "trafilatura"
                        if winning_label != known:
                            _host_fetcher[host] = winning_label
                        return text, winning_label
                except Exception:
                    pass

    # Race the remaining fetchers (all of them if no known-good, or the others
    # if the known one just missed this page).
    skip_labels = {known, "trafilatura"} if known else set()
    rest = [l for l in _FETCHERS if l not in skip_labels]
    
    if rest:
        text, label = await _race(url, rest)
        if text:
            _host_fetcher[host] = label
            return text, label

    if _is_cloudflare_domain(url):
        text = await _playwright_fetch(url)
        if text:
            return text, "playwright"

    # Snippet fallback only if it carries real text (>= 80 chars).
    if len(snippet.strip()) >= _MIN_SNIPPET:
        return snippet[:1000], "snippet"
    return "", "failed"
