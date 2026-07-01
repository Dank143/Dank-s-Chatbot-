import logging
from urllib.parse import urlparse, parse_qs
from .engines import _searxng_search, _ddg_search, _tavily_search

_log = logging.getLogger(__name__)

_YT_ALLOW = frozenset({"youtube.com", "youtu.be"})

def _is_youtube_video(url: str) -> bool:
    u = url.lower()
    return "youtube.com/watch" in u or "youtu.be/" in u

def _yt_video_id(url: str) -> str:
    """Extract the canonical video id so the same clip across subdomains dedupes."""
    p = urlparse(url)
    if "youtu.be" in (p.hostname or ""):
        return p.path.lstrip("/").split("/")[0]
    return parse_qs(p.query).get("v", [""])[0]

async def _fetch_media(rewritten: str, original: str, num_urls: int) -> tuple[str, dict]:
    """Media intent: return YouTube video links only — no scraping, no relevance gate."""
    search_query = f"site:youtube.com {rewritten}"
    results = await _searxng_search(search_query, max_results=num_urls + 4)
    
    if not results:
        results = await _ddg_search(
            rewritten, site="youtube.com", max_results=num_urls + 4, allowed=_YT_ALLOW,
        )
        
    if not results:
        results = await _tavily_search(search_query, max_results=num_urls + 4)

    vids, seen = [], set()
    for r in results:
        if not _is_youtube_video(r["url"]):
            continue
        vid = _yt_video_id(r["url"])
        if vid and vid in seen:
            continue
        seen.add(vid)
        vids.append(r)
        if len(vids) >= num_urls:
            break

    debug: dict = {
        "site": "youtube.com",
        "original_query": original,
        "rewritten_query": rewritten,
        "query": f"site:youtube.com {rewritten}",
        "fallback": False,
        "media": True,
        "sources": [
            {"url": r["url"], "method": "youtube", "chars": 0, "relevant": True}
            for r in vids
        ],
    }
    if not vids:
        return "", debug

    lines = [f"- {(r.get('title') or 'Video').strip()}: {r['url']}" for r in vids]
    ctx = (
        "=== YouTube Results (links only) ===\n\n"
        + "\n".join(lines)
        + "\n\n=== End of YouTube Results ==="
    )
    return ctx, debug
