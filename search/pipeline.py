import asyncio
import time
import logging
import re
import hashlib
from datetime import datetime

from config import load_config
from .fetcher import fetch_content, skip, warmup_browser, shutdown_browser
from .cache import _cache_get, _cache_set
from .engines import _searxng_search, _ddg_search, _tavily_search, _FANDOM_ALLOW
from .llm_processing import _rewrite_query, _rerank, _REGEX_INTENTS

_log = logging.getLogger(__name__)
_cfg = load_config()

_MAX_URLS = _cfg.get("defaults", {}).get("max_search_urls", 5)

_warmup_started = False
_warmup_done = asyncio.Event()


async def warmup() -> None:
    """Prime the DDG session at boot to avoid first-query ratelimit."""
    global _warmup_started
    _warmup_started = True
    try:
        import httpx
        # Warmup SearXNG so its internal docker workers spin up and resolve DNS
        async with httpx.AsyncClient(timeout=20.0) as client:
            await client.get("http://localhost:8888/search", params={"q": "wikipedia", "format": "json"})
        _log.debug("SearXNG warmup ok")
    except Exception:
        _log.debug("SearXNG warmup failed", exc_info=True)
        
    try:
        await warmup_browser()
    except Exception:
        pass

    finally:
        _warmup_done.set()

async def shutdown() -> None:
    await shutdown_browser()


async def _await_warmup() -> None:
    """Wait for boot warmup to finish so the first search hits a warm session."""
    if _warmup_started and not _warmup_done.is_set():
        try:
            await asyncio.wait_for(_warmup_done.wait(), timeout=6.0)
        except asyncio.TimeoutError:
            pass


def _clean(results: list[dict]) -> list[dict]:
    """Drop skip-listed domains (youtube/social/etc, fandom.com exempted) before
    they can win a race or waste a fetch slot. DDG already filters internally;
    this closes the same gap for SearXNG/Tavily, which don't."""
    return [r for r in results if r.get("url") and not skip(r["url"], _FANDOM_ALLOW)]


async def _staggered_search_cascade(
    searxng_q: str, search_query: str, site: str | None, max_results: int, is_fallback: bool = False
) -> tuple[list[dict], str]:
    """
    Run SearXNG -> DDG -> Tavily in a staggered race.
    Budget is halved if is_fallback=True.
    Returns (results, engine_used).
    """
    t1 = 1.0 if is_fallback else 2.0
    t2 = 1.0 if is_fallback else 2.0
    t_ddg = 3.0 if is_fallback else 6.0
    searxng_task = asyncio.ensure_future(_searxng_search(searxng_q, max_results))
    
    async def _ddg_with_timeout():
        try:
            return await asyncio.wait_for(
                _ddg_search(search_query, site=site, max_results=max_results, allowed=_FANDOM_ALLOW),
                timeout=t_ddg
            )
        except asyncio.TimeoutError:
            return []
            
    ddg_task = None
    tavily_task = None
    
    # Phase 1: wait up to t1 for SearXNG
    done, pending = await asyncio.wait([searxng_task], timeout=t1)
    if searxng_task in done:
        res = _clean(searxng_task.result())
        if res: return res, "SearXNG"
    
    # Phase 2: SearXNG didn't return, launch DDG
    ddg_task = asyncio.ensure_future(_ddg_with_timeout())
    pending = [t for t in (searxng_task, ddg_task) if not t.done()]
    
    if pending:
        done, pending = await asyncio.wait(pending, timeout=t2, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            res = _clean(task.result())
            if res:
                for p in pending: p.cancel()
                return res, "SearXNG" if task == searxng_task else "DuckDuckGo"
            
    # Phase 3: Still nothing, launch Tavily
    tavily_task = asyncio.ensure_future(_tavily_search(searxng_q, max_results=max_results))
    pending = [t for t in (searxng_task, ddg_task, tavily_task) if t and not t.done()]
    
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            res = _clean(task.result())
            if res:
                for p in pending: p.cancel()
                if task == searxng_task: return res, "SearXNG"
                if task == ddg_task: return res, "DuckDuckGo"
                return res, "Tavily"
                
    return [], ""


async def fetch_web_context(
    query: str, num_urls: int = _MAX_URLS, history_context: str = ""
) -> tuple[str, dict]:
    """Rewrite, route, and fetch web context."""
    t_start = time.monotonic()
    
    ctx_hash = hashlib.md5(history_context.encode("utf-8")).hexdigest() if history_context else ""
    cache_key = (query.lower().strip(), ctx_hash, num_urls)
    cached = _cache_get(cache_key)
    if cached is not None:
        _log.debug("Cache hit for %r", query)
        return cached

    t_rewrite_start = time.monotonic()
    
    intent = None
    rewritten = query
    for pattern, pattern_intent in _REGEX_INTENTS:
        if pattern.search(query):
            intent = pattern_intent
            break

    if intent is None:
        rewrite_res, _ = await asyncio.gather(
            _rewrite_query(query, history_context),
            _await_warmup(),
        )
        rewritten = rewrite_res.get("query", query)
        intent = rewrite_res.get("intent", "general")
    else:
        await _await_warmup()

    t_rewrite = int((time.monotonic() - t_rewrite_start) * 1000)

    year = "" if re.search(r"\b(19|20)\d{2}\b", rewritten) else str(datetime.now().year)

    site = None
    suffix = ""
    wiki_entity = False

    if intent == "documentation":
        suffix = "documentation"

    seed: list[dict] = []

    if intent == "wiki":
        wiki_entity = True
        
    if site or suffix:
        parts = [p for p in (rewritten, suffix, year) if p]
        search_query = " ".join(parts)
    else:
        parts = [rewritten, "wiki" if wiki_entity and not site else "", year]
        search_query = " ".join(p for p in parts if p).strip()

    searxng_q = f"site:{site} {search_query}" if site else search_query
    
    t_search_start = time.monotonic()
    
    found, engine_used = await _staggered_search_cascade(searxng_q, search_query, site, 10)

    seen = {r["url"] for r in seed}
    results = seed + [r for r in found if r["url"] not in seen]
    used_fallback = False

    # Only do a general fallback search when site-scoped results are thin.
    if len(results) < max(2, num_urls // 2):
        used_fallback = True
        seen = {r["url"] for r in results}
        
        general, gen_engine = await _staggered_search_cascade(rewritten, rewritten, None, 12, is_fallback=True)
        if general:
            engine_used = f"{gen_engine} (Fallback)" if engine_used else gen_engine
            results += [r for r in general if r["url"] not in seen]

    t_search = int((time.monotonic() - t_search_start) * 1000)

    debug: dict = {
        "site": site or "general",
        "intent": intent,
        "original_query": query,
        "rewritten_query": rewritten,
        "query": search_query,
        "fallback": used_fallback,
        "engine": engine_used,
        "sources": [],
        "t_rewrite": t_rewrite,
        "t_search": t_search,
    }

    if not results:
        return "", debug

    # Lead token anchors ranking to the entity so a site-scoped search can't drift.
    _noise_words = {"wiki", "the", "and", "for", "with", "from", "site"}
    _anchor = next((w.lower() for w in search_query.split() if len(w) > 1 and w.lower() not in _noise_words and not w.isdigit()), "")

    # On entity-wiki path, drop results that don't mention the entity at all.
    if wiki_entity and _anchor and len(_anchor) > 2:
        on_entity = [
            r for r in results
            if _anchor in r["url"].lower() or _anchor in (r.get("snippet") or "").lower()
        ]
        if on_entity:
            debug["entity_dropped"] = len(results) - len(on_entity)
            results = on_entity
        # Filter seeds strictly by anchor-in-URL.
        seed = [r for r in seed if _anchor in r["url"].lower()]

    def _is_junk(url: str) -> bool:
        u = url.lower()
        return any(ns in u for ns in
                   ("category:", "talk:", "file:", "special:", "user:", "/category"))

    def _priority(r: dict) -> int:
        url = r["url"].lower()
        # Demote wiki meta/user/namespace pages — noisy vs the main article.
        if _is_junk(url):
            return 3
        # Demote pages whose path doesn't mention the entity (off-topic drift).
        if _anchor and _anchor not in url:
            return 2
        if "wiki" in url or ".org" in url:
            return 0
        return 1

    # Relevance gate: require half the query terms repeated >= 1x.
    _noise = {"wiki", "documentation", "reddit", "guide"}
    _terms = [
        w.lower() for w in search_query.split()
        if len(w) > 3 and not w.isdigit() and w.lower() not in _noise
    ]
    _threshold = max(1, len(_terms) // 2)

    def _relevant(content: str) -> bool:
        if not _terms:
            return True
        cl = content.lower()
        # Require the anchor entity itself to avoid generic franchise term matches.
        if _anchor and _anchor not in cl:
            return False
        return sum(t in cl for t in _terms) >= _threshold

    async def _fetch_one(r: dict) -> tuple[dict, str, str]:
        u = r["url"].lower()
        if "youtube.com/watch" in u or "youtu.be/" in u:
            title = (r.get("title") or "").strip()
            snip = (r.get("snippet") or "").strip()
            content = f"Title: {title}\nDescription: {snip}"
            return r, content, "snippet"
        content, method = await fetch_content(r["url"], r["snippet"])
        return r, content, method

    def _accept(r: dict, content: str, method: str) -> None:
        ok = bool(content.strip()) and _relevant(content)
        debug["sources"].append({
            "url": r["url"], "method": method, "chars": len(content),
            "relevant": ok, "score": round(r.get("score", 0.0), 3),
        })
        fetched.append((r, content, ok))
        if ok and len(parts) < num_urls:
            parts.append(f"Source: {r['url']}\n{content}")

    parts: list[str] = []
    fetched: list[tuple[dict, str, bool]] = []

    # Phase 3.6: Eager Pre-fetching (overlaps fetch I/O with embedding rerank)
    t_rerank_start = time.monotonic()
    
    # 1. Eagerly kick off fetches for the heuristic top N
    heuristic_sorted = sorted(results, key=_priority)
    eager_batch = heuristic_sorted[:num_urls + 2]
    fetch_tasks = {r["url"]: asyncio.ensure_future(_fetch_one(r)) for r in eager_batch}

    # 2. Semantic rerank (runs concurrently with eager fetches!)
    if await _rerank(rewritten, results):
        debug["rerank"] = "embed"
        results.sort(key=lambda r: (_is_junk(r["url"]), -r.get("score", 0.0)))
    else:
        results.sort(key=_priority)
    debug["t_rerank"] = int((time.monotonic() - t_rerank_start) * 1000)

    t_wave1_wait = 0.0
    t_wave2_wait = 0.0
    
    # 3. Time-Boxed Hybrid Fetch Collection
    # Ensure all top semantic results have a running fetch task
    for r in results[:num_urls + 2]:
        if r["url"] not in fetch_tasks:
            fetch_tasks[r["url"]] = asyncio.ensure_future(_fetch_one(r))

    pending = set(fetch_tasks.values())
    completed_fetches = {}
    
    start_time = time.monotonic()
    budget = 5.0  # Max wait time for fetches
    
    while pending:
        elapsed = time.monotonic() - start_time
        remaining = budget - elapsed
        if remaining <= 0:
            break
            
        done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
            
        for task in done:
            try:
                res_r, content, method = task.result()
                completed_fetches[res_r["url"]] = (res_r, content, method)
            except Exception as e:
                _log.debug("Fetch task failed: %s", e)
                
        # Early exit: do we have enough acceptable results?
        accepted_count = sum(
            1 for u, (res_r, content, method) in completed_fetches.items()
            if bool(content.strip()) and _relevant(content)
        )
        if accepted_count >= num_urls:
            break

    # Cancel any remaining unused tasks to free I/O
    for task in pending:
        task.cancel()

    t_wave1_wait = time.monotonic() - start_time
    
    # Process the completed fetches in strict semantic order
    for r in results:
        if len(parts) >= num_urls:
            break
        url = r["url"]
        if url in completed_fetches:
            res_r, content, method = completed_fetches[url]
            _accept(res_r, content, method)

    debug["t_fetch_wave1"] = int(t_wave1_wait * 1000)
    debug["t_fetch_wave2"] = 0

    # Graceful degradation: fall back through looser tiers if strict gate rejected everything.
    if not parts:
        got = {r["url"]: content for r, content, _ok in fetched}
        # Tier 2: any content mentioning the anchor (relaxed threshold), or highly relevant semantically.
        for r in results[:num_urls + 2]:  # semantic order
            content = got.get(r["url"], "")
            if content.strip() and (not _anchor or _anchor in content.lower() or r.get("score", 0.0) > 0.4):
                parts.append(f"Source: {r['url']}\n{content}")
                if len(parts) >= num_urls:
                    break
        if parts:
            debug["degraded"] = "relaxed"
    if not parts:
        # Tier 3: DDG snippets as last resort (>= 40 chars).
        snips = [(r, (r.get("snippet") or "").strip()) for r in results[:num_urls + 2]]
        snips = [(r, s) for r, s in snips if len(s) >= 40]
        on_entity = [(r, s) for r, s in snips if not _anchor or _anchor in s.lower() or r.get("score", 0.0) > 0.4]
        for r, s in (on_entity or snips):
            parts.append(f"Source: {r['url']}\n{s}")
            if len(parts) >= num_urls:
                break
        if parts:
            debug["degraded"] = "snippet"
            
    if not parts:
        # Tier 4: literally any snippet or title we have. Guaranteed context if search returned *anything*.
        for r in results:
            s = (r.get("snippet") or r.get("title") or "").strip()
            if s:
                parts.append(f"Source: {r['url']}\n{s}")
                if len(parts) >= num_urls:
                    break
        if parts:
            debug["degraded"] = "any_snippet"

    if not parts:
        return "", debug

    # Enforce a hard character limit to prevent blowing out the model's context window.
    max_total_chars = 25000
    truncated_parts = []
    current_length = 0
    
    for p in parts:
        remaining = max_total_chars - current_length
        if remaining <= 0:
            break
        if len(p) > remaining:
            truncated_parts.append(p[:remaining] + "\n... [truncated to fit context window]")
            current_length += remaining
        else:
            truncated_parts.append(p)
            current_length += len(p)

    ctx = (
        "=== Web Search Results ===\n\n"
        + "\n\n---\n\n".join(truncated_parts)
        + "\n\n=== End of Web Results ==="
    )
    
    debug["t_total"] = int((time.monotonic() - t_start) * 1000)
    _log.info(
        "Search stats for %r: rewrite=%dms, search=%dms, rerank=%dms, wave1=%dms, wave2=%dms, total=%dms",
        query, debug.get("t_rewrite", 0), debug.get("t_search", 0), debug.get("t_rerank", 0),
        debug.get("t_fetch_wave1", 0), debug.get("t_fetch_wave2", 0), debug.get("t_total", 0)
    )
    
    result = (ctx, debug)
    _cache_set(cache_key, result)
    return result


def inject_web_context(messages: list[dict], web_ctx: str) -> None:
    """Append the web context + citation instructions to the system message for prompt caching."""
    if not web_ctx:
        return
    
    suffix = (
        f"\n\n{web_ctx}\n\n"
        "Use the search results above to answer accurately. "
        "Cite specific claims inline by enclosing the URL in angle brackets, exactly like this: (Source: <https://...>). "
        "If sources conflict, note the disagreement. "
        "Do not fabricate information not found in the results.\n"
        "CRITICAL INSTRUCTION: You MUST reply in the exact same language as the user's latest query, even if the search results are in a different language."
    )
    
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += suffix
    else:
        messages.insert(0, {"role": "system", "content": suffix.lstrip()})
