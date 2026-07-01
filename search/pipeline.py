import asyncio
import json
import math
import re
import time
import logging
import hashlib
import concurrent.futures
from datetime import datetime

from ddgs import DDGS
from config import load_config
from llm import race_models, get_client

from urllib.parse import urlparse, parse_qs


from .fetcher import fetch_content, skip, warmup_browser, shutdown_browser

_log = logging.getLogger(__name__)

_CACHE_TTL = 300

_cfg = load_config()

# DDG warmup guard: first query ratelimits without a warm session.
_warmup_started = False
_warmup_done = asyncio.Event()
_MAX_URLS = _cfg.get("defaults", {}).get("max_search_urls", 5)
_DDG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32)

_cache: dict[tuple, tuple] = {}

_FANDOM_ALLOW = frozenset({"fandom.com"})

_REGEX_INTENTS = [
    (re.compile(r'\b(youtube|trailer|soundtrack|gameplay|video|music|listen)\b', re.I), "media"),
    (re.compile(r'\b(traceback|pip install|error|exception|npm install|docs|documentation|api reference)\b', re.I), "documentation"),
    (re.compile(r'\b(reddit|best|vs|versus|should i|review|opinions?|recommendations?)\b', re.I), "opinion"),
    (re.compile(r'\b(meaning of|define|definition|synonym|translate|what does .* mean)\b', re.I), "dictionary"),
]


def _cache_get(key: tuple) -> "tuple[str, dict] | None":
    entry = _cache.get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.monotonic() - ts < _CACHE_TTL:
        return value
    del _cache[key]
    return None


def _cache_set(key: tuple, value: "tuple[str, dict]") -> None:
    now = time.monotonic()
    expired = [k for k, (v, ts) in _cache.items() if now - ts >= _CACHE_TTL]
    for k in expired:
        del _cache[k]
        
    if len(_cache) >= 200:
        oldest = min(_cache.keys(), key=lambda k: _cache[k][1])
        del _cache[oldest]
        
    _cache[key] = (value, now)


import os
import httpx

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


async def warmup() -> None:
    """Prime the DDG session at boot to avoid first-query ratelimit."""
    global _warmup_started
    _warmup_started = True
    try:
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


async def _rewrite_query(raw: str, context: str = "") -> dict:
    """LLM-rewrite to standalone search query with intent."""
    now = datetime.now()
    current_month = now.strftime("%B")
    current_year = now.year
    system_prompt = (
        f"The current date is {current_month} {current_year}. "
        "Analyze the user's message and generate a standalone web search query. "
        "Sentence case, resolve pronouns, keep proper nouns. 4-10 words. "
        "Do NOT add past years to the query unless explicitly requested. "
        "Also determine the optimal search intent.\n"
        "Output a JSON object with EXACTLY two keys:\n"
        '- "query": the rewritten search query string.\n'
        '- "intent": one of "wiki" (facts/entities), "media" (youtube/music/video), "documentation" (code/errors), "opinion" (reviews/reddit), "dictionary" (definitions/translations), or "general".'
    )
    
    if context:
        user_content = (
            f"Conversation so far:\n{context}\n\nLatest message: {raw}\nSearch query:"
        )
    else:
        user_content = raw

    async def _fetch(client, model) -> dict | None:
        try:
            _log.info("Query rewrite started using model %s", model)
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=60,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw_output = (resp.choices[0].message.content or "").strip()
            _log.info("[Background] %s query rewrite completed: %r", model, raw_output)
            try:
                parsed = json.loads(raw_output)
                return {"query": parsed.get("query", raw), "intent": parsed.get("intent", "general")}
            except json.JSONDecodeError:
                m = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        return {"query": parsed.get("query", raw), "intent": parsed.get("intent", "general")}
                    except json.JSONDecodeError:
                        pass
                return {"query": raw_output, "intent": "general"}
        except Exception as e:
            _log.warning("Query rewrite inner error for %s: %s", model, e)
            return None

    cfg = load_config()
    nim_model = cfg.get("rewrite_model_nim")
    ollama_model = cfg.get("rewrite_model_ollama")
    
    nim_task = asyncio.create_task(_fetch(get_client("nim"), nim_model)) if nim_model else None
    ollama_task = asyncio.create_task(_fetch(get_client("ollama"), ollama_model)) if ollama_model else None

    res = await race_models(
        ollama_task, nim_task, 
        timeout=5.0, logger=_log, task_name="query rewrite",
        primary_name="Ollama", backup_name="NIM"
    )
    if res:
        return res
    return {"query": raw, "intent": "general"}


_EMBED_TIMEOUT = 6.9


async def _embed(texts: list[str], input_type: str) -> "list[list[float]] | None":
    """Embed texts via Ollama or NIM."""
    if not texts:
        return []

    async def _fetch_embed(provider: str, model: str) -> list[list[float]] | None:
        try:
            client = get_client(provider)
            kwargs = {"model": model, "input": texts}
            if provider == "nim":
                kwargs["extra_body"] = {"input_type": input_type, "truncate": "END"}
            resp = await client.embeddings.create(**kwargs)
            return [d.embedding for d in resp.data]
        except Exception as e:
            _log.warning("%s embedding failed: %s", provider, e)
            return None

    cfg = load_config()
    nim_model = cfg.get("embed_model_nim")
    ollama_model = cfg.get("embed_model_ollama")

    nim_task = asyncio.create_task(_fetch_embed("nim", nim_model)) if nim_model else None
    ollama_task = asyncio.create_task(_fetch_embed("ollama", ollama_model)) if ollama_model else None

    if not nim_task and not ollama_task:
        return None

    res = await race_models(
        ollama_task, nim_task,
        timeout=5.0, logger=_log, task_name="semantic rerank",
        primary_name="Ollama", backup_name="NIM"
    )
    return res


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _rerank(query: str, results: list[dict]) -> bool:
    """Attach semantic score; return success."""
    if len(results) < 2:
        return False
    snippets = [
        ((r.get("snippet") or r.get("title") or r["url"])[:512]) for r in results
    ]
    qv, pv = await asyncio.gather(
        _embed([query], "query"), _embed(snippets, "passage")
    )
    if not qv or not pv or len(pv) != len(results):
        return False
    q = qv[0]
    for r, v in zip(results, pv):
        r["score"] = _cosine(q, v)
    return True


async def _staggered_search_cascade(
    searxng_q: str, search_query: str, site: str | None, max_results: int, is_fallback: bool = False
) -> tuple[list[dict], str]:
    """
    Run SearXNG -> DDG -> Tavily in a staggered race.
    Budget is halved if is_fallback=True.
    Returns (results, engine_used).
    """
    t1 = 1.5 if is_fallback else 3.0
    t2 = 1.5 if is_fallback else 3.0
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
        res = searxng_task.result()
        if res: return res, "SearXNG"
    
    # Phase 2: SearXNG didn't return, launch DDG
    ddg_task = asyncio.ensure_future(_ddg_with_timeout())
    pending = [t for t in (searxng_task, ddg_task) if not t.done()]
    
    if pending:
        done, pending = await asyncio.wait(pending, timeout=t2, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            res = task.result()
            if res:
                for p in pending: p.cancel()
                return res, "SearXNG" if task == searxng_task else "DuckDuckGo"
            
    # Phase 3: Still nothing, launch Tavily
    tavily_task = asyncio.ensure_future(_tavily_search(searxng_q, max_results=max_results))
    pending = [t for t in (searxng_task, ddg_task, tavily_task) if t and not t.done()]
    
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            res = task.result()
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

    if intent == "media":
        suffix = "video"
    elif intent == "documentation":
        suffix = "documentation"

    seed: list[dict] = []
    if suffix == "video":
        result = await _fetch_media(rewritten, query, num_urls)
        if result[0]:
            _cache_set(cache_key, result)
        return result
        
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
    
    # 3. Collect fetched content strictly in semantic order to guarantee highest quality wins!
    # (Avoids the 'as_completed' race where 0ms snippets starve 1000ms full-text scrapes)
    for r in results:
        if len(parts) >= num_urls:
            break
            
        url = r["url"]
        is_wave2 = False
        if url not in fetch_tasks:
            # Diamond in the rough found by semantic rerank; fetch it now.
            fetch_tasks[url] = asyncio.ensure_future(_fetch_one(r))
            is_wave2 = True
            
        try:
            w_start = time.monotonic()
            res_r, content, method = await fetch_tasks[url]
            w_elapsed = time.monotonic() - w_start
            
            if is_wave2:
                t_wave2_wait += w_elapsed
            else:
                t_wave1_wait += w_elapsed
                
            _accept(res_r, content, method)
        except Exception as e:
            _log.debug("Fetch failed for %s: %s", url, e)

    # Cancel any remaining unused tasks to free I/O
    for t in fetch_tasks.values():
        if not t.done():
            t.cancel()

    debug["t_fetch_wave1"] = int(t_wave1_wait * 1000)
    debug["t_fetch_wave2"] = int(t_wave2_wait * 1000)

    # Graceful degradation: fall back through looser tiers if strict gate
    # rejected everything.
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
    # We allocate 25,000 characters total across all parts.
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
    """Prepend the web context + citation instructions to the last user message."""
    if not web_ctx:
        return
    prefix = (
        f"{web_ctx}\n\n"
        "Use the search results above to answer accurately. "
        "Cite specific claims inline by enclosing the URL in angle brackets, exactly like this: (Source: <https://...>). "
        "If sources conflict, note the disagreement. "
        "Do not fabricate information not found in the results.\n"
        "CRITICAL INSTRUCTION: You MUST reply in the exact same language as the user's latest query below, even if the search results are in a different language.\n\n"
    )
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = prefix + last["content"]
    else:
        for part in last["content"]:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = prefix + part["text"]
                break
