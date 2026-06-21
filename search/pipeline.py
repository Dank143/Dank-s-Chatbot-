import asyncio
import math
import re
import time
import logging
from datetime import datetime

from openai import AsyncOpenAI
from ddgs import DDGS
from config import load_config, provider_api

from urllib.parse import urlparse, parse_qs

from .classifier import classify, clean_query, wants_wiki
from .fetcher import fetch_content, mediawiki_search, skip

_log = logging.getLogger(__name__)

_CACHE_TTL = 300

_cfg = load_config()
# DDG warmup guard: first query ratelimits without a warm session.
_warmup_started = False
_warmup_done = asyncio.Event()
_REWRITE_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
_MAX_URLS = _cfg.get("defaults", {}).get("max_search_urls", 5)

_rewriter_cache: dict[tuple, AsyncOpenAI] = {}


def _get_rewriter() -> AsyncOpenAI:
    """Rewriter client — always uses NIM (rewrite/embed models are NIM-hosted)."""
    api = provider_api("nim")
    key = (api["key"], api["base_url"])
    if key not in _rewriter_cache:
        _rewriter_cache[key] = AsyncOpenAI(api_key=key[0], base_url=key[1])
    return _rewriter_cache[key]

_cache: dict[tuple, tuple] = {}

# Entity lead-token -> discovered wiki host (or None); learned at runtime.
_wiki_host_cache: dict[str, "str | None"] = {}

# "wiki*" services that are not entity wikis, and foreign-language Wikipedias.
_NONWIKI_HOSTS = ("wikihow", "wikiedu", "wiktionary", "wikitravel", "wikiquote", "wikidata")


def _is_wiki_host(host: str) -> bool:
    h = host.lower()
    if "wiki" not in h:
        return False
    if any(b in h for b in _NONWIKI_HOSTS):
        return False
    # Exclude non-English Wikipedias/Wikimedia (en.* and en.m.* are fine).
    if (h.endswith("wikipedia.org") or h.endswith("wikimedia.org")) and not h.startswith(("en.", "en.m.")):
        return False
    return True


def discover_wiki_host(results: list[dict], terms: list[str]) -> "str | None":
    """Pick the wiki host matching the query via term-in-URL overlap + article-path bonus."""
    best, best_score = None, 0
    for r in results:
        u = r["url"].lower()
        parsed = urlparse(u)
        if not _is_wiki_host(parsed.netloc):
            continue
        score = sum(t in u for t in terms)
        if "/wiki/" in parsed.path or "/w/" in parsed.path:
            score += 1
        if score > best_score:
            best, best_score = parsed.netloc, score
    return best


async def _wiki_host_for(rewritten: str, terms: list[str]) -> "tuple[str | None, list[dict]]":
    """Discover the wiki host for an entity query; return (host, probe_results).
    Probe results are reused as seed URLs. Only successful discoveries are cached.
    """
    key = next((w.lower() for w in rewritten.split() if len(w) > 2), rewritten.lower())
    # Allow fandom in probe for host discovery; cap retries to save budget.
    probe = await _ddg_search(
        f"{rewritten} wiki", site=None, max_results=8,
        allowed=_FANDOM_ALLOW, max_attempts=2,
    )
    if key in _wiki_host_cache:
        return _wiki_host_cache[key], probe
    host = discover_wiki_host(probe, terms)
    if host:
        _wiki_host_cache[key] = host
    return host, probe


_FANDOM_ALLOW = frozenset({"fandom.com"})

# Quote/voiceline queries: append "audio" to surface wiki /Audio subpages.
_VOICE_HINT_RE = re.compile(
    r'\b(quotes?|voice ?lines?|voicelines?|sayings?|dialogue|dialog)\b', re.I
)


def _audio_hint(query: str, source: str) -> str:
    """Append 'audio' to wiki query if user asked for quotes/lines."""
    if "audio" in query.lower() or not _VOICE_HINT_RE.search(source):
        return query
    return f"{query} audio"


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
    _cache[key] = (value, time.monotonic())


async def _ddg_search(
    query: str, site: str | None = None, max_results: int = 4,
    allowed: "frozenset[str]" = frozenset(), max_attempts: int = 3,
) -> list[dict]:
    """DDG text search (off-thread), optionally site-scoped; skips junk domains.
    `allowed` whitelists otherwise-skipped domains. `max_attempts` bounds retries.
    """
    search_query = f"site:{site} {query}" if site else query
    # Retry with backoff on flaky DDG backends. Empty results get at most 1 retry.
    last_exc = False
    empty_seen = 0
    for attempt in range(max_attempts):
        try:
            results = await asyncio.to_thread(
                lambda: list(DDGS(timeout=8).text(search_query, max_results=max_results))
            )
            mapped = [
                {"url": r["href"], "snippet": r.get("body", ""), "title": r.get("title", "")}
                for r in (results or [])
                if r.get("href") and not skip(r["href"], allowed)
            ]
            if mapped:
                return mapped
            empty_seen += 1
            if empty_seen > 1:
                return mapped
        except Exception:
            last_exc = True
        # No backoff after the final attempt — it only delays the return.
        if attempt < max_attempts - 1:
            await asyncio.sleep(0.5 * (attempt + 1))
    if last_exc:
        _log.warning("DDG search failed for %r", search_query, exc_info=True)
    return []


async def warmup() -> None:
    """Prime the DDG session at boot to avoid first-query ratelimit."""
    global _warmup_started
    _warmup_started = True
    try:
        await asyncio.to_thread(lambda: list(DDGS(timeout=8).text("wikipedia", max_results=1)))
        _log.debug("DDG warmup ok")
    except Exception:
        _log.debug("DDG warmup failed", exc_info=True)
    finally:
        _warmup_done.set()


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
    results = await _ddg_search(
        rewritten, site="youtube.com", max_results=num_urls + 4, allowed=_YT_ALLOW,
    )
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


_REWRITE_SYSTEM = (
    "Rewrite the user's latest message into ONE standalone web search query. "
    "Resolve pronouns (e.g. replace 'his/it' with the actual subject). Keep proper nouns. "
    "4-10 words. No years, no 'build/guide/tips' words, no trailing punctuation. "
    "Output ONLY the query."
)


async def _rewrite_query(raw: str, context: str = "") -> str:
    """LLM-rewrite to standalone search query."""
    if context:
        user_content = (
            f"Conversation so far:\n{context}\n\nLatest message: {raw}\nSearch query:"
        )
    else:
        user_content = raw
    try:
        resp = await asyncio.wait_for(
            _get_rewriter().chat.completions.create(
                model=_REWRITE_MODEL,
                max_tokens=30,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            ),
            timeout=7.0,
        )
        return (resp.choices[0].message.content or "").strip() or raw
    except Exception:
        _log.warning("Query rewrite failed, using original", exc_info=True)
        return raw


_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
_EMBED_TIMEOUT = 6.0


async def _embed(texts: list[str], input_type: str) -> "list[list[float]] | None":
    """Embed texts via NIM."""
    if not texts:
        return []
    try:
        resp = await asyncio.wait_for(
            _get_rewriter().embeddings.create(
                model=_EMBED_MODEL,
                input=texts,
                extra_body={"input_type": input_type, "truncate": "END"},
            ),
            timeout=_EMBED_TIMEOUT,
        )
        return [d.embedding for d in resp.data]
    except Exception:
        _log.warning("Embedding failed (%s)", input_type, exc_info=True)
        return None


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


async def fetch_web_context(
    query: str, num_urls: int = _MAX_URLS, history_context: str = ""
) -> tuple[str, dict]:
    """Rewrite, route, and fetch web context."""
    cache_key = (query.lower().strip(), history_context[:200], num_urls)
    cached = _cache_get(cache_key)
    if cached is not None:
        _log.debug("Cache hit for %r", query)
        return cached

    rewritten, _ = await asyncio.gather(
        _rewrite_query(query, history_context),
        _await_warmup(),
    )
    year = "" if re.search(r"\b(19|20)\d{2}\b", rewritten) else str(datetime.now().year)

    site, suffix = classify(rewritten)
    seed: list[dict] = []
    if suffix == "video":
        result = await _fetch_media(rewritten, query, num_urls)
        if result[0]:
            _cache_set(cache_key, result)
        return result
    api_tasks: list = []
    wiki_entity = False
    if site or suffix:
        parts = [p for p in (rewritten, suffix, year) if p]
        search_query = " ".join(parts)
    elif wants_wiki(rewritten):
        terms = [w.lower() for w in rewritten.split() if len(w) > 3]
        host, probe = await _wiki_host_for(rewritten, terms)
        mw_query = clean_query(rewritten)
        api_query = _audio_hint(mw_query, rewritten)
        if host:
            site = host
            wiki_entity = True
            search_query = api_query
            seed = [r for r in probe if (urlparse(r["url"]).netloc or "") == host]
            api_tasks.append(asyncio.ensure_future(
                mediawiki_search(host, api_query, limit=num_urls)))
        else:
            search_query = f"{rewritten} {year}".strip()
        # NOTE: fandom API seed skipped — its content API returns empty extracts
        # and page scraping is Cloudflare-blocked. Recovered via DDG instead.
    else:
        search_query = f"{rewritten} {year}".strip()

    # Allow fandom through filter — _mediawiki_fetch reads it via open API.
    found = await _ddg_search(
        search_query, site=site, max_results=num_urls + 2, allowed=_FANDOM_ALLOW,
    )
    if api_tasks:
        seen_seed = {r["url"] for r in seed}
        for res in await asyncio.gather(*api_tasks):
            for r in res:
                if r["url"] not in seen_seed:
                    seed.append(r)
                    seen_seed.add(r["url"])
    seen = {r["url"] for r in seed}
    results = seed + [r for r in found if r["url"] not in seen]
    used_fallback = False

    # Only do a general fallback search when site-scoped results are thin.
    if len(results) < max(2, num_urls // 2):
        used_fallback = True
        seen = {r["url"] for r in results}
        general = await _ddg_search(
            rewritten, site=None, max_results=num_urls + 2,
            allowed=_FANDOM_ALLOW, max_attempts=2,
        )
        results += [r for r in general if r["url"] not in seen]

    debug: dict = {
        "site": site or "general",
        "original_query": query,
        "rewritten_query": rewritten,
        "query": search_query,
        "fallback": used_fallback,
        "sources": [],
    }

    if not results:
        return "", debug

    # Lead token anchors ranking to the entity so a site-scoped search can't drift.
    _anchor = next((w.lower() for w in search_query.split() if len(w) > 2), "")

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

    # Relevance gate: require half the query terms repeated >=3x.
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
        if _anchor and len(_anchor) > 3 and cl.count(_anchor) < 3:
            return False
        return sum(cl.count(t) >= 3 for t in _terms) >= _threshold

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
    seed_urls = {r["url"] for r in seed}

    # Wave 1: fetch seeds NOW (before rerank) so I/O overlaps the embed call.
    seed_tasks = [asyncio.ensure_future(_fetch_one(r)) for r in seed]

    # Semantic rerank: catches vocabulary mismatch keyword ranking misses.
    # Falls back to keyword priority if embedding unavailable.
    if await _rerank(rewritten, results):
        debug["rerank"] = "embed"
        results.sort(key=lambda r: (_is_junk(r["url"]), -r.get("score", 0.0)))
    else:
        results.sort(key=_priority)
    fetch_batch = results[:num_urls + 2]

    # Collect seed fetches (already in flight).
    if seed_tasks:
        for r, content, method in await asyncio.gather(*seed_tasks):
            _accept(r, content, method)

    # Wave 2: race remaining fetches, taking first relevant results until full.
    wave2 = [r for r in fetch_batch if r["url"] not in seed_urls]
    if len(parts) < num_urls and wave2:
        tasks = [asyncio.ensure_future(_fetch_one(r)) for r in wave2]
        try:
            for fut in asyncio.as_completed(tasks):
                r, content, method = await fut
                _accept(r, content, method)
                if len(parts) >= num_urls:
                    break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    # Graceful degradation: fall back through looser tiers if strict gate
    # rejected everything.
    if not parts:
        got = {r["url"]: content for r, content, _ok in fetched}
        # Tier 2: any content mentioning the anchor (relaxed threshold).
        for r in fetch_batch:  # priority order
            content = got.get(r["url"], "")
            if content.strip() and (not _anchor or _anchor in content.lower()):
                parts.append(f"Source: {r['url']}\n{content}")
                if len(parts) >= num_urls:
                    break
        if parts:
            debug["degraded"] = "relaxed"
    if not parts:
        # Tier 3: DDG snippets as last resort (>= 40 chars).
        snips = [(r, (r.get("snippet") or "").strip()) for r in fetch_batch]
        snips = [(r, s) for r, s in snips if len(s) >= 40]
        on_entity = [(r, s) for r, s in snips if not _anchor or _anchor in s.lower()]
        for r, s in (on_entity or snips):
            parts.append(f"Source: {r['url']}\n{s}")
            if len(parts) >= num_urls:
                break
        if parts:
            debug["degraded"] = "snippet"

    if not parts:
        return "", debug

    ctx = (
        "=== Web Search Results ===\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n\n=== End of Web Results ==="
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
        "Cite specific claims inline as (Source: <url>). "
        "If sources conflict, note the disagreement. "
        "Do not fabricate information not found in the results.\n\n"
    )
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = prefix + last["content"]
    else:
        for part in last["content"]:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = prefix + part["text"]
                break
