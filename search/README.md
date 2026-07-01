# Web Search Pipeline

Free-tier web search and content retrieval for the chatbot. Given a user query, it
discovers relevant sources, fetches their content, ranks it semantically, and returns
a context block that gets injected into the LLM prompt.

No paid APIs. Discovery uses DuckDuckGo (via `ddgs`) plus open MediaWiki APIs;
content is fetched via Jina Reader, direct extraction (trafilatura), MediaWiki APIs,
and Playwright; ranking uses NVIDIA NIM embeddings.

## Module layout

| File | Responsibility |
|------|----------------|
| `pipeline.py` | Orchestration: rewrite → route → search → rerank → fetch → assemble |
| `classifier.py` | Intent routing (media / reddit / dictionary / docs) and entity detection |
| `fetcher.py` | Content retrieval (Jina, trafilatura, MediaWiki, Playwright) + skip rules |

Entry point: `fetch_web_context(query, num_urls, history_context) -> (context, debug)`.

## End-to-end flow

```
query + history
   │
   ▼
[1] cache check ──hit──► return cached (context, debug)
   │ miss
   ▼
[2] LLM rewrite (NIM)            resolve "his/it", standalone query, +year
   │
   ▼
[3] classify(rewritten)
   │
   ├─ suffix == "video" ──► [3a] media path: YouTube links only (no scraping) ──► return
   ├─ intent (reddit/cambridge/docs) ──► build site-scoped query
   ├─ entity (proper noun) ──► [4] wiki discovery
   └─ else ──► general query
   │
   ▼
[4] discovery (entity queries)        ──┐
     DDG probe "{query} wiki"           │  parallel
     ├─ discover_wiki_host()            │  (gathered)
     ├─ official wiki MediaWiki API ────┤
     └─ fandom MediaWiki API ───────────┘
   │
   ▼
[5] DDG search (site-scoped)     runs concurrently with [4]'s API calls
   │  └─ general fallback only if coverage thin
   ▼
[6] merge: seed (probe + API hits) + DDG results, deduped
   │
   ▼
[7] embedding rerank (NIM nv-embedqa-e5-v5)
   │  cosine(query, snippet) → semantic order; junk pages demoted
   │  (falls back to keyword priority if embeddings unavailable)
   ▼
[8] fetch top N+2 concurrently
   │  wave 1: seed pages (gathered)   ─ high-value, never starved
   │  wave 2: rest (race, early-break) ─ fast
   ▼
[9] tiered acceptance (never return empty when URLs exist)
   │  strict   → anchor entity ≥3× + half query terms
   │  relaxed  → on-entity content (anchor present)
   │  snippet  → DDG snippets, prefer on-entity
   ▼
[10] assemble context block + cache → return
```

## Stage detail

### [1] Cache
In-memory, 5-minute TTL, keyed on `(query, history[:200], num_urls)`. Whole
`(context, debug)` result is cached.

### [2] Query rewrite
A NIM chat model rewrites the latest message into one standalone search query,
resolving pronouns from the conversation ("his voicelines" → "Pantheon voicelines").
The current year is appended for freshness unless the query already names one.
On failure (timeout/error) it falls back to the raw query.

### [3] Routing — `classify()`
Keyword rules map the query to an intent (first match wins):

| Intent | Target | Output |
|--------|--------|--------|
| media (trailer, MV, soundtrack, "youtube", "cinematic", …) | YouTube | links only |
| opinion (best, vs, "should I", VI equivalents) | reddit.com | site-scoped |
| dictionary (synonym, "meaning of", VI equivalents) | dictionary.cambridge.org | site-scoped |
| code (error:, traceback, pip install, …) | — | `documentation` suffix |
| entity (contains a proper noun) | runtime-discovered wiki | see [4] |
| none of the above | — | general search |

Patterns cover English and Vietnamese. `"what is X"` style queries skip the
dictionary route and go through entity detection instead.

#### [3a] Media path
YouTube is normally skip-listed (can't scrape). For media intent we instead query
`site:youtube.com` and return the **video links only** — no content fetch, no
relevance gate. Results are filtered to real watch URLs and deduped by video id.

### [4] Wiki discovery (entity queries)
1. A DDG probe (`"{query} wiki"`) finds candidate wiki hosts; `discover_wiki_host()`
   scores them by term-in-URL overlap + article-path bonus.
2. The discovered host's **MediaWiki search API** is queried for exact pages —
   deterministic, unlike DDG ranking. (Wikipedia + most wikis; Cloudflare-blocked
   wikis return nothing and fall back to DDG.)
3. **Fandom** is also queried via its MediaWiki API in parallel — it is skip-listed
   for scraping, but its API is open, covering the game/lore long tail.

The probe results on the discovered host are reused as **seed URLs** because DDG's
site-scoped search routinely drops key subpages (e.g. `.../Pantheon/Audio`).

Only a successfully discovered host is cached (per lead token), so a transient probe
failure retries next time instead of poisoning the cache.

### [5]/[6] DDG search + merge
The site-scoped DDG search runs concurrently with the API calls in [4]. Results are
merged: `seed (probe + API) + DDG`, deduplicated by URL. A second general (un-scoped)
search runs only when coverage is thin (`< num_urls/2`).

### [7] Embedding rerank
Query and result snippets are embedded with `nvidia/nv-embedqa-e5-v5`
(`input_type` query/passage; two calls, gathered). Results are ordered by cosine
similarity — this catches vocabulary mismatch that keyword ranking misses
("voicelines" ≈ a page titled "Audio"). Junk meta pages (Category:/Talk:/…) stay
demoted. If embeddings are unavailable, it falls back to a keyword priority heuristic.

### [8] Fetch — `fetch_content()`
The top `num_urls + 2` are fetched concurrently in two waves:
- **Wave 1** — seed pages, fully gathered. The highest-value page is often the
  largest (a transcript/`Audio` page) and thus slowest; this guarantees it isn't
  cancelled by completion-order racing.
- **Wave 2** — the rest, raced with early-break once enough relevant results are in.

Per URL, `fetch_content` prefers the host's known-good fetcher (learned cache) and
only races the others on a miss:
- **MediaWiki API** (`extracts`) for wikis with an open API
- **Jina Reader** (`r.jina.ai`) for JS-heavy pages — retries once on transient failure
- **trafilatura** for direct HTML extraction
- **Playwright** fallback for Cloudflare-gated hosts (e.g. reddit)
- **snippet** as last resort

Content is truncated at `_MAX_CHARS` (12345). A per-host cache records which fetcher
worked, cutting ~⅔ of fetch requests after warmup.

### [9] Tiered acceptance
To avoid returning an empty context when sources were found:
1. **Strict** — content must mention the anchor entity ≥3× and repeat half the query
   terms (rejects wrong-but-same-franchise pages).
2. **Relaxed** — any on-entity full content (anchor present at all).
3. **Snippet** — DDG snippets, preferring those that name the entity.

Empty is only possible when discovery returns zero URLs.

### [10] Injection — `inject_web_context()`
The assembled block is prepended to the last user message with instructions to cite
sources inline and not fabricate beyond the results.

## Reliability layers

`rewrite retry → DDG retry (3, backoff) → multi-source seeding (probe + official API
+ fandom API + DDG) → embedding rerank → keyword gate → tiered fallback → per-host
fetcher cache`.

- **Startup Warmup**: On server boot, a shared persistent `DDGS` session is initialized and warmed up to bypass initial DuckDuckGo anti-bot rate limits. Concurrently, a dummy query is sent to SearXNG to force its internal docker workers to spin up and resolve DNS, avoiding massive cold-start penalties on the first user query.
- **Strict Engine Timeouts**: Local `SearXNG` searches are wrapped in a strict `5.0s` timeout guard. If the SearXNG container hangs or takes too long to aggregate results, it aborts instantly and falls back to DuckDuckGo, guaranteeing the UI stream never freezes.

## Configuration

| Constant | File | Value | Meaning |
|----------|------|-------|---------|
| `_REWRITE_MODEL` | pipeline.py | `qwen/qwen3-next-80b-a3b-instruct` | query rewriter |
| `_EMBED_MODEL` | pipeline.py | `nvidia/nv-embedqa-e5-v5` | rerank embeddings |
| `_CACHE_TTL` | pipeline.py | 300 | result cache seconds |
| `_DDG_BACKENDS` | pipeline.py | duckduckgo, google, bing, brave, startpage | pinned (skips flaky mojeek) |
| `_MAX_URLS` | pipeline.py | config `defaults.max_search_urls` (5) | sources per query |
| `_MAX_CHARS` | fetcher.py | 12345 | chars per source |
| `_JINA_TIMEOUT` | fetcher.py | 10 | Jina request timeout |

## Known limits

- **Discovery ceiling** — finding *which* wiki/site exists for an entity relies on
  DuckDuckGo, which is nondeterministic and rate-limits. This is the dominant source
  of latency and the rare wrong-result. There is no free SERP alternative; breaking
  past it requires a paid search API. Everything *within* a discovered host is now
  deterministic (MediaWiki APIs).
- **Latency** is external-bound: ~4-6s healthy, higher when DDG backends are slow.
- **Cache is in-memory** — cleared on restart.
