# Web Search Pipeline

Free-tier and self-hosted web search and content retrieval for the chatbot. Given a user query, it discovers relevant sources, fetches their content, ranks it semantically, and returns a context block that gets injected into the LLM prompt.

No paid APIs. Discovery uses self-hosted SearXNG and DuckDuckGo (via `ddgs`) plus open MediaWiki APIs; content is fetched via Jina Reader, direct extraction (trafilatura), MediaWiki APIs, and Playwright; ranking uses local or NIM embeddings.

## Module Layout

| File | Responsibility |
|------|----------------|
| `pipeline.py` | Orchestration: rewrite → route → staggered search cascade → rerank → time-boxed fetch → assemble |
| `engines.py` | Search engine integrations: SearXNG (primary), DuckDuckGo (secondary), Tavily (tertiary). |
| `fetcher.py` | Content retrieval (Jina, trafilatura, MediaWiki, Playwright) + skip rules + host-fetcher caching. |
| `llm_processing.py` | LLM-based query rewriting, intent classification, and semantic reranking using embeddings. |
| `media.py` | YouTube media search and link extraction. |
| `cache.py` | Simple in-memory caching layer for search results. |

Entry point: `fetch_web_context(query, num_urls, history_context) -> (context, debug)`.

## End-to-End Flow

```text
query + history
   │
   ▼
[1] Cache Check ──hit──► return cached (context, debug)
   │ miss
   ▼
[2] Intent Classification & Query Rewrite (Regex Fast-Path / LLM)
   │  If regex matches (e.g., "opinion", "youtube", "error"), instant route!
   │  Else, race Ollama vs NIM to rewrite query & detect intent (5s timeout).
   ▼
[3] Routing
   ├─ intent == "media" ──► YouTube links only (no scraping) ──► return
   ├─ intent (reddit/cambridge/docs) ──► build site-scoped query
   ├─ intent == "wiki" (entity) ──► Wiki discovery (MediaWiki/Fandom APIs)
   └─ else ──► general query
   │
   ▼
[4] Staggered Search Cascade (SearXNG → DuckDuckGo → Tavily)
   │  SearXNG is queried first. If it takes > 2.0s (1.0s for fallbacks), DDG is
   │  launched concurrently. If both fail, Tavily is called.
   │  If site-scoped results are < num_urls/2, a general fallback cascade runs.
   ▼
[5] Merge Results
   │  Merge API-discovered seed URLs + Search Cascade results, deduped.
   ▼
[6] Semantic Reranking
   │  Query and result snippets are embedded using Ollama vs NIM embeddings.
   │  Results are sorted by cosine similarity; junk pages are demoted.
   ▼
[7] Time-Boxed Hybrid Fetch Collection
   │  The top N+2 results are fetched concurrently.
   │  The loop waits up to 5.0 seconds (budget) to collect fetches.
   │  It breaks early if `num_urls` acceptable results are successfully loaded.
   │  Fast responders are then processed strictly in semantic order!
   ▼
[8] Tiered Acceptance & Formatting
   │  Filters fetched content based on strict/relaxed heuristic rules (anchor filtering).
   │  Truncates total context to 25,000 characters to prevent context blowout.
   ▼
[9] Assemble & Return
```

## Stage Details

### [1] Caching
In-memory, 5-minute TTL, keyed on `(query, history_hash, num_urls)`. The entire `(context, debug)` tuple is cached to skip downstream work.

### [2] Query Rewrite & Intent (`llm_processing.py`)
A fast-path checks for basic intents using Regex (e.g., matching the word "opinion", "youtube", "traceback"). If matched, the query avoids the LLM entirely (0ms latency). 

If no match, a race between Ollama and NIM models determines the standalone search query and intent, resolving pronouns from conversation history (e.g., "his voicelines" → "Pantheon voicelines"). The current year is appended for freshness.

Keyword rules map the query to an intent:
- **media**: YouTube (links only)
- **opinion**: reddit.com (site-scoped)
- **dictionary**: dictionary.cambridge.org (site-scoped)
- **documentation**: adds `documentation` suffix
- **wiki**: Triggers specialized entity discovery
- **general**: Normal search

### [3] Wiki Discovery
For wiki intent, standard search engines often drop key subpages. The pipeline performs specialized discovery:
1. Searches the Fandom MediaWiki API in parallel (since Fandom HTML is skip-listed from scraping, but its API is open).
2. Probes for official wikis and uses their MediaWiki API to fetch exact pages.
These discovered URLs are used as **seed URLs** and injected directly into the fetch wave.

### [4] Staggered Search Cascade (`engines.py` & `pipeline.py`)
Instead of blasting all engines simultaneously (which causes rate limits) or waiting for them sequentially (which causes UI freezing):
- **SearXNG** is the primary engine (self-hosted).
- **DuckDuckGo** starts if SearXNG doesn't respond within 2.0s (1.0s for fallbacks).
- **Tavily** starts if both fail to return results.

### [5] Semantic Reranking (`llm_processing.py`)
Results are reranked using semantic embeddings. Ollama and NIM embedding models are raced concurrently. This ensures results conceptually similar to the query bubble up, even if keywords mismatch (e.g. "voicelines" ≈ a page titled "Audio"). Junk meta pages (Category:, Talk:) are demoted.

### [6] Time-Boxed Fetching (`fetcher.py` & `pipeline.py`)
To prevent a single slow website from freezing the chatbot, `pipeline.py` uses a **Time-Boxed Hybrid Loop**. 
- It kicks off fetches for the top `num_urls + 2` semantic hits concurrently.
- It waits a maximum of **5.0 seconds**.
- If `num_urls` fast websites finish before the timeout, it stops waiting early.
- To guarantee quality, the collected fast websites are ordered by their original semantic rank.

Fetch methods race or fallback gracefully:
- **MediaWiki API**: Used for known wikis.
- **Jina Reader**: Fallback for JS-heavy sites.
- **Trafilatura**: Fast direct HTML extraction.
- **Playwright**: Used specifically for Cloudflare-protected domains like Reddit.

A per-host cache remembers which fetcher succeeded last time, avoiding redundant fallback attempts and saving significant latency.

### [7] Tiered Acceptance
To avoid returning an empty context, content goes through layered filters:
1. **Strict**: Content must mention the anchor entity ≥3× and repeat half the query terms.
2. **Relaxed**: Any on-entity full content (anchor present at all).
3. **Snippet**: DDG snippets, preferring those that name the entity.

## Reliability Features
- **Concurrent Provider Racing**: LLM rewrites and embeddings race Ollama vs NIM, falling back seamlessly if one provider is down.
- **Hard Context Limits**: The final text is strictly truncated to 25,000 characters to protect the LLM context window.

## Configuration Defaults

- `defaults.max_search_urls`: 5 sources per query
- Fetch max chars: 20000
- Jina timeout: 5.0s
- Playwright timeout: 10.0s
- Rerank/Rewrite race timeouts: 5.0s
