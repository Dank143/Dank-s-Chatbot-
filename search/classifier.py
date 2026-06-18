import re

# Intent routing: opinion -> reddit, dictionary -> cambridge, code -> docs.
# Wiki routing handled at runtime in pipeline.
_INTENT_RULES: list[tuple[list[str], str | None, str]] = [
    (
        # Media intent -> YouTube, returned as links only (no scraping).
        # Mostly trailers, music videos, soundtracks. Kept specific to avoid
        # catching explanatory queries ("how video codecs work").
        ["trailer", "music video", "official mv", " mv ", "official video",
         "official audio", "lyric video", "lyrics video", "soundtrack",
         " ost ", "theme song", "listen to", "full song", "watch the",
         "youtube", "cinematic", "video clip", " clip ", "music mv",
         # Vietnamese media cues.
         "nhạc phim", "bài hát", "nghe nhạc", "lời bài hát", "xem trailer",
         "nhạc nền", "ca khúc", "xem video"],
        "youtube.com", "video",
    ),
    (
        ["best ", "vs ", " versus ", "recommend", "worth it", "should i",
         "which should i", "comparison", "buying advice", "thoughts on", "opinion on",
         # Vietnamese opinion/recommendation cues.
         "tốt nhất", "tốt hơn", "so với", "so sánh", "nên mua", "có nên",
         "đánh giá", "review", "nên chọn"],
        "reddit.com", "",
    ),
    (
        ["synonym", "pronunciation of", "how to pronounce", "ipa ", " grammar ",
         "meaning of", "definition of", "antonym", "how to say",
         # Vietnamese dictionary cues.
         "nghĩa là gì", "nghĩa của", "đồng nghĩa", "trái nghĩa",
         "cách phát âm", "phát âm của", "cách viết"],
        "dictionary.cambridge.org", "",
    ),
    (
        ["error:", "traceback", "exception:", "modulenotfounderror", "typeerror",
         "syntaxerror", "valueerror", "importerror", "api reference",
         "pip install", "npm install", "how to use library"],
        None, "documentation",
    ),
]

_DEFAULT_SITE = None

# Filler + follow-up pronouns dropped from queries ("Pantheon — his quotes?" -> "Pantheon quotes").
_STOPWORDS = {
    "in", "for", "the", "on", "about", "from", "of", "a", "an",
    "his", "her", "its", "their", "what", "whats", "is", "are", "and",
}


def _strip_keywords(query: str, keywords: list[str]) -> str:
    out = query
    for kw in keywords:
        out = re.sub(r'\b' + re.escape(kw) + r'\b', " ", out, flags=re.IGNORECASE)
    # Drop separators/punctuation that pollute the search (em/en dash, :, ?, !);
    # keep apostrophes and hyphens that occur inside names (Bel'Veth, Rek'Sai).
    out = re.sub(r"[—–:?!]", " ", out)
    words = [w for w in out.split() if w.lower() not in _STOPWORDS]
    return " ".join(words)


def clean_query(text: str) -> str:
    """Strip filler/punctuation from already entity-resolved terms (site-scoped search)."""
    return _strip_keywords(text, [])


# Matches capitalized names incl. mixed-case (Pantheon, LoL, PvP).
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][A-Za-z]+\b')

# "what is X" / "what does X" queries — route via proper-noun check, not Cambridge
_DEFINITIONAL_PREFIXES = ("what is ", "what are ", "what does ", "define ")


def _has_proper_nouns(query: str) -> bool:
    return bool(_PROPER_NOUN_RE.search(query))


def wants_wiki(query: str) -> bool:
    """True if the query names a proper-noun entity -> route to a wiki."""
    return _has_proper_nouns(query)


def classify(query: str) -> tuple[str | None, str]:
    """Intent route -> (site, suffix); (None, "") means no match (caller falls back)."""
    q = query.lower()
    is_definitional = any(q.startswith(p) for p in _DEFINITIONAL_PREFIXES)

    for patterns, site, suffix in _INTENT_RULES:
        if site == "dictionary.cambridge.org" and is_definitional:
            continue
        if any(p in q for p in patterns):
            return site, suffix

    return _DEFAULT_SITE, ""
