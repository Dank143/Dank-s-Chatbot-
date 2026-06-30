import re

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
