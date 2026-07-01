import time

_CACHE_TTL = 300
_cache: dict[tuple, tuple] = {}

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
