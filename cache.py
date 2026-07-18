"""
cache.py — GitOracle In-Memory TTL Cache
=========================================
Wraps GitHub API calls to avoid burning rate-limit quota on repeated queries.

Design:
  - Keyed on (path, frozen_params) tuples
  - TTL configurable via CFG.cache_ttl_seconds (default 60 s)
  - Thread-safe via a simple dict + timestamp check (no external deps)
  - Can be disabled entirely via CFG.cache_enabled = False

Usage (inside mcp_server.py):
    from cache import cached_get
    status, data = cached_get("/repos/facebook/react")
"""

import time
import hashlib
import json
from typing import Any

from config import CFG
from logger import get_logger

log = get_logger("Cache")

# Internal store: key -> (expires_at_unix, status_code, data)
_store: dict[str, tuple[float, int, Any]] = {}


def _make_key(path: str, params: dict | None) -> str:
    """Stable cache key from path + sorted params."""
    canonical = json.dumps({"path": path, "params": params or {}}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def cached_get(
    path: str,
    params: dict | None = None,
    *,
    ttl: int | None = None,
    fetch_fn,        # callable(path, params) -> (status_code, data)
) -> tuple[int, Any]:
    """
    Return (status, data) for the given GitHub path, using a cached value
    if one exists and has not expired.

    Args:
        path:      GitHub API path, e.g. "/repos/facebook/react"
        params:    Optional query parameters dict
        ttl:       Override TTL in seconds (defaults to CFG.cache_ttl_seconds)
        fetch_fn:  The real HTTP getter to call on a cache miss
    """
    if not CFG.cache_enabled:
        return fetch_fn(path, params)

    key = _make_key(path, params)
    now = time.monotonic()
    effective_ttl = ttl if ttl is not None else CFG.cache_ttl_seconds

    # Cache hit?
    if key in _store:
        expires_at, status, data = _store[key]
        if now < expires_at:
            log.cached(path)
            return status, data

    # Cache miss — fetch and store
    status, data = fetch_fn(path, params)
    if status == 200:                           # only cache successful responses
        _store[key] = (now + effective_ttl, status, data)
    return status, data


def invalidate(path: str, params: dict | None = None):
    """Manually evict a single cache entry."""
    key = _make_key(path, params)
    _store.pop(key, None)


def clear_all():
    """Flush the entire cache (useful in tests)."""
    _store.clear()


def stats() -> dict:
    """Return cache statistics."""
    now = time.monotonic()
    live = sum(1 for (exp, _, _) in _store.values() if now < exp)
    return {"total_entries": len(_store), "live_entries": live, "expired_entries": len(_store) - live}
