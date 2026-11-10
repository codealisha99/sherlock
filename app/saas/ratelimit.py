"""Sliding-window rate limiter keyed by (bucket, key), e.g. ("auth", "ip:1.2.3.4").

In-process and thread-safe. Memory is bounded: idle keys are evicted and the
total number of tracked keys is capped (least recently used go first).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Deque, Dict, Optional, Tuple

# bucket name -> (max requests, window seconds)
BUCKETS: Dict[str, Tuple[int, float]] = {
    "auth": (10, 60.0),
    "api": (120, 60.0),
    "ingest": (5, 60.0),
    "webhook": (300, 60.0),
    "chat": (20, 60.0),
}
DEFAULT_BUCKET = "api"
MAX_KEYS = 50_000
# ponytail: single global lock; shard per bucket if contention shows up under load.
_lock = threading.Lock()
_windows: "OrderedDict[Tuple[str, str], Deque[float]]" = OrderedDict()
_clock: Callable[[], float] = time.monotonic


def _disabled() -> bool:
    return os.getenv("SHERLOCK_RATE_LIMIT_DISABLED", "").lower() == "true"


def _resolve_bucket(bucket: str) -> Tuple[str, int, float]:
    name = bucket if bucket in BUCKETS else DEFAULT_BUCKET
    limit, window = BUCKETS[name]
    return name, limit, window


def _evict(now: float) -> None:
    """Drop idle keys from the LRU front, then enforce the global cap."""
    for _ in range(16):  # bounded work per call
        if not _windows:
            return
        (bucket, _key), stamps = next(iter(_windows.items()))
        window = BUCKETS[bucket][1]
        if stamps and stamps[-1] > now - window:
            break
        _windows.popitem(last=False)
    while len(_windows) > MAX_KEYS:
        _windows.popitem(last=False)


def check(bucket: str, key: str) -> Optional[float]:
    """Record a hit. Returns None if allowed, else seconds until a slot frees up."""
    if _disabled():
        return None
    name, limit, window = _resolve_bucket(bucket)
    now = _clock()
    with _lock:
        slot = (name, key)
        stamps = _windows.get(slot)
        if stamps is None:
            stamps = deque()
            _windows[slot] = stamps
        else:
            _windows.move_to_end(slot)
        while stamps and stamps[0] <= now - window:
            stamps.popleft()
        retry: Optional[float] = None
        if len(stamps) < limit:
            stamps.append(now)
        else:
            retry = max(stamps[0] + window - now, 0.001)
        _evict(now)  # after the hit is recorded, so the current key is never idle
        return retry


def reset() -> None:
    """Forget all hits (tests and admin tooling)."""
    with _lock:
        _windows.clear()


def set_clock(clock: Optional[Callable[[], float]]) -> None:
    """Inject a clock for tests; None restores time.monotonic."""
    global _clock
    _clock = clock or time.monotonic
