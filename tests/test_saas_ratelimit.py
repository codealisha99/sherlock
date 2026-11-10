"""Tests for app.saas.ratelimit: sliding window, buckets, bounded memory, reset."""

from __future__ import annotations

import threading

import pytest

from app.saas import ratelimit


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(ratelimit, "_clock", fake)
    monkeypatch.delenv("SHERLOCK_RATE_LIMIT_DISABLED", raising=False)
    ratelimit.reset()
    yield fake
    ratelimit.reset()


def test_allows_up_to_limit_then_blocks(clock):
    results = [ratelimit.check("auth", "ip:1") for _ in range(10)]
    assert results == [None] * 10
    retry = ratelimit.check("auth", "ip:1")
    assert retry is not None and 0 < retry <= 60


def test_retry_after_reflects_oldest_hit(clock):
    for _ in range(10):
        ratelimit.check("auth", "ip:1")
    clock.advance(20)
    retry = ratelimit.check("auth", "ip:1")
    assert retry == pytest.approx(40.0)


def test_window_slides_and_frees_slots(clock):
    for _ in range(10):
        ratelimit.check("auth", "ip:1")
    assert ratelimit.check("auth", "ip:1") is not None
    clock.advance(60.5)
    assert ratelimit.check("auth", "ip:1") is None


def test_keys_are_independent(clock):
    for _ in range(10):
        ratelimit.check("auth", "ip:1")
    assert ratelimit.check("auth", "ip:1") is not None
    assert ratelimit.check("auth", "ip:2") is None


def test_buckets_are_independent(clock):
    for _ in range(10):
        ratelimit.check("auth", "ip:1")
    assert ratelimit.check("auth", "ip:1") is not None
    assert ratelimit.check("api", "ip:1") is None


@pytest.mark.parametrize(
    "bucket, limit",
    [("auth", 10), ("api", 120), ("ingest", 5), ("webhook", 300), ("chat", 20)],
)
def test_documented_bucket_limits(clock, bucket, limit):
    assert [ratelimit.check(bucket, "k") for _ in range(limit)] == [None] * limit
    assert ratelimit.check(bucket, "k") is not None


def test_unknown_bucket_uses_api_limits(clock):
    assert [ratelimit.check("nonsense", "k") for _ in range(120)] == [None] * 120
    assert ratelimit.check("nonsense", "k") is not None
    # and shares the api counter, since unknown buckets resolve to api
    assert ratelimit.check("api", "k") is not None


def test_disabled_by_env(clock, monkeypatch):
    monkeypatch.setenv("SHERLOCK_RATE_LIMIT_DISABLED", "true")
    assert [ratelimit.check("auth", "ip:1") for _ in range(50)] == [None] * 50


def test_reset_clears_state(clock):
    for _ in range(10):
        ratelimit.check("auth", "ip:1")
    assert ratelimit.check("auth", "ip:1") is not None
    ratelimit.reset()
    assert ratelimit.check("auth", "ip:1") is None


def test_total_keys_are_capped(clock, monkeypatch):
    monkeypatch.setattr(ratelimit, "MAX_KEYS", 5)
    for i in range(50):
        ratelimit.check("api", f"ip:{i}")
    assert len(ratelimit._windows) <= 5
    # most recent key survives eviction
    assert ratelimit.check("api", "ip:49") is None
    assert ("api", "ip:49") in ratelimit._windows


def test_idle_keys_are_evicted(clock):
    ratelimit.check("api", "idle")
    clock.advance(120)
    ratelimit.check("api", "fresh")
    assert ("api", "idle") not in ratelimit._windows


def test_set_clock_none_restores_monotonic(monkeypatch):
    monkeypatch.setattr(ratelimit, "_clock", lambda: 0.0)
    ratelimit.set_clock(None)
    assert ratelimit._clock is ratelimit.time.monotonic
    ratelimit.reset()


def test_thread_safety_never_exceeds_limit(clock):
    allowed = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            if ratelimit.check("chat", "shared") is None:
                with lock:
                    allowed.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 20  # chat bucket limit
