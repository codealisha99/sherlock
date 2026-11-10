"""The shared embedding client must load its model exactly once under concurrency."""

import threading
import time

import app.llm.embedding_client as ec


def test_concurrent_first_use_loads_once(monkeypatch):
    loads = []

    class SlowModel:
        def __init__(self, *_a, **_k):
            loads.append(1)
            time.sleep(0.2)

    monkeypatch.setattr(ec, "SentenceTransformer", SlowModel)
    monkeypatch.setattr(ec, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
    client = ec.EmbeddingClient()
    results, errors = [], []

    def use():
        try:
            results.append(client._ensure_loaded())
        except Exception as exc:  # the old code raised AssertionError here
            errors.append(exc)

    threads = [threading.Thread(target=use) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(loads) == 1
    assert len({id(r) for r in results}) == 1


def test_default_client_is_shared():
    assert ec.get_default_client() is ec.get_default_client()
