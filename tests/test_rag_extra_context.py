"""RAG pipeline: caller-supplied extra_context and provider error markers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from app.rag.pipeline import RAGPipeline
from tests.test_rag_streaming import _FakeIndexer, _make_chunk


class _FakeGateway:
    """Stands in for LLMGateway.generate_stream: records the request, yields scripted chunks."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.requests: List[Any] = []

    async def generate_stream(self, request):
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk


def _marker(payload: Dict[str, Any]) -> str:
    return f"__error__:{json.dumps(payload)}__\n"


def _collect(pipeline: RAGPipeline, extra_context=None) -> List[Dict[str, Any]]:
    async def run():
        return [
            e async for e in pipeline.ask_stream(
                repo_id="repo-test", question="How does add work?", extra_context=extra_context,
            )
        ]

    return asyncio.run(run())


def _pipeline(gateway: _FakeGateway) -> RAGPipeline:
    indexer = _FakeIndexer(
        results=[
            _make_chunk("c1", "def add(a, b): return a + b", file_path="src/calculator.py", start_line=1, end_line=2),
            _make_chunk("c2", "def multiply(a, b): return a * b", file_path="src/operations.py", start_line=5, end_line=6),
        ]
    )
    return RAGPipeline(indexer=indexer, gateway=gateway)


def _focus_chunk() -> Dict[str, Any]:
    return {
        "id": "focus:node:sym:src/calculator.py::add",
        "repo_id": "repo-test",
        "file_path": "src/calculator.py",
        "start_line": 1,
        "end_line": 2,
        "content": "def add(a, b): return a + b",
        "chunk_type": "code",
    }


def _user_prompt(gateway: _FakeGateway) -> str:
    return gateway.requests[0].messages[1]["content"]


def test_extra_context_is_first_deduped_and_cited():
    gateway = _FakeGateway(["It adds [S1]."])
    # Same id as retrieved c1 must not appear twice.
    dup = {**_focus_chunk(), "id": "c1"}
    events = _collect(_pipeline(gateway), extra_context=[_focus_chunk(), dup])

    prompt = _user_prompt(gateway)
    # Extra chunks take S1 and S2; retrieved c1 is dropped as a duplicate, so c2 is S3.
    assert prompt.index("[S1] src/calculator.py") < prompt.index("[S2] src/calculator.py")
    assert prompt.index("[S2] src/calculator.py") < prompt.index("[S3] src/operations.py")
    assert "[S4]" not in prompt
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"]


def test_no_extra_context_keeps_retrieval_order():
    gateway = _FakeGateway(["ok"])
    _collect(_pipeline(gateway), extra_context=None)
    prompt = _user_prompt(gateway)
    assert prompt.index("src/calculator.py") < prompt.index("src/operations.py")


def test_provider_auth_marker_becomes_friendly_error():
    gateway = _FakeGateway([_marker({"message": "Provider returned HTTP 401.", "error_category": "provider"})])
    events = _collect(_pipeline(gateway))

    assert events[-1]["type"] == "error"
    assert "The AI provider rejected the API key (401)" in events[-1]["message"]
    assert "check it in Settings" in events[-1]["message"]
    assert not any(e["type"] == "token" for e in events)


def test_other_provider_messages_pass_through_without_slicing_errors():
    gateway = _FakeGateway([_marker({"message": "Invalid API key provided", "error_category": "provider"})])
    events = _collect(_pipeline(gateway))

    assert events[-1]["type"] == "error"
    assert "Invalid API key provided" in events[-1]["message"]
    assert "Extra data" not in events[-1]["message"]
