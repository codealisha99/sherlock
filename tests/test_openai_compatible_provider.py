"""Tests for the OpenAI-compatible provider (BYOK) and gateway override routing."""

import asyncio
import json

import httpx

from app.llm.gateway import LLMGateway
from app.llm.models import GenerationRequest
from app.llm.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
    ProviderError,
    parse_sse_line,
)

BASE = "https://api.example.com/v1"
KEY = "sk-test-0123456789abcdefABCD"


def _provider(handler, base_url=BASE, **kwargs):
    return OpenAICompatibleProvider(
        name="custom",
        base_url=base_url,
        api_key=KEY,
        model="model-1",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _request(model: str = "") -> GenerationRequest:
    return GenerationRequest(messages=[{"role": "user", "content": "Explain main.py"}], model=model)


def _sse(*lines: str) -> httpx.Response:
    body = "".join(f"{line}\n\n" for line in lines)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _collect(provider, request):
    async def run():
        return [chunk async for chunk in provider.generate_stream(request)]

    return asyncio.run(run())


def _error_payload(chunk: str) -> dict:
    assert chunk.startswith("__error__:") and chunk.endswith("__\n")
    return json.loads(chunk[len("__error__:"):-3])


# ------------------------------------------------------------------ generate
def test_generate_returns_text_and_provider_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "It starts the app."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            },
        )

    result = _provider(handler).generate(_request())

    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert result.error is None
    assert result.text == "It starts the app."
    assert result.usage.input == 42 and result.usage.output == 7
    assert result.finish_reason == "stop"


def test_request_model_is_ignored_in_favour_of_provider_model():
    seen = {}

    def handler(request):
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _provider(handler).generate(_request(model="openai/gpt-oss-120b"))
    assert seen["model"] == "model-1"


def test_generate_estimates_usage_when_provider_omits_it():
    handler = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "one two three."}}]})
    result = _provider(handler).generate(_request())
    assert result.usage.provider_input is None
    assert result.usage.estimated_input > 0 and result.usage.estimated_output > 0


def test_generate_429_maps_to_rate_limit_with_retry_after():
    handler = lambda r: httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
    result = _provider(handler).generate(_request())
    assert result.error_category == "rate_limit"
    assert result.error == "Provider rate limit exceeded"


def test_generate_http_error_is_safe_and_never_echoes_key():
    handler = lambda r: httpx.Response(500, json={"error": f"bad key {KEY}"})
    result = _provider(handler).generate(_request())
    assert result.error_category == "provider"
    assert result.error == "Provider returned HTTP 500."
    assert KEY not in (result.error or "")


def test_generate_timeout_maps_to_timeout_category():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    result = _provider(handler).generate(_request())
    assert result.error_category == "timeout"


def test_generate_malformed_body_is_provider_error():
    handler = lambda r: httpx.Response(200, json=["not", "an", "object"])
    result = _provider(handler).generate(_request())
    assert result.error_category == "provider"
    assert result.error == "Malformed provider response."


# ------------------------------------------------------------------ basics
def test_repr_and_health_never_expose_key():
    provider = _provider(lambda r: httpx.Response(200))
    assert KEY not in repr(provider)
    health = provider.check_health()
    assert health == {"provider": "custom", "model": "model-1", "available": True}
    assert KEY not in json.dumps(health)


def test_base_url_trailing_slash_is_normalised_for_anthropic_compat_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _provider(handler, base_url="https://api.anthropic.com/v1/").generate(_request())
    assert seen["url"] == "https://api.anthropic.com/v1/chat/completions"


def test_constructor_requires_key_base_url_and_model():
    for kwargs in ({"api_key": "", "base_url": BASE, "model": "m"},
                   {"api_key": KEY, "base_url": "", "model": "m"},
                   {"api_key": KEY, "base_url": BASE, "model": ""}):
        try:
            OpenAICompatibleProvider(name="x", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ------------------------------------------------------------------ SSE parsing
def test_parse_sse_line_cases():
    assert parse_sse_line("data: [DONE]") == (True, "")
    assert parse_sse_line(": keep-alive") == (False, "")
    assert parse_sse_line("data: {not json") == (False, "")
    assert parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}]}') == (False, "")
    assert parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}') == (False, "hi")
    try:
        parse_sse_line('data: {"error": {"message": "overloaded"}}')
    except ProviderError as exc:
        assert str(exc) == "overloaded"
    else:
        raise AssertionError("mid-stream error must raise")


def test_stream_yields_ttft_then_text_and_stops_at_done():
    body = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"SHOULD-NOT-APPEAR"}}]}',
    ]
    chunks = _collect(_provider(lambda r: _sse(*body)), _request())

    assert chunks[0].startswith("__ttft__:")
    assert "".join(c for c in chunks if not c.startswith("__")) == "Hello world"
    assert not any("SHOULD-NOT-APPEAR" in c for c in chunks)


def test_stream_http_401_yields_safe_error_event():
    chunks = _collect(
        _provider(lambda r: httpx.Response(401, json={"error": KEY})),
        _request(),
    )
    assert len(chunks) == 1
    payload = _error_payload(chunks[0])
    assert payload == {"type": "error", "message": "Provider returned HTTP 401.", "error_category": "provider"}
    assert KEY not in chunks[0]


def test_stream_429_yields_rate_limit_event_with_retry_after():
    chunks = _collect(
        _provider(lambda r: httpx.Response(429, headers={"retry-after": "12"})),
        _request(),
    )
    payload = _error_payload(chunks[0])
    assert payload["error_category"] == "rate_limit"
    assert payload["retry_after_seconds"] == 12.0


def test_stream_mid_stream_error_yields_error_event():
    chunks = _collect(
        _provider(lambda r: _sse('data: {"choices":[{"delta":{"content":"partial"}}]}',
                                 'data: {"error": {"message": "overloaded"}}')),
        _request(),
    )
    assert [c for c in chunks if not c.startswith("__ttft__")][0] == "partial"
    assert _error_payload(chunks[-1])["message"] == "overloaded"


# ------------------------------------------------------------------ gateway
def test_gateway_override_routes_to_provider_without_platform_quota():
    gateway = LLMGateway(override_provider=_provider(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "byok answer"}}]})
    ))

    result = gateway.generate(_request(model="openai/gpt-oss-120b"))

    assert result.text == "byok answer"
    assert result.provider == "custom"
    assert gateway._quota is None  # no platform QuotaManager, no shared budget consumed
    assert gateway.is_provider_available("custom")
    assert not gateway.is_provider_available("groq")


def test_gateway_override_stream_strips_ttft_and_keeps_error_contract():
    gateway = LLMGateway(override_provider=_provider(
        lambda r: _sse('data: {"choices":[{"delta":{"content":"streamed"}}]}', "data: [DONE]")
    ))

    async def run():
        return [chunk async for chunk in gateway.generate_stream(_request())]

    chunks = asyncio.run(run())
    assert chunks == ["streamed"]
