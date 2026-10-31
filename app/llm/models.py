"""Structured request/result types and provider protocol for the LLM Gateway.

Defines a small, provider-agnostic interface so Sherlock can route inference
through GroqCloud, local Ollama, or future providers without leaking
provider-specific details into the RAG pipeline or six-agent architecture.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol


@dataclass
class GenerationRequest:
    """A single LLM generation request.

    Attributes:
        messages: Chat-style message list (role/content dicts). When a
            provider only supports prompt/system_prompt, the gateway
            concatenates messages into a single prompt.
        model: Model identifier (e.g. "openai/gpt-oss-120b" or
            "qwen3-coder:30b"). Empty means use the provider default.
        temperature: Sampling temperature (0.0-2.0). None means provider
            default.
        max_output_tokens: Hard cap on generated tokens. None means provider
            default.
        request_id: Unique id for tracing. Auto-generated if omitted.
        inference_operation: Semantic category (e.g. "rag_answer",
            "agent_mapper", "agent_teacher"). Used for budget tracking and
            telemetry grouping.
        repo_id: Repository id the request relates to (for isolation and
            secret redaction). Optional.
        context_hash: Hash of retrieved context used to build this prompt,
            for cache keying and audit. Optional.
        metadata: Additional provider-agnostic metadata.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    model: str = ""
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    inference_operation: str = "unknown"
    repo_id: Optional[str] = None
    context_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Flatten messages into a single prompt string."""
        parts: List[str] = []
        for m in self.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)


@dataclass
class TokenUsage:
    """Token usage for a generation call.

    estimated_input / estimated_output are gateway-side approximations
    (tokenizers vary). provider_input / provider_output are values reported
    by the provider when available.
    """

    estimated_input: int = 0
    estimated_output: int = 0
    provider_input: Optional[int] = None
    provider_output: Optional[int] = None

    @property
    def input(self) -> int:
        return self.provider_input if self.provider_input is not None else self.estimated_input

    @property
    def output(self) -> int:
        return self.provider_output if self.provider_output is not None else self.estimated_output

    @property
    def total(self) -> int:
        return self.input + self.output


@dataclass
class GenerationResult:
    """Result of a generation call."""

    request_id: str
    provider: str
    model: str
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    timing: Dict[str, float] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    error_category: Optional[str] = None  # "budget", "provider", "network", "timeout", "rate_limit"

    @property
    def ttft_ms(self) -> Optional[float]:
        return self.timing.get("ttft_ms")

    @property
    def total_latency_ms(self) -> Optional[float]:
        return self.timing.get("total_latency_ms")

    @property
    def tokens_per_second(self) -> Optional[float]:
        total = self.timing.get("total_latency_ms")
        if not total or total <= 0:
            return None
        return self.usage.output / (total / 1000.0)


class LLMProvider(Protocol):
    """Provider-agnostic LLM interface.

    Implementations must NOT log, print, persist, or expose API keys.
    """

    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Non-streaming generation. Returns a complete GenerationResult."""
        ...

    def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text chunks as they arrive."""
        ...

    def check_health(self) -> Dict[str, Any]:
        """Return provider health status."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...
