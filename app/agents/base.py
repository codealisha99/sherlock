"""Base Agent for Sherlock Pipeline.

Defines the core interface, prompt execution, context preparation,
and resilient synthesis capabilities shared by all analysis agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Type, TypeVar, Optional, Dict, Any, List
from pydantic import BaseModel

from app.analysis.context import AnalysisContext
from app.llm.gateway import LLMGateway
from app.llm.models import GenerationRequest
from app.llm.ollama_client import OllamaClient
from config import config, SherlockConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """Abstract base class for all Sherlock specialized agents."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_client: Optional[OllamaClient] = None,
        gateway: Optional[LLMGateway] = None,
        sherlock_cfg: Optional[SherlockConfig] = None,
    ):
        self.name = name
        self.description = description
        self.cfg = sherlock_cfg or config
        self.llm = llm_client or OllamaClient(self.cfg)
        self.gateway = gateway

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent."""
        pass

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> Any:
        """Execute the agent's analysis using the shared context."""
        pass

    def _query_model_with_fallback(
        self,
        prompt: str,
        schema_class: Type[T],
        fallback_factory: Any,
        ctx: AnalysisContext,
        empty_fields: Optional[List[str]] = None,
    ) -> T:
        """
        Query the LLM (Gateway or Ollama) for structured output.
        If the LLM is unreachable (or offline), or if the specified fields
        are empty in the LLM result, use fallback_factory to generate
        a deterministic output from static AST/indexer data.
        """
        # Prefer the LLM Gateway when configured.
        if self.gateway is not None:
            try:
                result = self._query_gateway_structured(
                    prompt=prompt,
                    schema_class=schema_class,
                )
                if self._is_empty_result(result, empty_fields):
                    logger.info(
                        f"{self.name} LLM returned empty result for fields "
                        f"{empty_fields}. Using static AST synthesis fallback."
                    )
                    return fallback_factory(ctx)
                return result
            except Exception as e:
                logger.warning(
                    f"{self.name} gateway query encountered exception: {e}. "
                    f"Falling back to Ollama client or static AST synthesis."
                )

        # Fall back to OllamaClient.
        try:
            health = self.llm.check_health()
            if health["available"]:
                result = self.llm.generate_structured(
                    prompt=prompt,
                    schema_class=schema_class,
                    system_prompt=self.get_system_prompt(),
                )
                if self._is_empty_result(result, empty_fields):
                    logger.info(
                        f"{self.name} LLM returned empty result for fields "
                        f"{empty_fields}. Using static AST synthesis fallback."
                    )
                    return fallback_factory(ctx)
                return result
            else:
                logger.info(
                    f"Ollama not available ({health.get('error')}). Using static AST synthesis for {self.name}."
                )
                return fallback_factory(ctx)
        except Exception as e:
            logger.warning(
                f"{self.name} LLM query encountered exception: {e}. Falling back to static AST synthesis."
            )
            return fallback_factory(ctx)

    def _query_gateway_structured(
        self,
        prompt: str,
        schema_class: Type[T],
    ) -> T:
        """Query the LLM Gateway for structured output and parse JSON into a Pydantic model."""
        if self.gateway is None:
            raise RuntimeError("Gateway is not configured.")

        result = self.gateway.generate(
            GenerationRequest(
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_output_tokens=int(self.cfg.llm_max_output_tokens),
                inference_operation="agent_analysis",
            )
        )

        if result.error:
            raise RuntimeError(f"Gateway generation failed: {result.error}")

        # Parse JSON from the gateway response text.
        text = result.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
                data = json.loads(cleaned)
        else:
            data = json.loads(text)

        return schema_class.model_validate(data)

    @staticmethod
    def _is_empty_result(result: BaseModel, empty_fields: Optional[List[str]] = None) -> bool:
        """
        Return True if all (or specified) list fields on the Pydantic model are empty.
        Used to detect when the LLM returned no findings and the static
        fallback should be used instead.
        """
        fields_to_check = empty_fields or [f.name for f in result.model_fields.values()]
        for field_name in fields_to_check:
            value = getattr(result, field_name, None)
            if isinstance(value, list) and len(value) > 0:
                return False
            if isinstance(value, dict) and len(value) > 0:
                return False
        return True
