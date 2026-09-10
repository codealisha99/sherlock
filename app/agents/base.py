"""Base Agent for Sherlock Pipeline.

Defines the core interface, prompt execution, context preparation,
and resilient synthesis capabilities shared by all analysis agents.
"""

from abc import ABC, abstractmethod
import logging
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel

from app.analysis.context import AnalysisContext
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
        sherlock_cfg: Optional[SherlockConfig] = None,
    ):
        self.name = name
        self.description = description
        self.cfg = sherlock_cfg or config
        self.llm = llm_client or OllamaClient(self.cfg)

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
    ) -> T:
        """
        Query Ollama for structured output.
        If Ollama is unreachable (or offline), use fallback_factory to generate
        a deterministic output from static AST/indexer data.
        """
        try:
            health = self.llm.check_health()
            if health["available"]:
                return self.llm.generate_structured(
                    prompt=prompt,
                    schema_class=schema_class,
                    system_prompt=self.get_system_prompt(),
                )
            else:
                logger.info(f"Ollama not available ({health.get('error')}). Using static AST synthesis for {self.name}.")
                return fallback_factory(ctx)
        except Exception as e:
            logger.warning(f"{self.name} LLM query encountered exception: {e}. Falling back to static AST synthesis.")
            return fallback_factory(ctx)
