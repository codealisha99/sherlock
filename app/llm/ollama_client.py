"""Ollama Client Integration for Sherlock.

Provides robust interaction with local Ollama instances, JSON schema extraction,
Pydantic validation, retry/repair mechanics, and graceful offline fallback.
"""

import json
import logging
import re
from typing import Optional, Type, TypeVar, Any, Dict
import httpx
from pydantic import BaseModel, ValidationError

from config import config, SherlockConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """Client for querying local Ollama LLM with structured output validation."""

    def __init__(self, sherlock_cfg: Optional[SherlockConfig] = None):
        self.cfg = sherlock_cfg or config
        self.host = self.cfg.ollama_host.rstrip("/")
        self.model = self.cfg.model_name
        self.timeout = self.cfg.timeout_seconds
        self.max_retries = self.cfg.max_retries

    def check_health(self) -> Dict[str, Any]:
        """Check if local Ollama daemon is reachable and whether configured model is present."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.host}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_found = any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
                    return {
                        "available": True,
                        "installed_models": models,
                        "model_ready": model_found,
                        "configured_model": self.model,
                    }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "installed_models": [],
                "model_ready": False,
                "configured_model": self.model,
            }
        return {"available": False, "installed_models": [], "model_ready": False, "configured_model": self.model}

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Raw generation call to Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "").strip()
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            raise ConnectionError(
                f"Ollama is unreachable at {self.host}. Ensure 'ollama serve' is running locally."
            ) from e
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """
        Request structured JSON from Ollama, validate against Pydantic schema,
        and auto-retry with error feedback if output is invalid.
        """
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
        system_instruction = (system_prompt or "") + (
            f"\n\nCRITICAL INSTRUCTION:\nYou must respond ONLY with valid JSON conforming to the following JSON schema.\n"
            f"Do not include any explanation or commentary outside the JSON object.\n"
            f"JSON Schema:\n{schema_json}"
        )

        current_prompt = prompt
        last_error = None

        for attempt in range(self.max_retries + 1):
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": current_prompt},
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "format": "json",
                "stream": False,
                "options": {"temperature": temperature},
            }

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.host}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data.get("message", {}).get("content", "")

                parsed_dict = self._clean_and_parse_json(content)
                validated_model = schema_class.model_validate(parsed_dict)
                return validated_model

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                # Provide self-correction prompt
                current_prompt = (
                    f"{prompt}\n\n[PREVIOUS ATTEMPT FAILED]\n"
                    f"Error: {str(e)}\n"
                    f"Please output strictly valid JSON matching the schema."
                )
            except httpx.ConnectError:
                raise

        raise ValueError(
            f"Failed to obtain valid structured output matching {schema_class.__name__} after {self.max_retries + 1} attempts: {last_error}"
        )

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Strip markdown fences and extract raw JSON object."""
        text = raw_text.strip()
        # Strip ```json ... ```
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        # Find first { and last }
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Clean trailing commas
                cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
                return json.loads(cleaned)

        # Try direct parse
        return json.loads(text)
