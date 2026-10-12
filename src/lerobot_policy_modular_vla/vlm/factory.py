"""VLM Client Factory."""

import logging
from typing import Any

from .base import BaseVLMClient
from .mock_client import MockVLMClient
from .ollama_client import OllamaVLMClient
from .openai_client import OpenAIVLMClient
from .gemini_client import GeminiVLMClient
from .anthropic_client import AnthropicVLMClient
from .transformers_client import TransformersVLMClient

logger = logging.getLogger(__name__)


def make_vlm_client(
    provider: str = "mock",
    model_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.1,
    **kwargs: Any,
) -> BaseVLMClient:
    """Factory function to instantiate the requested VLM provider client.

    Args:
        provider: Provider name ('mock', 'ollama', 'openai', 'gemini', 'anthropic', 'transformers').
        model_name: Model identifier (e.g. 'qwen2.5-vl:7b', 'gpt-4o', 'gemini-2.0-flash').
        api_key: Optional API key.
        api_base: Optional base URL for Ollama or custom OpenAI endpoint.
        temperature: Generation temperature.

    Returns:
        Instance of BaseVLMClient.
    """
    prov = (provider or "mock").lower()

    if prov in ("mock", "rule", "dummy"):
        return MockVLMClient()

    elif prov == "ollama":
        return OllamaVLMClient(
            model_name=model_name or "qwen2.5-vl:7b",
            api_base=api_base or "http://localhost:11434",
            temperature=temperature,
        )

    elif prov in ("openai", "vllm", "litellm", "openrouter"):
        return OpenAIVLMClient(
            model_name=model_name or "gpt-4o",
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            temperature=temperature,
        )

    elif prov in ("gemini", "google"):
        return GeminiVLMClient(
            model_name=model_name or "gemini-2.0-flash",
            api_key=api_key,
            temperature=temperature,
        )

    elif prov in ("anthropic", "claude"):
        return AnthropicVLMClient(
            model_name=model_name or "claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=temperature,
        )

    elif prov in ("transformers", "hf", "local"):
        return TransformersVLMClient(
            model_name=model_name or "HuggingFaceTB/SmolVLM-Instruct",
            device=kwargs.get("device"),
        )

    else:
        logger.warning(f"Unknown VLM provider '{provider}'. Falling back to MockVLMClient.")
        return MockVLMClient()
