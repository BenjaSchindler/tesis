#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer for Phase I Robust

Supports multiple LLM providers for text generation:
- OpenAI (gpt-4o-mini, etc.)
- Google Gemini 3 Flash (gemini-3-flash-preview)

Usage:
    from llm_providers import create_provider

    provider = create_provider("google", "gemini-3-flash-preview")
    text, usage = provider.generate(messages, temperature=0.7)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import os
import logging
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, usage_dict)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "provider": "openai",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model


class Gemini3Provider(LLMProvider):
    """
    Google Gemini 3 Flash provider using new SDK.

    Model: gemini-3-flash-preview
    Context: 1,048,576 tokens input
    Output: 65,536 tokens max
    """

    def __init__(self, model: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        from google import genai

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to text prompt."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"Instructions: {content}\n")
            elif role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Previous response: {content}\n")
        return "\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        prompt = self._convert_messages(messages)

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            text = response.text

            # Extract usage if available
            usage_meta = getattr(response, 'usage_metadata', None)
            usage = {
                "input_tokens": getattr(usage_meta, 'prompt_token_count', 0) if usage_meta else 0,
                "output_tokens": getattr(usage_meta, 'candidates_token_count', 0) if usage_meta else 0,
                "total_tokens": getattr(usage_meta, 'total_token_count', 0) if usage_meta else 0,
                "provider": "google",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model


class GPT5Provider(LLMProvider):
    """
    OpenAI GPT-5 provider using Responses API.

    IMPORTANT: GPT-5 does NOT support temperature/top_p parameters.
    Uses reasoning.effort instead for controlling output quality.

    Models: gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2, gpt-5.2-pro
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        reasoning_effort: str = "medium"
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,  # IGNORED for GPT-5
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate using GPT-5 Responses API.

        Note: temperature is IGNORED - GPT-5 uses reasoning.effort instead.
        """
        # Extract content from messages (GPT-5 Responses API uses 'input')
        # Combine system and user messages
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"Instructions: {msg['content']}\n")
            elif msg["role"] == "user":
                prompt_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Previous response: {msg['content']}\n")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={"effort": self.reasoning_effort},
                text={"verbosity": "medium"},
                max_output_tokens=max_tokens
            )

            text = response.output_text

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', 0),
                "provider": "openai_gpt5",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"GPT-5 API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "openai_gpt5"


def create_provider(provider_name: str, model: str = None, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_name: "openai" or "google"
        model: Model name (uses defaults if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(
            model=model or "gpt-4o-mini",
            **kwargs
        )
    elif provider_name in ["gpt5", "openai_gpt5"]:
        return GPT5Provider(
            model=model or "gpt-5-mini",
            **kwargs
        )
    elif provider_name in ["google", "gemini"]:
        return Gemini3Provider(
            model=model or "gemini-3-flash-preview",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: openai, gpt5, google")


# Quick test function
def test_provider(provider_name: str = "google"):
    """Test a provider with a simple prompt."""
    try:
        provider = create_provider(provider_name)
        messages = [{"role": "user", "content": "Say 'Hello World' in 5 words or less."}]
        text, usage = provider.generate(messages, temperature=0.5, max_tokens=50)
        print(f"Provider: {provider.get_provider_name()}")
        print(f"Model: {provider.get_model_name()}")
        print(f"Response: {text}")
        print(f"Usage: {usage}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "google"
    test_provider(provider)
