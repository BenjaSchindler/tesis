#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer for Phase I

This module provides a unified interface for multiple LLM providers,
allowing easy switching between OpenAI, Anthropic, Google, DeepSeek, xAI, and Moonshot.

Usage:
    from llm_providers import create_provider

    provider = create_provider("anthropic", "claude-opus-4-5-20250514", use_thinking=True)
    text, usage = provider.generate(messages, temperature=0.5)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import os
import logging
import time

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Tuple of (generated_text, usage_dict)
            usage_dict contains at minimum: input_tokens, output_tokens
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass

    def supports_thinking(self) -> bool:
        """Return True if this provider supports CoT/extended thinking."""
        return False

    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        return self.__class__.__name__.replace("Provider", "").lower()


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider supporting both Chat Completions and Responses API.

    Supports:
    - gpt-4o, gpt-4o-mini: Standard Chat Completions API
    - gpt-5-mini, gpt-5.1: Responses API with reasoning parameter
    """

    def __init__(
        self,
        model: str,
        reasoning_effort: str = "none",
        api_key: Optional[str] = None
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

        # Determine if this is a reasoning model
        self._is_reasoning_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        try:
            if self._is_reasoning_model and self.reasoning_effort != "none":
                # Use Responses API for reasoning models
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    reasoning={"effort": self.reasoning_effort},
                )

                text = response.output_text
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', 0),
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "provider": "openai",
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort
                }
            else:
                # Use Chat Completions API
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
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
        suffix = f"_thinking_{self.reasoning_effort}" if self.reasoning_effort != "none" else ""
        return f"{self.model}{suffix}"

    def supports_thinking(self) -> bool:
        return self._is_reasoning_model


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider supporting Claude models with extended thinking.

    Supports:
    - claude-3-5-sonnet: Standard generation
    - claude-opus-4-5: Extended thinking capability
    """

    def __init__(
        self,
        model: str,
        use_thinking: bool = False,
        thinking_budget: int = 10000,
        api_key: Optional[str] = None
    ):
        import anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.use_thinking = use_thinking
        self.thinking_budget = thinking_budget

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        # Convert OpenAI format to Anthropic format
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": user_messages,
            }

            if system_content:
                params["system"] = system_content

            if self.use_thinking:
                # Extended thinking mode (requires specific models)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                # Note: temperature cannot be set with thinking mode
            else:
                params["temperature"] = temperature
                params["top_p"] = top_p

            response = self.client.messages.create(**params)

            # Extract text from response
            text = ""
            thinking_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    text = block.text
                elif hasattr(block, 'thinking'):
                    thinking_text = block.thinking

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "provider": "anthropic",
                "model": self.model,
                "thinking_used": self.use_thinking
            }

            if thinking_text:
                usage["thinking_tokens"] = len(thinking_text.split())  # Approximate

            return text, usage

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def get_model_name(self) -> str:
        suffix = "_thinking" if self.use_thinking else ""
        return f"{self.model}{suffix}"

    def supports_thinking(self) -> bool:
        return "opus" in self.model.lower() or "4-5" in self.model


class GoogleProvider(LLMProvider):
    """
    Google Gemini provider.

    Supports:
    - gemini-2.0-flash: Fast, standard generation
    - gemini-3-pro-preview: Advanced reasoning
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None
    ):
        import google.generativeai as genai

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"Instructions: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        return "\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        prompt = self._convert_messages(messages)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                }
            )

            text = response.text

            # Extract usage metadata if available
            usage_meta = getattr(response, 'usage_metadata', None)
            usage = {
                "input_tokens": getattr(usage_meta, 'prompt_token_count', 0) if usage_meta else 0,
                "output_tokens": getattr(usage_meta, 'candidates_token_count', 0) if usage_meta else 0,
                "total_tokens": getattr(usage_meta, 'total_token_count', 0) if usage_meta else 0,
                "provider": "google",
                "model": self.model_name
            }

            return text, usage

        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name

    def supports_thinking(self) -> bool:
        return "pro" in self.model_name.lower()


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek provider using OpenAI-compatible API.

    Supports:
    - deepseek-chat (V3): Standard chat
    - deepseek-reasoner: Extended reasoning
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "provider": "deepseek",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model

    def supports_thinking(self) -> bool:
        return "reasoner" in self.model.lower()


class XAIProvider(LLMProvider):
    """
    xAI Grok provider using OpenAI-compatible API.

    Supports:
    - grok-3: Standard generation
    - grok-4: Advanced reasoning
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content
            usage = {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
                "provider": "xai",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"xAI API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model

    def supports_thinking(self) -> bool:
        return "grok-4" in self.model.lower()


class MoonshotProvider(LLMProvider):
    """
    Moonshot Kimi provider.

    Supports:
    - kimi-k2-0711: Standard generation
    - kimi-k2-thinking: Extended thinking
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content
            usage = {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
                "provider": "moonshot",
                "model": self.model
            }

            return text, usage

        except Exception as e:
            logger.error(f"Moonshot API error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model

    def supports_thinking(self) -> bool:
        return "thinking" in self.model.lower()


# =============================================================================
# Factory Function
# =============================================================================

def create_provider(
    provider_name: str,
    model: str,
    use_thinking: bool = False,
    reasoning_effort: str = "none",
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider instances.

    Args:
        provider_name: One of 'openai', 'anthropic', 'google', 'deepseek', 'xai', 'moonshot'
        model: Model identifier string
        use_thinking: Enable CoT/thinking mode if supported
        reasoning_effort: For OpenAI models: 'none', 'low', 'medium', 'high'
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance

    Example:
        # OpenAI with thinking
        provider = create_provider("openai", "gpt-5-mini", reasoning_effort="high")

        # Anthropic with extended thinking
        provider = create_provider("anthropic", "claude-opus-4-5-20250514", use_thinking=True)

        # DeepSeek standard
        provider = create_provider("deepseek", "deepseek-chat")
    """

    provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(
            model=model,
            reasoning_effort=reasoning_effort if use_thinking else "none",
            **kwargs
        )
    elif provider_name == "anthropic":
        return AnthropicProvider(
            model=model,
            use_thinking=use_thinking,
            **kwargs
        )
    elif provider_name == "google":
        return GoogleProvider(
            model=model,
            **kwargs
        )
    elif provider_name == "deepseek":
        return DeepSeekProvider(
            model=model,
            **kwargs
        )
    elif provider_name == "xai":
        return XAIProvider(
            model=model,
            **kwargs
        )
    elif provider_name == "moonshot":
        return MoonshotProvider(
            model=model,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}. "
                        f"Supported: openai, anthropic, google, deepseek, xai, moonshot")


# =============================================================================
# Utility Functions
# =============================================================================

def list_providers() -> List[str]:
    """Return list of supported provider names."""
    return ["openai", "anthropic", "google", "deepseek", "xai", "moonshot"]


def get_provider_info(provider_name: str) -> Dict[str, Any]:
    """Return information about a provider."""
    info = {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5.1"],
            "thinking_support": True,
            "api_key_env": "OPENAI_API_KEY",
            "docs": "https://platform.openai.com/docs"
        },
        "anthropic": {
            "name": "Anthropic",
            "models": ["claude-3-5-sonnet-20241022", "claude-opus-4-5-20250514"],
            "thinking_support": True,
            "api_key_env": "ANTHROPIC_API_KEY",
            "docs": "https://docs.anthropic.com"
        },
        "google": {
            "name": "Google",
            "models": ["gemini-2.0-flash", "gemini-3-pro-preview"],
            "thinking_support": True,
            "api_key_env": "GOOGLE_API_KEY",
            "docs": "https://ai.google.dev/docs"
        },
        "deepseek": {
            "name": "DeepSeek",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "thinking_support": True,
            "api_key_env": "DEEPSEEK_API_KEY",
            "docs": "https://platform.deepseek.com/api-docs"
        },
        "xai": {
            "name": "xAI",
            "models": ["grok-3", "grok-4"],
            "thinking_support": True,
            "api_key_env": "XAI_API_KEY",
            "docs": "https://docs.x.ai"
        },
        "moonshot": {
            "name": "Moonshot",
            "models": ["kimi-k2-0711", "kimi-k2-thinking"],
            "thinking_support": True,
            "api_key_env": "MOONSHOT_API_KEY",
            "docs": "https://platform.moonshot.cn/docs"
        }
    }
    return info.get(provider_name.lower(), {})


if __name__ == "__main__":
    # Quick test
    print("LLM Providers Module")
    print("=" * 50)
    print(f"Supported providers: {list_providers()}")

    for provider in list_providers():
        info = get_provider_info(provider)
        print(f"\n{info['name']}:")
        print(f"  Models: {info['models']}")
        print(f"  Thinking: {info['thinking_support']}")
        print(f"  API Key: {info['api_key_env']}")
