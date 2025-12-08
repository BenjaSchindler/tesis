# Phase I: Multi-LLM Provider Comparison

## Overview

Phase I provides a unified abstraction layer for comparing synthetic data generation across multiple LLM providers. This enables experiments with different models (OpenAI, Anthropic, Google, DeepSeek, xAI, Moonshot) using a consistent API interface.

## Key Feature

**6 LLM providers with unified interface** - Switch between providers with a single parameter change, enabling fair comparisons of synthetic data quality across different foundation models.

## Problem Statement

Previous phases only used OpenAI models (gpt-4o-mini, gpt-5-mini). Questions arose:
- Do different LLM providers generate different quality synthetic data?
- Can reasoning/thinking models improve generation quality?
- What is the cost-quality tradeoff across providers?

## Supported Providers

| Provider | Standard Model | Reasoning Model | API Key Env |
|----------|---------------|-----------------|-------------|
| **OpenAI** | gpt-4o, gpt-4o-mini | gpt-5-mini, gpt-5.1 | `OPENAI_API_KEY` |
| **Anthropic** | claude-3-5-sonnet | claude-opus-4-5 | `ANTHROPIC_API_KEY` |
| **Google** | gemini-2.0-flash | gemini-3-pro-preview | `GOOGLE_API_KEY` |
| **DeepSeek** | deepseek-chat | deepseek-reasoner | `DEEPSEEK_API_KEY` |
| **xAI** | grok-3 | grok-4 | `XAI_API_KEY` |
| **Moonshot** | kimi-k2-0711 | kimi-k2-thinking | `MOONSHOT_API_KEY` |

## Architecture

### Provider Abstraction

```python
from llm_providers import create_provider

# OpenAI with reasoning
provider = create_provider("openai", "gpt-5-mini", reasoning_effort="high")

# Anthropic with extended thinking
provider = create_provider("anthropic", "claude-opus-4-5-20250514", use_thinking=True)

# DeepSeek standard
provider = create_provider("deepseek", "deepseek-chat")

# Generate text
text, usage = provider.generate(messages, temperature=0.5)
```

### Unified Interface

All providers implement the same interface:

```python
class LLMProvider(ABC):
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text from messages."""
        pass

    def get_model_name(self) -> str:
        """Return model identifier."""
        pass

    def supports_thinking(self) -> bool:
        """Return True if provider supports CoT/thinking."""
        pass
```

### Usage Statistics

Each generation returns usage statistics:

```python
usage = {
    "input_tokens": 150,
    "output_tokens": 500,
    "total_tokens": 650,
    "provider": "openai",
    "model": "gpt-5-mini",
    "reasoning_effort": "high"  # OpenAI only
}
```

## Thinking/Reasoning Modes

### OpenAI Reasoning

```python
# Uses reasoning_effort parameter
provider = create_provider("openai", "gpt-5-mini", reasoning_effort="high")

# Effort levels: "none", "low", "medium", "high"
# Higher = more reasoning tokens, higher cost, better quality
```

### Anthropic Extended Thinking

```python
# Uses thinking_budget parameter
provider = create_provider(
    "anthropic",
    "claude-opus-4-5-20250514",
    use_thinking=True,
    thinking_budget=10000  # tokens for internal reasoning
)
```

### DeepSeek Reasoning

```python
# Model-specific (deepseek-reasoner)
provider = create_provider("deepseek", "deepseek-reasoner")
```

## Directory Structure

```
phase_i/
  core/
    llm_providers.py        # Provider abstraction layer
    runner_phase2.py        # Main runner with multi-provider support
    ...
  configs/                  # Configuration files for each provider
  results/                  # Output files
  base_config.sh            # Template configuration
  kfold_evaluator.py        # K-Fold evaluation script
  run_model_ensemble.sh     # Multi-model run script
  README.md                 # This file
```

## Usage

### Single Provider Run

```bash
cd phase_i
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'

# OpenAI with reasoning
./run_provider.sh openai gpt-5-mini high 42

# Anthropic with thinking
./run_provider.sh anthropic claude-opus-4-5 thinking 42

# DeepSeek standard
./run_provider.sh deepseek deepseek-chat none 42
```

### Compare Providers

```bash
# Run same experiment across all providers
./run_model_ensemble.sh 42
```

### Provider Info

```python
from llm_providers import list_providers, get_provider_info

print(list_providers())
# ['openai', 'anthropic', 'google', 'deepseek', 'xai', 'moonshot']

print(get_provider_info('anthropic'))
# {
#     'name': 'Anthropic',
#     'models': ['claude-3-5-sonnet-20241022', 'claude-opus-4-5-20250514'],
#     'thinking_support': True,
#     'api_key_env': 'ANTHROPIC_API_KEY',
#     'docs': 'https://docs.anthropic.com'
# }
```

## Cost Comparison

Estimated costs per 1000 synthetic samples:

| Provider | Model | Cost/1K tokens (input) | Cost/1K tokens (output) |
|----------|-------|------------------------|-------------------------|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-5-mini | $0.075 | $0.30 |
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| Anthropic | claude-opus-4-5 | $15.00 | $75.00 |
| Google | gemini-2.0-flash | $0.075 | $0.30 |
| DeepSeek | deepseek-chat | $0.14 | $0.28 |

## Planned Experiments

### Phase I Roadmap

1. **Baseline Comparison**: Same prompt across all providers
2. **Reasoning Comparison**: Standard vs reasoning models
3. **Cost-Quality Analysis**: Quality per dollar spent
4. **Ensemble Generation**: Combine synthetics from multiple providers

### Metrics

- Macro F1 improvement (5-fold CV)
- Per-class F1 delta
- Synthetic acceptance rate
- Cost per synthetic sample
- Generation latency

## Technical Notes

### Message Format

All providers accept OpenAI-style messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Generate MBTI personality text."}
]
```

The provider abstraction layer handles format conversion internally.

### Error Handling

```python
try:
    text, usage = provider.generate(messages)
except Exception as e:
    logger.error(f"API error: {e}")
    # Retry logic, fallback provider, etc.
```

### Rate Limiting

Each provider has different rate limits. For high-volume generation:
- Use exponential backoff
- Batch requests appropriately
- Monitor usage quotas

## Files Reference

- [core/llm_providers.py](core/llm_providers.py) - Provider implementation (688 lines)
- [kfold_evaluator.py](kfold_evaluator.py) - K-Fold evaluation
- [base_config.sh](base_config.sh) - Configuration template

## Dependencies

```bash
pip install openai anthropic google-generativeai
```

## Environment Variables

```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
export GOOGLE_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export XAI_API_KEY='...'
export MOONSHOT_API_KEY='...'
```

## Created

2025-12-05
