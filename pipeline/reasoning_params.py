"""
Provider-specific litellm.completion() / litellm.acompletion() kwargs.

Centralised here so both api_client.py (item scoring) and behavioral_runner.py
(free-text collection) can reuse credential-injection and model-config logic
without duplication.
"""

from __future__ import annotations

import os


def get_provider_kwargs(model: dict, behavioral: bool = False) -> dict:
    """
    Return provider-specific extra kwargs to merge into a litellm call.

    behavioral=True skips extra_body params that disable reasoning/thinking mode —
    behavioral prompts should let models respond naturally. Credential injection
    still applies in both modes.
    """
    api_provider = model.get("api_provider", "")
    kwargs: dict = {}

    if api_provider == "azure":
        azure_base = model.get("azure_api_base") or os.environ.get("AZURE_API_BASE")
        if azure_base:
            kwargs["api_base"] = azure_base
        key_env = model.get("azure_api_key_env") or "AZURE_API_KEY"
        kwargs["api_key"] = os.environ.get(key_env)
        kwargs["api_version"] = os.environ.get("AZURE_API_VERSION", "2024-05-01-preview")
        # For item scoring: inject logprobs via extra_body (bypasses litellm's param filtering
        # for Azure AI Foundry models it doesn't recognise).
        if not behavioral and model.get("token_probabilities") is not False:
            kwargs["extra_body"] = {"logprobs": True, "top_logprobs": 5}

    elif api_provider == "xiaomi":
        kwargs["api_base"] = "https://api.xiaomimimo.com/v1"
        kwargs["api_key"] = os.environ.get("MIMO_API_KEY")
        # For item scoring: disable thinking to avoid massive token chains on Likert ratings.
        if not behavioral:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    elif api_provider == "openrouter":
        kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY")

    elif api_provider == "alibaba":
        # DashScope international endpoint (required outside mainland China).
        kwargs["api_base"] = os.environ.get("DASHSCOPE_API_BASE")
        kwargs["api_key"] = os.environ.get("DASHSCOPE_API_KEY")
        # For item scoring: disable thinking mode (Qwen 3.5 generates massive chains by default).
        if not behavioral:
            kwargs["extra_body"] = {"enable_thinking": False}

    elif api_provider == "bedrock":
        # Some Bedrock models are reasoning models whose thinking must be disabled
        # for item scoring to avoid CoT flooding the response and breaking JSON parsing.
        # Set disable_thinking=true in the registry to activate this.
        if not behavioral and model.get("disable_thinking"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # openai, google, xai, ai21, deepseek: litellm reads standard env vars
    # (OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, AWS_*, AI21_API_KEY, DEEPSEEK_API_KEY)
    # automatically — no extra kwargs needed.

    return kwargs
