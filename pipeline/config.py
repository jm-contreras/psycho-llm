"""
Load environment variables and model registry.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.rate_limiter import AsyncRateLimiter
from pipeline.token_budget import TokenBudget

# Repo root is one level up from this file
REPO_ROOT = Path(__file__).parent.parent

load_dotenv(REPO_ROOT / ".env")

_registry_path = REPO_ROOT / "model_registry.json"


def load_model_registry(providers: list[str] | None = None) -> list[dict]:
    """
    Return the list of active models from model_registry.json.

    Args:
        providers: If given, only return models whose api_provider is in this list.
                   E.g. ["bedrock"] for the v1 smoke test.
    """
    with open(_registry_path) as f:
        data = json.load(f)

    models = data["bedrock_models"]  # field name is a legacy misnomer; contains all 25

    if providers is not None:
        models = [m for m in models if m.get("api_provider") in providers]

    models = [m for m in models if not m.get("skip")]

    return models


def build_tpm_budgets(models: list[dict]) -> dict[str, TokenBudget]:
    """
    Return a mapping of resource_group -> TokenBudget for models that have
    both 'resource_group' and 'tpm_limit' set. One TokenBudget instance is
    shared by all models in the same group.
    """
    budgets: dict[str, TokenBudget] = {}
    for m in models:
        rg = m.get("resource_group")
        limit = m.get("tpm_limit")
        if rg and limit and rg not in budgets:
            budgets[rg] = TokenBudget(tpm_limit=int(int(limit) * 0.8))
    return budgets


def build_rate_limiters(models: list[dict]) -> dict[str, AsyncRateLimiter]:
    """
    Return a mapping of litellm_model_id -> AsyncRateLimiter.

    Each model gets its own limiter based on its registry fields:
      - requests_per_minute (RPM)
      - tpm_limit (tokens per minute)
      - tpd_limit (tokens per day, optional)

    Models without tpm_limit default to 100,000 TPM (conservative fallback).
    """
    limiters: dict[str, AsyncRateLimiter] = {}
    for m in models:
        mid = m["litellm_model_id"]
        if mid in limiters:
            continue
        rpm = m.get("requests_per_minute") or 60
        tpm = int((m.get("tpm_limit") or 100_000) * 0.8)  # 80% headroom against provider limit
        tpd = m.get("tpd_limit") or None  # None if not set or null
        rpd = m.get("requests_per_day") or None
        limiters[mid] = AsyncRateLimiter(rpm=rpm, tpm=tpm, tpd=tpd, rpd=rpd)
    return limiters


def filter_by_names(models: list[dict], model_names: list[str] | None) -> list[dict]:
    """
    Filter model list by model_name. For each search term, tries exact match
    first (case-insensitive); falls back to substring only when no exact match
    exists. This prevents "GPT-5.4" from matching "GPT-5.4 Mini"/"GPT-5.4 Nano".
    """
    if model_names is None:
        return models
    seen: set[str] = set()
    result: list[dict] = []
    for name in model_names:
        n_lower = name.lower()
        exact = [m for m in models if m["model_name"].lower() == n_lower]
        candidates = exact if exact else [m for m in models if n_lower in m["model_name"].lower()]
        for m in candidates:
            mid = m["litellm_model_id"]
            if mid not in seen:
                seen.add(mid)
                result.append(m)
    return result
