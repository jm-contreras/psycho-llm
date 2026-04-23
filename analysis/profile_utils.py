"""Shared helpers for model-profile figures and tables.

Used by make_hero_profile, make_method_convergence, make_ocean_profile,
make_metadata_aggregation, make_unified_profile_table, and make_paired_profiles.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

from analysis.data_loader import get_short_model_name

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FACTORS = ["RE", "DE", "BO", "GU", "VB"]

FACTOR_FULL_NAMES = {
    "RE": "Responsiveness",
    "DE": "Deference",
    "BO": "Boldness",
    "GU": "Guardedness",
    "VB": "Verbosity",
}

# Direction of the high pole (for per-panel "↑ = ..." annotations).
FACTOR_HIGH_POLE = {
    "RE": "more responsive",
    "DE": "more deferent",
    "BO": "bolder",
    "GU": "more guarded",
    "VB": "more verbose",
}

# Display-name overrides for figures (cleaner than get_short_model_name).
MODEL_DISPLAY_NAMES = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Claude Opus 4.6",
    "bedrock/us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash",
    "bedrock/google.gemma-3-27b-it": "Gemma 3 27B",
    "xai/grok-4.20-beta-0309-non-reasoning": "Grok 4.20",
    "openai/deepseek-v3.2": "DeepSeek V3.2",
    "deepseek/deepseek-reasoner": "DeepSeek R1",
    "dashscope/qwen3.5-plus": "Qwen 3.5",
    "bedrock/converse/moonshotai.kimi-k2.5": "Kimi K2.5",
    "bedrock/converse/zai.glm-5": "GLM-5",
    "bedrock/converse/minimax.minimax-m2.5": "MiniMax M2.5",
    "openai/mimo-v2-pro": "MiMo V2 Pro",
    "openai/mistral-large-3": "Mistral Large 3",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "openai/cohere-command-a": "Command A",
    "bedrock/amazon.nova-pro-v1:0": "Nova 2 Pro",
    "openai/phi-4": "Phi 4",
    "ai21/jamba-large-1.7": "Jamba Large 1.7",
    "bedrock/converse/nvidia.nemotron-super-3-120b": "Nemotron 3 Super",
}


def display_name(model_id: str) -> str:
    """Human-readable model name for figures. Falls back to short name."""
    if model_id in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_id]
    return get_short_model_name(model_id)

R1_CANONICAL = "deepseek/deepseek-reasoner"
R1_ALIASES = {"openai/deepseek-r1": R1_CANONICAL}

# Curated popular-9 subset — model_id strings (canonical post-alias).
POPULAR_MODEL_IDS = [
    "bedrock/us.anthropic.claude-opus-4-6-v1",
    "openai/gpt-5.4",
    "gemini/gemini-3.1-pro-preview",
    "xai/grok-4.20-beta-0309-non-reasoning",
    "deepseek/deepseek-reasoner",
    "dashscope/qwen3.5-plus",
    "bedrock/converse/moonshotai.kimi-k2.5",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "openai/mistral-large-3",
]


def z_score(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        col = out[c]
        std = col.std(ddof=1)
        out[c] = (col - col.mean()) / std if std > 0 else 0.0
    return out


def load_instrument_profile() -> pd.DataFrame:
    """Per-model mean factor scores from the EFA solution (all 30 runs)."""
    path = os.path.join(ROOT, "analysis", "output", "factor_scores.csv")
    df = pd.read_csv(path)
    df["model_id"] = df["model_id"].replace(R1_ALIASES)
    df = df.groupby("model_id", as_index=False)[FACTORS].mean()
    df["short_name"] = df["model_id"].map(get_short_model_name)
    return df


def load_judge_profile() -> pd.DataFrame:
    """Per-model judge-rated profile: for each factor F, mean of score_F across
    prompts targeting F (matched-factor ratings), from judge_ratings table.
    """
    db_path = os.path.join(ROOT, "data", "raw", "responses.db")
    conn = sqlite3.connect(db_path)
    jr = pd.read_sql_query(
        "SELECT subject_model_id, prompt_id, "
        "score_RE, score_DE, score_BO, score_GU, score_VB "
        "FROM judge_ratings WHERE parse_status='success'",
        conn,
    )
    conn.close()
    jr["subject_model_id"] = jr["subject_model_id"].replace(R1_ALIASES)
    jr["prompt_factor"] = jr["prompt_id"].str.slice(0, 2)

    rows = []
    for model_id, mdf in jr.groupby("subject_model_id"):
        row = {"model_id": model_id, "short_name": get_short_model_name(model_id)}
        for f in FACTORS:
            on_factor = mdf[mdf["prompt_factor"] == f]
            row[f] = on_factor[f"score_{f}"].mean() if len(on_factor) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_human_profile(on_target_only: bool = False) -> pd.DataFrame:
    """Per-model human-rated profile: for each factor F, mean of corrected_F
    across Prolific ratings.

    Default: all ratings contribute to every factor (each response was rated on
    all 5 factors). Set ``on_target_only=True`` to restrict to prompts whose
    target factor matches F.

    Joins prolific_ratings.behavioral_response_id → behavioral_responses.model_id.
    Uses `corrected_*` (reverse-coded into positive direction) per the existing
    prolific_analysis pipeline. Excludes ratings from participants flagged by
    gold-item QA.
    """
    # Model lookup via responses.db
    resp_path = os.path.join(ROOT, "data", "raw", "responses.db")
    with sqlite3.connect(resp_path) as conn:
        br = pd.read_sql_query(
            "SELECT id AS behavioral_response_id, model_id, prompt_id "
            "FROM behavioral_responses",
            conn,
        )

    # Human ratings
    prolific_path = os.path.join(ROOT, "data", "prolific", "prolific.db")
    with sqlite3.connect(prolific_path) as conn:
        pr = pd.read_sql_query(
            "SELECT behavioral_response_id, prompt_id, "
            "corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB, "
            "participant_flagged "
            "FROM prolific_ratings "
            "WHERE participant_flagged = 0",
            conn,
        )

    merged = pr.merge(
        br[["behavioral_response_id", "model_id"]],
        on="behavioral_response_id", how="inner",
    )
    merged["model_id"] = merged["model_id"].replace(R1_ALIASES)
    merged["prompt_factor"] = merged["prompt_id"].str.slice(0, 2)

    rows = []
    for model_id, mdf in merged.groupby("model_id"):
        row = {"model_id": model_id, "short_name": get_short_model_name(model_id)}
        for f in FACTORS:
            subset = mdf[mdf["prompt_factor"] == f] if on_target_only else mdf
            vals = subset[f"corrected_{f}"].dropna()
            row[f] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_se(
    values: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Bootstrap SE of the mean. Resamples `values` with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))
