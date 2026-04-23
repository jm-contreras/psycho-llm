"""Section 1: Engineering checks — completeness, parse rates, score ranges."""

import pandas as pd
import numpy as np

from .data_loader import get_short_model_name


def model_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary of call outcomes and coverage.

    Returns DataFrame with: model_id, short_name, total, success, parse_error,
    api_error, refusal, success_rate, refusal_rate, parse_error_rate,
    primary_scoring_method, n_items, n_runs.
    """
    status_counts = (
        df.groupby(["model_id", "status"]).size().unstack(fill_value=0)
    )
    for col in ["success", "parse_error", "api_error", "refusal"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    status_counts["total"] = status_counts.sum(axis=1)
    status_counts["success_rate"] = status_counts["success"] / status_counts["total"]
    status_counts["refusal_rate"] = status_counts["refusal"] / status_counts["total"]
    status_counts["parse_error_rate"] = (
        status_counts["parse_error"] / status_counts["total"]
    )

    # Primary scoring method per model (mode of text_scoring_method for success rows)
    scoring = (
        df[df["status"] == "success"]
        .groupby("model_id")["text_scoring_method"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A")
        .rename("primary_scoring_method")
    )

    items = df.groupby("model_id")["item_id"].nunique().rename("n_items")
    max_run = df.groupby("model_id")["run_number"].max().rename("n_runs")

    summary = status_counts.join(scoring).join(items).join(max_run)
    summary = summary.reset_index()
    summary["short_name"] = summary["model_id"].apply(get_short_model_name)
    summary = summary.sort_values("total", ascending=False).reset_index(drop=True)

    cols = [
        "short_name", "model_id", "total", "success", "parse_error", "api_error",
        "refusal", "success_rate", "refusal_rate", "parse_error_rate",
        "primary_scoring_method", "n_items", "n_runs",
    ]
    return summary[cols]


def flag_problematic_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Flag models with refusal_rate >25% or parse_error_rate >5%.

    Returns subset with added 'flag_reason' column.
    """
    flags = []
    for _, row in summary.iterrows():
        reasons = []
        if row["refusal_rate"] > 0.25:
            reasons.append(f"refusal_rate={row['refusal_rate']:.1%}")
        if row["parse_error_rate"] > 0.05:
            reasons.append(f"parse_error_rate={row['parse_error_rate']:.1%}")
        if reasons:
            flags.append({**row, "flag_reason": "; ".join(reasons)})

    if not flags:
        return pd.DataFrame(columns=[*summary.columns, "flag_reason"])
    return pd.DataFrame(flags)


def check_score_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Check that parsed_score is within expected range per item_type.

    Direct items: [1, 5]. Scenario items: [1, 4].
    Returns DataFrame of out-of-range rows (empty if clean).
    """
    success = df[(df["status"] == "success") & df["parsed_score"].notna()]

    direct_oor = success[
        (success["item_type"] == "direct")
        & ((success["parsed_score"] < 1) | (success["parsed_score"] > 5))
    ]
    scenario_oor = success[
        (success["item_type"] == "scenario")
        & ((success["parsed_score"] < 1) | (success["parsed_score"] > 4))
    ]
    oor = pd.concat([direct_oor, scenario_oor])
    if len(oor) > 0:
        oor = oor[["model_id", "item_id", "item_type", "parsed_score", "run_number"]].copy()
    return oor


def run_engineering_checks(df: pd.DataFrame) -> dict:
    """Run all Section 1 checks.

    Returns dict with keys: 'model_summary', 'flagged_models', 'out_of_range'.
    """
    summary = model_summary_table(df)
    flagged = flag_problematic_models(summary)
    oor = check_score_ranges(df)

    return {
        "model_summary": summary,
        "flagged_models": flagged,
        "out_of_range": oor,
    }
