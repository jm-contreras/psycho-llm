"""BFI-44 analysis module.

Produces Big Five trait scores per model, reliability estimates, profile
visualisations, and a preliminary convergent/discriminant correlation matrix
against the AI-native candidate dimensions.

Usage:
    python -m analysis.bfi_analysis
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .data_loader import (
    DB_PATH,
    DIMENSION_CODES,
    OUTPUT_DIR,
    PLOTS_DIR,
    compute_model_item_means,
    ensure_output_dirs,
    filter_success,
    get_short_model_name,
    load_responses,
    pivot_score_matrix,
    recode_reverse_items,
)
from .dimension_coherence import cronbachs_alpha
from .report import df_to_markdown

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BFI_DIMENSIONS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
BFI_DIM_SHORT = {"Openness": "O", "Conscientiousness": "C", "Extraversion": "E",
                 "Agreeableness": "A", "Neuroticism": "N"}

# Scoring columns: OCEAN plus forward-only Extraversion
SCORE_DIMS = list("OCEAN") + ["E_fwd"]

# AI-native dimensions (excluding BFI)
AI_NATIVE_DIMENSIONS = list(DIMENSION_CODES.values())


def _is_bfi(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for BFI rows."""
    return df["item_id"].str.startswith("BFI-")


def _is_ai_native(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for AI-native rows (non-BFI)."""
    return ~_is_bfi(df)


# ===========================================================================
# Section 1: Engineering Checks
# ===========================================================================

def bfi_engineering_checks(df_all: pd.DataFrame) -> dict:
    """Per-model BFI-specific health checks.

    Returns dict with 'model_summary' (DataFrame) and 'flagged_models' (DataFrame).
    """
    bfi = df_all[_is_bfi(df_all)].copy()

    rows = []
    for model_id, mdf in bfi.groupby("model_id"):
        total = len(mdf)
        success = (mdf["status"] == "success").sum()
        parse_error = (mdf["status"] == "parse_error").sum()
        refusal = (mdf["status"] == "refusal").sum()
        api_error = (mdf["status"] == "api_error").sum()
        n_items = mdf["item_id"].nunique()
        max_run = mdf["run_number"].max()
        missing_items = 44 - n_items

        rows.append({
            "model_id": model_id,
            "short_name": get_short_model_name(model_id),
            "total_bfi_calls": total,
            "success": success,
            "parse_error": parse_error,
            "refusal": refusal,
            "api_error": api_error,
            "success_rate": success / total if total > 0 else 0,
            "parse_error_rate": parse_error / total if total > 0 else 0,
            "refusal_rate": refusal / total if total > 0 else 0,
            "n_items": n_items,
            "missing_items": missing_items,
            "max_run": max_run,
        })

    summary = pd.DataFrame(rows).sort_values("short_name").reset_index(drop=True)

    # Flag models
    flags = []
    for _, row in summary.iterrows():
        reasons = []
        if row["parse_error_rate"] > 0.10:
            reasons.append(f"parse_error_rate={row['parse_error_rate']:.1%}")
        if row["refusal_rate"] > 0.05:
            reasons.append(f"refusal_rate={row['refusal_rate']:.1%}")
        if row["missing_items"] > 0:
            reasons.append(f"missing {row['missing_items']}/44 items")
        if reasons:
            flags.append({**row.to_dict(), "flag_reason": "; ".join(reasons)})

    flagged = pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=list(summary.columns) + ["flag_reason"]
    )

    return {"model_summary": summary, "flagged_models": flagged}


# ===========================================================================
# Section 1b: Acquiescence Diagnostic
# ===========================================================================

def acquiescence_diagnostic(df_success: pd.DataFrame) -> pd.DataFrame:
    """Compare raw (pre-recode) forward vs reverse item means per dimension.

    A small gap between forward and reverse raw means indicates acquiescence
    (agreeing with both poles).  For a functioning scale, forward items should
    be high and reverse items low (or vice versa for low-trait respondents).

    Returns DataFrame with: dimension, fwd_raw_mean, rev_raw_mean, gap,
    n_fwd_items, n_rev_items, verdict.
    """
    bfi = df_success[_is_bfi(df_success)].copy()

    rows = []
    for dim in BFI_DIMENSIONS:
        d = bfi[bfi["dimension"] == dim]
        fwd = d[d["keying"] == "+"]
        rev = d[d["keying"] == "-"]

        fwd_mean = fwd["parsed_score"].mean() if len(fwd) > 0 else np.nan
        rev_mean = rev["parsed_score"].mean() if len(rev) > 0 else np.nan
        gap = fwd_mean - rev_mean if not (np.isnan(fwd_mean) or np.isnan(rev_mean)) else np.nan

        # Verdict: gap < 1.0 is concerning for a 1-5 scale
        if np.isnan(gap):
            verdict = "insufficient data"
        elif abs(gap) < 1.0:
            verdict = "ACQUIESCENCE — reverse items not functioning"
        elif abs(gap) < 1.5:
            verdict = "weak differentiation"
        else:
            verdict = "OK"

        rows.append({
            "dimension": dim,
            "fwd_raw_mean": fwd_mean,
            "rev_raw_mean": rev_mean,
            "gap": gap,
            "n_fwd_items": fwd["item_id"].nunique(),
            "n_rev_items": rev["item_id"].nunique(),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


def acquiescence_per_model(df_success: pd.DataFrame) -> pd.DataFrame:
    """Forward vs reverse raw means per model for Extraversion.

    This dimension shows the most severe acquiescence.
    """
    bfi = df_success[_is_bfi(df_success) & (df_success["dimension"] == "Extraversion")].copy()

    rows = []
    for model_id, mdf in bfi.groupby("model_id"):
        fwd = mdf[mdf["keying"] == "+"]["parsed_score"]
        rev = mdf[mdf["keying"] == "-"]["parsed_score"]
        fwd_mean = fwd.mean() if len(fwd) > 0 else np.nan
        rev_mean = rev.mean() if len(rev) > 0 else np.nan
        gap = fwd_mean - rev_mean if not (np.isnan(fwd_mean) or np.isnan(rev_mean)) else np.nan

        rows.append({
            "short_name": get_short_model_name(model_id),
            "E_fwd_raw": fwd_mean,
            "E_rev_raw": rev_mean,
            "gap": gap,
        })

    return pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)


# ===========================================================================
# Section 2: BFI Scoring
# ===========================================================================

def bfi_dimension_scores(df_success: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Big Five dimension scores per model.

    Returns (scores_table, sd_table):
    - scores_table: rows=models, columns=O/C/E/E_fwd/A/N
    - sd_table: rows=models, columns=O/C/E/E_fwd/A/N

    E uses all 8 items (with reverse coding).
    E_fwd uses only the 5 forward-keyed Extraversion items (no reverse coding
    needed), as a robustness check given acquiescence on reverse E items.
    """
    bfi = df_success[_is_bfi(df_success)].copy()

    score_rows = []
    sd_rows = []

    for model_id, mdf in bfi.groupby("model_id"):
        score_entry = {"model_id": model_id, "short_name": get_short_model_name(model_id)}
        sd_entry = {"model_id": model_id, "short_name": get_short_model_name(model_id)}

        for dim in BFI_DIMENSIONS:
            dim_data = mdf[mdf["dimension"] == dim]
            if len(dim_data) == 0:
                score_entry[BFI_DIM_SHORT[dim]] = np.nan
                sd_entry[BFI_DIM_SHORT[dim]] = np.nan
                continue

            # Per-run dimension score: mean of item scores within the run
            run_scores = dim_data.groupby("run_number")["score"].mean()
            score_entry[BFI_DIM_SHORT[dim]] = run_scores.mean()
            sd_entry[BFI_DIM_SHORT[dim]] = run_scores.std(ddof=1)

        # E_fwd: forward-keyed Extraversion items only (score = parsed_score, no recode needed)
        e_fwd = mdf[(mdf["dimension"] == "Extraversion") & (mdf["keying"] == "+")]
        if len(e_fwd) > 0:
            run_scores_fwd = e_fwd.groupby("run_number")["score"].mean()
            score_entry["E_fwd"] = run_scores_fwd.mean()
            sd_entry["E_fwd"] = run_scores_fwd.std(ddof=1)
        else:
            score_entry["E_fwd"] = np.nan
            sd_entry["E_fwd"] = np.nan

        score_rows.append(score_entry)
        sd_rows.append(sd_entry)

    scores = pd.DataFrame(score_rows).sort_values("short_name").reset_index(drop=True)
    sds = pd.DataFrame(sd_rows).sort_values("short_name").reset_index(drop=True)
    return scores, sds


# ===========================================================================
# Section 3: Reliability
# ===========================================================================

def _alpha_from_matrix(obs: pd.DataFrame) -> float:
    """Compute Cronbach's alpha from an observations × items matrix."""
    k = obs.shape[1]
    n = obs.shape[0]
    if k < 2 or n < 2:
        return np.nan
    item_vars = obs.var(axis=0, ddof=1)
    total_var = obs.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def bfi_reliability_model_means(df_success: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha per BFI dimension using the model-level means matrix.

    This is the PRIMARY reliability metric.  Each row is a model (N = number
    of models), each column is an item, values are mean scores across runs.
    This matches the actual unit of analysis for the MTMM.

    Also computes E_fwd (forward-keyed Extraversion items only).
    """
    bfi = df_success[_is_bfi(df_success)].copy()

    # Build model × item means matrix
    means = bfi.groupby(["model_id", "item_id", "dimension", "keying"]).agg(
        mean_score=("score", "mean"),
    ).reset_index()

    rows = []
    for dim in BFI_DIMENSIONS:
        dim_means = means[means["dimension"] == dim]
        mat = dim_means.pivot_table(index="model_id", columns="item_id", values="mean_score").dropna()

        alpha = _alpha_from_matrix(mat)
        rows.append({
            "dimension": dim,
            "short": BFI_DIM_SHORT[dim],
            "n_items": mat.shape[1],
            "n_models": mat.shape[0],
            "alpha": alpha,
            "flagged": alpha < 0.70 if not np.isnan(alpha) else True,
        })

    # E_fwd: forward-keyed Extraversion items only
    e_fwd = means[(means["dimension"] == "Extraversion") & (means["keying"] == "+")]
    mat_fwd = e_fwd.pivot_table(index="model_id", columns="item_id", values="mean_score").dropna()
    alpha_fwd = _alpha_from_matrix(mat_fwd)
    rows.append({
        "dimension": "Extraversion (fwd-only)",
        "short": "E_fwd",
        "n_items": mat_fwd.shape[1],
        "n_models": mat_fwd.shape[0],
        "alpha": alpha_fwd,
        "flagged": alpha_fwd < 0.70 if not np.isnan(alpha_fwd) else True,
    })

    return pd.DataFrame(rows)


def bfi_reliability_pooled(df_success: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha per BFI dimension, pooled across models.

    SECONDARY metric (provided for comparison).  Each (model, run) is one
    observation row — but note that within-model runs are near-identical for
    most LLMs, so this inflates N without adding information.
    """
    bfi = df_success[_is_bfi(df_success)].copy()

    rows = []
    for dim in BFI_DIMENSIONS:
        dim_data = bfi[bfi["dimension"] == dim]
        if len(dim_data) == 0:
            rows.append({"dimension": dim, "short": BFI_DIM_SHORT[dim],
                         "n_items": 0, "alpha": np.nan, "flagged": True})
            continue

        obs = dim_data.pivot_table(
            index=["model_id", "run_number"],
            columns="item_id",
            values="score",
        ).dropna()

        alpha = _alpha_from_matrix(obs)

        rows.append({
            "dimension": dim,
            "short": BFI_DIM_SHORT[dim],
            "n_items": obs.shape[1],
            "n_obs": obs.shape[0],
            "alpha": alpha,
            "flagged": alpha < 0.70 if not np.isnan(alpha) else True,
        })

    return pd.DataFrame(rows)


def bfi_reliability_per_model(df_success: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha per BFI dimension per model.

    Only includes models with complete data (all 44 items observed).
    """
    bfi = df_success[_is_bfi(df_success)].copy()

    rows = []
    for model_id, mdf in bfi.groupby("model_id"):
        n_items = mdf["item_id"].nunique()
        if n_items < 44:
            continue

        for dim in BFI_DIMENSIONS:
            dim_data = mdf[mdf["dimension"] == dim]
            obs = dim_data.pivot_table(
                index="run_number", columns="item_id", values="score"
            ).dropna()

            item_ids = sorted(obs.columns.tolist())
            k = len(item_ids)
            n = len(obs)

            if k < 2 or n < 2:
                alpha = np.nan
            else:
                item_vars = obs[item_ids].var(axis=0, ddof=1)
                total_var = obs[item_ids].sum(axis=1).var(ddof=1)
                alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else np.nan

            rows.append({
                "model_id": model_id,
                "short_name": get_short_model_name(model_id),
                "dimension": dim,
                "short": BFI_DIM_SHORT[dim],
                "n_runs": n,
                "alpha": alpha,
            })

    return pd.DataFrame(rows)


# ===========================================================================
# Section 4: Visualizations
# ===========================================================================

def plot_radar_profiles(scores: pd.DataFrame, output_path: str) -> None:
    """Radar/spider plot of Big Five z-score profiles per model.

    Uses E_fwd instead of E for Extraversion, since full-scale E is
    compromised by acquiescence on reverse-coded items.
    """
    dims = ["O", "C", "E_fwd", "A", "N"]
    dim_labels = ["Openness", "Conscientiousness", "Extraversion\n(fwd-only)", "Agreeableness", "Neuroticism"]

    # Compute z-scores across models
    z_scores = scores.copy()
    for d in dims:
        col = z_scores[d]
        z_scores[d] = (col - col.mean()) / col.std(ddof=1) if col.std(ddof=1) > 0 else 0

    n_models = len(z_scores)
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              subplot_kw=dict(polar=True))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.cm.tab20
    for idx, (_, row) in enumerate(z_scores.iterrows()):
        ax = axes[idx]
        values = [row[d] for d in dims]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, color=cmap(idx % 20))
        ax.fill(angles, values, alpha=0.15, color=cmap(idx % 20))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=7)
        ax.set_ylim(-3, 3)
        ax.set_title(row["short_name"], fontsize=9, pad=12)
        ax.tick_params(axis="y", labelsize=6)

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Big Five Profiles (z-scores across models)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile_correlation_heatmap(scores: pd.DataFrame, output_path: str) -> None:
    """Heatmap of inter-model profile correlations (uses E_fwd)."""
    dims = ["O", "C", "E_fwd", "A", "N"]
    matrix = scores.set_index("short_name")[dims]
    corr = matrix.T.corr()  # model × model correlation

    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.5), max(7, len(corr) * 0.45)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=len(corr) <= 25, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        xticklabels=corr.columns, yticklabels=corr.index,
    )
    ax.set_title("Inter-Model Big Five Profile Correlations", fontsize=12)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_within_model_sd(sd_table: pd.DataFrame, output_path: str) -> None:
    """Bar chart of within-model SD per BFI dimension."""
    dims = ["O", "C", "E_fwd", "A", "N"]
    dim_names = ["Openness", "Conscientiousness", "Extraversion (fwd)", "Agreeableness", "Neuroticism"]

    fig, ax = plt.subplots(figsize=(max(10, len(sd_table) * 0.5), 5))

    x = np.arange(len(sd_table))
    width = 0.15
    offsets = np.arange(len(dims)) - (len(dims) - 1) / 2

    colors = sns.color_palette("Set2", len(dims))
    for i, (d, name) in enumerate(zip(dims, dim_names)):
        vals = sd_table[d].fillna(0).values
        ax.bar(x + offsets[i] * width, vals, width, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(sd_table["short_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Within-Model SD (across runs)")
    ax.set_title("Run-to-Run Variability by BFI Dimension")
    ax.legend(fontsize=8, ncol=len(dims))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section 5: Convergent / Discriminant Preview
# ===========================================================================

def convergent_discriminant_preview(
    df_success: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlate BFI dimension scores with AI-native dimension scores across models.

    Returns (corr_matrix, detail_df):
    - corr_matrix: rows = AI-native dims, columns = BFI traits (O/C/E/A/N),
      values = Pearson r
    - detail_df: one row per (AI-native dim, BFI dim) pair with r, p, CI, N
    """
    # --- BFI model-level means ---
    bfi = df_success[_is_bfi(df_success)].copy()
    bfi_scores = {}
    for model_id, mdf in bfi.groupby("model_id"):
        for dim in BFI_DIMENSIONS:
            dim_data = mdf[mdf["dimension"] == dim]
            if len(dim_data) > 0:
                bfi_scores.setdefault(model_id, {})[BFI_DIM_SHORT[dim]] = dim_data["score"].mean()
        # E_fwd: forward-keyed Extraversion only
        e_fwd = mdf[(mdf["dimension"] == "Extraversion") & (mdf["keying"] == "+")]
        if len(e_fwd) > 0:
            bfi_scores.setdefault(model_id, {})["E_fwd"] = e_fwd["score"].mean()
    bfi_df = pd.DataFrame.from_dict(bfi_scores, orient="index")

    # --- AI-native model-level means ---
    ai = df_success[_is_ai_native(df_success)].copy()
    ai_scores = {}
    for model_id, mdf in ai.groupby("model_id"):
        for dim in AI_NATIVE_DIMENSIONS:
            dim_data = mdf[mdf["dimension"] == dim]
            if len(dim_data) > 0:
                ai_scores.setdefault(model_id, {})[dim] = dim_data["score"].mean()
    ai_df = pd.DataFrame.from_dict(ai_scores, orient="index")

    # Models with both
    common_models = sorted(set(bfi_df.index) & set(ai_df.index))
    n_models = len(common_models)

    if n_models < 4:
        empty_corr = pd.DataFrame(
            np.nan, index=AI_NATIVE_DIMENSIONS,
            columns=list("OCEAN")
        )
        return empty_corr, pd.DataFrame()

    bfi_common = bfi_df.loc[common_models]
    ai_common = ai_df.loc[common_models]

    detail_rows = []
    corr_vals = {}
    for ai_dim in AI_NATIVE_DIMENSIONS:
        if ai_dim not in ai_common.columns:
            continue
        corr_vals[ai_dim] = {}
        ai_col = ai_common[ai_dim].dropna()
        for bfi_dim in ["O", "C", "E", "E_fwd", "A", "N"]:
            if bfi_dim not in bfi_common.columns:
                continue
            bfi_col = bfi_common[bfi_dim].dropna()
            shared = ai_col.index.intersection(bfi_col.index)
            n = len(shared)
            if n < 4:
                corr_vals[ai_dim][bfi_dim] = np.nan
                continue

            r, p = stats.pearsonr(ai_col[shared], bfi_col[shared])
            corr_vals[ai_dim][bfi_dim] = r

            # Fisher z CI
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            ci_lo = np.tanh(z - 1.96 * se)
            ci_hi = np.tanh(z + 1.96 * se)

            detail_rows.append({
                "ai_native_dim": ai_dim,
                "bfi_dim": bfi_dim,
                "r": r,
                "p": p,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_models": n,
                "notable": abs(r) > 0.50,
            })

    corr_matrix = pd.DataFrame(corr_vals).T
    # Reorder columns
    corr_matrix = corr_matrix.reindex(columns=["O", "C", "E", "E_fwd", "A", "N"])
    # Keep only AI-native dims that are present
    present_ai_dims = [d for d in AI_NATIVE_DIMENSIONS if d in corr_matrix.index]
    corr_matrix = corr_matrix.loc[present_ai_dims]

    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    return corr_matrix, detail_df


def plot_convergent_discriminant_heatmap(
    corr_matrix: pd.DataFrame, output_path: str
) -> None:
    """Heatmap of BFI × AI-native dimension correlations."""
    if corr_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(6, len(corr_matrix) * 0.45)))

    # Short labels for AI-native dims
    y_labels = [d.replace("/", "/\n") for d in corr_matrix.index]
    x_labels = [{"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
                  "E_fwd": "Extraversion\n(fwd-only)", "A": "Agreeableness",
                  "N": "Neuroticism"}.get(c, c) for c in corr_matrix.columns]

    sns.heatmap(
        corr_matrix.values.astype(float),
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=x_labels, yticklabels=y_labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("BFI × AI-Native Dimension Correlations\n(model-level means)", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Report generation
# ===========================================================================

def generate_bfi_report(
    engineering: dict,
    acquiescence: pd.DataFrame,
    acquiescence_e: pd.DataFrame,
    scores: pd.DataFrame,
    sd_table: pd.DataFrame,
    reliability_model_means: pd.DataFrame,
    reliability_pooled: pd.DataFrame,
    reliability_per_model: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_detail: pd.DataFrame,
    plot_paths: dict,
    output_path: str,
) -> None:
    """Write the BFI analysis report as markdown."""
    report_dir = str(Path(output_path).parent)
    lines = []
    lines.append("# BFI-44 Analysis Report")
    lines.append("")
    lines.append("*Auto-generated by `python -m analysis.bfi_analysis`*")
    lines.append("")

    # --- Section 1: Engineering Checks ---
    lines.append("## 1. Engineering Checks (BFI-Specific)")
    lines.append("")
    summary = engineering["model_summary"]
    display_cols = ["short_name", "total_bfi_calls", "success", "parse_error",
                    "refusal", "api_error", "success_rate", "parse_error_rate",
                    "refusal_rate", "n_items", "missing_items", "max_run"]
    lines.append(df_to_markdown(summary[display_cols]))
    lines.append("")

    flagged = engineering["flagged_models"]
    if len(flagged) > 0:
        lines.append("### Flagged Models")
        lines.append("")
        flag_cols = ["short_name", "flag_reason"]
        lines.append(df_to_markdown(flagged[flag_cols]))
        lines.append("")
    else:
        lines.append("No models flagged.")
        lines.append("")

    # --- Section 1b: Acquiescence Diagnostic ---
    lines.append("### Acquiescence Diagnostic")
    lines.append("")
    lines.append("Forward vs reverse raw item means per dimension (before reverse coding). "
                 "A small gap indicates acquiescence — models agreeing with both poles.")
    lines.append("")
    lines.append(df_to_markdown(acquiescence))
    lines.append("")

    acq_broken = acquiescence[acquiescence["verdict"].str.startswith("ACQUIESCENCE")]
    if len(acq_broken) > 0:
        broken_dims = ", ".join(acq_broken["dimension"].tolist())
        lines.append(f"**Reverse items not functioning:** {broken_dims}")
        lines.append("")

    lines.append("#### Extraversion Forward vs Reverse Per Model")
    lines.append("")
    lines.append(df_to_markdown(acquiescence_e))
    lines.append("")

    # --- Section 2: BFI Scoring ---
    lines.append("## 2. Big Five Dimension Scores")
    lines.append("")
    lines.append("Mean scores per model (1–5 scale, reverse-coded items flipped). "
                 "E_fwd = Extraversion scored from forward-keyed items only "
                 "(recommended due to acquiescence on reverse E items).")
    lines.append("")
    score_cols = ["short_name", "O", "C", "E", "E_fwd", "A", "N"]
    lines.append(df_to_markdown(scores[score_cols]))
    lines.append("")

    lines.append("### Within-Model Variability (SD Across Runs)")
    lines.append("")
    lines.append(df_to_markdown(sd_table[score_cols]))
    lines.append("")

    sd_plot = plot_paths.get("within_model_sd")
    if sd_plot:
        rel = str(Path(sd_plot).relative_to(Path(report_dir)))
        lines.append(f"![Within-model SD]({rel})")
        lines.append("")

    # --- Section 3: Reliability ---
    lines.append("## 3. Reliability")
    lines.append("")

    lines.append("### Model-Means Alpha (PRIMARY)")
    lines.append("")
    lines.append("Cronbach's alpha computed on the model-level means matrix "
                 "(rows = models, columns = items). This is the correct unit "
                 "of analysis for the MTMM, where each model contributes one "
                 "score per dimension.")
    lines.append("")
    lines.append(df_to_markdown(reliability_model_means))
    lines.append("")
    flagged_mm = reliability_model_means[reliability_model_means["flagged"]]
    if len(flagged_mm) > 0:
        dim_list = ", ".join(flagged_mm["dimension"].tolist())
        lines.append(f"**Flagged (alpha < 0.70):** {dim_list}")
        lines.append("")

    lines.append("### Pooled Alpha (secondary, for comparison)")
    lines.append("")
    lines.append("Each (model, run) is one observation. Note: inflated N due "
                 "to near-deterministic within-model responding.")
    lines.append("")
    lines.append(df_to_markdown(reliability_pooled))
    lines.append("")

    if len(reliability_per_model) > 0:
        lines.append("### Per-Model Alpha (informational)")
        lines.append("")
        lines.append("Near-zero values expected for deterministic responders — "
                     "this metric is not meaningful when within-model variance ≈ 0.")
        lines.append("")
        pm_wide = reliability_per_model.pivot_table(
            index="short_name", columns="short", values="alpha"
        ).reindex(columns=list("OCEAN")).reset_index()
        pm_wide.columns.name = None
        lines.append(df_to_markdown(pm_wide))
        lines.append("")

    # --- Section 4: Visualizations ---
    lines.append("## 4. BFI Profile Visualizations")
    lines.append("")
    lines.append("All profile plots use E_fwd (forward-keyed Extraversion) "
                 "instead of full-scale E.")
    lines.append("")

    radar_plot = plot_paths.get("radar")
    if radar_plot:
        rel = str(Path(radar_plot).relative_to(Path(report_dir)))
        lines.append("### Radar Profiles (z-scores)")
        lines.append(f"![Radar profiles]({rel})")
        lines.append("")

    corr_heatmap = plot_paths.get("profile_corr")
    if corr_heatmap:
        rel = str(Path(corr_heatmap).relative_to(Path(report_dir)))
        lines.append("### Inter-Model Profile Correlations")
        lines.append(f"![Profile correlations]({rel})")
        lines.append("")

    # --- Section 5: Convergent / Discriminant ---
    lines.append("## 5. Convergent / Discriminant Preview")
    lines.append("")
    lines.append("Pearson correlations between BFI dimension scores and AI-native "
                 "candidate dimension scores across models (model-level means). "
                 "Both E (full, compromised) and E_fwd (forward-only, recommended) "
                 "are shown.")
    lines.append("")

    cd_plot = plot_paths.get("convergent_discriminant")
    if cd_plot:
        rel = str(Path(cd_plot).relative_to(Path(report_dir)))
        lines.append(f"![Convergent/discriminant heatmap]({rel})")
        lines.append("")

    if len(corr_detail) > 0:
        n_models = corr_detail["n_models"].iloc[0]
        lines.append(f"N = {n_models} models with both BFI and AI-native data.")
        lines.append("")

        notable = corr_detail[corr_detail["notable"]]
        if len(notable) > 0:
            lines.append("### Notable Correlations (|r| > 0.50)")
            lines.append("")
            notable_display = notable[["ai_native_dim", "bfi_dim", "r", "p",
                                       "ci_lo", "ci_hi", "n_models"]].copy()
            notable_display = notable_display.sort_values("r", key=abs, ascending=False)
            lines.append(df_to_markdown(notable_display))
            lines.append("")
        else:
            lines.append("No correlations exceeded |r| > 0.50.")
            lines.append("")

        lines.append("### Full Correlation Detail")
        lines.append("")
        full = corr_detail[["ai_native_dim", "bfi_dim", "r", "ci_lo", "ci_hi",
                            "n_models"]].sort_values(["ai_native_dim", "bfi_dim"])
        lines.append(df_to_markdown(full))
        lines.append("")

        lines.append("**Note:** With N = {} models, statistical power is limited. "
                     "Effect sizes and confidence intervals are more informative "
                     "than p-values at this sample size.".format(n_models))
        lines.append("")
    else:
        lines.append("Insufficient overlapping models to compute correlations.")
        lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"BFI report written to {output_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run BFI-44 analysis")
    parser.parse_args()

    start = time.time()
    ensure_output_dirs()
    plots_dir = str(PLOTS_DIR)
    output_dir = str(OUTPUT_DIR)
    plot_paths = {}

    # Load data
    print("Loading data...")
    df_all = load_responses()
    df_bfi_all = df_all[_is_bfi(df_all)]
    print(f"  {len(df_bfi_all)} BFI rows across {df_bfi_all['model_id'].nunique()} models")

    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)

    # Section 1: Engineering checks
    print("\n=== Section 1: BFI Engineering Checks ===")
    engineering = bfi_engineering_checks(df_all)
    n_flagged = len(engineering["flagged_models"])
    print(f"  {len(engineering['model_summary'])} models, {n_flagged} flagged")

    # Section 1b: Acquiescence diagnostic
    print("\n=== Section 1b: Acquiescence Diagnostic ===")
    acq = acquiescence_diagnostic(df_success)
    for _, row in acq.iterrows():
        print(f"  {row['dimension']}: fwd={row['fwd_raw_mean']:.2f} rev={row['rev_raw_mean']:.2f} "
              f"gap={row['gap']:+.2f} → {row['verdict']}")
    acq_e = acquiescence_per_model(df_success)

    # Section 2: BFI scoring
    print("\n=== Section 2: BFI Scoring ===")
    scores, sd_table = bfi_dimension_scores(df_success)
    print(f"  Dimension scores for {len(scores)} models")
    for d in SCORE_DIMS:
        col = scores[d].dropna()
        if len(col) > 0:
            print(f"    {d}: mean={col.mean():.2f}, range=[{col.min():.2f}, {col.max():.2f}]")

    # Section 3: Reliability
    print("\n=== Section 3: Reliability ===")

    print("  Model-means alpha (PRIMARY):")
    rel_model_means = bfi_reliability_model_means(df_success)
    for _, row in rel_model_means.iterrows():
        flag = " *** FLAGGED" if row["flagged"] else ""
        print(f"    {row['dimension']}: α={row['alpha']:.3f} (N={row['n_models']} models, "
              f"k={row['n_items']} items){flag}")

    print("  Pooled alpha (secondary):")
    rel_pooled = bfi_reliability_pooled(df_success)
    for _, row in rel_pooled.iterrows():
        flag = " *** FLAGGED" if row["flagged"] else ""
        print(f"    {row['dimension']}: α={row['alpha']:.3f} (n_obs={row['n_obs']}){flag}")

    rel_per_model = bfi_reliability_per_model(df_success)
    n_complete = rel_per_model["model_id"].nunique() if len(rel_per_model) > 0 else 0
    print(f"  Per-model alpha: {n_complete} models with complete data")

    # Section 4: Visualizations
    print("\n=== Section 4: Visualizations ===")

    radar_path = f"{plots_dir}/bfi_radar_profiles.png"
    plot_radar_profiles(scores, radar_path)
    plot_paths["radar"] = radar_path
    print(f"  Radar plot: {radar_path}")

    corr_heatmap_path = f"{plots_dir}/bfi_profile_correlations.png"
    plot_profile_correlation_heatmap(scores, corr_heatmap_path)
    plot_paths["profile_corr"] = corr_heatmap_path
    print(f"  Profile correlation heatmap: {corr_heatmap_path}")

    sd_plot_path = f"{plots_dir}/bfi_within_model_sd.png"
    plot_within_model_sd(sd_table, sd_plot_path)
    plot_paths["within_model_sd"] = sd_plot_path
    print(f"  Within-model SD bar chart: {sd_plot_path}")

    # Section 5: Convergent / discriminant
    print("\n=== Section 5: Convergent / Discriminant Preview ===")
    corr_matrix, corr_detail = convergent_discriminant_preview(df_success)
    if len(corr_detail) > 0:
        n_notable = corr_detail["notable"].sum()
        print(f"  {len(corr_detail)} pairs computed, {n_notable} with |r| > 0.50")

        cd_path = f"{plots_dir}/bfi_convergent_discriminant.png"
        plot_convergent_discriminant_heatmap(corr_matrix, cd_path)
        plot_paths["convergent_discriminant"] = cd_path
        print(f"  Heatmap: {cd_path}")
    else:
        print("  Insufficient data for convergent/discriminant analysis")

    # Generate report
    print("\n=== Generating Report ===")
    report_path = f"{output_dir}/bfi_report.md"
    generate_bfi_report(
        engineering=engineering,
        acquiescence=acq,
        acquiescence_e=acq_e,
        scores=scores,
        sd_table=sd_table,
        reliability_model_means=rel_model_means,
        reliability_pooled=rel_pooled,
        reliability_per_model=rel_per_model,
        corr_matrix=corr_matrix,
        corr_detail=corr_detail,
        plot_paths=plot_paths,
        output_path=report_path,
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
