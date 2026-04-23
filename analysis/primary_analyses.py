"""Week 2 primary analyses: EFA → item selection → CFA → reliability → profiles → validity.

Usage:
    python -m analysis.primary_analyses
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

from .data_loader import (
    DB_PATH,
    DIMENSION_CODES,
    OUTPUT_DIR,
    PLOTS_DIR,
    compute_model_item_means,
    ensure_output_dirs,
    filter_success,
    get_models_for_section,
    get_short_model_name,
    load_responses,
    pivot_score_matrix,
    recode_reverse_items,
)
from .dimension_coherence import cronbachs_alpha
from .factor_structure import (
    build_pooled_matrix,
    compute_icc,
    parallel_analysis,
    plot_icc_distribution,
    plot_scree,
    _weighted_corr,
)
from .bfi_analysis import (
    bfi_dimension_scores,
    convergent_discriminant_preview,
    BFI_DIMENSIONS,
    BFI_DIM_SHORT,
    AI_NATIVE_DIMENSIONS,
    _is_bfi,
    _is_ai_native,
)
from .report import df_to_markdown

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFA_RUNS = list(range(1, 16))   # runs 1-15 for exploration
CFA_RUNS = list(range(16, 31))  # runs 16-30 for confirmation

PRIMARY_LOADING_THRESHOLD = 0.40
CROSS_LOADING_THRESHOLD = 0.30

# Forced 5-factor solution — selected after systematic k=5–9 comparison (2026-03-26).
# Best ESEM CFI (0.935), most retained items (96/240), cleanest factor balance.
FORCED_N_FACTORS = 5
FACTOR_LABELS = {
    "Factor1": "Responsiveness",
    "Factor2": "Deference",
    "Factor3": "Boldness",
    "Factor4": "Guardedness",
    "Factor5": "Verbosity",
}


# ---------------------------------------------------------------------------
# 0. Data preparation with split-half
# ---------------------------------------------------------------------------

def split_half_data(
    df_success: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into EFA half (runs 1-15) and CFA half (runs 16-30).

    Excludes BFI items — EFA/CFA operate on AI-native items only.
    """
    ai_only = df_success[_is_ai_native(df_success)]
    efa_df = ai_only[ai_only["run_number"].isin(EFA_RUNS)].copy()
    cfa_df = ai_only[ai_only["run_number"].isin(CFA_RUNS)].copy()
    return efa_df, cfa_df


# ---------------------------------------------------------------------------
# 1. EFA on exploration half
# ---------------------------------------------------------------------------

def run_efa_exploration(
    efa_df: pd.DataFrame,
    eligible_models: list[str],
    plots_dir: str,
    forced_n_factors: int | None = None,
) -> dict:
    """Run EFA on the exploration half (runs 1-15).

    Steps: build pooled matrix → parallel analysis → PAF with oblimin.
    If forced_n_factors is set, uses that instead of parallel analysis suggestion.
    """
    print("\n=== 1. EFA on Exploration Half (runs 1-15) ===")

    # Build pooled matrix from EFA half only
    obs_matrix, weights = build_pooled_matrix(efa_df, eligible_models, "direct")
    print(f"  Pooled matrix: {obs_matrix.shape[0]} obs × {obs_matrix.shape[1]} items "
          f"from {len(eligible_models)} models")

    # KMO and Bartlett's test
    filled = obs_matrix.fillna(obs_matrix.mean())
    try:
        kmo_all, kmo_model = calculate_kmo(filled)
        print(f"  KMO: {kmo_model:.3f}")
    except Exception:
        kmo_model = None
        print("  KMO: could not compute")

    try:
        chi2, p_val = calculate_bartlett_sphericity(filled)
        print(f"  Bartlett's test: χ²={chi2:.1f}, p={p_val:.2e}")
    except Exception:
        chi2, p_val = None, None

    # Parallel analysis
    print("  Running parallel analysis (1000 iterations)...")
    pa = parallel_analysis(obs_matrix, weights)
    n_factors = pa["n_factors_suggested"]
    print(f"  Parallel analysis suggests {n_factors} factors")

    scree_path = f"{plots_dir}/efa_scree_parallel_analysis.png"
    plot_scree(pa["real_eigenvalues"], pa["random_eigenvalues_95"], n_factors, scree_path)

    if forced_n_factors is not None:
        n_factors_pa = n_factors  # save parallel analysis suggestion
        n_factors = forced_n_factors
        print(f"  Parallel analysis suggested {n_factors_pa}; FORCING {n_factors} factors")
    else:
        if n_factors < 1:
            n_factors = 12
            print(f"  Fallback: using {n_factors} factors")
        n_factors = min(n_factors, 20)

    # EFA with PAF + oblimin
    print(f"  Running EFA with {n_factors} factors (PAF + oblimin)...")
    sqrt_w = np.sqrt(weights)
    wmean = np.average(filled.values, axis=0, weights=weights)
    weighted_centered = (filled.values - wmean) * sqrt_w[:, np.newaxis]
    weighted_df = pd.DataFrame(weighted_centered, columns=filled.columns)

    method_used = "minres"
    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)
        loadings = fa.loadings_
    except Exception as e:
        method_used = "principal_fallback"
        print(f"  minres failed ({e}), falling back to principal...")
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)
        loadings = fa.loadings_

    loadings_df = pd.DataFrame(
        loadings, index=obs_matrix.columns,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    factor_corr = None
    if hasattr(fa, "phi_") and fa.phi_ is not None:
        factor_corr = pd.DataFrame(
            fa.phi_, index=loadings_df.columns, columns=loadings_df.columns,
        )

    # Communalities
    communalities = fa.get_communalities()
    comm_df = pd.DataFrame({
        "item_id": obs_matrix.columns,
        "communality": communalities,
    })

    # Eigenvalues (common factor)
    eigenvalues = fa.get_eigenvalues()

    # Variance explained
    var_explained = fa.get_factor_variance()

    print(f"  EFA method: {method_used}")

    return {
        "obs_matrix": obs_matrix,
        "weights": weights,
        "n_factors": n_factors,
        "parallel_analysis": pa,
        "loadings": loadings_df,
        "factor_correlation": factor_corr,
        "communalities": comm_df,
        "eigenvalues": eigenvalues,
        "variance_explained": var_explained,
        "method_used": method_used,
        "kmo": kmo_model,
        "bartlett_chi2": chi2,
        "bartlett_p": p_val,
        "scree_plot": scree_path,
    }


# ---------------------------------------------------------------------------
# 2. Item selection
# ---------------------------------------------------------------------------

def select_items(
    loadings_df: pd.DataFrame,
    means_df: pd.DataFrame,
    communalities_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Select items based on EFA loadings.

    Drops items with:
    - Primary loading < 0.40
    - Cross-loading > 0.30

    Returns (report_df, retained_items, dropped_items).
    """
    print("\n=== 2. Item Selection ===")

    dim_map = (
        means_df[["item_id", "dimension"]].drop_duplicates()
        .set_index("item_id")["dimension"]
    )

    rows = []
    for item_id in loadings_df.index:
        abs_row = loadings_df.loc[item_id].abs()
        sorted_vals = abs_row.sort_values(ascending=False)
        primary_factor = sorted_vals.index[0]
        primary_loading = sorted_vals.iloc[0]
        cross_loading = sorted_vals.iloc[1] if len(sorted_vals) > 1 else 0.0

        # Use signed loading for primary
        signed_loading = loadings_df.loc[item_id, primary_factor]

        flags = []
        if primary_loading < PRIMARY_LOADING_THRESHOLD:
            flags.append(f"low_primary ({primary_loading:.2f})")
        if cross_loading >= CROSS_LOADING_THRESHOLD:
            flags.append(f"cross_load ({sorted_vals.index[1]}={cross_loading:.2f})")

        comm = communalities_df.loc[
            communalities_df["item_id"] == item_id, "communality"
        ]
        comm_val = comm.iloc[0] if len(comm) > 0 else np.nan

        rows.append({
            "item_id": item_id,
            "dimension": dim_map.get(item_id, "unknown"),
            "primary_factor": primary_factor,
            "primary_loading": signed_loading,
            "abs_primary_loading": primary_loading,
            "cross_loading": cross_loading,
            "communality": comm_val,
            "flag": "; ".join(flags) if flags else "",
            "retained": len(flags) == 0,
        })

    report = pd.DataFrame(rows)
    retained = report[report["retained"]]["item_id"].tolist()
    dropped = report[~report["retained"]]["item_id"].tolist()

    n_total = len(report)
    n_retained = len(retained)
    n_dropped = len(dropped)
    print(f"  {n_retained}/{n_total} direct items retained, {n_dropped} dropped")

    # Summary by dimension
    dim_counts = report.groupby("dimension").agg(
        total=("retained", "count"),
        retained=("retained", "sum"),
    ).reset_index()
    dim_counts["dropped"] = dim_counts["total"] - dim_counts["retained"]
    print("  Per-dimension retention:")
    for _, r in dim_counts.iterrows():
        print(f"    {r['dimension']}: {int(r['retained'])}/{int(r['total'])}")

    # Summary by factor
    factor_counts = report[report["retained"]].groupby("primary_factor").size()
    print(f"  Items per factor (retained): {dict(factor_counts)}")

    return report, retained, dropped


# ---------------------------------------------------------------------------
# 3. CFA on confirmation half
# ---------------------------------------------------------------------------

def run_cfa_confirmation(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    retained_items: list[str],
    item_selection_report: pd.DataFrame,
) -> dict:
    """Run CFA on the confirmation half (runs 16-30) using semopy.

    Builds the measurement model from EFA-derived factor → item assignments.
    Reports fit indices: CFI, TLI, RMSEA, SRMR.
    """
    import semopy

    print("\n=== 3. CFA on Confirmation Half (runs 16-30) ===")

    # Build pooled matrix from CFA half
    obs_matrix, weights = build_pooled_matrix(cfa_df, eligible_models, "direct")

    # Restrict to retained items that exist in the CFA matrix
    available = [c for c in retained_items if c in obs_matrix.columns]
    obs_cfa = obs_matrix[available].copy()
    print(f"  CFA matrix: {obs_cfa.shape[0]} obs × {obs_cfa.shape[1]} items")

    # Fill NaN with column means for CFA
    obs_cfa = obs_cfa.fillna(obs_cfa.mean())

    # Build factor → items mapping from item selection
    retained_report = item_selection_report[
        item_selection_report["item_id"].isin(available)
    ]
    factor_items = {}
    for _, row in retained_report.iterrows():
        factor = row["primary_factor"]
        factor_items.setdefault(factor, []).append(row["item_id"])

    # Drop factors with < 3 items (can't estimate)
    factor_items = {f: items for f, items in factor_items.items() if len(items) >= 3}
    print(f"  Factors with ≥3 items: {len(factor_items)}")
    for f, items in sorted(factor_items.items()):
        print(f"    {f}: {len(items)} items")

    if len(factor_items) < 2:
        print("  ERROR: Fewer than 2 factors with ≥3 items — cannot run CFA")
        return {"error": "Insufficient factors for CFA"}

    # Sanitize item IDs for lavaan-style syntax (replace hyphens with underscores)
    rename_map = {item: item.replace("-", "_") for item in available}
    obs_cfa_renamed = obs_cfa.rename(columns=rename_map)

    # Build model specification (lavaan-style syntax for semopy)
    model_lines = []
    factor_names = []
    for factor, items in sorted(factor_items.items()):
        fname = factor.replace(" ", "")  # e.g. "Factor1"
        factor_names.append(fname)
        sanitized_items = [rename_map[i] for i in items if i in rename_map]
        model_lines.append(f"{fname} =~ {' + '.join(sanitized_items)}")

    model_spec = "\n".join(model_lines)
    print(f"  Model specification: {len(model_lines)} factors")

    # Fit CFA
    try:
        model = semopy.Model(model_spec)
        result = model.fit(obs_cfa_renamed)
        fit_stats = semopy.calc_stats(model)

        # Extract key indices — semopy returns stats as columns, index='Value'
        fit_dict = {}
        for stat_name in ["CFI", "TLI", "RMSEA", "SRMR", "chi2", "DoF"]:
            if stat_name in fit_stats.columns:
                fit_dict[stat_name] = fit_stats.loc["Value", stat_name]
            else:
                # Case-insensitive fallback
                for col in fit_stats.columns:
                    if col.lower() == stat_name.lower():
                        fit_dict[stat_name] = fit_stats.loc["Value", col]
                        break

        print(f"  Fit indices:")
        for k, v in fit_dict.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # Get factor loadings from CFA
        estimates = model.inspect()

        return {
            "fit_stats": fit_stats,
            "fit_dict": fit_dict,
            "estimates": estimates,
            "model_spec": model_spec,
            "factor_items": factor_items,
            "n_obs": obs_cfa.shape[0],
            "n_items": obs_cfa.shape[1],
        }

    except Exception as e:
        print(f"  CFA failed: {e}")
        return {"error": str(e), "model_spec": model_spec, "factor_items": factor_items}


def run_cfa_trimmed(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    item_selection_report: pd.DataFrame,
    top_n: int = 6,
) -> dict:
    """Run CFA on a trimmed item set (top N items per factor by abs loading).

    Reduces model complexity for better fit while retaining strongest markers.
    """
    import semopy

    print(f"\n=== 3b. Trimmed CFA (top {top_n} items/factor) ===")

    # Select top N items per factor
    retained = item_selection_report[item_selection_report["retained"]].copy()
    trimmed_items = []
    trimmed_factor_items = {}
    for factor in sorted(retained["primary_factor"].unique()):
        factor_rows = retained[retained["primary_factor"] == factor]
        top = factor_rows.nlargest(top_n, "abs_primary_loading")
        items = top["item_id"].tolist()
        trimmed_items.extend(items)
        trimmed_factor_items[factor] = items
        label = FACTOR_LABELS.get(factor, factor)
        print(f"  {factor} [{label}]: {len(items)} items")

    print(f"  Total trimmed items: {len(trimmed_items)}")

    # Build CFA matrix
    obs_matrix, weights = build_pooled_matrix(cfa_df, eligible_models, "direct")
    available = [c for c in trimmed_items if c in obs_matrix.columns]
    obs_cfa = obs_matrix[available].fillna(obs_matrix[available].mean())
    print(f"  CFA matrix: {obs_cfa.shape[0]} obs × {obs_cfa.shape[1]} items")

    # Sanitize and build model spec
    rename_map = {item: item.replace("-", "_") for item in available}
    obs_cfa_renamed = obs_cfa.rename(columns=rename_map)

    model_lines = []
    for factor, items in sorted(trimmed_factor_items.items()):
        fname = factor.replace(" ", "")
        sanitized = [rename_map[i] for i in items if i in rename_map]
        if len(sanitized) >= 3:
            model_lines.append(f"{fname} =~ {' + '.join(sanitized)}")

    model_spec = "\n".join(model_lines)

    try:
        model = semopy.Model(model_spec)
        model.fit(obs_cfa_renamed)
        fit_stats = semopy.calc_stats(model)

        fit_dict = {}
        for stat_name in ["CFI", "TLI", "RMSEA", "SRMR", "chi2", "DoF"]:
            if stat_name in fit_stats.columns:
                fit_dict[stat_name] = fit_stats.loc["Value", stat_name]
            else:
                for col in fit_stats.columns:
                    if col.lower() == stat_name.lower():
                        fit_dict[stat_name] = fit_stats.loc["Value", col]
                        break

        print(f"  Trimmed CFA fit:")
        for k, v in fit_dict.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        return {
            "fit_dict": fit_dict,
            "fit_stats": fit_stats,
            "model_spec": model_spec,
            "factor_items": trimmed_factor_items,
            "n_obs": obs_cfa.shape[0],
            "n_items": obs_cfa.shape[1],
            "top_n": top_n,
        }
    except Exception as e:
        print(f"  Trimmed CFA failed: {e}")
        return {"error": str(e), "model_spec": model_spec, "top_n": top_n}


# ---------------------------------------------------------------------------
# 4. Reliability
# ---------------------------------------------------------------------------

def compute_mcdonalds_omega(
    score_matrix: pd.DataFrame,
    item_ids: list[str],
) -> float:
    """Compute McDonald's omega (hierarchical) for a set of items.

    Uses single-factor model to estimate omega = (sum of loadings)^2 / total var.
    """
    sub = score_matrix[[c for c in item_ids if c in score_matrix.columns]].dropna()
    k = sub.shape[1]
    n = sub.shape[0]

    if k < 3 or n < k:
        return np.nan

    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(sub)
        loadings = fa.loadings_.flatten()
        uniquenesses = fa.get_uniquenesses()

        # omega = (sum of loadings)^2 / ((sum of loadings)^2 + sum of uniquenesses)
        sum_loadings = np.sum(loadings)
        omega = sum_loadings**2 / (sum_loadings**2 + np.sum(uniquenesses))
        return omega
    except Exception:
        return np.nan


def _align_to_factor(
    score_matrix: pd.DataFrame,
    item_ids: list[str],
    item_loading_signs: dict[str, float],
) -> pd.DataFrame:
    """Flip item scores so all items align with the factor direction.

    Items with negative EFA loadings are reverse-scored (6 - score for 1-5 scale)
    so that higher values always mean "more of the factor".
    """
    sub = score_matrix[[c for c in item_ids if c in score_matrix.columns]].copy()
    for col in sub.columns:
        if item_loading_signs.get(col, 1.0) < 0:
            sub[col] = 6 - sub[col]
    return sub


def compute_reliability_full(
    df_success: pd.DataFrame,
    means_df: pd.DataFrame,
    score_matrix_direct: pd.DataFrame,
    retained_items: list[str],
    item_selection_report: pd.DataFrame,
    eligible_models: list[str],
) -> dict:
    """Compute comprehensive reliability metrics.

    - Cronbach's α per EFA-derived factor (across models)
    - McDonald's ω per factor (across models)
    - Per-model α and ω
    - Split-half reliability (Spearman-Brown corrected)
    - Cross-run stability (odd/even run split)

    Items are aligned to factor direction (negative-loading items are flipped)
    before computing α and split-half, so mixed-sign factors work correctly.
    """
    print("\n=== 4. Reliability ===")

    # Build factor → items mapping and loading signs
    retained_report = item_selection_report[item_selection_report["retained"]]
    factor_items = {}
    # Map item_id → signed loading for factor-alignment
    item_loading_signs: dict[str, float] = {}
    for _, row in retained_report.iterrows():
        factor = row["primary_factor"]
        if row["item_id"] in retained_items:
            factor_items.setdefault(factor, []).append(row["item_id"])
            item_loading_signs[row["item_id"]] = row["primary_loading"]

    # Also build dimension → items mapping for the original dimensions
    dim_items = {}
    for _, row in retained_report.iterrows():
        dim = row["dimension"]
        if row["item_id"] in retained_items:
            dim_items.setdefault(dim, []).append(row["item_id"])

    # --- 4a. Factor-level α and ω across models ---
    print("  Factor-level reliability (model-means matrix):")
    sm_eligible = score_matrix_direct.loc[
        score_matrix_direct.index.isin(eligible_models)
    ]

    factor_rel_rows = []
    for factor in sorted(factor_items.keys()):
        items = factor_items[factor]
        present = [i for i in items if i in sm_eligible.columns]
        if len(present) < 3:
            continue
        # Align items to factor direction before computing reliability
        aligned = _align_to_factor(sm_eligible, present, item_loading_signs)
        alpha = cronbachs_alpha(aligned, list(aligned.columns))
        omega = compute_mcdonalds_omega(aligned, list(aligned.columns))
        label = FACTOR_LABELS.get(factor, factor)
        print(f"    {factor} [{label}] ({len(present)} items): α={alpha:.3f}, ω={omega:.3f}")
        factor_rel_rows.append({
            "factor": factor,
            "label": label,
            "n_items": len(present),
            "alpha": alpha,
            "omega": omega,
        })
    factor_reliability = pd.DataFrame(factor_rel_rows)

    # --- 4b. Dimension-level α and ω across models (original dimensions) ---
    print("  Dimension-level reliability (model-means matrix):")
    dim_rel_rows = []
    for dim in sorted(dim_items.keys()):
        items = dim_items[dim]
        present = [i for i in items if i in sm_eligible.columns]
        if len(present) < 3:
            continue
        alpha = cronbachs_alpha(sm_eligible, present)
        omega = compute_mcdonalds_omega(sm_eligible, present)
        print(f"    {dim} ({len(present)} items): α={alpha:.3f}, ω={omega:.3f}")
        dim_rel_rows.append({
            "dimension": dim,
            "n_items": len(present),
            "alpha": alpha,
            "omega": omega,
        })
    dimension_reliability = pd.DataFrame(dim_rel_rows)

    # --- 4c. Per-model reliability ---
    print("  Per-model reliability...")
    ai_native = df_success[_is_ai_native(df_success)]
    per_model_rows = []

    for model_id in eligible_models:
        model_data = ai_native[ai_native["model_id"] == model_id]
        if len(model_data) == 0:
            continue

        for factor, items in sorted(factor_items.items()):
            present = [i for i in items if i in model_data["item_id"].unique()]
            if len(present) < 3:
                continue

            # Build runs × items matrix for this model
            factor_data = model_data[model_data["item_id"].isin(present)]
            obs = factor_data.pivot_table(
                index="run_number", columns="item_id", values="score",
            ).dropna()

            if obs.shape[0] < 3 or obs.shape[1] < 3:
                continue

            # Align items to factor direction
            obs = _align_to_factor(obs, list(obs.columns), item_loading_signs)
            alpha = cronbachs_alpha(obs, list(obs.columns))
            omega = compute_mcdonalds_omega(obs, list(obs.columns))

            per_model_rows.append({
                "model_id": model_id,
                "short_name": get_short_model_name(model_id),
                "factor": factor,
                "n_runs": obs.shape[0],
                "n_items": obs.shape[1],
                "alpha": alpha,
                "omega": omega,
            })

    per_model_reliability = pd.DataFrame(per_model_rows)
    if len(per_model_reliability) > 0:
        n_models = per_model_reliability["model_id"].nunique()
        print(f"    {n_models} models with per-model reliability")

    # --- 4d. Split-half reliability (Spearman-Brown corrected) ---
    print("  Split-half reliability...")
    split_half_rows = []

    for factor, items in sorted(factor_items.items()):
        present = [i for i in items if i in sm_eligible.columns]
        if len(present) < 4:
            continue

        # Split items into two halves (odd/even by sorted position)
        sorted_items = sorted(present)
        half1 = [sorted_items[i] for i in range(0, len(sorted_items), 2)]
        half2 = [sorted_items[i] for i in range(1, len(sorted_items), 2)]

        # Compute composite scores for each half (aligned to factor direction)
        aligned = _align_to_factor(sm_eligible, present, item_loading_signs)
        sub = aligned.dropna()
        if sub.shape[0] < 3:
            continue
        score1 = sub[half1].mean(axis=1)
        score2 = sub[half2].mean(axis=1)

        r, _ = stats.pearsonr(score1, score2)
        # Spearman-Brown prophecy formula
        sb = (2 * r) / (1 + r) if (1 + r) != 0 else np.nan

        split_half_rows.append({
            "factor": factor,
            "n_items_half1": len(half1),
            "n_items_half2": len(half2),
            "r_halves": r,
            "spearman_brown": sb,
        })
        print(f"    {factor}: r_halves={r:.3f}, SB={sb:.3f}")

    split_half_reliability = pd.DataFrame(split_half_rows)

    # --- 4e. Cross-run stability (odd vs even runs) ---
    print("  Cross-run stability (odd vs even runs)...")
    cross_run_rows = []

    for factor, items in sorted(factor_items.items()):
        present = [i for i in items if i in ai_native["item_id"].unique()]
        if len(present) < 3:
            continue

        model_stabilities = []
        for model_id in eligible_models:
            model_data = ai_native[
                (ai_native["model_id"] == model_id) &
                (ai_native["item_id"].isin(present))
            ]
            if len(model_data) == 0:
                continue

            odd_runs = model_data[model_data["run_number"] % 2 == 1]
            even_runs = model_data[model_data["run_number"] % 2 == 0]

            odd_score = odd_runs.groupby("item_id")["score"].mean()
            even_score = even_runs.groupby("item_id")["score"].mean()

            common = odd_score.index.intersection(even_score.index)
            if len(common) < 3:
                continue

            # Factor score = mean across items
            odd_factor = odd_score[common].mean()
            even_factor = even_score[common].mean()
            model_stabilities.append((odd_factor, even_factor))

        if len(model_stabilities) >= 3:
            odd_scores = [s[0] for s in model_stabilities]
            even_scores = [s[1] for s in model_stabilities]
            r, _ = stats.pearsonr(odd_scores, even_scores)
            cross_run_rows.append({
                "factor": factor,
                "n_models": len(model_stabilities),
                "r_odd_even": r,
            })
            print(f"    {factor}: r_odd_even={r:.3f} (n={len(model_stabilities)} models)")

    cross_run_stability = pd.DataFrame(cross_run_rows)

    return {
        "factor_reliability": factor_reliability,
        "dimension_reliability": dimension_reliability,
        "per_model_reliability": per_model_reliability,
        "split_half_reliability": split_half_reliability,
        "cross_run_stability": cross_run_stability,
        "factor_items": factor_items,
    }


# ---------------------------------------------------------------------------
# 5. Model personality profiles
# ---------------------------------------------------------------------------

def compute_model_profiles(
    df_success: pd.DataFrame,
    eligible_models: list[str],
    factor_items: dict[str, list[str]],
    plots_dir: str,
) -> dict:
    """Compute standardized dimension scores and generate radar plots.

    Returns z-scored profiles per model for each EFA-derived factor.
    """
    print("\n=== 5. Model Personality Profiles ===")

    ai_native = df_success[_is_ai_native(df_success)]

    # Compute factor scores per model (mean across items × runs)
    profile_rows = []
    for model_id in eligible_models:
        model_data = ai_native[ai_native["model_id"] == model_id]
        if len(model_data) == 0:
            continue

        row = {
            "model_id": model_id,
            "short_name": get_short_model_name(model_id),
        }
        for factor, items in sorted(factor_items.items()):
            factor_data = model_data[model_data["item_id"].isin(items)]
            if len(factor_data) > 0:
                row[factor] = factor_data["score"].mean()
            else:
                row[factor] = np.nan
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows)
    factors = sorted(factor_items.keys())

    # Z-score across models
    z_profiles = profiles.copy()
    for f in factors:
        col = z_profiles[f]
        std = col.std(ddof=1)
        z_profiles[f] = (col - col.mean()) / std if std > 0 else 0.0

    print(f"  Profiles computed for {len(profiles)} models across {len(factors)} factors")

    # Summary stats
    for f in factors:
        col = profiles[f].dropna()
        print(f"    {f}: mean={col.mean():.2f}, sd={col.std():.2f}, "
              f"range=[{col.min():.2f}, {col.max():.2f}]")

    # Radar plots
    radar_path = f"{plots_dir}/model_profiles_radar.png"
    _plot_radar_profiles(z_profiles, factors, radar_path)

    # Heatmap of z-scores
    heatmap_path = f"{plots_dir}/model_profiles_heatmap.png"
    _plot_profile_heatmap(z_profiles, factors, heatmap_path)

    return {
        "profiles": profiles,
        "z_profiles": z_profiles,
        "factors": factors,
        "radar_plot": radar_path,
        "heatmap_plot": heatmap_path,
    }


def _plot_radar_profiles(
    z_profiles: pd.DataFrame,
    factors: list[str],
    output_path: str,
) -> None:
    """Radar/spider plot of z-scored profiles per model."""
    n_models = len(z_profiles)
    n_factors = len(factors)

    # Short factor labels — use descriptive names if available
    labels = [FACTOR_LABELS.get(f, f.replace("Factor", "F")) for f in factors]

    angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
    angles += angles[:1]

    n_cols = min(5, n_models)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
        subplot_kw=dict(polar=True),
    )
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.cm.tab20
    for idx, (_, row) in enumerate(z_profiles.iterrows()):
        ax = axes[idx]
        values = [row[f] for f in factors] + [row[factors[0]]]
        ax.plot(angles, values, "o-", linewidth=1.2, color=cmap(idx % 20), markersize=3)
        ax.fill(angles, values, alpha=0.15, color=cmap(idx % 20))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylim(-3, 3)
        ax.set_title(row["short_name"], fontsize=8, pad=10)
        ax.tick_params(axis="y", labelsize=5)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Model Personality Profiles (z-scores across models)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_profile_heatmap(
    z_profiles: pd.DataFrame,
    factors: list[str],
    output_path: str,
) -> None:
    """Heatmap of model z-scores across factors."""
    plot_data = z_profiles.set_index("short_name")[factors]

    fig, ax = plt.subplots(figsize=(max(8, len(factors) * 0.8), max(6, len(plot_data) * 0.35)))
    sns.heatmap(
        plot_data, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
        vmin=-3, vmax=3, linewidths=0.5, ax=ax,
        xticklabels=[FACTOR_LABELS.get(f, f.replace("Factor", "F")) for f in factors],
    )
    ax.set_title("Model Personality Profiles (z-scores)", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Convergent/Discriminant Validity (MTMM)
# ---------------------------------------------------------------------------

def compute_mtmm(
    df_success: pd.DataFrame,
    plots_dir: str,
) -> dict:
    """Full multitrait-multimethod matrix: AI-native dimensions × BFI-44 traits.

    Reuses the BFI analysis module's convergent_discriminant_preview.
    """
    print("\n=== 6. Convergent/Discriminant Validity (MTMM) ===")

    corr_matrix, detail_df = convergent_discriminant_preview(df_success)

    if len(detail_df) > 0:
        n_models = detail_df["n_models"].iloc[0]
        n_notable = detail_df["notable"].sum()
        print(f"  {len(detail_df)} pairs, {n_notable} with |r| > 0.50, N={n_models} models")

        # Enhanced MTMM heatmap
        mtmm_path = f"{plots_dir}/mtmm_heatmap.png"
        _plot_mtmm_heatmap(corr_matrix, detail_df, mtmm_path)
    else:
        print("  Insufficient data for MTMM")
        mtmm_path = None

    return {
        "corr_matrix": corr_matrix,
        "detail_df": detail_df,
        "mtmm_plot": mtmm_path,
    }


def _plot_mtmm_heatmap(
    corr_matrix: pd.DataFrame,
    detail_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Enhanced MTMM heatmap with significance markers."""
    if corr_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(7, len(corr_matrix) * 0.5)))

    y_labels = [d.replace("/", "/\n") for d in corr_matrix.index]
    x_labels = [
        {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "E_fwd": "Extraversion\n(fwd-only)", "A": "Agreeableness",
         "N": "Neuroticism"}.get(c, c) for c in corr_matrix.columns
    ]

    # Build annotation matrix with significance stars
    annot = corr_matrix.copy().astype(object)
    for _, row in detail_df.iterrows():
        ai_dim = row["ai_native_dim"]
        bfi_dim = row["bfi_dim"]
        if ai_dim in annot.index and bfi_dim in annot.columns:
            r_val = row["r"]
            stars = ""
            if row["p"] < 0.001:
                stars = "***"
            elif row["p"] < 0.01:
                stars = "**"
            elif row["p"] < 0.05:
                stars = "*"
            annot.loc[ai_dim, bfi_dim] = f"{r_val:.2f}{stars}"

    sns.heatmap(
        corr_matrix.values.astype(float),
        annot=annot.values, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=x_labels, yticklabels=y_labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("MTMM: AI-Native × BFI-44 Correlations\n(model-level means, * p<.05 ** p<.01 *** p<.001)",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Method convergence (direct vs scenario)
# ---------------------------------------------------------------------------

def compute_method_convergence(
    df_success: pd.DataFrame,
    eligible_models: list[str],
    plots_dir: str,
) -> dict:
    """Correlate direct self-report vs scenario-based scores per dimension."""
    print("\n=== 7. Method Convergence (Direct vs Scenario) ===")

    ai_native = df_success[_is_ai_native(df_success)]
    ai_native = ai_native[ai_native["model_id"].isin(eligible_models)]

    rows = []
    for dim in AI_NATIVE_DIMENSIONS:
        dim_data = ai_native[ai_native["dimension"] == dim]

        direct_scores = {}
        scenario_scores = {}

        for model_id in eligible_models:
            model_data = dim_data[dim_data["model_id"] == model_id]

            direct = model_data[model_data["item_type"] == "direct"]
            scenario = model_data[model_data["item_type"] == "scenario"]

            if len(direct) > 0:
                direct_scores[model_id] = direct["score"].mean()
            if len(scenario) > 0:
                scenario_scores[model_id] = scenario["parsed_score"].mean()

        common = set(direct_scores.keys()) & set(scenario_scores.keys())
        if len(common) < 4:
            rows.append({
                "dimension": dim,
                "n_models": len(common),
                "r": np.nan,
                "p": np.nan,
            })
            continue

        d_vals = [direct_scores[m] for m in sorted(common)]
        s_vals = [scenario_scores[m] for m in sorted(common)]
        r, p = stats.pearsonr(d_vals, s_vals)

        rows.append({
            "dimension": dim,
            "n_models": len(common),
            "r": r,
            "p": p,
        })
        print(f"  {dim}: r={r:.3f}, p={p:.3f}, n={len(common)}")

    method_conv = pd.DataFrame(rows)

    # Bar plot
    bar_path = f"{plots_dir}/method_convergence.png"
    _plot_method_convergence(method_conv, bar_path)

    mean_r = method_conv["r"].dropna().mean()
    print(f"  Mean direct-scenario r: {mean_r:.3f}")

    return {
        "method_convergence": method_conv,
        "mean_r": mean_r,
        "bar_plot": bar_path,
    }


def _plot_method_convergence(method_conv: pd.DataFrame, output_path: str) -> None:
    """Bar plot of direct vs scenario correlations by dimension."""
    valid = method_conv.dropna(subset=["r"]).copy()
    if len(valid) == 0:
        return

    valid = valid.sort_values("r", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(valid) * 0.35)))
    colors = ["#2196F3" if r >= 0.50 else "#FFC107" if r >= 0.30 else "#F44336"
              for r in valid["r"]]
    ax.barh(range(len(valid)), valid["r"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["dimension"], fontsize=8)
    ax.set_xlabel("Pearson r (Direct vs Scenario)")
    ax.set_title("Method Convergence: Direct Self-Report vs Scenario-Based Scores")
    ax.axvline(0.50, color="gray", linestyle="--", alpha=0.5, label="r=0.50")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Scoring method convergence (repeated sampling vs log-prob)
# ---------------------------------------------------------------------------

def compute_scoring_convergence(
    df_success: pd.DataFrame,
    eligible_models: list[str],
    plots_dir: str,
) -> dict:
    """For models with log-prob data, correlate text-based vs log-prob scores."""
    print("\n=== 8. Scoring Method Convergence ===")

    ai_native = df_success[_is_ai_native(df_success)]
    ai_native = ai_native[ai_native["model_id"].isin(eligible_models)]

    # Find models with log-prob data
    logprob_models = (
        ai_native[ai_native["logprob_available"] == 1]["model_id"].unique()
    )
    print(f"  Models with log-prob data: {len(logprob_models)}")

    if len(logprob_models) < 3:
        print("  Insufficient log-prob models for convergence analysis")
        return {"scoring_convergence": pd.DataFrame(), "logprob_models": list(logprob_models)}

    # For each dimension × model, compute text-based and logprob-based dimension scores
    rows = []
    for dim in AI_NATIVE_DIMENSIONS:
        text_scores = {}
        logprob_scores = {}

        for model_id in logprob_models:
            model_dim = ai_native[
                (ai_native["model_id"] == model_id) &
                (ai_native["dimension"] == dim) &
                (ai_native["item_type"] == "direct") &
                (ai_native["logprob_score"].notna())
            ]

            if len(model_dim) < 5:
                continue

            text_scores[model_id] = model_dim["score"].mean()

            # For logprob, we need to apply reverse coding manually
            lp = model_dim.copy()
            lp["lp_score"] = lp["logprob_score"]
            reverse_mask = lp["keying"] == "-"
            lp.loc[reverse_mask, "lp_score"] = 6 - lp.loc[reverse_mask, "logprob_score"]
            logprob_scores[model_id] = lp["lp_score"].mean()

        common = set(text_scores.keys()) & set(logprob_scores.keys())
        if len(common) < 3:
            continue

        t_vals = [text_scores[m] for m in sorted(common)]
        l_vals = [logprob_scores[m] for m in sorted(common)]
        r, p = stats.pearsonr(t_vals, l_vals)

        rows.append({
            "dimension": dim,
            "n_models": len(common),
            "r": r,
            "p": p,
        })
        print(f"  {dim}: r={r:.3f}, n={len(common)}")

    scoring_conv = pd.DataFrame(rows)

    if len(scoring_conv) > 0:
        mean_r = scoring_conv["r"].dropna().mean()
        print(f"  Mean text-logprob r: {mean_r:.3f}")
    else:
        mean_r = np.nan

    return {
        "scoring_convergence": scoring_conv,
        "logprob_models": list(logprob_models),
        "mean_r": mean_r,
    }


# ---------------------------------------------------------------------------
# 9. Factor loading heatmap (improved)
# ---------------------------------------------------------------------------

def plot_efa_loadings(
    loadings_df: pd.DataFrame,
    means_df: pd.DataFrame,
    item_selection_report: pd.DataFrame,
    output_path: str,
) -> None:
    """Improved heatmap: items grouped by dimension, retained items highlighted."""
    dim_map = (
        means_df[["item_id", "dimension"]].drop_duplicates()
        .set_index("item_id")["dimension"]
    )

    # Sort items by dimension then item_id
    items_with_dim = [(dim_map.get(item, "zzz"), item) for item in loadings_df.index]
    items_with_dim.sort()
    ordered_items = [item for _, item in items_with_dim]

    ordered_loadings = loadings_df.loc[ordered_items]

    fig, ax = plt.subplots(
        figsize=(max(8, loadings_df.shape[1] * 1.2), max(12, len(ordered_items) * 0.12))
    )
    sns.heatmap(
        ordered_loadings, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        yticklabels=False, ax=ax,
    )

    # Dimension separators
    dims_ordered = [dim_map.get(item, "") for item in ordered_items]
    prev_dim = None
    dim_positions = []
    for i, dim in enumerate(dims_ordered):
        if dim != prev_dim and prev_dim is not None:
            ax.axhline(i, color="black", linewidth=0.5)
            dim_positions.append((i, prev_dim))
        prev_dim = dim
    if prev_dim:
        dim_positions.append((len(dims_ordered), prev_dim))

    ax.set_title("EFA Factor Loadings (items grouped by candidate dimension)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Items (grouped by dimension)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Report generation
# ---------------------------------------------------------------------------

def generate_primary_report(
    efa_results: dict,
    item_selection: tuple,
    cfa_results: dict,
    cfa_trimmed: dict,
    reliability_results: dict,
    profile_results: dict,
    mtmm_results: dict,
    method_conv_results: dict,
    scoring_conv_results: dict,
    output_path: str,
) -> None:
    """Generate the comprehensive primary analysis report."""
    report_dir = str(Path(output_path).parent)
    lines = []
    lines.append("# Primary Analysis Report")
    lines.append("")
    lines.append("*Auto-generated by `python -m analysis.primary_analyses`*")
    lines.append("")
    lines.append("**Design:** Split-half cross-validation — EFA on runs 1-15, "
                 "CFA on runs 16-30 (stratified by model).")
    lines.append("")

    item_report, retained, dropped = item_selection

    # =====================================================
    # 1. EFA
    # =====================================================
    lines.append("## 1. Exploratory Factor Analysis (Runs 1-15)")
    lines.append("")

    kmo = efa_results.get("kmo")
    if kmo is not None:
        lines.append(f"- **KMO:** {kmo:.3f}")
    chi2 = efa_results.get("bartlett_chi2")
    if chi2 is not None:
        lines.append(f"- **Bartlett's test:** χ²={chi2:.1f}, p={efa_results['bartlett_p']:.2e}")

    pa = efa_results["parallel_analysis"]
    lines.append(f"- **Parallel analysis:** {pa['n_factors_suggested']} factors suggested")
    lines.append(f"- **EFA method:** {efa_results['method_used']} with oblimin rotation")
    lines.append(f"- **Factors extracted:** {efa_results['n_factors']}"
                 f" (forced; parallel analysis suggested {pa['n_factors_suggested']})")
    lines.append("")

    # Factor labels
    if FACTOR_LABELS:
        lines.append("### Factor Labels")
        lines.append("")
        lines.append("| Factor | Label |")
        lines.append("|--------|-------|")
        for f_key in sorted(FACTOR_LABELS.keys()):
            lines.append(f"| {f_key} | {FACTOR_LABELS[f_key]} |")
        lines.append("")

    scree = efa_results.get("scree_plot")
    if scree:
        rel = str(Path(scree).relative_to(Path(report_dir)))
        lines.append(f"![Scree plot]({rel})")
        lines.append("")

    # Variance explained
    var_exp = efa_results.get("variance_explained")
    if var_exp is not None:
        var_df = pd.DataFrame({
            "Factor": [f"Factor{i+1}" for i in range(len(var_exp[0]))],
            "SS_Loadings": var_exp[0],
            "Proportion_Variance": var_exp[1],
            "Cumulative_Variance": var_exp[2],
        })
        lines.append("### Variance Explained")
        lines.append("")
        lines.append(df_to_markdown(var_df))
        lines.append("")

    # Factor correlations
    fc = efa_results.get("factor_correlation")
    if fc is not None:
        lines.append("### Factor Correlation Matrix")
        lines.append("")
        lines.append(df_to_markdown(fc.reset_index()))
        lines.append("")

    # =====================================================
    # 2. Item Selection
    # =====================================================
    lines.append("## 2. Item Selection")
    lines.append("")
    lines.append(f"**{len(retained)}/{len(item_report)} direct items retained** "
                 f"(primary loading ≥ {PRIMARY_LOADING_THRESHOLD}, "
                 f"cross-loading < {CROSS_LOADING_THRESHOLD})")
    lines.append("")

    # Per-dimension summary
    dim_summary = item_report.groupby("dimension").agg(
        total=("retained", "count"),
        retained=("retained", "sum"),
    ).reset_index()
    dim_summary["dropped"] = dim_summary["total"] - dim_summary["retained"]
    dim_summary = dim_summary.sort_values("dimension")
    lines.append("### Retention by Dimension")
    lines.append("")
    lines.append(df_to_markdown(dim_summary))
    lines.append("")

    # Per-factor summary
    factor_summary = item_report[item_report["retained"]].groupby("primary_factor").size().reset_index()
    factor_summary.columns = ["factor", "n_retained_items"]
    lines.append("### Retained Items per Factor")
    lines.append("")
    lines.append(df_to_markdown(factor_summary))
    lines.append("")

    # Dropped items
    if len(dropped) > 0:
        dropped_df = item_report[~item_report["retained"]][
            ["item_id", "dimension", "primary_factor", "abs_primary_loading", "cross_loading", "flag"]
        ].sort_values("dimension")
        lines.append("### Dropped Items")
        lines.append("")
        lines.append(df_to_markdown(dropped_df))
        lines.append("")

    # =====================================================
    # 3. CFA
    # =====================================================
    lines.append("## 3. Confirmatory Factor Analysis (Runs 16-30)")
    lines.append("")

    if "error" in cfa_results:
        lines.append(f"**CFA failed:** {cfa_results['error']}")
        lines.append("")
        if "model_spec" in cfa_results:
            lines.append("Model specification:")
            lines.append("```")
            lines.append(cfa_results["model_spec"])
            lines.append("```")
            lines.append("")
    else:
        n_obs = cfa_results.get("n_obs", "?")
        n_items = cfa_results.get("n_items", "?")
        lines.append(f"- **Observations:** {n_obs}")
        lines.append(f"- **Items:** {n_items}")
        lines.append(f"- **Factors:** {len(cfa_results.get('factor_items', {}))}")
        lines.append("")

        fit = cfa_results.get("fit_dict", {})
        if fit:
            lines.append("### Fit Indices")
            lines.append("")
            lines.append("| Index | Value | Threshold | Verdict |")
            lines.append("|-------|-------|-----------|---------|")

            cfi = fit.get("CFI")
            if cfi is not None:
                verdict = "Good" if cfi >= 0.95 else "Acceptable" if cfi >= 0.90 else "Poor"
                lines.append(f"| CFI | {cfi:.4f} | ≥ 0.95 (good), ≥ 0.90 (acceptable) | {verdict} |")

            tli = fit.get("TLI")
            if tli is not None:
                verdict = "Good" if tli >= 0.95 else "Acceptable" if tli >= 0.90 else "Poor"
                lines.append(f"| TLI | {tli:.4f} | ≥ 0.95 (good), ≥ 0.90 (acceptable) | {verdict} |")

            rmsea = fit.get("RMSEA")
            if rmsea is not None:
                verdict = "Good" if rmsea <= 0.06 else "Acceptable" if rmsea <= 0.08 else "Poor"
                lines.append(f"| RMSEA | {rmsea:.4f} | ≤ 0.06 (good), ≤ 0.08 (acceptable) | {verdict} |")

            srmr = fit.get("SRMR")
            if srmr is not None:
                verdict = "Good" if srmr <= 0.08 else "Poor"
                lines.append(f"| SRMR | {srmr:.4f} | ≤ 0.08 (good) | {verdict} |")

            chi2 = fit.get("chi2")
            dof = fit.get("DoF")
            if chi2 is not None and dof is not None:
                lines.append(f"| χ² | {chi2:.2f} | — | DoF={dof:.0f} |")
            lines.append("")

        # Model specification
        lines.append("### Model Specification")
        lines.append("")
        lines.append("```")
        lines.append(cfa_results.get("model_spec", ""))
        lines.append("```")
        lines.append("")

    # =====================================================
    # 3b. Trimmed CFA
    # =====================================================
    lines.append("## 3b. Trimmed CFA (Short Form)")
    lines.append("")

    if "error" in cfa_trimmed:
        lines.append(f"**Trimmed CFA failed:** {cfa_trimmed['error']}")
        lines.append("")
    else:
        top_n = cfa_trimmed.get("top_n", "?")
        n_obs = cfa_trimmed.get("n_obs", "?")
        n_items = cfa_trimmed.get("n_items", "?")
        lines.append(f"- **Design:** Top {top_n} items per factor by absolute loading")
        lines.append(f"- **Observations:** {n_obs}")
        lines.append(f"- **Items:** {n_items}")
        lines.append("")

        fit = cfa_trimmed.get("fit_dict", {})
        if fit:
            lines.append("### Fit Indices (Trimmed)")
            lines.append("")
            lines.append("| Index | Value | Threshold | Verdict |")
            lines.append("|-------|-------|-----------|---------|")

            cfi = fit.get("CFI")
            if cfi is not None:
                verdict = "Good" if cfi >= 0.95 else "Acceptable" if cfi >= 0.90 else "Poor"
                lines.append(f"| CFI | {cfi:.4f} | ≥ 0.95 (good), ≥ 0.90 (acceptable) | {verdict} |")

            tli = fit.get("TLI")
            if tli is not None:
                verdict = "Good" if tli >= 0.95 else "Acceptable" if tli >= 0.90 else "Poor"
                lines.append(f"| TLI | {tli:.4f} | ≥ 0.95 (good), ≥ 0.90 (acceptable) | {verdict} |")

            rmsea = fit.get("RMSEA")
            if rmsea is not None:
                verdict = "Good" if rmsea <= 0.06 else "Acceptable" if rmsea <= 0.08 else "Poor"
                lines.append(f"| RMSEA | {rmsea:.4f} | ≤ 0.06 (good), ≤ 0.08 (acceptable) | {verdict} |")

            chi2 = fit.get("chi2")
            dof = fit.get("DoF")
            if chi2 is not None and dof is not None:
                lines.append(f"| χ² | {chi2:.2f} | — | DoF={dof:.0f} |")
            lines.append("")

    # =====================================================
    # 4. Reliability
    # =====================================================
    lines.append("## 4. Reliability")
    lines.append("")

    # Factor-level
    fr = reliability_results["factor_reliability"]
    if len(fr) > 0:
        lines.append("### Factor-Level Reliability (Model-Means Matrix)")
        lines.append("")
        lines.append(df_to_markdown(fr))
        lines.append("")

    # Dimension-level
    dr = reliability_results["dimension_reliability"]
    if len(dr) > 0:
        lines.append("### Dimension-Level Reliability (Model-Means Matrix)")
        lines.append("")
        lines.append(df_to_markdown(dr))
        lines.append("")

    # Per-model
    pmr = reliability_results["per_model_reliability"]
    if len(pmr) > 0:
        # Pivot: models × factors for alpha
        alpha_wide = pmr.pivot_table(
            index="short_name", columns="factor", values="alpha"
        ).reset_index()
        alpha_wide.columns.name = None
        lines.append("### Per-Model Cronbach's α")
        lines.append("")
        lines.append(df_to_markdown(alpha_wide))
        lines.append("")

        # Pivot for omega
        omega_wide = pmr.pivot_table(
            index="short_name", columns="factor", values="omega"
        ).reset_index()
        omega_wide.columns.name = None
        lines.append("### Per-Model McDonald's ω")
        lines.append("")
        lines.append(df_to_markdown(omega_wide))
        lines.append("")

    # Split-half
    shr = reliability_results["split_half_reliability"]
    if len(shr) > 0:
        lines.append("### Split-Half Reliability (Spearman-Brown Corrected)")
        lines.append("")
        lines.append(df_to_markdown(shr))
        lines.append("")

    # Cross-run stability
    crs = reliability_results["cross_run_stability"]
    if len(crs) > 0:
        lines.append("### Cross-Run Stability (Odd vs Even Runs)")
        lines.append("")
        lines.append(df_to_markdown(crs))
        lines.append("")

    # =====================================================
    # 5. Model Profiles
    # =====================================================
    lines.append("## 5. Model Personality Profiles")
    lines.append("")

    profiles = profile_results["profiles"]
    factors = profile_results["factors"]
    display_cols = ["short_name"] + factors
    lines.append("### Raw Scores")
    lines.append("")
    lines.append(df_to_markdown(profiles[display_cols]))
    lines.append("")

    z_profiles = profile_results["z_profiles"]
    lines.append("### Z-Scores")
    lines.append("")
    lines.append(df_to_markdown(z_profiles[display_cols]))
    lines.append("")

    radar = profile_results.get("radar_plot")
    if radar:
        rel = str(Path(radar).relative_to(Path(report_dir)))
        lines.append(f"![Radar profiles]({rel})")
        lines.append("")

    heatmap = profile_results.get("heatmap_plot")
    if heatmap:
        rel = str(Path(heatmap).relative_to(Path(report_dir)))
        lines.append(f"![Profile heatmap]({rel})")
        lines.append("")

    # =====================================================
    # 6. MTMM
    # =====================================================
    lines.append("## 6. Convergent/Discriminant Validity (MTMM)")
    lines.append("")

    mtmm_plot = mtmm_results.get("mtmm_plot")
    if mtmm_plot:
        rel = str(Path(mtmm_plot).relative_to(Path(report_dir)))
        lines.append(f"![MTMM heatmap]({rel})")
        lines.append("")

    detail = mtmm_results.get("detail_df")
    if detail is not None and len(detail) > 0:
        n_models = detail["n_models"].iloc[0]
        lines.append(f"N = {n_models} models.")
        lines.append("")

        notable = detail[detail["notable"]]
        if len(notable) > 0:
            lines.append("### Notable Correlations (|r| > 0.50)")
            lines.append("")
            notable_disp = notable[
                ["ai_native_dim", "bfi_dim", "r", "p", "ci_lo", "ci_hi", "n_models"]
            ].sort_values("r", key=abs, ascending=False)
            lines.append(df_to_markdown(notable_disp))
            lines.append("")
        else:
            lines.append("No correlations exceeded |r| > 0.50.")
            lines.append("")

    # =====================================================
    # 7. Method Convergence
    # =====================================================
    lines.append("## 7. Method Convergence (Direct vs Scenario)")
    lines.append("")

    mc = method_conv_results.get("method_convergence")
    if mc is not None and len(mc) > 0:
        lines.append(f"Mean r = {method_conv_results.get('mean_r', np.nan):.3f}")
        lines.append("")
        lines.append(df_to_markdown(mc))
        lines.append("")

    mc_plot = method_conv_results.get("bar_plot")
    if mc_plot:
        rel = str(Path(mc_plot).relative_to(Path(report_dir)))
        lines.append(f"![Method convergence]({rel})")
        lines.append("")

    # =====================================================
    # 8. Scoring Method Convergence
    # =====================================================
    lines.append("## 8. Scoring Method Convergence (Text vs Log-Prob)")
    lines.append("")

    sc = scoring_conv_results.get("scoring_convergence")
    lp_models = scoring_conv_results.get("logprob_models", [])
    lines.append(f"Log-prob models: {len(lp_models)}")
    if lp_models:
        lines.append(f"  ({', '.join(get_short_model_name(m) for m in lp_models)})")
    lines.append("")

    if sc is not None and len(sc) > 0:
        lines.append(f"Mean r = {scoring_conv_results.get('mean_r', np.nan):.3f}")
        lines.append("")
        lines.append(df_to_markdown(sc))
        lines.append("")
    else:
        lines.append("Insufficient log-prob data for convergence analysis.")
        lines.append("")

    # Write
    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Week 2 primary analyses")
    parser.parse_args()

    start = time.time()
    ensure_output_dirs()
    plots_dir = str(PLOTS_DIR)
    output_dir = str(OUTPUT_DIR)

    # Load data
    print("Loading data...")
    df_all = load_responses()
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)

    ai_native = df_success[_is_ai_native(df_success)]
    n_models = ai_native["model_id"].nunique()
    n_items = ai_native["item_id"].nunique()
    n_rows = len(ai_native)
    print(f"  AI-native: {n_rows} rows, {n_models} models, {n_items} items")

    eligible_models = get_models_for_section(df_all, section=4)
    print(f"  Eligible models (≥200 direct items): {len(eligible_models)}")

    # Compute means for item mapping
    means_df = compute_model_item_means(df_success)
    score_matrix_direct = pivot_score_matrix(means_df, item_type="direct")
    score_matrix_direct_eligible = score_matrix_direct.loc[
        score_matrix_direct.index.isin(eligible_models)
    ]

    # Split data
    efa_df, cfa_df = split_half_data(df_success)
    print(f"  EFA half: {len(efa_df)} rows (runs 1-15)")
    print(f"  CFA half: {len(cfa_df)} rows (runs 16-30)")

    # 1. EFA (forced 7-factor solution)
    efa_results = run_efa_exploration(
        efa_df, eligible_models, plots_dir, forced_n_factors=FORCED_N_FACTORS,
    )

    # EFA loadings plot
    if "loadings" in efa_results:
        loadings_path = f"{plots_dir}/efa_factor_loadings.png"
        plot_efa_loadings(efa_results["loadings"], means_df, None, loadings_path)
        efa_results["loadings_plot"] = loadings_path

    # 2. Item selection
    item_report, retained, dropped = select_items(
        efa_results["loadings"], means_df, efa_results["communalities"],
    )

    # 3. CFA (full)
    cfa_results = run_cfa_confirmation(
        cfa_df, eligible_models, retained, item_report,
    )

    # 3b. CFA (trimmed — top 6 items per factor)
    cfa_trimmed = run_cfa_trimmed(
        cfa_df, eligible_models, item_report, top_n=6,
    )

    # 4. Reliability
    reliability_results = compute_reliability_full(
        df_success, means_df, score_matrix_direct_eligible,
        retained, item_report, eligible_models,
    )

    # 5. Model profiles
    profile_results = compute_model_profiles(
        df_success, eligible_models,
        reliability_results["factor_items"], plots_dir,
    )

    # 6. MTMM
    mtmm_results = compute_mtmm(df_success, plots_dir)

    # 7. Method convergence
    method_conv_results = compute_method_convergence(
        df_success, eligible_models, plots_dir,
    )

    # 8. Scoring convergence
    scoring_conv_results = compute_scoring_convergence(
        df_success, eligible_models, plots_dir,
    )

    # 9. Report
    report_path = f"{output_dir}/primary_analysis_report.md"
    generate_primary_report(
        efa_results=efa_results,
        item_selection=(item_report, retained, dropped),
        cfa_results=cfa_results,
        cfa_trimmed=cfa_trimmed,
        reliability_results=reliability_results,
        profile_results=profile_results,
        mtmm_results=mtmm_results,
        method_conv_results=method_conv_results,
        scoring_conv_results=scoring_conv_results,
        output_path=report_path,
    )

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
