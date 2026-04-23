"""Robustness: model-level EFA (N=25) vs. the primary observation-weighted EFA.

Gemini review raised concern that pooling 25 models x 15 runs = 375 observations
inflates the effective N. The primary analysis already uses observation weighting
(each model weighted equally, effective N=25), but this script adds a stricter
robustness check: aggregate each model's 15 runs to a per-item mean, producing
a 25 x 240 matrix, then run the same k=5 EFA pipeline on that model-level matrix.

We then ask:
  - Do the same items load on the same factors?
  - Do the resulting per-model factor scores match the primary scores?

Output: analysis/output/model_level_efa.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from analysis.data_loader import (
    DB_PATH, OUTPUT_DIR, filter_success, load_responses, recode_reverse_items,
    get_models_for_section,
)
from analysis.primary_analyses import select_items, split_half_data
from analysis.judge_analysis import load_instrument_factor_scores, _EFA_FACTOR_TO_CODE


FACTORS = ["RE", "DE", "BO", "GU", "VB"]
FACTOR_NAMES = {
    "RE": "Responsiveness", "DE": "Deference", "BO": "Boldness",
    "GU": "Guardedness", "VB": "Verbosity",
}


def _is_ai_native(df: pd.DataFrame) -> pd.Series:
    return ~df["item_id"].str.startswith("BFI-")


def _is_direct(df: pd.DataFrame) -> pd.Series:
    return df["item_type"] == "direct"


def build_model_level_matrix(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Aggregate exploration half (runs 1-15) to model-level per-item means.

    Returns: DataFrame of shape (n_models, n_items) with one row per model.
    """
    exp = df[df["run_number"].between(1, 15)]
    ai = exp[_is_ai_native(exp) & _is_direct(exp)]
    ai = ai[ai["model_id"].isin(models)]
    mat = ai.groupby(["model_id", "item_id"])["score"].mean().unstack("item_id")
    mat = mat.loc[models]  # preserve model order
    return mat


def tucker_phi(L1: np.ndarray, L2: np.ndarray) -> float:
    """Tucker's congruence between two loading vectors."""
    num = np.dot(L1, L2)
    den = np.sqrt(np.sum(L1**2) * np.sum(L2**2))
    return float(num / den) if den != 0 else np.nan


def match_factors(pooled_loadings: pd.DataFrame, ml_loadings: pd.DataFrame) -> dict:
    """Greedy match between model-level factors and pooled factors.

    For each pooled factor, find the model-level factor with highest |Tucker phi|.
    Returns dict: ml_factor -> pooled_factor (and the phi, with sign).
    """
    common = pooled_loadings.index.intersection(ml_loadings.index)
    P = pooled_loadings.loc[common].values  # (n_items, n_factors_pooled)
    M = ml_loadings.loc[common].values
    n_pooled = P.shape[1]
    n_ml = M.shape[1]

    # All pairwise phi values
    phis = np.zeros((n_ml, n_pooled))
    for i in range(n_ml):
        for j in range(n_pooled):
            phis[i, j] = tucker_phi(M[:, i], P[:, j])

    # Greedy assignment on absolute values
    mapping = {}
    used_pooled = set()
    order = np.argsort(-np.max(np.abs(phis), axis=1))
    for i in order:
        j_sorted = np.argsort(-np.abs(phis[i]))
        for j in j_sorted:
            if j not in used_pooled:
                mapping[i] = (j, float(phis[i, j]))
                used_pooled.add(j)
                break
    return mapping


def main():
    print("Loading data...")
    df = load_responses(DB_PATH)
    df = filter_success(df)
    df = recode_reverse_items(df)
    models = get_models_for_section(df, section=4)
    print(f"  {len(models)} models")

    # Build model-level matrix (exploration half only, to match primary EFA)
    mat = build_model_level_matrix(df, models)
    # Drop columns with any NaN (items where at least one model is missing)
    mat = mat.dropna(axis=1)
    print(f"  Model-level matrix: {mat.shape}")

    # Run EFA on the model-level matrix
    print("Running model-level EFA (k=5, PAF, oblimin)...")
    try:
        fa = FactorAnalyzer(n_factors=5, rotation="oblimin", method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(mat.values)
    except Exception:
        fa = FactorAnalyzer(n_factors=5, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(mat.values)

    ml_loadings = pd.DataFrame(
        fa.loadings_, index=mat.columns,
        columns=[f"F{i+1}" for i in range(5)],
    )
    ml_var = fa.get_factor_variance()[1]  # proportion variance explained

    # Re-run pooled EFA (observation-weighted) for comparison
    print("Running pooled (observation-weighted) EFA for comparison...")
    from analysis.judge_analysis import build_pooled_matrix
    FORCED_N_FACTORS = 5
    efa_df, _ = split_half_data(df)
    obs_matrix, weights = build_pooled_matrix(efa_df, models, "direct")
    filled = obs_matrix.fillna(obs_matrix.mean())
    sqrt_w = np.sqrt(weights)
    wmean = np.average(filled.values, axis=0, weights=weights)
    weighted_centered = (filled.values - wmean) * sqrt_w[:, np.newaxis]
    weighted_df = pd.DataFrame(weighted_centered, columns=filled.columns)
    try:
        fa_p = FactorAnalyzer(n_factors=FORCED_N_FACTORS, rotation="oblimin", method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa_p.fit(weighted_df)
    except Exception:
        fa_p = FactorAnalyzer(n_factors=FORCED_N_FACTORS, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa_p.fit(weighted_df)
    pooled_loadings = pd.DataFrame(
        fa_p.loadings_, index=obs_matrix.columns,
        columns=[f"F{i+1}" for i in range(5)],
    )

    # Match model-level factors to pooled factors via Tucker's phi
    print("Computing factor congruence...")
    mapping = match_factors(pooled_loadings, ml_loadings)

    # Compute per-factor recovery: items primary on the same factor?
    ml_primary = ml_loadings.abs().idxmax(axis=1)
    pooled_primary = pooled_loadings.abs().idxmax(axis=1)
    common_items = ml_primary.index.intersection(pooled_primary.index)

    # Build a reverse mapping for pooled factor name -> ml factor name
    ml_to_pooled = {f"F{i+1}": f"F{mapping[i][0]+1}" for i in mapping}
    pooled_to_ml = {v: k for k, v in ml_to_pooled.items()}

    # For each item, check whether its primary at ml matches its primary at pooled (mapped)
    recovered = 0
    total = len(common_items)
    for item in common_items:
        ml_f = ml_primary.loc[item]
        pooled_f = pooled_primary.loc[item]
        if ml_to_pooled.get(ml_f) == pooled_f:
            recovered += 1

    # Factor score correlations: reload primary, recompute ml-based scores using
    # the same factor-> items assignment from the primary pipeline
    primary_scores = load_instrument_factor_scores(DB_PATH).set_index("model_id")

    # Compute ml-level factor scores by retaining items and averaging per model
    # on the same retained set used by the primary analysis.
    # Read the retained item assignment from primary factor loadings
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from analysis.data_loader import compute_model_item_means
        means_df = compute_model_item_means(df)
    # Use pooled loadings + retained items mapping to get factor -> items
    communalities_df = pd.DataFrame({
        "item_id": obs_matrix.columns, "communality": fa_p.get_communalities(),
    })
    item_report, retained, _ = select_items(pooled_loadings.rename(
        columns={f"F{i+1}": f"Factor{i+1}" for i in range(5)}
    ), means_df, communalities_df)
    factor_items: dict[str, list[str]] = {}
    for _, row in item_report[item_report["retained"]].iterrows():
        factor_items.setdefault(row["primary_factor"], []).append(row["item_id"])

    # Compute factor scores from model-level matrix (exploration half only, since
    # that's what the matrix uses)
    ml_factor_scores = pd.DataFrame(index=mat.index)
    for factor_label, items in factor_items.items():
        code = _EFA_FACTOR_TO_CODE.get(factor_label, factor_label[:2])
        shared = [i for i in items if i in mat.columns]
        if not shared:
            continue
        ml_factor_scores[code] = mat[shared].mean(axis=1)

    # Correlate ml factor scores vs primary factor scores
    common_m = ml_factor_scores.index.intersection(primary_scores.index)
    corr_rows = []
    for f in FACTORS:
        if f not in ml_factor_scores.columns or f not in primary_scores.columns:
            corr_rows.append({"factor": f, "r": np.nan})
            continue
        x = ml_factor_scores.loc[common_m, f].values
        y = primary_scores.loc[common_m, f].values
        r = float(np.corrcoef(x, y)[0, 1])
        corr_rows.append({"factor": f, "r": r, "n": len(common_m)})

    # Write report
    out = OUTPUT_DIR / "model_level_efa.md"
    lines = []
    lines.append("# Model-Level EFA Robustness Check")
    lines.append("")
    lines.append(
        "Aggregates each model's 15 exploration-half runs to a per-item mean, "
        "yielding a 25 x N matrix. Runs k=5 EFA (oblimin) on this matrix and "
        "compares to the primary observation-weighted EFA."
    )
    lines.append("")
    lines.append(f"**Matrix shape:** {mat.shape[0]} models x {mat.shape[1]} items")
    lines.append("")

    lines.append("## Variance explained (model-level EFA)")
    lines.append("")
    lines.append("| Factor | Proportion |")
    lines.append("|---|---:|")
    for i, v in enumerate(ml_var):
        lines.append(f"| F{i+1} | {v:.3f} |")
    lines.append(f"| **Total** | **{sum(ml_var):.3f}** |")
    lines.append("")

    lines.append("## Factor congruence (Tucker's phi)")
    lines.append("")
    lines.append("Greedy matching of model-level factors to pooled factors. "
                 "|phi| >= .95 indicates strong equivalence.")
    lines.append("")
    lines.append("| Model-level factor | Matched pooled factor | Tucker's phi |")
    lines.append("|---|---|---:|")
    for i in range(5):
        j, phi = mapping[i]
        lines.append(f"| F{i+1} | F{j+1} | {phi:+.3f} |")
    lines.append("")

    lines.append("## Item recovery")
    lines.append("")
    lines.append(
        f"**{recovered}/{total} items ({100 * recovered / total:.1f}%)** have the "
        f"same primary factor under model-level EFA as under pooled EFA "
        f"(after matching factor labels)."
    )
    lines.append("")

    lines.append("## Factor score correlations (model-level vs primary, N=25)")
    lines.append("")
    lines.append("| Factor | Pearson r |")
    lines.append("|---|---:|")
    for row in corr_rows:
        r = row.get("r", np.nan)
        lines.append(f"| {FACTOR_NAMES[row['factor']]} | {r:+.3f} |")

    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
