"""ESEM (Exploratory Structural Equation Modeling) diagnostic.

Answers one question: is the 7-factor structure defensible, or does it need
restructuring?  Three complementary approaches:

1. **Tucker congruence**: Run EFA on both halves independently and compare
   loadings.  Congruence ≥ 0.95 = factor replicates.
2. **CFA-half EFA fit**: Run EFA on the CFA half (runs 16-30) and compute
   residual-matrix-based fit indices (SRMR-equivalent).
3. **ESEM via semopy**: Specify all cross-loadings freely estimated.  If
   ESEM fit is adequate (CFI ≥ 0.90) but CFA fit was poor, the factors
   are real — the problem was zero cross-loading constraints.

Usage:
    python -m analysis.esem
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import (
    DB_PATH,
    OUTPUT_DIR,
    PLOTS_DIR,
    compute_model_item_means,
    ensure_output_dirs,
    filter_success,
    get_models_for_section,
    load_responses,
    pivot_score_matrix,
    recode_reverse_items,
)
from .factor_structure import (
    build_pooled_matrix,
    run_efa,
    _weighted_corr,
)
from .primary_analyses import (
    EFA_RUNS,
    CFA_RUNS,
    FORCED_N_FACTORS,
    FACTOR_LABELS,
    PRIMARY_LOADING_THRESHOLD,
    CROSS_LOADING_THRESHOLD,
    split_half_data,
    select_items,
)
from .bfi_analysis import _is_ai_native


# ---------------------------------------------------------------------------
# 1. Tucker's congruence coefficient
# ---------------------------------------------------------------------------

def tucker_congruence(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Tucker's congruence coefficient between two loading matrices.

    Returns a (k × k) matrix where entry (i, j) is the congruence between
    factor i of A and factor j of B.  Diagonal after optimal alignment
    gives per-factor congruence.

    Interpretation (Lorenzo-Seva & ten Berge, 2006):
      ≥ 0.95  =  equal / excellent replication
      0.85–0.94 = fair
      < 0.85  =  poor
    """
    # phi(a, b) = sum(a*b) / sqrt(sum(a^2) * sum(b^2))
    numer = A.T @ B                                   # k_a × k_b
    denom = np.sqrt(
        (A ** 2).sum(axis=0)[:, None] *
        (B ** 2).sum(axis=0)[None, :]
    )
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return numer / denom


def align_factors(
    loadings_target: pd.DataFrame,
    loadings_source: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Align source factors to target factors using Tucker congruence.

    Greedy matching: assign each source factor to the target factor
    with the highest absolute congruence, then flip sign if needed.

    Returns (aligned_source, congruence_vector).
    """
    # Restrict to shared items
    shared = loadings_target.index.intersection(loadings_source.index)
    A = loadings_target.loc[shared].values
    B = loadings_source.loc[shared].values

    k = A.shape[1]
    phi = tucker_congruence(A, B)

    # Greedy matching by |congruence|
    used_src = set()
    mapping = {}  # target_idx → source_idx
    signs = {}

    abs_phi = np.abs(phi)
    for _ in range(k):
        # Find best remaining pair
        best_val = -1.0
        best_t, best_s = -1, -1
        for t in range(k):
            if t in mapping:
                continue
            for s in range(k):
                if s in used_src:
                    continue
                if abs_phi[t, s] > best_val:
                    best_val = abs_phi[t, s]
                    best_t, best_s = t, s
        mapping[best_t] = best_s
        used_src.add(best_s)
        signs[best_t] = np.sign(phi[best_t, best_s])

    # Reorder and flip source columns
    new_order = [mapping[t] for t in range(k)]
    aligned = loadings_source.iloc[:, new_order].copy()
    aligned.columns = loadings_target.columns

    for t in range(k):
        if signs[t] < 0:
            aligned.iloc[:, t] *= -1

    congruence = np.array([
        abs_phi[t, mapping[t]] for t in range(k)
    ])

    return aligned, congruence


# ---------------------------------------------------------------------------
# 2. Residual-based fit from EFA on CFA half
# ---------------------------------------------------------------------------

def efa_residual_fit(
    obs_matrix: pd.DataFrame,
    weights: np.ndarray,
    loadings: np.ndarray,
    factor_corr: np.ndarray | None = None,
) -> dict:
    """Compute fit indices from the residual correlation matrix.

    Given observed weighted correlation R and reproduced correlation
    R_hat = Λ Φ Λ' + Ψ, compute SRMR and related indices.
    """
    R = _weighted_corr(obs_matrix, weights)
    p = R.shape[0]

    if factor_corr is None:
        factor_corr = np.eye(loadings.shape[1])

    # Reproduced correlation
    R_hat = loadings @ factor_corr @ loadings.T
    # Add uniquenesses on diagonal so R_hat diagonal = 1
    np.fill_diagonal(R_hat, 1.0)

    # Residual matrix (off-diagonal only)
    residual = R - R_hat
    np.fill_diagonal(residual, 0.0)

    # SRMR: sqrt of mean squared off-diagonal residuals
    n_off_diag = p * (p - 1) / 2
    sum_sq = np.sum(np.triu(residual, k=1) ** 2)
    srmr = np.sqrt(sum_sq / n_off_diag)

    # Max absolute residual
    max_resid = np.max(np.abs(np.triu(residual, k=1)))

    # Mean absolute residual
    mean_abs_resid = np.sum(np.abs(np.triu(residual, k=1))) / n_off_diag

    # Proportion of residuals > 0.05
    resid_upper = np.triu(residual, k=1)
    n_large = np.sum(np.abs(resid_upper[resid_upper != 0]) > 0.05)
    prop_large = n_large / n_off_diag

    return {
        "srmr": srmr,
        "max_residual": max_resid,
        "mean_abs_residual": mean_abs_resid,
        "prop_residuals_gt_05": prop_large,
        "residual_matrix": residual,
    }


# ---------------------------------------------------------------------------
# 3. ESEM via semopy (all cross-loadings free)
# ---------------------------------------------------------------------------

def build_esem_spec(
    factor_items: dict[str, list[str]],
    all_items: list[str],
    rename_map: dict[str, str],
) -> str:
    """Build semopy model spec with all cross-loadings freely estimated.

    Each factor loads on ALL items.  The first item per factor (the one with
    the highest EFA loading) is the reference indicator (loading fixed to 1
    for identification, which semopy handles as default for the first item).
    Non-target loadings are freely estimated (not fixed to zero as in CFA).
    """
    factors = sorted(factor_items.keys())

    # All unique items across factors
    all_sanitized = [rename_map.get(i, i.replace("-", "_")) for i in all_items]

    model_lines = []
    for factor in factors:
        # Put target items first (primary indicators), then non-target
        target = [rename_map.get(i, i.replace("-", "_"))
                  for i in factor_items[factor] if i in rename_map]
        non_target = [s for s in all_sanitized if s not in target]
        items_ordered = target + non_target
        model_lines.append(f"{factor} =~ {' + '.join(items_ordered)}")

    return "\n".join(model_lines)


def select_top_items(
    item_selection_report: pd.DataFrame,
    top_n: int,
) -> tuple[list[str], dict[str, list[str]]]:
    """Select top N items per factor by absolute loading.

    Returns (item_list, factor_items_dict).
    """
    retained = item_selection_report[item_selection_report["flag"] == ""].copy()
    retained["abs_loading"] = retained["primary_loading"].abs()

    all_items = []
    factor_items = {}
    for factor in sorted(retained["primary_factor"].unique()):
        rows = retained[retained["primary_factor"] == factor]
        top = rows.nlargest(top_n, "abs_loading")
        items = top["item_id"].tolist()
        all_items.extend(items)
        factor_items[factor] = items

    return all_items, factor_items


def run_cfa_for_items(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    factor_items: dict[str, list[str]],
    label: str = "CFA",
) -> dict:
    """Run strict CFA for a given set of factor→items and return fit indices."""
    import semopy

    obs_matrix, weights = build_pooled_matrix(cfa_df, eligible_models, "direct")

    all_items = [i for items in factor_items.values() for i in items]
    available = [c for c in all_items if c in obs_matrix.columns]
    obs = obs_matrix[available].copy().fillna(obs_matrix[available].mean())

    rename_map = {item: item.replace("-", "_") for item in available}
    obs_renamed = obs.rename(columns=rename_map)

    model_lines = []
    for factor, items in sorted(factor_items.items()):
        fname = factor.replace(" ", "")
        sanitized = [rename_map[i] for i in items if i in rename_map]
        if len(sanitized) >= 3:
            model_lines.append(f"{fname} =~ {' + '.join(sanitized)}")

    model_spec = "\n".join(model_lines)

    try:
        model = semopy.Model(model_spec)
        model.fit(obs_renamed)
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

        return {"fit_dict": fit_dict, "n_items": len(available), "n_obs": obs.shape[0]}
    except Exception as e:
        return {"error": str(e)}


def run_esem_for_items(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    factor_items: dict[str, list[str]],
) -> dict:
    """Run ESEM (all cross-loadings free) for a given factor→items mapping."""
    import semopy

    obs_matrix, weights = build_pooled_matrix(cfa_df, eligible_models, "direct")

    all_items = [i for items in factor_items.values() for i in items]
    available = [c for c in all_items if c in obs_matrix.columns]
    obs = obs_matrix[available].copy().fillna(obs_matrix[available].mean())

    rename_map = {item: item.replace("-", "_") for item in available}
    obs_renamed = obs.rename(columns=rename_map)

    model_spec = build_esem_spec(factor_items, available, rename_map)

    try:
        model = semopy.Model(model_spec)
        model.fit(obs_renamed)
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

        return {"fit_dict": fit_dict, "n_items": len(available), "n_obs": obs.shape[0]}
    except Exception as e:
        return {"error": str(e)}


def run_trim_sweep(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    item_selection_report: pd.DataFrame,
    trim_levels: list[int] = [3, 4, 5, 6, 8, 10],
    output_dir: str = "",
    plots_dir: str = "",
) -> pd.DataFrame:
    """Sweep across trim levels, running both CFA and ESEM at each.

    Returns a DataFrame with fit indices per (trim_level, method).
    """
    print("\n=== Trim Sweep: CFA & ESEM across item counts ===\n")

    rows = []
    for top_n in trim_levels:
        items, factor_items = select_top_items(item_selection_report, top_n)
        n_factors = len(factor_items)
        n_items = sum(len(v) for v in factor_items.values())
        print(f"--- top {top_n}/factor → {n_items} items, {n_factors} factors ---")

        # CFA
        print(f"  CFA...", end=" ", flush=True)
        cfa = run_cfa_for_items(cfa_df, eligible_models, factor_items, f"CFA-{top_n}")
        if "error" in cfa:
            print(f"FAILED: {cfa['error']}")
            rows.append({
                "top_n": top_n, "method": "CFA", "n_items": n_items,
                "CFI": np.nan, "TLI": np.nan, "RMSEA": np.nan,
            })
        else:
            fd = cfa["fit_dict"]
            print(f"CFI={fd.get('CFI', 'N/A'):.3f}, RMSEA={fd.get('RMSEA', 'N/A'):.3f}")
            rows.append({
                "top_n": top_n, "method": "CFA", "n_items": n_items,
                "CFI": fd.get("CFI"), "TLI": fd.get("TLI"), "RMSEA": fd.get("RMSEA"),
                "chi2": fd.get("chi2"), "DoF": fd.get("DoF"),
            })

        # ESEM
        print(f"  ESEM...", end=" ", flush=True)
        esem = run_esem_for_items(cfa_df, eligible_models, factor_items)
        if "error" in esem:
            print(f"FAILED: {esem['error']}")
            rows.append({
                "top_n": top_n, "method": "ESEM", "n_items": n_items,
                "CFI": np.nan, "TLI": np.nan, "RMSEA": np.nan,
            })
        else:
            fd = esem["fit_dict"]
            print(f"CFI={fd.get('CFI', 'N/A'):.3f}, RMSEA={fd.get('RMSEA', 'N/A'):.3f}")
            rows.append({
                "top_n": top_n, "method": "ESEM", "n_items": n_items,
                "CFI": fd.get("CFI"), "TLI": fd.get("TLI"), "RMSEA": fd.get("RMSEA"),
                "chi2": fd.get("chi2"), "DoF": fd.get("DoF"),
            })

    sweep_df = pd.DataFrame(rows)

    # Plot
    if plots_dir:
        plot_trim_sweep(sweep_df, f"{plots_dir}/trim_sweep_fit.png")

    # Write summary table
    if output_dir:
        summary_path = f"{output_dir}/trim_sweep_results.md"
        lines = ["# Trim Sweep: CFA vs ESEM Fit by Items-per-Factor\n"]
        lines.append("| top_n | method | n_items | CFI | TLI | RMSEA |")
        lines.append("|-------|--------|---------|-----|-----|-------|")
        for _, row in sweep_df.iterrows():
            cfi = f"{row['CFI']:.3f}" if pd.notna(row.get("CFI")) else "—"
            tli = f"{row['TLI']:.3f}" if pd.notna(row.get("TLI")) else "—"
            rmsea = f"{row['RMSEA']:.3f}" if pd.notna(row.get("RMSEA")) else "—"
            lines.append(
                f"| {row['top_n']} | {row['method']} | {row['n_items']} | {cfi} | {tli} | {rmsea} |"
            )

        # Find best
        valid = sweep_df.dropna(subset=["CFI"])
        if len(valid):
            best = valid.loc[valid["CFI"].idxmax()]
            lines.append(f"\n**Best CFI:** {best['CFI']:.3f} ({best['method']}, top {int(best['top_n'])} items/factor, {int(best['n_items'])} total)")

            acceptable = valid[valid["CFI"] >= 0.90]
            if len(acceptable):
                lines.append(f"\n**Configurations reaching CFI ≥ 0.90:** {len(acceptable)}")
            else:
                closest = valid.loc[valid["CFI"].idxmax()]
                lines.append(f"\n**No configuration reached CFI ≥ 0.90.** Closest: CFI={closest['CFI']:.3f}")

        Path(summary_path).write_text("\n".join(lines))
        print(f"\nSweep results written to {summary_path}")

    return sweep_df


def plot_trim_sweep(sweep_df: pd.DataFrame, output_path: str) -> None:
    """Plot CFI and RMSEA across trim levels for CFA vs ESEM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method, marker, color in [("CFA", "o", "#d62728"), ("ESEM", "s", "#2ca02c")]:
        sub = sweep_df[sweep_df["method"] == method].dropna(subset=["CFI"])
        if len(sub) == 0:
            continue
        axes[0].plot(sub["top_n"], sub["CFI"], f"{marker}-", color=color, label=method, linewidth=2)
        axes[1].plot(sub["top_n"], sub["RMSEA"], f"{marker}-", color=color, label=method, linewidth=2)

    axes[0].axhline(0.90, color="gray", linestyle="--", alpha=0.7, label="Acceptable (0.90)")
    axes[0].axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="Good (0.95)")
    axes[0].set_xlabel("Items per factor")
    axes[0].set_ylabel("CFI")
    axes[0].set_title("CFI by trim level")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)

    axes[1].axhline(0.08, color="gray", linestyle="--", alpha=0.7, label="Acceptable (0.08)")
    axes[1].axhline(0.06, color="gray", linestyle=":", alpha=0.5, label="Good (0.06)")
    axes[1].set_xlabel("Items per factor")
    axes[1].set_ylabel("RMSEA")
    axes[1].set_title("RMSEA by trim level")
    axes[1].legend(fontsize=9)

    plt.suptitle("Fit indices: CFA vs ESEM across instrument lengths", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_esem(
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    retained_items: list[str],
    item_selection_report: pd.DataFrame,
) -> dict:
    """Run ESEM: CFA with all cross-loadings freely estimated.

    This gives proper SEM fit indices (CFI, TLI, RMSEA) while allowing
    the flexibility of EFA.  If ESEM fit >> CFA fit, the factors are
    defensible and the CFA failure was due to unrealistic zero constraints.
    """
    import semopy

    print("\n=== ESEM (all cross-loadings free) ===")

    obs_matrix, weights = build_pooled_matrix(cfa_df, eligible_models, "direct")
    available = [c for c in retained_items if c in obs_matrix.columns]
    obs = obs_matrix[available].copy().fillna(obs_matrix[available].mean())

    print(f"  Matrix: {obs.shape[0]} obs × {obs.shape[1]} items")

    # Build factor → items mapping
    retained_report = item_selection_report[
        item_selection_report["item_id"].isin(available)
    ]
    factor_items = {}
    for _, row in retained_report.iterrows():
        factor = row["primary_factor"]
        factor_items.setdefault(factor, []).append(row["item_id"])
    factor_items = {f: items for f, items in factor_items.items() if len(items) >= 3}

    rename_map = {item: item.replace("-", "_") for item in available}
    obs_renamed = obs.rename(columns=rename_map)

    # Build ESEM spec (all cross-loadings free)
    model_spec = build_esem_spec(factor_items, available, rename_map)

    n_factors = len(factor_items)
    n_items = len(available)
    n_params_cfa = n_items * 1 + n_items  # 1 loading + 1 uniqueness per item (approx)
    n_params_esem = n_items * n_factors + n_items  # all loadings + uniquenesses
    print(f"  Factors: {n_factors}, Items: {n_items}")
    print(f"  CFA params (approx): {n_params_cfa}, ESEM params (approx): {n_params_esem}")

    try:
        model = semopy.Model(model_spec)
        result = model.fit(obs_renamed)
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

        print(f"  ESEM fit indices:")
        for k, v in fit_dict.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        estimates = model.inspect()

        return {
            "fit_dict": fit_dict,
            "fit_stats": fit_stats,
            "estimates": estimates,
            "model_spec": model_spec,
            "n_obs": obs.shape[0],
            "n_items": obs.shape[1],
        }

    except Exception as e:
        print(f"  ESEM failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_esem_report(
    congruence_results: dict,
    efa_fit_results: dict,
    esem_results: dict,
    cfa_fit_dict: dict | None,
    output_path: str,
) -> None:
    """Generate markdown report comparing CFA vs ESEM fit."""
    lines = []
    lines.append("# ESEM Diagnostic Report\n")
    lines.append("*Does the 7-factor structure hold, or does it need restructuring?*\n")

    # --- Tucker congruence ---
    lines.append("## 1. Factor Replicability (Tucker Congruence)\n")
    lines.append("EFA run independently on both halves (runs 1-15 vs 16-30),")
    lines.append("then factors aligned by maximum congruence.\n")
    lines.append("| Factor | Label | Congruence | Verdict |")
    lines.append("|--------|-------|------------|---------|")

    for i, (factor, congruence) in enumerate(
        zip(congruence_results["factors"], congruence_results["congruences"])
    ):
        label = FACTOR_LABELS.get(factor, "")
        if congruence >= 0.95:
            verdict = "Excellent"
        elif congruence >= 0.85:
            verdict = "Fair"
        else:
            verdict = "Poor"
        lines.append(f"| {factor} | {label} | {congruence:.3f} | {verdict} |")

    mean_cong = np.mean(congruence_results["congruences"])
    lines.append(f"\n**Mean congruence: {mean_cong:.3f}**\n")

    if mean_cong >= 0.95:
        lines.append("> All factors replicate excellently across halves — the structure is stable.\n")
    elif mean_cong >= 0.85:
        lines.append("> Factors show fair to good replication — structure is mostly stable.\n")
    else:
        lines.append("> Some factors do not replicate well — consider reducing or merging factors.\n")

    # --- EFA residual fit ---
    lines.append("## 2. EFA Residual Fit (CFA Half)\n")
    lines.append("EFA on the CFA half with 7 factors — how well does the model reproduce")
    lines.append("the observed correlations?\n")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")

    srmr = efa_fit_results["srmr"]
    srmr_verdict = "Good" if srmr <= 0.08 else "Acceptable" if srmr <= 0.10 else "Poor"
    lines.append(f"| SRMR (from residuals) | {srmr:.4f} | {srmr_verdict} |")
    lines.append(f"| Max |residual| | {efa_fit_results['max_residual']:.4f} | — |")
    lines.append(f"| Mean |residual| | {efa_fit_results['mean_abs_residual']:.4f} | — |")
    lines.append(f"| % residuals > 0.05 | {efa_fit_results['prop_residuals_gt_05']:.1%} | — |")
    lines.append("")

    # --- ESEM vs CFA comparison ---
    lines.append("## 3. ESEM vs CFA Fit Comparison\n")

    if "error" in esem_results:
        lines.append(f"ESEM failed: {esem_results['error']}\n")
        lines.append("Falling back to Tucker congruence + EFA residual fit for the verdict.\n")
    else:
        lines.append("| Index | CFA (strict) | ESEM (free cross-loadings) | Threshold |")
        lines.append("|-------|-------------|---------------------------|-----------|")

        for idx in ["CFI", "TLI", "RMSEA"]:
            cfa_val = cfa_fit_dict.get(idx, float("nan")) if cfa_fit_dict else float("nan")
            esem_val = esem_results["fit_dict"].get(idx, float("nan"))

            if idx in ("CFI", "TLI"):
                threshold = "≥ 0.90 acceptable, ≥ 0.95 good"
            else:
                threshold = "≤ 0.08 acceptable, ≤ 0.06 good"

            lines.append(
                f"| {idx} | {cfa_val:.4f} | {esem_val:.4f} | {threshold} |"
            )

        lines.append("")

    # --- Verdict ---
    lines.append("## 4. Verdict\n")

    good_congruence = mean_cong >= 0.85
    good_srmr = srmr <= 0.08
    esem_ok = (
        not ("error" in esem_results)
        and esem_results.get("fit_dict", {}).get("CFI", 0) >= 0.90
    )

    if good_congruence and good_srmr:
        lines.append("**The 7-factor structure is defensible.**\n")
        lines.append("- Factors replicate across halves (Tucker congruence ≥ 0.85)")
        lines.append("- EFA residual fit is adequate (SRMR ≤ 0.08)")
        if esem_ok:
            lines.append("- ESEM fit is acceptable — CFA failure was due to zero cross-loading constraints")
        lines.append("\n**Recommendation:** Report ESEM as the primary measurement model in the paper,")
        lines.append("with CFA as a strict lower bound. The factors are real; cross-loadings are")
        lines.append("expected in personality research (Marsh et al., 2014). Proceed to Phase 3.\n")
    elif good_congruence and not good_srmr:
        lines.append("**Factors replicate but fit is marginal.**\n")
        lines.append("- Tucker congruence is adequate — the structure is consistent across halves")
        lines.append("- But EFA residual fit suggests more factors may be needed or some items are problematic")
        lines.append("\n**Recommendation:** Consider testing 5-6 and 8-9 factor solutions. If a nearby")
        lines.append("solution improves fit substantially, adopt it. Otherwise, report the 7-factor")
        lines.append("structure with appropriate caveats.\n")
    else:
        lines.append("**The structure needs revision.**\n")
        lines.append("- Factors do not replicate well across halves")
        lines.append("\n**Recommendation:** Re-run EFA exploration with different factor counts (3-10).")
        lines.append("Consider whether structural compression (Big Two recovery) better fits the data.\n")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text)
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_congruence_heatmap(
    loadings_efa: pd.DataFrame,
    loadings_cfa: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot Tucker congruence matrix (EFA-half factors vs CFA-half factors)."""
    shared = loadings_efa.index.intersection(loadings_cfa.index)
    phi = tucker_congruence(
        loadings_efa.loc[shared].values,
        loadings_cfa.loc[shared].values,
    )

    labels = [FACTOR_LABELS.get(f, f) for f in loadings_efa.columns]
    phi_df = pd.DataFrame(phi, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        phi_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title("Tucker Congruence: EFA-half vs CFA-half factors")
    ax.set_xlabel("CFA-half EFA factors")
    ax.set_ylabel("EFA-half factors (reference)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_distribution(
    residual_matrix: np.ndarray,
    output_path: str,
) -> None:
    """Histogram of off-diagonal residual correlations."""
    upper = residual_matrix[np.triu_indices_from(residual_matrix, k=1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(upper, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.05, color="orange", linestyle="--", label="|r| = 0.05")
    ax.axvline(-0.05, color="orange", linestyle="--")
    ax.set_xlabel("Residual correlation")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of residual correlations (7-factor EFA, CFA half)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ESEM diagnostic for 7-factor structure")
    parser.add_argument("--sweep", action="store_true",
                        help="Run trim sweep (CFA & ESEM at 3-10 items/factor)")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Run ONLY the trim sweep (skip Tucker/residual/full ESEM)")
    args = parser.parse_args()

    start = time.time()
    ensure_output_dirs()
    plots_dir = str(PLOTS_DIR)
    output_dir = str(OUTPUT_DIR)

    # ---- Load data ----
    print("Loading data...")
    df_all = load_responses()
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)

    ai_native = df_success[_is_ai_native(df_success)]
    eligible_models = get_models_for_section(df_all, section=4)
    means_df = compute_model_item_means(df_success)

    efa_df, cfa_df = split_half_data(df_success)
    print(f"  EFA half: {len(efa_df)} rows, CFA half: {len(cfa_df)} rows")

    # ---- Get EFA loadings (needed for item selection in sweep) ----
    obs_efa, w_efa = build_pooled_matrix(efa_df, eligible_models, "direct")
    obs_cfa, w_cfa = build_pooled_matrix(cfa_df, eligible_models, "direct")

    shared_items = obs_efa.columns.intersection(obs_cfa.columns)
    obs_efa = obs_efa[shared_items]
    obs_cfa = obs_cfa[shared_items]

    print("  Running EFA on EFA half...")
    efa_result_a = run_efa(obs_efa, w_efa, FORCED_N_FACTORS)
    if "error" in efa_result_a:
        print(f"  EFA-half EFA failed: {efa_result_a['error']}")
        return

    loadings_a = efa_result_a["loadings"]

    from .factor_structure import loading_report
    item_report_df = loading_report(
        loadings_a, means_df,
        threshold_primary=PRIMARY_LOADING_THRESHOLD,
        threshold_cross=CROSS_LOADING_THRESHOLD,
    )
    retained_mask = item_report_df["flag"] == ""
    retained_items = item_report_df.loc[retained_mask, "item_id"].tolist()
    print(f"  Retained items: {len(retained_items)}")

    # ---- Sweep mode ----
    if args.sweep or args.sweep_only:
        sweep_df = run_trim_sweep(
            cfa_df, eligible_models, item_report_df,
            trim_levels=[3, 4, 5, 6, 8, 10],
            output_dir=output_dir,
            plots_dir=plots_dir,
        )
        if args.sweep_only:
            elapsed = time.time() - start
            print(f"\nTotal time: {elapsed:.1f}s")
            return

    # ---- Full diagnostic (Tucker + residual + ESEM) ----
    print("\n=== 1. Tucker Congruence (cross-half EFA replication) ===")
    print(f"  Shared items: {len(shared_items)}")

    print("  Running EFA on CFA half...")
    efa_result_b = run_efa(obs_cfa, w_cfa, FORCED_N_FACTORS)
    if "error" in efa_result_b:
        print(f"  CFA-half EFA failed: {efa_result_b['error']}")
        return

    loadings_b = efa_result_b["loadings"]
    aligned_b, congruences = align_factors(loadings_a, loadings_b)

    factors = list(loadings_a.columns)
    for i, f in enumerate(factors):
        label = FACTOR_LABELS.get(f, "")
        verdict = "excellent" if congruences[i] >= 0.95 else "fair" if congruences[i] >= 0.85 else "POOR"
        print(f"  {f} ({label}): congruence = {congruences[i]:.3f} [{verdict}]")
    print(f"  Mean congruence: {np.mean(congruences):.3f}")

    congruence_results = {
        "factors": factors,
        "congruences": congruences,
        "loadings_efa": loadings_a,
        "loadings_cfa": aligned_b,
    }

    cong_plot_path = f"{plots_dir}/tucker_congruence_heatmap.png"
    plot_congruence_heatmap(loadings_a, loadings_b, cong_plot_path)

    # ---- EFA residual fit ----
    print("\n=== 2. EFA Residual Fit (CFA half) ===")

    factor_corr_b = efa_result_b.get("factor_correlation")
    if factor_corr_b is not None:
        factor_corr_b = factor_corr_b.values
    else:
        factor_corr_b = np.eye(FORCED_N_FACTORS)

    efa_fit = efa_residual_fit(
        obs_cfa, w_cfa, aligned_b.loc[shared_items].values, factor_corr_b,
    )
    print(f"  SRMR: {efa_fit['srmr']:.4f}")
    print(f"  Max |residual|: {efa_fit['max_residual']:.4f}")
    print(f"  Mean |residual|: {efa_fit['mean_abs_residual']:.4f}")
    print(f"  % residuals > 0.05: {efa_fit['prop_residuals_gt_05']:.1%}")

    resid_plot_path = f"{plots_dir}/esem_residual_distribution.png"
    plot_residual_distribution(efa_fit["residual_matrix"], resid_plot_path)

    # ---- ESEM (full 96 items) ----
    esem_results = run_esem(cfa_df, eligible_models, retained_items, item_report_df)

    # ---- CFA fit from existing report ----
    cfa_fit_dict = None
    report_path = Path(output_dir) / "primary_analysis_report.md"
    if report_path.exists():
        import re
        text = report_path.read_text()
        cfa_fit_dict = {}
        for stat in ["CFI", "TLI", "RMSEA"]:
            pattern = rf"\| {stat} \| ([\d.]+)"
            match = re.search(pattern, text)
            if match:
                cfa_fit_dict[stat] = float(match.group(1))
        if not cfa_fit_dict:
            cfa_fit_dict = None
        else:
            print(f"\n  CFA fit (from existing report): {cfa_fit_dict}")

    # ---- Generate report ----
    esem_report_path = f"{output_dir}/esem_diagnostic_report.md"
    generate_esem_report(
        congruence_results=congruence_results,
        efa_fit_results=efa_fit,
        esem_results=esem_results,
        cfa_fit_dict=cfa_fit_dict,
        output_path=esem_report_path,
    )

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
