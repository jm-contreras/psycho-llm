"""Forced-factor EFA exploration: compare 8, 10, 12-factor solutions.

Generates dimension × factor cross-tabs and applies multiple threshold levels
to help determine the best defensible factor count.

Usage:
    python -m analysis.forced_factor_exploration
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer

from .data_loader import (
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
from .factor_structure import build_pooled_matrix, parallel_analysis
from .bfi_analysis import _is_ai_native
from .primary_analyses import EFA_RUNS, split_half_data
from .report import df_to_markdown


# ---------------------------------------------------------------------------
# Core: run forced EFA at a given factor count
# ---------------------------------------------------------------------------

def run_forced_efa(
    obs_matrix: pd.DataFrame,
    weights: np.ndarray,
    n_factors: int,
) -> dict:
    """Run EFA forced to n_factors with PAF + oblimin."""
    filled = obs_matrix.fillna(obs_matrix.mean())
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
    except Exception as e:
        method_used = "principal_fallback"
        print(f"    minres failed ({e}), falling back to principal...")
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)

    loadings_df = pd.DataFrame(
        fa.loadings_,
        index=obs_matrix.columns,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )

    factor_corr = None
    if hasattr(fa, "phi_") and fa.phi_ is not None:
        factor_corr = pd.DataFrame(
            fa.phi_,
            index=loadings_df.columns,
            columns=loadings_df.columns,
        )

    communalities = fa.get_communalities()
    var_explained = fa.get_factor_variance()

    return {
        "loadings": loadings_df,
        "factor_correlation": factor_corr,
        "communalities": communalities,
        "variance_explained": var_explained,
        "method_used": method_used,
    }


# ---------------------------------------------------------------------------
# Dimension × Factor cross-tab
# ---------------------------------------------------------------------------

def dimension_factor_crosstab(
    loadings_df: pd.DataFrame,
    dim_map: pd.Series,
    threshold_primary: float = 0.40,
    threshold_cross: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a cross-tab: how many items from each dimension load on each factor.

    Returns:
        (crosstab_df, item_assignments_df)
        - crosstab_df: dimensions × factors count matrix
        - item_assignments_df: per-item detail (item_id, dimension, primary factor, loading, flags)
    """
    rows = []
    for item_id in loadings_df.index:
        abs_row = loadings_df.loc[item_id].abs()
        sorted_vals = abs_row.sort_values(ascending=False)
        primary_factor = sorted_vals.index[0]
        primary_loading = sorted_vals.iloc[0]
        cross_loading = sorted_vals.iloc[1] if len(sorted_vals) > 1 else 0.0
        signed_loading = loadings_df.loc[item_id, primary_factor]

        flags = []
        if primary_loading < threshold_primary:
            flags.append("low_primary")
        if cross_loading >= threshold_cross:
            flags.append(f"cross:{sorted_vals.index[1]}")

        rows.append({
            "item_id": item_id,
            "dimension": dim_map.get(item_id, "unknown"),
            "primary_factor": primary_factor,
            "signed_loading": signed_loading,
            "abs_loading": primary_loading,
            "cross_loading_val": cross_loading,
            "cross_factor": sorted_vals.index[1] if len(sorted_vals) > 1 else "",
            "retained": len(flags) == 0,
            "flags": "; ".join(flags),
        })

    assignments = pd.DataFrame(rows)

    # Cross-tab: count items from each dimension assigned to each factor
    # Only count items that pass thresholds
    retained = assignments[assignments["retained"]]
    crosstab = pd.crosstab(
        retained["dimension"],
        retained["primary_factor"],
        margins=True,
    )

    return crosstab, assignments


def dimension_factor_crosstab_all(
    loadings_df: pd.DataFrame,
    dim_map: pd.Series,
) -> pd.DataFrame:
    """Cross-tab using ALL items (no threshold filtering) — shows where items naturally cluster."""
    rows = []
    for item_id in loadings_df.index:
        abs_row = loadings_df.loc[item_id].abs()
        primary_factor = abs_row.idxmax()
        rows.append({
            "item_id": item_id,
            "dimension": dim_map.get(item_id, "unknown"),
            "primary_factor": primary_factor,
        })

    df = pd.DataFrame(rows)
    crosstab = pd.crosstab(df["dimension"], df["primary_factor"], margins=True)
    return crosstab


# ---------------------------------------------------------------------------
# Factor interpretability summary
# ---------------------------------------------------------------------------

def factor_composition_summary(
    assignments: pd.DataFrame,
    n_factors: int,
) -> pd.DataFrame:
    """For each factor, show the dominant dimension(s) and item composition."""
    retained = assignments[assignments["retained"]]

    rows = []
    for f in [f"F{i+1}" for i in range(n_factors)]:
        f_items = retained[retained["primary_factor"] == f]
        n_items = len(f_items)
        if n_items == 0:
            rows.append({
                "factor": f,
                "n_items": 0,
                "dominant_dim": "—",
                "dim_purity": 0.0,
                "composition": "empty",
            })
            continue

        dim_counts = f_items["dimension"].value_counts()
        dominant = dim_counts.index[0]
        purity = dim_counts.iloc[0] / n_items

        # Composition string: "SocialAlignment(8), Warmth(2)"
        comp_parts = [f"{dim}({c})" for dim, c in dim_counts.items()]
        composition = ", ".join(comp_parts[:4])
        if len(comp_parts) > 4:
            composition += f", +{len(comp_parts)-4} more"

        rows.append({
            "factor": f,
            "n_items": n_items,
            "dominant_dim": dominant,
            "dim_purity": purity,
            "composition": composition,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison summary across solutions
# ---------------------------------------------------------------------------

def compare_solutions(solutions: dict[int, dict], dim_map: pd.Series) -> pd.DataFrame:
    """Compare key metrics across forced-factor solutions."""
    rows = []
    for k, sol in sorted(solutions.items()):
        loadings_df = sol["loadings"]
        var_exp = sol["variance_explained"]

        # Total variance explained
        total_var = var_exp[2][-1] if len(var_exp[2]) > 0 else np.nan

        # Item retention at standard thresholds
        _, assignments = dimension_factor_crosstab(loadings_df, dim_map, 0.40, 0.30)
        n_retained_40 = assignments["retained"].sum()

        # Item retention at strict thresholds
        _, assignments_strict = dimension_factor_crosstab(loadings_df, dim_map, 0.50, 0.20)
        n_retained_50 = assignments_strict["retained"].sum()

        # Number of factors with ≥3 retained items
        factor_counts = assignments[assignments["retained"]].groupby("primary_factor").size()
        n_viable_factors = (factor_counts >= 3).sum()

        factor_counts_strict = assignments_strict[assignments_strict["retained"]].groupby("primary_factor").size()
        n_viable_strict = (factor_counts_strict >= 3).sum()

        # Mean absolute loading for retained items
        mean_loading = assignments[assignments["retained"]]["abs_loading"].mean()

        # Factor correlation stats
        fc = sol.get("factor_correlation")
        if fc is not None:
            # Off-diagonal correlations
            mask = np.triu(np.ones(fc.shape, dtype=bool), k=1)
            off_diag = fc.values[mask]
            mean_factor_r = np.abs(off_diag).mean()
            max_factor_r = np.abs(off_diag).max()
        else:
            mean_factor_r = np.nan
            max_factor_r = np.nan

        rows.append({
            "n_factors": k,
            "total_var_explained": total_var,
            "retained_items_40_30": int(n_retained_40),
            "retained_items_50_20": int(n_retained_50),
            "viable_factors_40_30": int(n_viable_factors),
            "viable_factors_50_20": int(n_viable_strict),
            "mean_abs_loading": mean_loading,
            "mean_factor_|r|": mean_factor_r,
            "max_factor_|r|": max_factor_r,
            "method": sol["method_used"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forced_loadings_heatmap(
    loadings_df: pd.DataFrame,
    dim_map: pd.Series,
    title: str,
    output_path: str,
) -> None:
    """Heatmap of loadings grouped by dimension for a forced-factor solution."""
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

    # Dimension separators + labels
    dims_ordered = [dim_map.get(item, "") for item in ordered_items]
    prev_dim = None
    dim_starts = []
    for i, dim in enumerate(dims_ordered):
        if dim != prev_dim:
            if prev_dim is not None:
                ax.axhline(i, color="black", linewidth=0.8)
            dim_starts.append((i, dim))
            prev_dim = dim

    # Add dimension labels on the right
    for idx, (start, dim) in enumerate(dim_starts):
        end = dim_starts[idx + 1][0] if idx + 1 < len(dim_starts) else len(ordered_items)
        mid = (start + end) / 2
        short = dim.split("/")[0][:20]
        ax.text(loadings_df.shape[1] + 0.3, mid, short, fontsize=6, va="center")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Items (grouped by dimension)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_crosstab_heatmap(
    crosstab: pd.DataFrame,
    title: str,
    output_path: str,
) -> None:
    """Heatmap of dimension × factor cross-tab."""
    # Remove margins row/col for the heatmap
    data = crosstab.copy()
    if "All" in data.index:
        data = data.drop("All")
    if "All" in data.columns:
        data = data.drop("All", axis=1)

    fig, ax = plt.subplots(figsize=(max(8, len(data.columns) * 0.9), max(6, len(data) * 0.5)))
    sns.heatmap(
        data, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Item count"},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Hypothesized Dimension")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    pa_results: dict,
    solutions: dict[int, dict],
    dim_map: pd.Series,
    comparison: pd.DataFrame,
    output_path: str,
    plots_dir: str,
) -> None:
    """Generate markdown report for forced-factor exploration."""
    lines = []
    lines.append("# Forced-Factor EFA Exploration")
    lines.append("")
    lines.append("*Auto-generated by `python -m analysis.forced_factor_exploration`*")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("Parallel analysis suggested 18 factors — likely over-extraction due to method artifacts, ")
    lines.append("correlated dimensions, and extreme response patterns in LLM data. This analysis compares ")
    lines.append("forced 8-, 10-, and 12-factor solutions to find the most interpretable structure that maps ")
    lines.append("to theoretically motivated dimensions.")
    lines.append("")

    # Comparison table
    lines.append("## Solution Comparison")
    lines.append("")
    lines.append(df_to_markdown(comparison))
    lines.append("")
    lines.append("- **retained_items_40_30**: Primary loading ≥ 0.40, cross-loading < 0.30")
    lines.append("- **retained_items_50_20**: Strict thresholds (≥ 0.50, < 0.20)")
    lines.append("- **viable_factors**: Factors with ≥ 3 retained items")
    lines.append("")

    report_dir = str(Path(output_path).parent)

    for k in sorted(solutions.keys()):
        sol = solutions[k]
        loadings_df = sol["loadings"]

        lines.append(f"---")
        lines.append(f"## {k}-Factor Solution")
        lines.append("")

        # Variance explained
        var_exp = sol["variance_explained"]
        total_var = var_exp[2][-1] if len(var_exp[2]) > 0 else 0
        lines.append(f"- Method: {sol['method_used']}")
        lines.append(f"- Total variance explained: {total_var:.1%}")
        lines.append("")

        # Cross-tab (all items, no threshold)
        crosstab_all = dimension_factor_crosstab_all(loadings_df, dim_map)
        lines.append(f"### Dimension × Factor Cross-Tab (all items, highest loading)")
        lines.append("")
        lines.append(df_to_markdown(crosstab_all.reset_index()))
        lines.append("")

        # Cross-tab (threshold-filtered)
        crosstab, assignments = dimension_factor_crosstab(loadings_df, dim_map, 0.40, 0.30)
        lines.append(f"### Dimension × Factor Cross-Tab (loading ≥ 0.40, cross < 0.30)")
        lines.append("")
        lines.append(df_to_markdown(crosstab.reset_index()))
        lines.append("")

        # Factor composition
        comp = factor_composition_summary(assignments, k)
        lines.append(f"### Factor Composition")
        lines.append("")
        lines.append(df_to_markdown(comp))
        lines.append("")

        # Strict thresholds
        crosstab_strict, assignments_strict = dimension_factor_crosstab(
            loadings_df, dim_map, 0.50, 0.20,
        )
        lines.append(f"### Dimension × Factor Cross-Tab (strict: loading ≥ 0.50, cross < 0.20)")
        lines.append("")
        lines.append(df_to_markdown(crosstab_strict.reset_index()))
        lines.append("")

        comp_strict = factor_composition_summary(assignments_strict, k)
        lines.append(f"### Factor Composition (strict thresholds)")
        lines.append("")
        lines.append(df_to_markdown(comp_strict))
        lines.append("")

        # Heatmap plot
        hm_path = f"{plots_dir}/forced_{k}f_loadings.png"
        plot_forced_loadings_heatmap(
            loadings_df, dim_map,
            f"Forced {k}-Factor EFA Loadings", hm_path,
        )
        rel = str(Path(hm_path).relative_to(Path(report_dir)))
        lines.append(f"![{k}-factor loadings heatmap]({rel})")
        lines.append("")

        # Cross-tab heatmap (all items)
        ct_path = f"{plots_dir}/forced_{k}f_crosstab.png"
        plot_crosstab_heatmap(
            crosstab_all,
            f"Dimension × Factor Cross-Tab ({k} factors, all items)",
            ct_path,
        )
        rel = str(Path(ct_path).relative_to(Path(report_dir)))
        lines.append(f"![{k}-factor cross-tab]({rel})")
        lines.append("")

        # Factor correlation matrix
        fc = sol.get("factor_correlation")
        if fc is not None:
            lines.append("### Factor Correlation Matrix")
            lines.append("")
            fc_display = fc.round(3)
            lines.append(df_to_markdown(fc_display.reset_index()))
            lines.append("")

        # Per-factor top items (top 5 by absolute loading, retained only)
        lines.append("### Top Items per Factor (retained, top 5 by |loading|)")
        lines.append("")
        retained = assignments[assignments["retained"]].copy()
        for f in [f"F{i+1}" for i in range(k)]:
            f_items = retained[retained["primary_factor"] == f].nlargest(5, "abs_loading")
            if len(f_items) == 0:
                lines.append(f"**{f}**: no retained items")
                lines.append("")
                continue
            lines.append(f"**{f}** ({len(retained[retained['primary_factor'] == f])} items):")
            lines.append("")
            for _, row in f_items.iterrows():
                lines.append(f"- `{row['item_id']}` ({row['dimension']}): {row['signed_loading']:.3f}")
            lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    ensure_output_dirs()
    plots_dir = str(PLOTS_DIR)

    # Load and prepare data
    print("Loading data...")
    df_all = load_responses()
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)

    eligible_models = get_models_for_section(df_all, section=4)
    print(f"  Eligible models: {len(eligible_models)}")

    # Split — use EFA half only
    efa_df, _ = split_half_data(df_success)
    print(f"  EFA half: {len(efa_df)} rows (runs 1-15)")

    # Build pooled matrix
    obs_matrix, weights = build_pooled_matrix(efa_df, eligible_models, "direct")
    print(f"  Pooled matrix: {obs_matrix.shape[0]} obs × {obs_matrix.shape[1]} items")

    # Build dimension map
    means_df = compute_model_item_means(df_success)
    dim_map = (
        means_df[["item_id", "dimension"]].drop_duplicates()
        .set_index("item_id")["dimension"]
    )

    # Parallel analysis (for reference)
    print("\nRunning parallel analysis...")
    pa = parallel_analysis(obs_matrix, weights)
    print(f"  Parallel analysis suggests: {pa['n_factors_suggested']} factors")

    # Run forced solutions
    factor_counts = list(range(6, 13))
    solutions = {}

    for k in factor_counts:
        print(f"\n=== Forced {k}-Factor EFA ===")
        sol = run_forced_efa(obs_matrix, weights, k)
        solutions[k] = sol

        # Quick summary
        _, assignments = dimension_factor_crosstab(sol["loadings"], dim_map, 0.40, 0.30)
        n_retained = assignments["retained"].sum()
        total_var = sol["variance_explained"][2][-1]
        print(f"  Method: {sol['method_used']}")
        print(f"  Variance explained: {total_var:.1%}")
        print(f"  Retained items (0.40/0.30): {n_retained}/240")

        # Show cross-tab (all items)
        ct_all = dimension_factor_crosstab_all(sol["loadings"], dim_map)
        print(f"\n  Dimension × Factor cross-tab (all items):")
        print(ct_all.to_string())

    # Comparison
    comparison = compare_solutions(solutions, dim_map)
    print(f"\n=== Solution Comparison ===")
    print(comparison.to_string(index=False))

    # Generate report
    report_path = f"{str(OUTPUT_DIR)}/forced_factor_report.md"
    generate_report(pa, solutions, dim_map, comparison, report_path, plots_dir)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
