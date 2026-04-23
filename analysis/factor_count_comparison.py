"""Compare 6-factor vs 7-factor ESEM fit to decide whether F5 and F7 should merge.

Usage:
    python -m analysis.factor_count_comparison
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import (
    OUTPUT_DIR,
    PLOTS_DIR,
    ensure_output_dirs,
    filter_success,
    get_models_for_section,
    load_responses,
    compute_model_item_means,
    recode_reverse_items,
)
from .factor_structure import (
    build_pooled_matrix,
    run_efa,
    loading_report,
    _weighted_corr,
)
from .primary_analyses import (
    CFA_RUNS,
    EFA_RUNS,
    FORCED_N_FACTORS,
    PRIMARY_LOADING_THRESHOLD,
    CROSS_LOADING_THRESHOLD,
    split_half_data,
)
from .esem import (
    select_top_items,
    run_cfa_for_items,
    run_esem_for_items,
)
from .bfi_analysis import _is_ai_native


def run_k_factor_esem(
    efa_df: pd.DataFrame,
    cfa_df: pd.DataFrame,
    eligible_models: list[str],
    means_df: pd.DataFrame,
    k: int,
    top_n: int = 4,
) -> dict:
    """Run k-factor EFA on EFA half, select top_n items/factor, fit ESEM on CFA half."""
    obs_efa, w_efa = build_pooled_matrix(efa_df, eligible_models, "direct")

    print(f"\n--- {k}-factor EFA ---")
    efa_result = run_efa(obs_efa, w_efa, k)
    if "error" in efa_result:
        return {"k": k, "error": efa_result["error"]}

    loadings = efa_result["loadings"]
    factor_corr = efa_result.get("factor_correlation")

    # Item selection
    item_report = loading_report(
        loadings, means_df,
        threshold_primary=PRIMARY_LOADING_THRESHOLD,
        threshold_cross=CROSS_LOADING_THRESHOLD,
    )
    retained = item_report[item_report["flag"] == ""]
    n_retained = len(retained)

    # Top N per factor
    items, factor_items = select_top_items(item_report, top_n)
    n_factors_with_items = len(factor_items)
    n_items = sum(len(v) for v in factor_items.values())

    print(f"  Retained (full): {n_retained}, Top-{top_n}: {n_items} items across {n_factors_with_items} factors")

    # Per-factor item summary
    for f in sorted(factor_items.keys()):
        f_items = factor_items[f]
        f_report = item_report[item_report["item_id"].isin(f_items)]
        dims = f_report["dimension"].unique() if "dimension" in f_report.columns else []
        print(f"  {f}: {', '.join(f_items)} (dims: {', '.join(str(d) for d in dims)})")

    # CFA fit
    print(f"  Fitting CFA...", end=" ", flush=True)
    cfa = run_cfa_for_items(cfa_df, eligible_models, factor_items, f"CFA-{k}f")
    if "error" in cfa:
        print(f"FAILED: {cfa['error']}")
        cfa_fit = {}
    else:
        cfa_fit = cfa["fit_dict"]
        print(f"CFI={cfa_fit.get('CFI', 'N/A'):.3f}")

    # ESEM fit
    print(f"  Fitting ESEM...", end=" ", flush=True)
    esem = run_esem_for_items(cfa_df, eligible_models, factor_items)
    if "error" in esem:
        print(f"FAILED: {esem['error']}")
        esem_fit = {}
    else:
        esem_fit = esem["fit_dict"]
        print(f"CFI={esem_fit.get('CFI', 'N/A'):.3f}")

    # Factor correlations
    max_corr = None
    max_corr_pair = None
    if factor_corr is not None:
        fc = (factor_corr.values if hasattr(factor_corr, 'values') else factor_corr).copy()
        np.fill_diagonal(fc, 0)
        idx = np.unravel_index(np.argmax(np.abs(fc)), fc.shape)
        max_corr = fc[idx]
        cols = factor_corr.columns if hasattr(factor_corr, 'columns') else [f"F{i+1}" for i in range(k)]
        max_corr_pair = f"{cols[idx[0]]}-{cols[idx[1]]}"

    return {
        "k": k,
        "n_retained_full": n_retained,
        "n_items_trimmed": n_items,
        "n_factors": n_factors_with_items,
        "cfa_fit": cfa_fit,
        "esem_fit": esem_fit,
        "factor_items": factor_items,
        "factor_corr": factor_corr,
        "max_corr": max_corr,
        "max_corr_pair": max_corr_pair,
        "item_report": item_report,
    }


def generate_comparison_report(results: list[dict], output_path: str) -> None:
    """Generate markdown comparison report."""
    lines = []
    lines.append("# Factor Count Comparison: 5–9 Factors\n")
    lines.append("*Does merging F5 (Passive Epistemic Deference) and F7 (Concise Directness) improve fit?*\n")
    lines.append(f"**Method:** EFA (oblimin) on runs 1-15, top 4 items/factor, CFA & ESEM on runs 16-30.\n")

    # Summary table
    lines.append("## Fit Comparison (ESEM, top-4 items/factor)\n")
    lines.append("| k | Items | CFI | TLI | RMSEA | Max |r| between factors |")
    lines.append("|---|-------|-----|-----|-------|--------------------------|")

    for r in sorted(results, key=lambda x: x["k"]):
        if "error" in r:
            lines.append(f"| {r['k']} | — | ERROR | — | — | — |")
            continue
        ef = r["esem_fit"]
        cfi = f"{ef['CFI']:.3f}" if ef.get("CFI") is not None else "—"
        tli = f"{ef['TLI']:.3f}" if ef.get("TLI") is not None else "—"
        rmsea = f"{ef['RMSEA']:.3f}" if ef.get("RMSEA") is not None else "—"
        max_r = f"{r['max_corr']:.3f} ({r['max_corr_pair']})" if r.get("max_corr") is not None else "—"
        lines.append(f"| {r['k']} | {r['n_items_trimmed']} | {cfi} | {tli} | {rmsea} | {max_r} |")

    lines.append("")

    # CFA comparison table
    lines.append("## CFA Comparison (strict, top-4 items/factor)\n")
    lines.append("| k | Items | CFI | TLI | RMSEA |")
    lines.append("|---|-------|-----|-----|-------|")

    for r in sorted(results, key=lambda x: x["k"]):
        if "error" in r:
            continue
        cf = r["cfa_fit"]
        cfi = f"{cf['CFI']:.3f}" if cf.get("CFI") is not None else "—"
        tli = f"{cf['TLI']:.3f}" if cf.get("TLI") is not None else "—"
        rmsea = f"{cf['RMSEA']:.3f}" if cf.get("RMSEA") is not None else "—"
        lines.append(f"| {r['k']} | {r['n_items_trimmed']} | {cfi} | {tli} | {rmsea} |")

    lines.append("")

    # Per-solution factor content
    for r in sorted(results, key=lambda x: x["k"]):
        if "error" in r:
            continue
        lines.append(f"## {r['k']}-Factor Solution: Item Content\n")
        for f in sorted(r["factor_items"].keys()):
            items = r["factor_items"][f]
            ir = r["item_report"]
            f_rows = ir[ir["item_id"].isin(items)]
            if "dimension" in f_rows.columns:
                dims = f_rows["dimension"].unique()
                dim_str = ", ".join(str(d) for d in dims)
            else:
                dim_str = "—"
            lines.append(f"**{f}** ({dim_str}): {', '.join(items)}")
        lines.append("")

    # Verdict
    lines.append("## Verdict\n")

    valid = [r for r in results if "error" not in r and r["esem_fit"].get("CFI") is not None]
    if valid:
        best = max(valid, key=lambda r: r["esem_fit"]["CFI"])
        k7 = next((r for r in valid if r["k"] == 7), None)
        k6 = next((r for r in valid if r["k"] == 6), None)

        lines.append(f"**Best ESEM CFI:** {best['esem_fit']['CFI']:.3f} ({best['k']}-factor solution)\n")

        if k7 and k6:
            delta_cfi = k6["esem_fit"]["CFI"] - k7["esem_fit"]["CFI"]
            lines.append(f"**6 vs 7 factor delta CFI:** {delta_cfi:+.3f}")
            if delta_cfi > 0.01:
                lines.append("→ 6-factor solution is meaningfully better. Consider merging.\n")
            elif delta_cfi < -0.01:
                lines.append("→ 7-factor solution is meaningfully better. Keep separate factors.\n")
            else:
                lines.append("→ Difference is negligible (< 0.01). Prefer the more parsimonious (6-factor) solution.\n")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text)
    print(f"\nReport written to {output_path}")


def main():
    start = time.time()
    ensure_output_dirs()
    warnings.filterwarnings("ignore")

    print("Loading data...")
    df_all = load_responses()
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)
    eligible_models = get_models_for_section(df_all, section=4)
    means_df = compute_model_item_means(df_success)

    efa_df, cfa_df = split_half_data(df_success)
    print(f"  EFA half: {len(efa_df)} rows, CFA half: {len(cfa_df)} rows")

    results = []
    for k in [5, 6, 7, 8, 9]:
        r = run_k_factor_esem(efa_df, cfa_df, eligible_models, means_df, k, top_n=4)
        results.append(r)

    output_path = str(Path(OUTPUT_DIR) / "factor_count_comparison.md")
    generate_comparison_report(results, output_path)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
