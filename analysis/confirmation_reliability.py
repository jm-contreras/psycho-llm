"""Reliability of the retained instrument on the confirmation half only.

The published reliability table computes alpha/omega/split-half on the
model-means matrix built from all 30 runs, but items were selected on the
exploration half (runs 1-15), so those coefficients are not fully independent
of the selection step. This script recomputes the same coefficients on a
model-means matrix built from the confirmation half (runs 16-30) only,
using the retained item-factor assignments and loading signs from the
released scale (data/scale_v1_items.csv).

Usage:
    python -m analysis.confirmation_reliability

Output:
    analysis/output/confirmation_reliability.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import (
    OUTPUT_DIR, load_responses, filter_success, recode_reverse_items,
    compute_model_item_means, pivot_score_matrix,
)
from analysis.dimension_coherence import cronbachs_alpha
from analysis.primary_analyses import compute_mcdonalds_omega
from analysis.predictive_validity import FACTOR_NAMES

SCALE_CSV = Path(__file__).parent.parent / "data" / "scale_v1_items.csv"


def main() -> None:
    scale = pd.read_csv(SCALE_CSV)
    factor_items: dict[str, list[str]] = {}
    signs: dict[str, float] = {}
    for _, row in scale.iterrows():
        code = row["primary_factor_code"].replace("Factor", "")
        fac = row["final_factor_code"].split("-")[0]
        factor_items.setdefault(fac, []).append(row["item_id"])
        signs[row["item_id"]] = row["primary_loading_standardized"]

    df = load_responses()
    df = filter_success(df)
    df = df[(df["item_type"] == "direct") & (df["run_number"] >= 16)]
    df = recode_reverse_items(df)
    means = compute_model_item_means(df)
    sm = pivot_score_matrix(means, item_type="direct")

    lines = [
        "# Reliability on the Confirmation Half (Runs 16-30 Only)",
        "",
        "Model-means matrix (N = 25 models) built exclusively from runs 16-30; "
        "items and loading signs fixed to the released scale "
        "(selected on runs 1-15). Split-half = odd/even items, "
        "Spearman-Brown corrected.",
        "",
        "| Factor | n items | alpha | omega | split-half SB |",
        "|---|---:|---:|---:|---:|",
    ]

    for fac in ["RE", "DE", "GU", "BO", "VB"]:
        items = [i for i in factor_items.get(fac, []) if i in sm.columns]
        aligned = sm[items].copy()
        for col in aligned.columns:
            if signs.get(col, 1.0) < 0:
                aligned[col] = 6 - aligned[col]
        alpha = cronbachs_alpha(aligned, list(aligned.columns))
        omega = compute_mcdonalds_omega(aligned, list(aligned.columns))

        sorted_items = sorted(items)
        h1 = [sorted_items[i] for i in range(0, len(sorted_items), 2)]
        h2 = [sorted_items[i] for i in range(1, len(sorted_items), 2)]
        sub = aligned.dropna()
        r, _ = stats.pearsonr(sub[h1].mean(axis=1), sub[h2].mean(axis=1))
        sb = (2 * r) / (1 + r)

        omega_str = f"{omega:.3f}" if not np.isnan(omega) else "---"
        lines.append(
            f"| {FACTOR_NAMES[fac]} | {len(items)} | {alpha:.3f} "
            f"| {omega_str} | {sb:.3f} |"
        )

    out = OUTPUT_DIR / "confirmation_reliability.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
