"""Acquiescence audit for the retained 100-item AI-native instrument.

For each of the 5 factors, compute mean of forward-keyed items vs. mean of
reverse-keyed items (pre-recode) at the item-response level. A small gap
indicates acquiescence bias. Compare against BFI-44 Extraversion (gap = 0.635,
alpha = .167) as the reference point flagged in the paper.

Usage:
    python -m analysis.acquiescence_audit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

from .data_loader import (
    filter_success,
    load_responses,
    recode_reverse_items,
)
from .bfi_analysis import _is_ai_native
from .primary_analyses import (
    EFA_RUNS,
    FORCED_N_FACTORS,
    FACTOR_LABELS,
    run_efa_exploration,
    select_items,
)
from .factor_structure import build_pooled_matrix


def main():
    df = load_responses()
    df_success = filter_success(df)
    df_success_recoded = recode_reverse_items(df_success)

    ai_recoded = df_success_recoded[_is_ai_native(df_success_recoded)]
    ai_efa = ai_recoded[ai_recoded["run_number"].isin(EFA_RUNS)].copy()

    # Means (needed by select_items)
    means_df = (ai_efa.groupby(["item_id", "dimension"], as_index=False)["score"].mean())

    # Reproduce EFA + item selection (EFA needs recoded 'score' column)
    eligible = sorted(ai_efa["model_id"].unique().tolist())
    efa = run_efa_exploration(ai_efa, eligible, plots_dir="/tmp", forced_n_factors=FORCED_N_FACTORS)
    report, retained, _ = select_items(efa["loadings"], means_df, efa["communalities"])

    retained_df = report[report["retained"]][["item_id", "primary_factor", "primary_loading"]].copy()
    retained_df["factor_label"] = retained_df["primary_factor"].map(FACTOR_LABELS)
    # Sign of loading on primary factor: + → high-factor endorses, − → high-factor rejects
    retained_df["loading_sign"] = np.where(retained_df["primary_loading"] >= 0, "+", "-")

    # Pull raw (pre-recode) responses for retained items
    raw = df_success[df_success["item_id"].isin(retained_df["item_id"])][
        ["item_id", "keying", "parsed_score"]
    ].copy()
    raw = raw.merge(
        retained_df[["item_id", "factor_label", "loading_sign"]],
        on="item_id",
        how="left",
    )

    print("\n=== Acquiescence audit: retained 100-item instrument ===\n")
    print("Grouping items by SIGN OF LOADING on primary factor (not by original keying),")
    print("since EFA re-sorts items across factors. For a functioning factor, positively-loading")
    print("items should have higher raw means than negatively-loading items across the pool.\n")
    print(f"{'Factor':<18}{'n_pos':>6}{'n_neg':>6}{'pos_raw':>10}{'neg_raw':>10}{'gap':>8}   verdict")

    factor_order = ["Responsiveness", "Deference", "Boldness", "Guardedness", "Verbosity"]
    rows = []
    for fac in factor_order:
        sub = raw[raw["factor_label"] == fac]
        pos = sub[sub["loading_sign"] == "+"]
        neg = sub[sub["loading_sign"] == "-"]
        n_pos = pos["item_id"].nunique()
        n_neg = neg["item_id"].nunique()
        pos_mean = pos["parsed_score"].mean() if len(pos) else np.nan
        neg_mean = neg["parsed_score"].mean() if len(neg) else np.nan
        gap = pos_mean - neg_mean if not (np.isnan(pos_mean) or np.isnan(neg_mean)) else np.nan

        if np.isnan(gap) or n_neg == 0 or n_pos == 0:
            verdict = f"no counter-keyed items (n_pos={n_pos}, n_neg={n_neg})"
        elif abs(gap) < 0.3:
            verdict = "concerning gap (<0.3)"
        elif abs(gap) < 0.6:
            verdict = "modest"
        else:
            verdict = "OK"

        pos_s = f"{pos_mean:.3f}" if not np.isnan(pos_mean) else "  nan"
        neg_s = f"{neg_mean:.3f}" if not np.isnan(neg_mean) else "  nan"
        gap_s = f"{gap:.3f}" if not np.isnan(gap) else "  nan"
        print(f"{fac:<18}{n_pos:>6}{n_neg:>6}{pos_s:>10}{neg_s:>10}{gap_s:>8}   {verdict}")
        rows.append({
            "factor": fac,
            "n_pos_loading": n_pos,
            "n_neg_loading": n_neg,
            "pos_raw_mean": pos_mean,
            "neg_raw_mean": neg_mean,
            "gap": gap,
            "verdict": verdict,
        })

    # Cleaner test: within each model × factor, compute the gap between
    # positively-loading and negatively-loading items. Acquiescence = models
    # agreeing with both poles, i.e. pos_mean ≈ neg_mean within-model even though
    # EFA identifies these as counter-keyed on the same factor.
    print("\n--- Within-model loading-sign gap per factor (mean ± SD across 25 models) ---")
    print("Large positive gap = items behave as EFA says (functioning scale).")
    print("Gap near zero = within-model acquiescence (models agree with both sides).\n")
    print(f"{'Factor':<18}{'mean_gap':>10}{'sd_gap':>10}{'min':>8}{'max':>8}{'n<0':>6}")

    raw_with_model = df_success[df_success["item_id"].isin(retained_df["item_id"])][
        ["model_id", "item_id", "parsed_score"]
    ].merge(
        retained_df[["item_id", "factor_label", "loading_sign"]],
        on="item_id",
        how="left",
    )

    within_rows = []
    for fac in factor_order:
        sub = raw_with_model[raw_with_model["factor_label"] == fac]
        gaps = []
        for model_id, mdf in sub.groupby("model_id"):
            pos = mdf[mdf["loading_sign"] == "+"]["parsed_score"].mean()
            neg = mdf[mdf["loading_sign"] == "-"]["parsed_score"].mean()
            if not (np.isnan(pos) or np.isnan(neg)):
                gaps.append(pos - neg)
        gaps = np.array(gaps)
        if len(gaps) == 0:
            continue
        mean_gap = gaps.mean()
        sd_gap = gaps.std(ddof=1) if len(gaps) > 1 else 0.0
        print(f"{fac:<18}{mean_gap:>10.3f}{sd_gap:>10.3f}{gaps.min():>8.3f}{gaps.max():>8.3f}{int((gaps < 0).sum()):>6}")
        within_rows.append({
            "factor": fac,
            "mean_within_model_gap": mean_gap,
            "sd_within_model_gap": sd_gap,
            "min_gap": gaps.min(),
            "max_gap": gaps.max(),
            "n_models_negative_gap": int((gaps < 0).sum()),
            "n_models": len(gaps),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(within_rows).to_csv(
        OUTPUT_DIR / "acquiescence_within_model.csv",
        index=False,
    )

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "acquiescence_audit.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Reference: BFI Extraversion gap = 0.635 (alpha=.167, flagged in paper)
    print("\nFor comparison: BFI Extraversion gap = 0.635 (alpha=0.167) — flagged as acquiescent in paper.")


if __name__ == "__main__":
    main()
