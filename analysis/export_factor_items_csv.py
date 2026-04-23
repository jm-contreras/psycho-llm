"""Export top items per factor per k-solution to CSV for manual review.

Usage:
    python -m analysis.export_factor_items_csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

from .data_loader import (
    OUTPUT_DIR,
    compute_model_item_means,
    ensure_output_dirs,
    filter_success,
    get_models_for_section,
    load_responses,
    recode_reverse_items,
)
from .factor_structure import build_pooled_matrix
from .bfi_analysis import _is_ai_native
from .primary_analyses import split_half_data
from pipeline.item_loader import load_items


def run_forced_efa(obs_matrix, weights, n_factors):
    """Minimal forced EFA returning loadings DataFrame."""
    filled = obs_matrix.fillna(obs_matrix.mean())
    sqrt_w = np.sqrt(weights)
    wmean = np.average(filled.values, axis=0, weights=weights)
    weighted_centered = (filled.values - wmean) * sqrt_w[:, np.newaxis]
    weighted_df = pd.DataFrame(weighted_centered, columns=filled.columns)

    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)
    except Exception:
        fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)

    loadings_df = pd.DataFrame(
        fa.loadings_,
        index=obs_matrix.columns,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )
    return loadings_df


def main():
    ensure_output_dirs()

    # Load item texts
    items = load_items()
    text_map = {item["item_id"]: item["text"] for item in items}

    # Load data
    print("Loading data...")
    df_all = load_responses()
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)

    eligible_models = get_models_for_section(df_all, section=4)
    efa_df, _ = split_half_data(df_success)
    obs_matrix, weights = build_pooled_matrix(efa_df, eligible_models, "direct")

    # Dimension map
    means_df = compute_model_item_means(df_success)
    dim_map = (
        means_df[["item_id", "dimension"]].drop_duplicates()
        .set_index("item_id")["dimension"]
    )

    # Keying map
    keying_map = {item["item_id"]: item["keying"] for item in items}

    all_rows = []
    top_n = 10  # top items per factor

    for k in range(6, 13):
        print(f"  k={k}...")
        loadings_df = run_forced_efa(obs_matrix, weights, k)

        for item_id in loadings_df.index:
            abs_row = loadings_df.loc[item_id].abs()
            sorted_vals = abs_row.sort_values(ascending=False)
            primary_factor = sorted_vals.index[0]
            primary_loading = loadings_df.loc[item_id, primary_factor]
            abs_primary = sorted_vals.iloc[0]
            second_factor = sorted_vals.index[1] if len(sorted_vals) > 1 else ""
            cross_loading = sorted_vals.iloc[1] if len(sorted_vals) > 1 else 0.0

            all_rows.append({
                "k": k,
                "item_id": item_id,
                "dimension": dim_map.get(item_id, ""),
                "keying": keying_map.get(item_id, ""),
                "item_text": text_map.get(item_id, ""),
                "primary_factor": primary_factor,
                "loading": round(primary_loading, 4),
                "abs_loading": round(abs_primary, 4),
                "second_factor": second_factor,
                "cross_loading": round(cross_loading, 4),
            })

    df = pd.DataFrame(all_rows)

    # Sort: by k, then factor (numeric), then abs_loading descending
    df["factor_num"] = df["primary_factor"].str.replace("F", "").astype(int)
    df = df.sort_values(["k", "factor_num", "abs_loading"], ascending=[True, True, False])

    # Add rank within each k × factor group
    df["rank_in_factor"] = df.groupby(["k", "primary_factor"]).cumcount() + 1

    # Keep top N per factor
    df_top = df[df["rank_in_factor"] <= top_n].copy()

    # Final column order
    df_top = df_top[[
        "k", "primary_factor", "rank_in_factor", "item_id", "dimension", "keying",
        "loading", "abs_loading", "cross_loading", "second_factor", "item_text",
    ]]

    out_path = Path(OUTPUT_DIR) / "forced_factor_top_items.csv"
    df_top.to_csv(out_path, index=False)
    print(f"\nWritten {len(df_top)} rows to {out_path}")

    # Also export the full (unfiltered) version
    df_full = df[[
        "k", "primary_factor", "rank_in_factor", "item_id", "dimension", "keying",
        "loading", "abs_loading", "cross_loading", "second_factor", "item_text",
    ]]
    full_path = Path(OUTPUT_DIR) / "forced_factor_all_items.csv"
    df_full.to_csv(full_path, index=False)
    print(f"Written {len(df_full)} rows (full) to {full_path}")


if __name__ == "__main__":
    main()
