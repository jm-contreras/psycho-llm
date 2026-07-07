"""Export the final 100-item scale (v1) as a CSV for data release.

Reproduces the EFA + item selection from primary_analyses (forced k=5, primary
loading >= 0.40, cross-loading < 0.30), joins in item text from the item pool,
and writes data/scale_v1_items.csv with the fields needed to score and reuse
the instrument.

Usage:
    python -m analysis.make_scale_v1_csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.item_loader import load_items

from .data_loader import filter_success, load_responses, recode_reverse_items
from .bfi_analysis import _is_ai_native
from .judge_analysis import _EFA_FACTOR_TO_CODE
from .primary_analyses import (
    EFA_RUNS,
    FACTOR_LABELS,
    FORCED_N_FACTORS,
    run_efa_exploration,
    select_items,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "scale_v1_items.csv"


def main() -> None:
    # 1. Reproduce EFA half + select_items
    df = load_responses()
    df_success = filter_success(df)
    df_success_recoded = recode_reverse_items(df_success)

    ai = df_success_recoded[_is_ai_native(df_success_recoded)]
    ai_efa = ai[ai["run_number"].isin(EFA_RUNS)].copy()

    means_df = ai_efa.groupby(["item_id", "dimension"], as_index=False)["score"].mean()
    eligible = sorted(ai_efa["model_id"].unique().tolist())

    efa = run_efa_exploration(
        ai_efa, eligible, plots_dir="/tmp", forced_n_factors=FORCED_N_FACTORS,
    )
    report, _, _ = select_items(efa["loadings"], means_df, efa["communalities"])
    loadings = efa["loadings"]  # rows=item_id, cols=Factor1..Factor5

    retained = report[report["retained"]].copy()
    retained["factor_label"] = retained["primary_factor"].map(FACTOR_LABELS)

    # 2. Secondary (cross) loading: largest |loading| on a non-primary factor
    secondary_factor = []
    secondary_loading = []
    for item_id, primary in zip(retained["item_id"], retained["primary_factor"]):
        row = loadings.loc[item_id].drop(primary)
        sec_factor = row.abs().idxmax()
        secondary_factor.append(sec_factor)
        secondary_loading.append(loadings.loc[item_id, sec_factor])
    retained["secondary_factor"] = secondary_factor
    retained["secondary_factor_label"] = pd.Series(secondary_factor).map(FACTOR_LABELS).values
    retained["secondary_loading"] = secondary_loading

    # 3. Join item text + original keying from the item pool
    item_pool = pd.DataFrame(load_items())
    item_pool = item_pool[item_pool["item_type"] == "direct"][
        ["item_id", "dimension_code", "text", "keying"]
    ].rename(columns={"keying": "original_keying", "text": "item_text"})

    out = retained.merge(item_pool, on="item_id", how="left")

    # 4. Final column layout + sort
    out = out.rename(columns={
        "dimension": "a_priori_dimension",
        "primary_factor": "primary_factor_code",
        "factor_label": "primary_factor",
        "primary_loading": "primary_loading_standardized",
        "secondary_factor": "secondary_factor_code",
        "secondary_factor_label": "secondary_factor",
        "secondary_loading": "secondary_loading_standardized",
    })

    cols = [
        "item_id",
        "item_text",
        "a_priori_dimension",
        "dimension_code",
        "primary_factor",
        "primary_factor_code",
        "primary_loading_standardized",
        "secondary_factor",
        "secondary_factor_code",
        "secondary_loading_standardized",
        "original_keying",
    ]
    out = out[cols].copy()

    factor_order = ["Responsiveness", "Deference", "Boldness", "Guardedness", "Verbosity"]
    out["_f"] = out["primary_factor"].apply(factor_order.index)
    out = out.sort_values(
        ["_f", "primary_loading_standardized"],
        key=lambda s: s.abs() if s.name == "primary_loading_standardized" else s,
        ascending=[True, False],
    ).drop(columns=["_f"])

    # Final-factor item codes: <CODE>-NN within each factor, ordered by |loading| desc
    # (i.e., the current sort order). Keeps the a-priori item_id for traceability.
    two_letter = out["primary_factor_code"].map(_EFA_FACTOR_TO_CODE)
    seq = (out.groupby("primary_factor_code").cumcount() + 1).map("{:02d}".format)
    out.insert(1, "final_factor_code", (two_letter + "-" + seq).values)

    # Round loadings for readability
    for c in ("primary_loading_standardized", "secondary_loading_standardized"):
        out[c] = out[c].round(3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} items to {OUT_PATH}")
    print(out.groupby("primary_factor").size().to_string())


if __name__ == "__main__":
    main()
