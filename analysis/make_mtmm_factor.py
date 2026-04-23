"""Regenerate MTMM heatmap at the factor level (5 AI factors x 5 BFI traits).

Uses forward-keyed-only Extraversion in place of the full scale (since full-E
has poor reliability, alpha = .17), but labels the column simply "Extraversion".
Drops the redundant "Extraversion (fwd-only)" column that previously appeared.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analysis.make_appendix_tables import RETAINED  # type: ignore

R1_CANONICAL = "deepseek/deepseek-reasoner"
R1_ALIASES = {"openai/deepseek-r1": R1_CANONICAL}

BFI_DIMENSIONS = [
    "Openness", "Conscientiousness", "Extraversion",
    "Agreeableness", "Neuroticism",
]
BFI_DIM_SHORT = {
    "Openness": "O", "Conscientiousness": "C", "Extraversion": "E",
    "Agreeableness": "A", "Neuroticism": "N",
}
FACTOR_CODE = {
    "Factor1": "Responsiveness", "Factor2": "Deference", "Factor3": "Boldness",
    "Factor4": "Guardedness", "Factor5": "Verbosity",
}


def load_model_scores() -> pd.DataFrame:
    conn = sqlite3.connect(os.path.join(ROOT, "data", "raw", "responses.db"))
    df = pd.read_sql_query(
        "SELECT model_id, item_id, dimension, keying, parsed_score "
        "FROM responses WHERE status='success' AND parsed_score IS NOT NULL",
        conn,
    )
    conn.close()
    df["model_id"] = df["model_id"].replace(R1_ALIASES)
    df["score_eff"] = np.where(df["keying"] == "-", 6 - df["parsed_score"], df["parsed_score"])
    return df


def build_ai_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    # AI-native items are those with dimension not in BFI_DIMENSIONS
    ai = df[~df["dimension"].isin(BFI_DIMENSIONS)].copy()
    rows = []
    for mid, mdf in ai.groupby("model_id"):
        row = {"model_id": mid}
        for fkey, items_u in RETAINED.items():
            items = [i.replace("_", "-") for i in items_u]
            vals = mdf[mdf["item_id"].isin(items)]["score_eff"]
            row[FACTOR_CODE[fkey]] = vals.mean() if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def build_bfi_scores(df: pd.DataFrame) -> pd.DataFrame:
    bfi = df[df["dimension"].isin(BFI_DIMENSIONS)].copy()
    rows = []
    for mid, mdf in bfi.groupby("model_id"):
        row = {"model_id": mid}
        for dim in BFI_DIMENSIONS:
            sub = mdf[mdf["dimension"] == dim]
            if dim == "Extraversion":
                # Use forward-keyed-only (full E has alpha = .17)
                sub = sub[sub["keying"] == "+"]
            row[BFI_DIM_SHORT[dim]] = sub["score_eff"].mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def main() -> None:
    df = load_model_scores()
    ai = build_ai_factor_scores(df)
    bfi = build_bfi_scores(df)

    common = ai.index.intersection(bfi.index)
    ai = ai.loc[common]
    bfi = bfi.loc[common]
    n = len(common)
    print(f"Models with both AI-native + BFI: {n}")

    factor_names = list(FACTOR_CODE.values())
    bfi_names = [BFI_DIM_SHORT[d] for d in BFI_DIMENSIONS]

    corr = pd.DataFrame(index=factor_names, columns=bfi_names, dtype=float)
    pvals = pd.DataFrame(index=factor_names, columns=bfi_names, dtype=float)
    for f in factor_names:
        for b in bfi_names:
            x = ai[f].to_numpy()
            y = bfi[b].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 4:
                corr.loc[f, b] = np.nan
                pvals.loc[f, b] = np.nan
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            corr.loc[f, b] = r
            pvals.loc[f, b] = p

    annot = corr.copy().astype(object)
    for f in factor_names:
        for b in bfi_names:
            r_val = corr.loc[f, b]
            annot.loc[f, b] = "" if np.isnan(r_val) else f"{r_val:.2f}"

    full_bfi_labels = ["Openness", "Conscientiousness", "Extraversion",
                       "Agreeableness", "Neuroticism"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        corr.to_numpy(dtype=float),
        annot=annot.to_numpy(), fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=full_bfi_labels, yticklabels=factor_names,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(
        f"MTMM: AI-Native Factors × BFI-44 Traits (N = {n} models)",
        fontsize=10,
    )
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    out = os.path.join(ROOT, "analysis", "output", "plots", "mtmm_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
