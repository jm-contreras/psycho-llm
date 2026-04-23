"""Generate a paired model-profile heatmap: self-report (left) vs. judge-derived (right).

Row ordering and color scale shared; both z-scored across the 25 models.
Output: analysis/output/plots/model_profiles_paired.png
"""
from __future__ import annotations

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analysis.primary_analyses import FACTOR_LABELS  # type: ignore
from analysis.data_loader import get_short_model_name  # type: ignore

FACTORS = ["RE", "DE", "BO", "GU", "VB"]

FACTOR_FROM_PROMPT_PREFIX = {"RE": "RE", "DE": "DE", "BO": "BO", "GU": "GU", "VB": "VB"}


R1_CANONICAL = "deepseek/deepseek-reasoner"
R1_ALIASES = {"openai/deepseek-r1": R1_CANONICAL}


def load_instrument_profile() -> pd.DataFrame:
    path = os.path.join(ROOT, "analysis", "output", "factor_scores.csv")
    df = pd.read_csv(path)
    df["model_id"] = df["model_id"].replace(R1_ALIASES)
    # Average any duplicates created by the merge
    df = df.groupby("model_id", as_index=False)[FACTORS].mean()
    df["short_name"] = df["model_id"].map(get_short_model_name)
    return df


def load_judge_profile() -> pd.DataFrame:
    """Judge profile: per model, for each factor F, mean of score_F across prompts
    whose target factor is F (matched-factor ratings).

    Pulls from the SQLite judge_ratings table (full dataset).
    """
    db_path = os.path.join(ROOT, "data", "raw", "responses.db")
    conn = sqlite3.connect(db_path)
    jr = pd.read_sql_query(
        "SELECT subject_model_id, prompt_id, score_RE, score_DE, score_BO, score_GU, score_VB "
        "FROM judge_ratings WHERE parse_status='success'",
        conn,
    )
    conn.close()
    jr["subject_model_id"] = jr["subject_model_id"].replace(R1_ALIASES)
    jr["prompt_factor"] = jr["prompt_id"].str.slice(0, 2)

    rows = []
    for model_id, model_df in jr.groupby("subject_model_id"):
        row = {"model_id": model_id, "short_name": get_short_model_name(model_id)}
        for f in FACTORS:
            on_factor = model_df[model_df["prompt_factor"] == f]
            if len(on_factor) == 0:
                row[f] = np.nan
            else:
                row[f] = on_factor[f"score_{f}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def z_score(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        col = out[c]
        std = col.std(ddof=1)
        out[c] = (col - col.mean()) / std if std > 0 else 0.0
    return out


def plot_paired(instrument: pd.DataFrame, judge: pd.DataFrame, output_path: str) -> None:
    # Align both to the same models (inner join)
    merged_models = sorted(
        set(instrument["short_name"]).intersection(set(judge["short_name"]))
    )
    inst = instrument[instrument["short_name"].isin(merged_models)].copy()
    jdg = judge[judge["short_name"].isin(merged_models)].copy()

    inst_z = z_score(inst, FACTORS).set_index("short_name")[FACTORS]
    jdg_z = z_score(jdg, FACTORS).set_index("short_name")[FACTORS]

    # Row order: descending self-report Responsiveness
    order = inst_z["RE"].sort_values(ascending=False).index
    inst_z = inst_z.loc[order]
    jdg_z = jdg_z.loc[order]

    xticklabels = [FACTOR_LABELS.get(f, f) for f in FACTORS]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(6, len(order) * 0.35)), sharey=True
    )
    vmin, vmax = -3, 3

    sns.heatmap(
        inst_z, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, linewidths=0.5, ax=axes[0], cbar=False,
        xticklabels=xticklabels,
    )
    axes[0].set_title("Self-report (instrument)", fontsize=12)
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", labelsize=8)

    sns.heatmap(
        jdg_z, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, linewidths=0.5, ax=axes[1],
        cbar_kws={"label": "z-score"},
        xticklabels=xticklabels,
    )
    axes[1].set_title("Judge ensemble (behavioral)", fontsize=12)
    axes[1].set_ylabel("")

    fig.suptitle(
        "Model profiles: self-report vs. judge-rated behavior "
        "(z-scores, rows ordered by self-report Responsiveness)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired profile heatmap → {output_path}")


if __name__ == "__main__":
    instrument = load_instrument_profile()
    judge = load_judge_profile()
    out = os.path.join(ROOT, "analysis", "output", "plots", "model_profiles_paired.png")
    plot_paired(instrument, judge, out)
