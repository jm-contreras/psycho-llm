"""Compute per-model reliability for each of the 5 factors.

Within a single model, Cronbach's alpha is dominated by between-item (not
between-run) variance, so we report two complementary coefficients:

  alpha  : Cronbach's alpha across items, using item-level means computed over
           that model's 30 runs (one row per item), with reverse-coded items
           flipped. Reflects how consistently the model's items hang together
           for that model. Undefined (NaN) if the item means have zero variance.
  split  : Spearman-Brown corrected split-half correlation between odd- and
           even-indexed runs' factor mean scores. Reflects how stable the
           model's factor-level mean is across runs.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analysis.data_loader import get_short_model_name  # type: ignore
from analysis.make_appendix_tables import RETAINED, FACTOR_NAMES, tex_escape  # type: ignore

OUT_DIR = os.path.join(ROOT, "paper", "appendix_fragments")
DB_PATH = os.path.join(ROOT, "data", "raw", "responses.db")


def cronbach_alpha(matrix: np.ndarray) -> float:
    """Cronbach's alpha across items (columns), with rows as observations.

    For per-model reliability we treat each of the model's runs as one
    observation and each retained item as a column. When a model is fully
    deterministic (zero run-to-run variance on every item), we report 1.00
    since the model's responses are perfectly self-consistent.
    """
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float("nan")
    # Perfect-consistency case: every run identical on every item.
    if np.all(matrix.var(axis=0, ddof=0) == 0):
        return 1.0
    k = matrix.shape[1]
    item_vars = matrix.var(axis=0, ddof=1)
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 1.0
    return (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)


def split_half_sb(run_means: pd.Series) -> float:
    """Spearman-Brown corrected split-half: correlate odd vs. even runs' factor means.

    If a model is fully deterministic across runs, return 1.00 (perfect stability).
    """
    run_means = run_means.dropna()
    if len(run_means) < 4:
        return float("nan")
    odd = run_means[run_means.index % 2 == 1]
    even = run_means[run_means.index % 2 == 0]
    n = min(len(odd), len(even))
    if n < 2:
        return float("nan")
    odd = odd.iloc[:n].to_numpy()
    even = even.iloc[:n].to_numpy()
    if odd.std() == 0 and even.std() == 0 and np.allclose(odd.mean(), even.mean()):
        return 1.0
    if odd.std() == 0 or even.std() == 0:
        return float("nan")
    r = np.corrcoef(odd, even)[0, 1]
    if np.isnan(r):
        return float("nan")
    return (2 * r) / (1 + r) if (1 + r) != 0 else float("nan")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT model_id, item_id, run_number, parsed_score, keying, status "
        "FROM responses WHERE status='success' AND parsed_score IS NOT NULL",
        conn,
    )
    conn.close()
    print(f"Loaded {len(df):,} rows from {DB_PATH}")

    # Merge DeepSeek R1 endpoints: treat the Azure (openai/deepseek-r1) and
    # DeepSeek-native (deepseek/deepseek-reasoner) pools as one model.
    df["model_id"] = df["model_id"].replace(
        {"openai/deepseek-r1": "deepseek/deepseek-reasoner"}
    )

    # Reverse-code: if keying == '-', use 6 - score (5-point scale)
    df["score_eff"] = np.where(df["keying"] == "-", 6 - df["parsed_score"], df["parsed_score"])

    rows = []
    for model_id, mdf in df.groupby("model_id"):
        row = {"model_id": model_id, "short_name": get_short_model_name(model_id)}
        for fkey, item_ids_u in RETAINED.items():
            items = [i.replace("_", "-") for i in item_ids_u]
            sub = mdf[mdf["item_id"].isin(items)]
            if len(sub) == 0:
                row[f"{fkey}_alpha"] = float("nan")
                row[f"{fkey}_sb"] = float("nan")
                continue
            pivot = sub.pivot_table(
                index="run_number", columns="item_id", values="score_eff", aggfunc="mean"
            ).reindex(columns=items)
            # Alpha treats each run as an observation, items as columns.
            row[f"{fkey}_alpha"] = cronbach_alpha(pivot.to_numpy())
            # Split-half: factor mean per run, correlate odd vs even runs.
            row[f"{fkey}_sb"] = split_half_sb(pivot.mean(axis=1))
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("short_name")
    csv_path = os.path.join(ROOT, "analysis", "output", "per_model_reliability.csv")
    result.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    factors = ["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"]
    # LaTeX table: two-row header (factor name | alpha / SB), one row per model.
    lines = [
        "% Auto-generated by analysis/make_per_model_reliability.py",
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}l" + ("rr" * len(factors)) + r"@{}}",
        r"\toprule",
        r"Model & " + " & ".join(
            r"\multicolumn{2}{c}{" + FACTOR_NAMES[f] + "}" for f in factors
        ) + r" \\",
        r" & " + " & ".join(r"$\alpha$ & SB" for _ in factors) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in result.iterrows():
        cells = []
        for f in factors:
            a = r[f"{f}_alpha"]
            s = r[f"{f}_sb"]
            cells.append("--" if pd.isna(a) else f"{a:.2f}")
            cells.append("--" if pd.isna(s) else f"{s:.2f}")
        lines.append(tex_escape(r["short_name"]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    lines.append(
        r"\noindent\footnotesize\emph{Note.} $\alpha$ = Cronbach's alpha across retained items for that factor, treating each of the model's 30 runs as an observation; negative or near-zero values indicate that the model's run-to-run variation is small relative to between-item variation (high determinism). SB = Spearman-Brown corrected split-half correlation between odd- and even-indexed runs' factor mean scores; high values indicate that the model's factor-level mean is stable across runs."
    )
    out = os.path.join(OUT_DIR, "per_model_reliability.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
