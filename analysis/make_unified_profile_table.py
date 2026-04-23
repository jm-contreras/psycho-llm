"""Appendix Table: unified per-model z-score table across methods.

25 rows (models) × 15 columns (5 factors × 3 methods: self-report, human
raters, LLM judge). Each cell is a pool-relative z-score; with --se, each
cell is ``z ± SE`` where SE is the bootstrap SE of the raw (pre-z) mean
rescaled into z-units via the pool SD.

Output:
    paper/appendix_fragments/unified_model_profiles.tex
    analysis/output/unified_model_profiles.csv

Usage:
    python -m analysis.make_unified_profile_table            # with SEs
    python -m analysis.make_unified_profile_table --no-se    # z only
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

from analysis.data_loader import (
    filter_success,
    load_responses,
    recode_reverse_items,
)
from analysis.make_hero_profile import MODEL_FAMILY_ORDER
from analysis.profile_utils import (
    FACTORS,
    R1_ALIASES,
    ROOT,
    bootstrap_se,
    display_name,
    load_human_profile,
    load_instrument_profile,
    load_judge_profile,
)


# ---------------------------------------------------------------------------
# Bootstrap SE computation (pre-z, then rescaled by pool SD)
# ---------------------------------------------------------------------------
def _instrument_se(models: list[str]) -> pd.DataFrame:
    """Bootstrap SE of the self-report factor score, resampling over runs."""
    path = os.path.join(ROOT, "analysis", "output", "factor_scores.csv")
    df = pd.read_csv(path)
    df["model_id"] = df["model_id"].replace(R1_ALIASES)
    rng = np.random.default_rng(7)
    rows = []
    for m in models:
        sub = df[df["model_id"] == m]
        row = {"model_id": m}
        for f in FACTORS:
            row[f] = bootstrap_se(sub[f].values, n_boot=1000, rng=rng)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def _human_se(models: list[str]) -> pd.DataFrame:
    """Bootstrap SE of per-factor mean human rating (all-ratings), resampling ratings."""
    resp_path = os.path.join(ROOT, "data", "raw", "responses.db")
    with sqlite3.connect(resp_path) as conn:
        br = pd.read_sql_query(
            "SELECT id AS behavioral_response_id, model_id FROM behavioral_responses",
            conn,
        )
    prolific_path = os.path.join(ROOT, "data", "prolific", "prolific.db")
    with sqlite3.connect(prolific_path) as conn:
        pr = pd.read_sql_query(
            "SELECT behavioral_response_id, "
            "corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB "
            "FROM prolific_ratings WHERE participant_flagged = 0",
            conn,
        )
    merged = pr.merge(br, on="behavioral_response_id", how="inner")
    merged["model_id"] = merged["model_id"].replace(R1_ALIASES)
    rng = np.random.default_rng(11)
    rows = []
    for m in models:
        sub = merged[merged["model_id"] == m]
        row = {"model_id": m}
        for f in FACTORS:
            row[f] = bootstrap_se(sub[f"corrected_{f}"].values, n_boot=1000, rng=rng)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def _judge_se(models: list[str]) -> pd.DataFrame:
    """Bootstrap SE of per-factor on-target judge rating, resampling judge rows."""
    db_path = os.path.join(ROOT, "data", "raw", "responses.db")
    conn = sqlite3.connect(db_path)
    jr = pd.read_sql_query(
        "SELECT subject_model_id, prompt_id, "
        "score_RE, score_DE, score_BO, score_GU, score_VB "
        "FROM judge_ratings WHERE parse_status='success'",
        conn,
    )
    conn.close()
    jr["subject_model_id"] = jr["subject_model_id"].replace(R1_ALIASES)
    jr["prompt_factor"] = jr["prompt_id"].str.slice(0, 2)
    rng = np.random.default_rng(13)
    rows = []
    for m in models:
        sub = jr[jr["subject_model_id"] == m]
        row = {"model_id": m}
        for f in FACTORS:
            on = sub[sub["prompt_factor"] == f]
            row[f] = bootstrap_se(on[f"score_{f}"].values, n_boot=1000, rng=rng)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def _z_with_se(
    raw_means: pd.DataFrame, raw_ses: pd.DataFrame, factors: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (z-scored means, SEs rescaled to z-units)."""
    z_means = raw_means.copy()
    z_ses = raw_ses.copy()
    for f in factors:
        col = raw_means[f]
        sd = col.std(ddof=1)
        if sd > 0:
            z_means[f] = (col - col.mean()) / sd
            z_ses[f] = raw_ses[f] / sd
        else:
            z_means[f] = 0.0
            z_ses[f] = np.nan
    return z_means, z_ses


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------
METHODS = [("Self", "self"), ("Human", "human"), ("Judge", "judge")]


def build_table(with_se: bool) -> pd.DataFrame:
    inst = load_instrument_profile().set_index("model_id")
    hum = load_human_profile(on_target_only=False).set_index("model_id")
    jdg = load_judge_profile().set_index("model_id")

    models = [m for m in MODEL_FAMILY_ORDER
              if m in inst.index and m in hum.index and m in jdg.index]

    inst = inst.loc[models, FACTORS]
    hum = hum.loc[models, FACTORS]
    jdg = jdg.loc[models, FACTORS]

    if with_se:
        inst_se = _instrument_se(models).loc[models, FACTORS]
        hum_se = _human_se(models).loc[models, FACTORS]
        jdg_se = _judge_se(models).loc[models, FACTORS]
    else:
        inst_se = hum_se = jdg_se = pd.DataFrame(
            0.0, index=models, columns=FACTORS
        )

    inst_z, inst_sez = _z_with_se(inst, inst_se, FACTORS)
    hum_z, hum_sez = _z_with_se(hum, hum_se, FACTORS)
    jdg_z, jdg_sez = _z_with_se(jdg, jdg_se, FACTORS)

    rows = []
    for m in models:
        row = {"Model": display_name(m)}
        for f in FACTORS:
            for method_label, (z_df, se_df) in zip(
                ("Self", "Human", "Judge"),
                (
                    (inst_z, inst_sez),
                    (hum_z, hum_sez),
                    (jdg_z, jdg_sez),
                ),
            ):
                z_val = z_df.loc[m, f]
                se_val = se_df.loc[m, f]
                if pd.isna(z_val):
                    cell = "—"
                elif with_se and not pd.isna(se_val):
                    cell = f"{z_val:+.2f} ± {se_val:.2f}"
                else:
                    cell = f"{z_val:+.2f}"
                row[f"{f}·{method_label}"] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, output_path: str, with_se: bool) -> None:
    factor_names = {
        "RE": "Responsiveness",
        "DE": "Deference",
        "BO": "Boldness",
        "GU": "Guardedness",
        "VB": "Verbosity",
    }
    col_count = 1 + 3 * len(FACTORS)
    col_spec = "l" + "r" * (3 * len(FACTORS))

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    cap = (
        r"\caption{Per-model z-scores across five AI-native factors and three "
        r"measurement methods (self-report, human raters, LLM-judge ensemble). "
        r"Z-scores are pool-relative (25-model pool) within each method; both "
        r"endpoints of a comparison live in the same scale."
    )
    if with_se:
        cap += (
            r" Cells show $z \pm \mathrm{SE}$ (bootstrap SE, 1{,}000 resamples: "
            r"runs for self-report; ratings for human; judge responses for LLM). "
            r"SE widths are not directly comparable across methods because the "
            r"resampling unit differs."
        )
    cap += r"}"
    lines.append(cap)
    lines.append(r"\label{tab:unified-profiles}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    top = ["", ]
    for f in FACTORS:
        top.append(r"\multicolumn{3}{c}{" + factor_names[f] + "}")
    lines.append(" & ".join(top) + r" \\")
    cmid_parts = []
    for i in range(len(FACTORS)):
        start = 2 + i * 3
        end = start + 2
        cmid_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append("".join(cmid_parts))
    sub = ["Model"]
    for _ in FACTORS:
        sub += ["Self", "Human", "Judge"]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = [row["Model"]]
        for f in FACTORS:
            for m in ("Self", "Human", "Judge"):
                cells.append(row[f"{f}·{m}"])
        # Escape any ± if needed — LaTeX handles ± as unicode with inputenc.
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-se", action="store_true",
                    help="Omit bootstrap SEs (cleaner but less informative).")
    args = ap.parse_args()
    with_se = not args.no_se

    df = build_table(with_se=with_se)

    csv_path = os.path.join(ROOT, "analysis", "output", "unified_model_profiles.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    tex_path = os.path.join(
        ROOT, "paper", "appendix_fragments", "unified_model_profiles.tex"
    )
    to_latex(df, tex_path, with_se=with_se)


if __name__ == "__main__":
    main()
