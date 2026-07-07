"""Criterion reliability and attenuation-corrected predictive validity.

Addresses the objection that the weak instrument × human convergence could be
an artifact of criterion unreliability (item-level human ICCs of .18-.43):

  1. Estimates the reliability of the model-level human criterion directly,
     via repeated random split-halves of each model's ratings (Spearman-Brown
     corrected correlation between the two half-sample model-mean vectors).
  2. Reports attenuation-corrected instrument × human correlations
     (r_true = r_obs / sqrt(rel_instrument × rel_human)) and the maximum
     observable correlation implied by the two reliabilities.

Instrument reliability at the model level uses cross-run stability
(correlation of model-level factor scores between runs 1-15 and 16-30),
taken from the primary analysis report.

Usage:
    python -m analysis.attenuation_analysis

Output:
    analysis/output/attenuation_report.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import OUTPUT_DIR
from analysis.predictive_validity import (
    FACTORS, FACTOR_NAMES, PROMPT_FACTOR,
    load_human_ratings, load_instrument_scores, model_level_human_scores,
)

SEED = 20260705
N_SPLITS = 1_000

# Model-level instrument reliability: cross-run stability r between factor
# scores computed on runs 1-15 vs runs 16-30 (primary_analysis_report.md,
# Table "Reliability"; identical values in paper Table tab:reliability).
INSTRUMENT_RELIABILITY = {
    "RE": 0.991, "DE": 0.965, "GU": 0.975, "BO": 0.992, "VB": 0.994,
}


def split_half_reliability(
    hr: pd.DataFrame, factor: str, subset: str, rng: np.random.Generator,
) -> dict:
    """Reliability of the model-level human mean for one factor.

    Repeatedly splits each model's ratings into random halves, computes the
    correlation across models between the two half-sample mean vectors, and
    applies the Spearman-Brown prophecy formula. Returns mean/median SB and
    the mean number of ratings per model.
    """
    col = f"corrected_{factor}"
    df = hr.copy()
    if subset == "on_target":
        df["target_factor"] = df["prompt_id"].map(PROMPT_FACTOR)
        df = df[df["target_factor"] == factor]
    df = df[["model_id", col]].dropna()

    groups = {m: g[col].to_numpy() for m, g in df.groupby("model_id")}
    groups = {m: v for m, v in groups.items() if len(v) >= 2}
    n_per_model = np.mean([len(v) for v in groups.values()])

    sb_vals = []
    for _ in range(N_SPLITS):
        a_means, b_means = [], []
        for v in groups.values():
            idx = rng.permutation(len(v))
            half = len(v) // 2
            a_means.append(v[idx[:half]].mean())
            b_means.append(v[idx[half:]].mean())
        r, _ = stats.pearsonr(a_means, b_means)
        if not np.isnan(r) and r > -1:
            sb_vals.append((2 * r) / (1 + r))
    sb_vals = np.array(sb_vals)
    return {
        "n_models": len(groups),
        "mean_ratings_per_model": n_per_model,
        "reliability_mean": float(np.mean(sb_vals)),
        "reliability_median": float(np.median(sb_vals)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    hr = load_human_ratings()
    inst = load_instrument_scores().set_index("model_id")

    lines = [
        "# Criterion Reliability and Attenuation-Corrected Validity",
        "",
        f"Split-half reliability of the model-level human criterion: {N_SPLITS} "
        "random splits of each model's ratings into halves; Pearson r between "
        "the two half-sample model-mean vectors (N = 25 models), Spearman-Brown "
        "corrected. Instrument reliability = cross-run stability "
        "(runs 1-15 vs 16-30 factor scores).",
        f"Seed = {SEED}.",
        "",
    ]

    for subset, label in [("all", "All prompts"), ("on_target", "On-target prompts only")]:
        human = model_level_human_scores(hr, subset)
        merged = inst.join(human, lsuffix="_inst", rsuffix="_human", how="inner")

        lines += [
            f"## {label}",
            "",
            "| Factor | ratings/model | rel(human) | rel(inst) | r_obs | max observable r | disattenuated r |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for f in FACTORS:
            rel = split_half_reliability(hr, f, subset, rng)
            rel_h = rel["reliability_mean"]
            rel_i = INSTRUMENT_RELIABILITY[f]
            x = merged[f"{f}_inst"].to_numpy()
            y = merged[f"{f}_human"].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            r_obs, _ = stats.pearsonr(x[mask], y[mask])
            max_r = np.sqrt(max(rel_h, 0) * rel_i)
            r_corr = r_obs / max_r if max_r > 0 else np.nan
            lines.append(
                f"| {FACTOR_NAMES[f]} | {rel['mean_ratings_per_model']:.1f} "
                f"| {rel_h:.3f} | {rel_i:.3f} | {r_obs:+.3f} "
                f"| {max_r:.3f} | {r_corr:+.3f} |"
            )
        lines.append("")

    out = OUTPUT_DIR / "attenuation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
