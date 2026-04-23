"""Within-prompt rank correlation sensitivity analysis.

For each of 20 behavioral prompts, rank the 25 models by mean human rating
and by instrument factor score, then compute Kendall tau (and Pearson r on
means). Aggregate across prompts to test whether instrument scores rank-order
models consistently within a single elicitation context.

This complements the model-level Table 7 analysis (which averages over prompts)
by preserving prompt-by-prompt structure and giving 20 pseudo-replicates per
factor instead of 1.

Usage:
    python -m analysis.within_prompt_validity

Output:
    analysis/output/within_prompt_validity_report.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.predictive_validity import (
    FACTORS,
    FACTOR_NAMES,
    PROMPT_FACTOR,
    load_human_ratings,
    load_instrument_scores,
)
from analysis.data_loader import OUTPUT_DIR


def per_prompt_rankings(hr: pd.DataFrame) -> pd.DataFrame:
    """Return model-mean human rating per (prompt_id, factor) cell.

    Rows: (model_id, prompt_id). Columns: corrected_RE..corrected_VB.
    """
    score_cols = [f"corrected_{f}" for f in FACTORS]
    return (
        hr.groupby(["prompt_id", "model_id"])[score_cols]
        .mean()
        .reset_index()
        .rename(columns={f"corrected_{f}": f for f in FACTORS})
    )


def within_prompt_correlations(
    per_prompt: pd.DataFrame, inst: pd.DataFrame
) -> pd.DataFrame:
    """Per-prompt rank correlation of instrument score vs human rating.

    Returns one row per (prompt_id, instrument_factor, rated_factor) with
    Pearson r, Kendall tau, and n.
    """
    inst_idx = inst.set_index("model_id")[FACTORS]
    rows = []
    for prompt_id, sub in per_prompt.groupby("prompt_id"):
        target = PROMPT_FACTOR.get(prompt_id)
        merged = sub.merge(
            inst_idx.reset_index().rename(
                columns={f: f"inst_{f}" for f in FACTORS}
            ),
            on="model_id",
            how="inner",
        )
        if len(merged) < 4:
            continue
        for inst_f in FACTORS:
            for rated_f in FACTORS:
                x = merged[f"inst_{inst_f}"].to_numpy(dtype=float)
                y = merged[rated_f].to_numpy(dtype=float)
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 4:
                    continue
                r, p_r = stats.pearsonr(x[mask], y[mask])
                tau, p_t = stats.kendalltau(x[mask], y[mask])
                rows.append({
                    "prompt_id": prompt_id,
                    "target_factor": target,
                    "instrument_factor": inst_f,
                    "rated_factor": rated_f,
                    "r": r,
                    "tau": tau,
                    "p_r": p_r,
                    "p_tau": p_t,
                    "n": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def aggregate_across_prompts(wp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-prompt tau/r across prompts.

    For each (instrument_factor, rated_factor) pair:
      - mean_r, mean_tau across the 20 prompts
      - sd_r, sd_tau
      - one-sample t-test against 0 over the 20 prompts
      - sign test (number of negative prompts / 20)
    """
    rows = []
    for (inst_f, rated_f), sub in wp.groupby(["instrument_factor", "rated_factor"]):
        taus = sub["tau"].to_numpy()
        rs = sub["r"].to_numpy()
        n_prompts = len(sub)
        t_tau, p_t_tau = stats.ttest_1samp(taus, 0.0) if n_prompts >= 3 else (np.nan, np.nan)
        t_r, p_t_r = stats.ttest_1samp(rs, 0.0) if n_prompts >= 3 else (np.nan, np.nan)
        n_neg = int((rs < 0).sum())
        rows.append({
            "instrument_factor": inst_f,
            "rated_factor": rated_f,
            "n_prompts": n_prompts,
            "mean_r": rs.mean(),
            "sd_r": rs.std(ddof=1) if n_prompts > 1 else np.nan,
            "mean_tau": taus.mean(),
            "sd_tau": taus.std(ddof=1) if n_prompts > 1 else np.nan,
            "t_r": t_r,
            "p_t_r": p_t_r,
            "t_tau": t_tau,
            "p_t_tau": p_t_tau,
            "n_negative_r": n_neg,
        })
    return pd.DataFrame(rows)


def on_target_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Filter aggregate to diagonal (instrument_factor == rated_factor)."""
    diag = agg[agg["instrument_factor"] == agg["rated_factor"]].copy()
    diag = diag.rename(columns={"instrument_factor": "factor"}).drop(columns=["rated_factor"])
    diag["factor_name"] = diag["factor"].map(FACTOR_NAMES)
    return diag[
        ["factor", "factor_name", "n_prompts", "mean_r", "sd_r",
         "mean_tau", "sd_tau", "t_r", "p_t_r", "n_negative_r"]
    ]


def on_target_restricted(wp: pd.DataFrame) -> pd.DataFrame:
    """For each factor, restrict to the 4 on-target prompts; aggregate."""
    rows = []
    for f in FACTORS:
        sub = wp[(wp["instrument_factor"] == f)
                 & (wp["rated_factor"] == f)
                 & (wp["target_factor"] == f)]
        n_prompts = len(sub)
        if n_prompts == 0:
            continue
        rs = sub["r"].to_numpy()
        taus = sub["tau"].to_numpy()
        t_r, p_t_r = stats.ttest_1samp(rs, 0.0) if n_prompts >= 3 else (np.nan, np.nan)
        rows.append({
            "factor": f,
            "factor_name": FACTOR_NAMES[f],
            "n_prompts": n_prompts,
            "mean_r": rs.mean(),
            "sd_r": rs.std(ddof=1) if n_prompts > 1 else np.nan,
            "mean_tau": taus.mean(),
            "t_r": t_r,
            "p_t_r": p_t_r,
            "n_negative_r": int((rs < 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    hr = load_human_ratings()
    inst = load_instrument_scores()

    per_prompt = per_prompt_rankings(hr)
    wp = within_prompt_correlations(per_prompt, inst)
    agg = aggregate_across_prompts(wp)
    diag = on_target_summary(agg)
    on_tgt = on_target_restricted(wp)

    n_unique_prompts = wp["prompt_id"].nunique()
    n_unique_models = per_prompt["model_id"].nunique()

    lines: list[str] = []
    lines.append("# Within-Prompt Rank Correlation Sensitivity Analysis\n")
    lines.append(
        f"For each of {n_unique_prompts} behavioral prompts, models (N="
        f"{n_unique_models}) were ranked by mean human rating and by instrument "
        "factor score, then rank-correlated. Aggregating across prompts gives "
        "20 pseudo-replicates per factor pair instead of the single model-level "
        "correlation of Table 7.\n"
    )
    lines.append(
        "Each prompt contributes one correlation coefficient. The one-sample "
        "t-test asks whether the mean correlation across prompts differs from "
        "zero; this is **not** equivalent to the model-level test (which loses "
        "prompt-by-prompt variation by averaging first) and trades "
        "between-prompt variance for a larger effective n.\n"
    )

    lines.append("## 1. Diagonal (convergent): instrument_F × human_F, all 20 prompts\n")
    diag_show = diag.copy()
    for col in ["mean_r", "sd_r", "mean_tau", "sd_tau", "t_r", "p_t_r"]:
        diag_show[col] = diag_show[col].round(3)
    lines.append(diag_show.to_markdown(index=False))
    lines.append("")

    lines.append("## 2. Diagonal restricted to on-target prompts (4 prompts/factor)\n")
    on_tgt_show = on_tgt.copy()
    for col in ["mean_r", "sd_r", "mean_tau", "t_r", "p_t_r"]:
        on_tgt_show[col] = on_tgt_show[col].round(3)
    lines.append(on_tgt_show.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Full 5×5 convergent-discriminant matrix (mean r across 20 prompts)\n")
    mat_r = agg.pivot(index="instrument_factor", columns="rated_factor", values="mean_r").round(3)
    mat_r = mat_r.reindex(index=FACTORS, columns=FACTORS)
    lines.append(mat_r.to_markdown())
    lines.append("")
    lines.append(f"**Mean convergent (diagonal):** {np.diag(mat_r.values).mean():.3f}  ")
    off = mat_r.values[~np.eye(5, dtype=bool)]
    lines.append(f"**Mean discriminant (off-diagonal):** {off.mean():.3f}\n")

    lines.append("## 4. Full 5×5 matrix (mean tau across 20 prompts)\n")
    mat_t = agg.pivot(index="instrument_factor", columns="rated_factor", values="mean_tau").round(3)
    mat_t = mat_t.reindex(index=FACTORS, columns=FACTORS)
    lines.append(mat_t.to_markdown())
    lines.append("")

    lines.append("## 5. Per-prompt convergent correlations (diagonal cells only)\n")
    diag_long = wp[wp["instrument_factor"] == wp["rated_factor"]].copy()
    diag_long = diag_long.sort_values(["rated_factor", "prompt_id"])
    show = diag_long[["prompt_id", "target_factor", "rated_factor", "n", "r", "tau", "p_r"]].copy()
    show["r"] = show["r"].round(3)
    show["tau"] = show["tau"].round(3)
    show["p_r"] = show["p_r"].round(4)
    show = show.rename(columns={"rated_factor": "factor"})
    lines.append(show.to_markdown(index=False))
    lines.append("")

    out_path = OUTPUT_DIR / "within_prompt_validity_report.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
