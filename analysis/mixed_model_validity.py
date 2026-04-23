"""Mixed-model / cluster-robust sensitivity analysis for predictive validity.

The Table 7 analysis aggregates human ratings to per-model means (N=25) and
correlates with instrument factor scores. Because the instrument score is
constant within model, the effective sample size for the fixed effect is
bounded by the number of models regardless of how many ratings are pooled.

This script runs two rating-level sensitivity analyses that use all 906
ratings while properly accounting for clustering:

  1. OLS with cluster-robust SEs (CR1) clustered by model. Standard
     econometric approach for level-2 predictors. Avoids the boundary /
     singularity problems of random-intercept models when the predictor is
     constant within cluster.

  2. Mixed model with crossed REs: model + sample + rater. Captures sample
     and rater variance components; model-level residual variance is
     typically absorbed by the fixed effect, so this is used as a secondary
     check that the rater/sample structure doesn't change inference.

Both are run on (a) all prompts and (b) on-target prompts only, per factor.

Usage:
    python -m analysis.mixed_model_validity

Output:
    analysis/output/mixed_model_report.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analysis.data_loader import OUTPUT_DIR
from analysis.predictive_validity import (
    FACTORS,
    FACTOR_NAMES,
    PROMPT_FACTOR,
    load_instrument_scores,
    load_human_ratings,
)

REPO_ROOT = Path(__file__).parent.parent


# ── Data prep ────────────────────────────────────────────────────────────────

def build_rating_level_df(inst: pd.DataFrame, hr: pd.DataFrame) -> pd.DataFrame:
    """Merge rating-level human data with per-model instrument scores.

    Returns DataFrame with one row per human rating, columns:
      model_id, behavioral_response_id, prompt_id, prolific_pid,
      corrected_RE..corrected_VB (outcomes),
      inst_RE..inst_VB (predictors, constant within model),
      target_factor (which factor this prompt targets).
    """
    inst_renamed = inst.rename(columns={f: f"inst_{f}" for f in FACTORS})
    df = hr.merge(inst_renamed, on="model_id", how="inner")
    df["target_factor"] = df["prompt_id"].map(PROMPT_FACTOR)
    return df


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd > 0 else s - mu


# ── Estimators ───────────────────────────────────────────────────────────────

def fit_ols_cluster_robust(df: pd.DataFrame, factor: str) -> dict:
    """OLS with CR1 cluster-robust SEs clustered by model.

    Spec: corrected_F ~ instrument_F (z-scored), SEs clustered on model_id.
    Beta is directly comparable to a model-level Pearson r in magnitude.
    """
    y_col = f"corrected_{factor}"
    x_col = f"inst_{factor}"
    sub = df[[y_col, x_col, "model_id"]].dropna().copy()
    if len(sub) < 50 or sub["model_id"].nunique() < 5:
        return {"factor": factor, "converged": False, "reason": "too few data"}

    sub["x_z"] = _zscore(sub[x_col])

    # z-score outcome too so beta ≈ r
    sub["y_z"] = _zscore(sub[y_col])

    model = smf.ols("y_z ~ x_z", data=sub)
    fit = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["model_id"].values},
    )

    beta = float(fit.params["x_z"])
    se = float(fit.bse["x_z"])
    p = float(fit.pvalues["x_z"])
    ci_lo, ci_hi = fit.conf_int().loc["x_z"].tolist()

    return {
        "factor": factor,
        "factor_name": FACTOR_NAMES[factor],
        "beta_z": round(beta, 3),
        "se": round(se, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "p": round(p, 4),
        "n_obs": len(sub),
        "n_models": int(sub["model_id"].nunique()),
        "converged": True,
    }


def fit_cluster_bootstrap(df: pd.DataFrame, factor: str, n_boot: int = 2000,
                           seed: int = 42) -> dict:
    """Cluster bootstrap of OLS beta, resampling models with replacement.

    Resamples *models* (with all their ratings) to generate the sampling
    distribution of the beta. This is the bootstrap analog of CR1 and handles
    small-cluster (N=25) cases where asymptotic cluster-robust SEs can be
    liberal.
    """
    y_col = f"corrected_{factor}"
    x_col = f"inst_{factor}"
    sub = df[[y_col, x_col, "model_id"]].dropna().copy()
    if len(sub) < 50 or sub["model_id"].nunique() < 5:
        return {"factor": factor, "converged": False, "reason": "too few data"}

    # Precompute
    sub["x_z"] = _zscore(sub[x_col])
    sub["y_z"] = _zscore(sub[y_col])

    # Point estimate
    fit = smf.ols("y_z ~ x_z", data=sub).fit()
    beta_hat = float(fit.params["x_z"])

    # Cluster bootstrap
    rng = np.random.default_rng(seed)
    models = sub["model_id"].unique()
    n_models = len(models)
    # Pre-split
    by_model = {m: sub[sub["model_id"] == m] for m in models}

    betas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_models, n_models)
        resampled = pd.concat([by_model[models[i]] for i in idx], ignore_index=True)
        # Re-z-score within the resample to preserve scale
        resampled["x_z"] = _zscore(resampled[x_col])
        resampled["y_z"] = _zscore(resampled[y_col])
        try:
            b_fit = smf.ols("y_z ~ x_z", data=resampled).fit()
            betas[b] = b_fit.params["x_z"]
        except Exception:
            betas[b] = np.nan

    betas = betas[~np.isnan(betas)]
    ci_lo, ci_hi = np.percentile(betas, [2.5, 97.5])

    return {
        "factor": factor,
        "factor_name": FACTOR_NAMES[factor],
        "beta_z": round(beta_hat, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "n_obs": len(sub),
        "n_models": int(n_models),
        "n_boot": len(betas),
        "converged": True,
    }


def fit_crossed_mixed(df: pd.DataFrame, factor: str) -> dict:
    """Mixed model: rating ~ inst_F + (1|model) + (1|sample) + (1|rater).

    The model-level random intercept is often absorbed into the fixed effect
    when the predictor is constant within cluster; the sample and rater VCs
    capture the remaining clustering.
    """
    y_col = f"corrected_{factor}"
    x_col = f"inst_{factor}"
    needed = [y_col, x_col, "model_id", "behavioral_response_id", "prolific_pid"]
    sub = df[needed].dropna().copy()
    if len(sub) < 50 or sub["model_id"].nunique() < 5:
        return {"factor": factor, "converged": False, "reason": "too few data"}

    sub["x_z"] = _zscore(sub[x_col])
    sub["y_z"] = _zscore(sub[y_col])
    sub["sample_id"] = sub["behavioral_response_id"].astype(str)
    sub["rater_id"] = sub["prolific_pid"].astype(str)

    vc_formula = {"sample": "0 + C(sample_id)", "rater": "0 + C(rater_id)"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            md = smf.mixedlm(
                "y_z ~ x_z", data=sub, groups=sub["model_id"],
                vc_formula=vc_formula, re_formula="1",
            )
            mdf = md.fit(method="lbfgs", reml=True)
        except Exception as e:
            return {"factor": factor, "converged": False, "reason": str(e)[:80]}

    if not mdf.converged:
        return {"factor": factor, "converged": False, "reason": "did not converge"}

    beta = float(mdf.fe_params["x_z"])
    se = float(mdf.bse_fe["x_z"])
    if not np.isfinite(se) or se <= 0:
        return {"factor": factor, "converged": False, "reason": "non-finite SE"}

    p = float(mdf.pvalues["x_z"])
    ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se

    return {
        "factor": factor,
        "factor_name": FACTOR_NAMES[factor],
        "beta_z": round(beta, 3),
        "se": round(se, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "p": round(p, 4),
        "n_obs": len(sub),
        "n_models": int(sub["model_id"].nunique()),
        "n_samples": int(sub["sample_id"].nunique()),
        "n_raters": int(sub["rater_id"].nunique()),
        "converged": True,
    }


# ── Formatting ───────────────────────────────────────────────────────────────

def _fmt_row(r: dict) -> dict:
    if not r.get("converged", False):
        return {
            "factor": r["factor"],
            "β_z": "—",
            "95% CI": f"failed: {r.get('reason', '')[:40]}",
            "p": "—",
            "n_obs": "—",
            "n_models": "—",
        }
    return {
        "factor": r["factor"],
        "β_z": r["beta_z"],
        "95% CI": f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]",
        "p": r.get("p", "—"),
        "n_obs": r["n_obs"],
        "n_models": r["n_models"],
    }


def _run_set(sub: pd.DataFrame, label: str, fn) -> pd.DataFrame:
    return pd.DataFrame([_fmt_row(fn(sub, f)) for f in FACTORS])


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report() -> str:
    lines = ["# Rating-Level Sensitivity Analysis: Predictive Validity\n"]
    lines.append(
        "The Table 7 model-level correlations are tests of a level-2 "
        "relationship (instrument score × behavior) with effective sample "
        "size bounded by the number of models ($N = 25$). This report refits "
        "the same question at the rating level ($N = 906$) using "
        "clustering-aware estimators to verify that model-level aggregation "
        "does not hide structure the model-level test cannot see.\n"
    )
    lines.append("### Estimators\n")
    lines.append(
        "**OLS / cluster-robust.** `y_z ~ x_z` on all ratings, with CR1 "
        "cluster-robust SEs clustered on `model_id`. `y_z` and `x_z` are "
        "z-scored so `β_z` is directly comparable to a model-level Pearson `r`.\n"
    )
    lines.append(
        "**Cluster bootstrap (2 000 reps).** Resamples *models* with "
        "replacement (block bootstrap) and refits OLS. Percentile CI. "
        "More conservative than CR1 at small cluster counts.\n"
    )
    lines.append(
        "**Crossed mixed model.** `y_z ~ x_z + (1 | model) + (1 | sample) + "
        "(1 | rater)`. Residual ML via `statsmodels.MixedLM` with VC formulas.\n"
    )

    inst = load_instrument_scores()
    hr = load_human_ratings()
    df = build_rating_level_df(inst, hr)
    lines.append(
        f"**Data:** {len(df):,} rating-level observations across "
        f"{df['model_id'].nunique()} models, "
        f"{df['behavioral_response_id'].nunique()} samples, "
        f"{df['prolific_pid'].nunique()} raters.\n"
    )

    # ── All prompts ──
    lines.append("## 1. All Prompts\n")
    lines.append("### 1a. OLS with cluster-robust SEs\n")
    lines.append(_run_set(df, "all", fit_ols_cluster_robust).to_markdown(index=False))
    lines.append("")

    lines.append("### 1b. Cluster bootstrap (percentile CI, 2 000 reps)\n")
    lines.append(_run_set(df, "all", fit_cluster_bootstrap).to_markdown(index=False))
    lines.append("")

    lines.append("### 1c. Crossed mixed model\n")
    lines.append(_run_set(df, "all", fit_crossed_mixed).to_markdown(index=False))
    lines.append("")

    # ── On-target ──
    lines.append("## 2. On-Target Prompts Only\n")
    lines.append(
        "Each factor fit on only the four prompts designed to elicit that "
        "factor (typically 4 prompts × 5 runs × ~2 raters ≈ 150–220 ratings).\n"
    )

    lines.append("### 2a. OLS with cluster-robust SEs\n")
    ols_on_rows = []
    for f in FACTORS:
        sub = df[df["target_factor"] == f]
        ols_on_rows.append(_fmt_row(fit_ols_cluster_robust(sub, f)))
    lines.append(pd.DataFrame(ols_on_rows).to_markdown(index=False))
    lines.append("")

    lines.append("### 2b. Cluster bootstrap (percentile CI, 2 000 reps)\n")
    boot_on_rows = []
    for f in FACTORS:
        sub = df[df["target_factor"] == f]
        boot_on_rows.append(_fmt_row(fit_cluster_bootstrap(sub, f)))
    lines.append(pd.DataFrame(boot_on_rows).to_markdown(index=False))
    lines.append("")

    lines.append("### 2c. Crossed mixed model\n")
    mm_on_rows = []
    for f in FACTORS:
        sub = df[df["target_factor"] == f]
        mm_on_rows.append(_fmt_row(fit_crossed_mixed(sub, f)))
    lines.append(pd.DataFrame(mm_on_rows).to_markdown(index=False))
    lines.append("")

    # ── Comparison to Table 7 model-level r ──
    lines.append("## 3. Comparison to Model-Level Pearson `r`\n")
    lines.append(
        "If rating-level estimators give substantively different answers "
        "from the model-level correlation, the model-level test is leaving "
        "signal on the table. If they agree, the bottleneck is genuinely "
        "the 25-model sample size.\n"
    )
    from analysis.predictive_validity import (
        model_level_human_scores,
        per_factor_correlations,
    )
    inst_idx = inst.set_index("model_id")
    human_all = model_level_human_scores(hr, "all")
    human_on = model_level_human_scores(hr, "on_target")
    pf_all = per_factor_correlations(inst_idx, human_all)
    pf_on = per_factor_correlations(inst_idx, human_on)

    rows = []
    for f in FACTORS:
        r_all = pf_all[pf_all["factor"] == f]["r"].iloc[0]
        r_on = pf_on[pf_on["factor"] == f]["r"].iloc[0]
        ols_all = fit_ols_cluster_robust(df, f)
        ols_on = fit_ols_cluster_robust(df[df["target_factor"] == f], f)
        rows.append({
            "factor": f,
            "r (all)": f"{r_all:+.2f}",
            "β_z OLS/CR1 (all)": f"{ols_all.get('beta_z', float('nan')):+.2f}"
                if ols_all.get("converged") else "—",
            "CI OLS/CR1 (all)": f"[{ols_all.get('ci_lo', 0):+.2f}, {ols_all.get('ci_hi', 0):+.2f}]"
                if ols_all.get("converged") else "—",
            "r (on-target)": f"{r_on:+.2f}",
            "β_z OLS/CR1 (on-target)": f"{ols_on.get('beta_z', float('nan')):+.2f}"
                if ols_on.get("converged") else "—",
            "CI OLS/CR1 (on-target)": f"[{ols_on.get('ci_lo', 0):+.2f}, {ols_on.get('ci_hi', 0):+.2f}]"
                if ols_on.get("converged") else "—",
        })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report()
    out = OUTPUT_DIR / "mixed_model_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[mixed_model_validity] Report saved to {out}")


if __name__ == "__main__":
    main()
