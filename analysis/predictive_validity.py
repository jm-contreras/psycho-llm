"""Predictive validity: instrument factor scores × human behavioral ratings.

Correlates Phase 1/2 instrument factor scores (from EFA on Likert items)
with Phase 3 human behavioral ratings (Prolific) at the model level.

Analyses:
  1. Model-level correlations — all prompts and on-target only
  2. Convergent-discriminant (MTMM) matrix — instrument × human factor → r
  3. On-target vs off-target advantage — do on-dim prompts predict better?
  4. Comparison with LLM-as-judge ratings

Usage:
    python -m analysis.predictive_validity

Output:
    analysis/output/predictive_validity_report.md
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import OUTPUT_DIR, _load_group_map

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
PROLIFIC_DB = REPO_ROOT / "data" / "prolific" / "prolific.db"
RESPONSES_DB = REPO_ROOT / "data" / "raw" / "responses.db"
FACTOR_SCORES_PATH = OUTPUT_DIR / "factor_scores.csv"

FACTORS = ["RE", "DE", "BO", "GU", "VB"]
FACTOR_NAMES = {
    "RE": "Responsiveness", "DE": "Deference", "BO": "Boldness",
    "GU": "Guardedness", "VB": "Verbosity",
}

PROMPT_FACTOR = {
    "RE-BP01": "RE", "RE-BP02": "RE", "RE-BP03": "RE", "RE-BP04": "RE",
    "DE-BP01": "DE", "DE-BP02": "DE", "DE-BP03": "DE", "DE-BP04": "DE",
    "BO-BP01": "BO", "BO-BP02": "BO", "BO-BP03": "BO", "BO-BP04": "BO",
    "GU-BP01": "GU", "GU-BP02": "GU", "GU-BP03": "GU", "GU-BP04": "GU",
    "VB-BP01": "VB", "VB-BP02": "VB", "VB-BP03": "VB", "VB-BP04": "VB",
}

# Anonymized rater IDs to exclude (manual rejections);
# see pipeline/prolific/config.py for the salt used for hashing.
EXCLUDED_PIDS = {
    "ca0b05c20ff3",
    "6f523c30c8cf",
    "bf53ae6e33a4",
    "22658e748c45",
    "de38314720fd",
    "5401bb5bf3ed",
    "413aff8a004b",
    "0a56fec18ea4",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_instrument_scores() -> pd.DataFrame:
    """Load cached instrument factor scores (model_id, RE..VB)."""
    if not FACTOR_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"{FACTOR_SCORES_PATH} not found. Run `python -m analysis.judge_analysis` first."
        )
    return pd.read_csv(FACTOR_SCORES_PATH)


def load_human_ratings() -> pd.DataFrame:
    """Load Prolific human ratings joined with behavioral_responses for model_id.

    Returns DataFrame with: behavioral_response_id, model_id, prompt_id,
    dimension_code, corrected_RE..corrected_VB (one row per rating).
    """
    # Prolific ratings
    pconn = sqlite3.connect(str(PROLIFIC_DB))
    pr = pd.read_sql_query(
        "SELECT r.behavioral_response_id, r.prompt_id, r.prolific_pid, r.is_gold, "
        "r.corrected_RE, r.corrected_DE, r.corrected_BO, r.corrected_GU, r.corrected_VB "
        "FROM prolific_ratings r "
        "JOIN prolific_sessions s ON r.prolific_pid = s.prolific_pid "
        "WHERE s.status = 'complete' AND r.is_gold = 0",
        pconn,
    )
    pconn.close()

    # Exclude rejected participants
    pr = pr[~pr["prolific_pid"].isin(EXCLUDED_PIDS)].copy()

    # Join with behavioral_responses for model_id and dimension_code
    rconn = sqlite3.connect(str(RESPONSES_DB))
    br = pd.read_sql_query(
        "SELECT id, model_id, dimension_code FROM behavioral_responses", rconn,
    )
    rconn.close()

    pr = pr.merge(br, left_on="behavioral_response_id", right_on="id", how="inner")

    group_map = _load_group_map()
    if group_map:
        pr["model_id"] = pr["model_id"].map(lambda x: group_map.get(x, x))
    return pr


def load_judge_ensemble() -> pd.DataFrame:
    """Load judge ensemble scores per model per factor (from judge_analysis)."""
    from analysis.judge_analysis import load_judge_ratings, compute_ensemble_scores
    df = load_judge_ratings(RESPONSES_DB)
    if df.empty:
        return pd.DataFrame()
    ensemble = compute_ensemble_scores(df)

    group_map = _load_group_map()
    if group_map:
        ensemble["subject_model_id"] = ensemble["subject_model_id"].map(
            lambda x: group_map.get(x, x)
        )

    model_judge = (
        ensemble.groupby(["subject_model_id", "factor_code"])["ensemble_score"]
        .mean()
        .reset_index()
        .pivot(index="subject_model_id", columns="factor_code", values="ensemble_score")
    )
    return model_judge


# ── Analysis functions ───────────────────────────────────────────────────────

def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r, Spearman rho, and p-values for aligned arrays."""
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < 4:
        return {"r": np.nan, "rho": np.nan, "p_r": np.nan, "p_rho": np.nan, "n": n}
    r, p_r = stats.pearsonr(x[mask], y[mask])
    rho, p_rho = stats.spearmanr(x[mask], y[mask])
    return {"r": round(r, 3), "rho": round(rho, 3),
            "p_r": round(p_r, 4), "p_rho": round(p_rho, 4), "n": n}


def model_level_human_scores(hr: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """Aggregate human ratings to model-level means per factor.

    subset: "all" = all prompts, "on_target" = only prompts targeting each factor.
    Returns DataFrame indexed by model_id with columns RE..VB.
    """
    score_cols = [f"corrected_{f}" for f in FACTORS]
    hr = hr.copy()
    hr["target_factor"] = hr["prompt_id"].map(PROMPT_FACTOR)

    if subset == "all":
        return (
            hr.groupby("model_id")[score_cols]
            .mean()
            .rename(columns=lambda c: c.replace("corrected_", ""))
        )

    # On-target: for each factor, only use prompts targeting that factor
    rows = {}
    for model_id in hr["model_id"].unique():
        mdata = hr[hr["model_id"] == model_id]
        row = {}
        for f in FACTORS:
            fdata = mdata.loc[mdata["target_factor"] == f, f"corrected_{f}"]
            row[f] = fdata.mean() if len(fdata) > 0 else np.nan
        rows[model_id] = row
    return pd.DataFrame(rows).T


def convergent_discriminant_matrix(
    inst: pd.DataFrame, human: pd.DataFrame
) -> pd.DataFrame:
    """Instrument factor × human factor → Pearson r (model-level).

    inst: indexed by model_id, columns = factor codes
    human: indexed by model_id, columns = factor codes
    Returns: DataFrame (rows=instrument factors, cols=human factors).
    """
    common = inst.index.intersection(human.index)
    inst = inst.loc[common]
    human = human.loc[common]

    matrix = {}
    for ifact in FACTORS:
        matrix[ifact] = {}
        for hfact in FACTORS:
            x = inst[ifact].values.astype(float)
            y = human[hfact].values.astype(float)
            result = _corr(x, y)
            matrix[ifact][hfact] = result["r"]

    df = pd.DataFrame(matrix).T
    df.index.name = "instrument"
    df.columns.name = "human"
    return df


def per_factor_correlations(
    inst: pd.DataFrame, human: pd.DataFrame, label: str = ""
) -> pd.DataFrame:
    """Per-factor convergent correlations (diagonal of MTMM)."""
    common = inst.index.intersection(human.index)
    inst = inst.loc[common]
    human = human.loc[common]

    rows = []
    for f in FACTORS:
        x = inst[f].values.astype(float)
        y = human[f].values.astype(float)
        result = _corr(x, y)
        result["factor"] = f
        result["factor_name"] = FACTOR_NAMES[f]
        rows.append(result)
    return pd.DataFrame(rows)[["factor", "factor_name", "r", "rho", "p_r", "p_rho", "n"]]


def on_vs_off_target(inst: pd.DataFrame, hr: pd.DataFrame) -> pd.DataFrame:
    """Per-prompt-dimension on/off factor analysis.

    For each prompt dimension and each instrument factor, correlate
    instrument scores with human-rated scores at the model level.
    """
    inst_idx = inst.set_index("model_id") if "model_id" in inst.columns else inst

    rows = []
    for prompt_dim in FACTORS:
        # Get human scores for prompts targeting this dimension
        dim_prompts = [p for p, f in PROMPT_FACTOR.items() if f == prompt_dim]
        dim_data = hr[hr["prompt_id"].isin(dim_prompts)]

        for rating_factor in FACTORS:
            col = f"corrected_{rating_factor}"
            model_means = dim_data.groupby("model_id")[col].mean()
            common = model_means.index.intersection(inst_idx.index)
            if len(common) < 5:
                continue
            x = inst_idx.loc[common, rating_factor].values.astype(float)
            y = model_means.loc[common].values.astype(float)
            result = _corr(x, y)
            rows.append({
                "prompt_dim": prompt_dim,
                "instrument_factor": rating_factor,
                "r": result["r"],
                "p": result["p_r"],
                "n": result["n"],
                "on_target": prompt_dim == rating_factor,
            })

    return pd.DataFrame(rows)


def human_judge_agreement(hr: pd.DataFrame) -> dict:
    """Direct agreement between human and LLM-as-judge ratings on the same samples.

    For each behavioral_response_id rated by both humans and judges, compute
    mean human score and judge ensemble score, then correlate at item-level
    and model-level. Also compute ICC(2,1) treating human-mean and judge-ensemble
    as two raters.

    Returns dict with:
      'item_level': DataFrame — factor, r, rho, p_r, p_rho, n, icc
      'model_level': DataFrame — factor, r, rho, p_r, p_rho, n
      'flagged': list of factors with item-level r < 0.65
    """
    from analysis.judge_analysis import load_judge_ratings, compute_ensemble_scores, _compute_icc21

    # Load judge data (already reverse-scored)
    jdf = load_judge_ratings(RESPONSES_DB)
    if jdf.empty:
        return {"item_level": pd.DataFrame(), "model_level": pd.DataFrame(), "flagged": []}

    # Compute judge ensemble per sample per factor
    ensemble = compute_ensemble_scores(jdf)

    # Pivot ensemble to wide: behavioral_response_id × factor → score
    judge_wide = ensemble.pivot_table(
        index="behavioral_response_id",
        columns="factor_code",
        values="ensemble_score",
        aggfunc="mean",
    )

    # Compute mean human rating per sample per factor
    score_cols = [f"corrected_{f}" for f in FACTORS]
    human_means = (
        hr.groupby("behavioral_response_id")[score_cols]
        .mean()
        .rename(columns=lambda c: c.replace("corrected_", ""))
    )

    # Align on common behavioral_response_ids
    common_rids = human_means.index.intersection(judge_wide.index)
    human_means = human_means.loc[common_rids]
    judge_wide = judge_wide.loc[common_rids]

    # Also need model_id for model-level aggregation
    rid_to_model = hr.drop_duplicates("behavioral_response_id").set_index(
        "behavioral_response_id"
    )["model_id"]

    item_rows = []
    model_rows = []
    flagged = []

    for f in FACTORS:
        if f not in human_means.columns or f not in judge_wide.columns:
            continue

        h = human_means[f].values.astype(float)
        j = judge_wide[f].values.astype(float)
        mask = ~(np.isnan(h) | np.isnan(j))
        n = int(mask.sum())

        if n < 4:
            item_rows.append({"factor": f, "factor_name": FACTOR_NAMES[f],
                              "r": np.nan, "rho": np.nan, "p_r": np.nan,
                              "p_rho": np.nan, "n": n, "icc": np.nan})
            continue

        r, p_r = stats.pearsonr(h[mask], j[mask])
        rho, p_rho = stats.spearmanr(h[mask], j[mask])

        # ICC(2,1): treat human-mean and judge-ensemble as 2 raters
        icc_data = np.column_stack([h[mask], j[mask]])
        icc_val, _, _ = _compute_icc21(icc_data)

        item_rows.append({
            "factor": f, "factor_name": FACTOR_NAMES[f],
            "r": round(r, 3), "rho": round(rho, 3),
            "p_r": round(p_r, 4), "p_rho": round(p_rho, 4),
            "n": n, "icc": round(icc_val, 3),
        })

        if r < 0.65:
            flagged.append(f)

        # Model-level: aggregate human and judge means per model, then correlate
        aligned = pd.DataFrame({
            "human": human_means.loc[common_rids, f],
            "judge": judge_wide.loc[common_rids, f],
            "model_id": rid_to_model.reindex(common_rids).values,
        }).dropna()

        model_agg = aligned.groupby("model_id")[["human", "judge"]].mean()
        if len(model_agg) >= 4:
            result = _corr(model_agg["human"].values, model_agg["judge"].values)
            result["factor"] = f
            result["factor_name"] = FACTOR_NAMES[f]
            model_rows.append(result)

    return {
        "item_level": pd.DataFrame(item_rows),
        "model_level": pd.DataFrame(model_rows),
        "flagged": flagged,
    }


# ── Report generation ────────────────────────────────────────────────────────

def generate_report() -> str:
    lines = ["# Predictive Validity: Instrument × Human Behavioral Ratings\n"]

    # Load data
    inst = load_instrument_scores()
    hr = load_human_ratings()
    inst_idx = inst.set_index("model_id")

    n_models = hr["model_id"].nunique()
    n_ratings = len(hr)
    n_raters = hr["prolific_pid"].nunique()
    lines.append(f"**Data:** {n_ratings} human ratings from {n_raters} raters "
                 f"across {n_models} models, {len(inst)} models with instrument scores.\n")

    common = inst_idx.index.intersection(hr["model_id"].unique())
    lines.append(f"**Models in common:** {len(common)}\n")

    # ── Section 1: All-prompts model-level correlations ──
    lines.append("## 1. Model-Level Convergent Validity (All Prompts)\n")
    lines.append("Mean human rating per model per factor (across all 20 prompts) "
                 "correlated with instrument factor scores.\n")

    human_all = model_level_human_scores(hr, "all")
    pf_all = per_factor_correlations(inst_idx, human_all)
    lines.append(pf_all.to_markdown(index=False))
    lines.append("")

    # ── Section 2: On-target model-level correlations ──
    lines.append("## 2. Model-Level Convergent Validity (On-Target Prompts Only)\n")
    lines.append("For each factor, only human ratings from prompts designed to elicit "
                 "that factor (4 prompts per factor).\n")

    human_on = model_level_human_scores(hr, "on_target")
    pf_on = per_factor_correlations(inst_idx, human_on)
    lines.append(pf_on.to_markdown(index=False))
    lines.append("")

    # ── Section 3: Full MTMM — all prompts ──
    lines.append("## 3. Convergent-Discriminant Matrix (All Prompts)\n")
    lines.append("Instrument factor (rows) × human-rated factor (columns) → Pearson r. "
                 "Diagonal = convergent validity.\n")

    cd_all = convergent_discriminant_matrix(inst_idx, human_all)
    lines.append(cd_all.round(3).to_markdown())
    lines.append("")

    # Highlight convergent vs discriminant
    diag = [cd_all.loc[f, f] for f in FACTORS]
    off_diag = []
    for i, fi in enumerate(FACTORS):
        for j, fj in enumerate(FACTORS):
            if i != j:
                off_diag.append(cd_all.loc[fi, fj])
    lines.append(f"**Mean convergent (diagonal):** {np.nanmean(diag):.3f}")
    lines.append(f"**Mean discriminant (off-diagonal):** {np.nanmean(off_diag):.3f}\n")

    # ── Section 4: Full MTMM — on-target only ──
    lines.append("## 4. Convergent-Discriminant Matrix (On-Target Prompts Only)\n")
    lines.append("Same as above but human ratings restricted to prompts targeting "
                 "each dimension. Only the diagonal changes — each cell uses human "
                 "ratings of factor X from prompts designed for factor X.\n")

    cd_on = convergent_discriminant_matrix(inst_idx, human_on)
    lines.append(cd_on.round(3).to_markdown())
    lines.append("")

    diag_on = [cd_on.loc[f, f] for f in FACTORS]
    lines.append(f"**Mean convergent (diagonal):** {np.nanmean(diag_on):.3f}")
    off_diag_on = []
    for i, fi in enumerate(FACTORS):
        for j, fj in enumerate(FACTORS):
            if i != j:
                off_diag_on.append(cd_on.loc[fi, fj])
    lines.append(f"**Mean discriminant (off-diagonal):** {np.nanmean(off_diag_on):.3f}\n")

    # ── Section 5: On-target vs off-target advantage ──
    lines.append("## 5. On-Target vs Off-Target Advantage\n")
    lines.append("For each factor: does the instrument predict human ratings "
                 "better when prompts target that factor?\n")

    onoff = on_vs_off_target(inst, hr)
    if not onoff.empty:
        summary_rows = []
        for f in FACTORS:
            sub = onoff[onoff["instrument_factor"] == f]
            on = sub[sub["on_target"]]
            off = sub[~sub["on_target"]]
            on_r = on["r"].iloc[0] if len(on) > 0 else np.nan
            on_p = on["p"].iloc[0] if len(on) > 0 else np.nan
            off_mean_r = round(float(off["r"].mean()), 3) if len(off) > 0 else np.nan
            advantage = round(on_r - off_mean_r, 3) if not (np.isnan(on_r) or np.isnan(off_mean_r)) else np.nan
            summary_rows.append({
                "factor": f,
                "on_target_r": on_r,
                "on_target_p": on_p,
                "off_target_mean_r": off_mean_r,
                "advantage": advantage,
            })
        summary_df = pd.DataFrame(summary_rows)
        lines.append(summary_df.to_markdown(index=False))
        lines.append("")

        lines.append("\n**Full detail (prompt_dim × instrument_factor):**\n")
        display = onoff.copy()
        display["on_target"] = display["on_target"].map({True: "**ON**", False: ""})
        lines.append(display.to_markdown(index=False))
        lines.append("")

    # ── Section 6: Comparison with LLM-as-judge ──
    lines.append("## 6. Comparison: Human vs LLM-as-Judge Ratings\n")
    lines.append("Same instrument factor scores correlated with judge ensemble "
                 "ratings (for comparison).\n")

    try:
        judge_model = load_judge_ensemble()
        if not judge_model.empty:
            pf_judge = per_factor_correlations(inst_idx, judge_model)
            lines.append("**Judge ensemble (model-level, all prompts):**\n")
            lines.append(pf_judge.to_markdown(index=False))
            lines.append("")

            # Side-by-side
            lines.append("**Side-by-side convergent r (instrument × criterion):**\n")
            lines.append("| Factor | Human (all) | Human (on-target) | Judge (all) |")
            lines.append("|--------|-------------|-------------------|-------------|")
            for f in FACTORS:
                r_h_all = pf_all[pf_all["factor"] == f]["r"].iloc[0] if len(pf_all[pf_all["factor"] == f]) else np.nan
                r_h_on = pf_on[pf_on["factor"] == f]["r"].iloc[0] if len(pf_on[pf_on["factor"] == f]) else np.nan
                r_j = pf_judge[pf_judge["factor"] == f]["r"].iloc[0] if len(pf_judge[pf_judge["factor"] == f]) else np.nan
                lines.append(f"| {f} | {r_h_all:.3f} | {r_h_on:.3f} | {r_j:.3f} |")
            lines.append("")
    except Exception as e:
        lines.append(f"*Judge data unavailable: {e}*\n")

    # ── Section 7: Human-Judge Agreement ──
    lines.append("## 7. Human–Judge Agreement\n")
    lines.append("Direct comparison of human (Prolific) and LLM-as-judge ratings "
                 "on the same 300 behavioral samples. Per-dimension Pearson r, "
                 "Spearman rho, and ICC(2,1) treating human-mean and judge-ensemble "
                 "as two raters.\n")

    hj_agreement = human_judge_agreement(hr)

    if not hj_agreement["item_level"].empty:
        lines.append("### Item-Level (per-sample agreement)\n")
        lines.append(hj_agreement["item_level"].to_markdown(index=False))
        lines.append("")

        if hj_agreement["flagged"]:
            flags = ", ".join(hj_agreement["flagged"])
            lines.append(f"**Flagged (r < 0.65):** {flags}\n")
        else:
            lines.append("**All factors meet r >= 0.65 threshold.**\n")

    if not hj_agreement["model_level"].empty:
        lines.append("### Model-Level (aggregated per model)\n")
        cols = ["factor", "factor_name", "r", "rho", "p_r", "p_rho", "n"]
        lines.append(hj_agreement["model_level"][cols].to_markdown(index=False))
        lines.append("")

    # ── Section 8: Model-level scatter data ──
    lines.append("## 8. Model-Level Scores\n")
    lines.append("Instrument and human-rated factor scores per model (for inspection).\n")

    merged = inst_idx.loc[common].copy()
    merged.columns = [f"inst_{c}" for c in merged.columns]
    human_all_common = human_all.loc[human_all.index.isin(common)]
    for f in FACTORS:
        merged[f"human_{f}"] = human_all_common[f]

    # Sort by model name for readability
    merged = merged.sort_index()
    lines.append(merged.round(3).to_markdown())
    lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report()
    report_path = OUTPUT_DIR / "predictive_validity_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[predictive_validity] Report saved to {report_path}")


if __name__ == "__main__":
    main()
