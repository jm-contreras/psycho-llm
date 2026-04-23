"""
LLM-as-judge analysis for Phase 3 predictive validity.

Sections:
  1. Engineering checks  — coverage, parse error rates, per-judge breakdown
  2. Reverse-scoring     — applies keying corrections to raw judge scores
  3. Inter-judge agreement — pairwise r, Spearman ρ, ICC(2,1) per factor
  4. Ensemble scores     — mean corrected scores across available judges per sample
  5. Predictive validity — instrument factor scores × ensemble judge scores

Usage:
  python -m analysis.judge_analysis

Output:
  analysis/output/judge_report.md
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import (
    DB_PATH, OUTPUT_DIR, ensure_output_dirs,
    filter_success, get_models_for_section, load_responses, recode_reverse_items,
)
from analysis.bfi_analysis import _is_ai_native
from analysis.factor_structure import build_pooled_matrix
from analysis.primary_analyses import split_half_data, select_items

# ── Constants ─────────────────────────────────────────────────────────────────

FACTOR_ORDER = ["RE", "DE", "BO", "GU", "VB"]
FACTOR_NAMES = {
    "RE": "Responsiveness",
    "DE": "Deference",
    "BO": "Boldness",
    "GU": "Guardedness",
    "VB": "Verbosity",
}
JUDGE_SCORE_COLS = [f"score_{f}" for f in FACTOR_ORDER]

# Agreement threshold below which only human-rated result is treated as valid
AGREEMENT_THRESHOLD = 0.65

# EFA factor number → behavioral factor code (from 5-factor solution, 2026-03-26)
_EFA_FACTOR_TO_CODE = {
    "Factor1": "RE",
    "Factor2": "DE",
    "Factor3": "BO",
    "Factor4": "GU",
    "Factor5": "VB",
}

FORCED_N_FACTORS = 5

# Cross-model exclusion: judge provider → excluded subject provider
_JUDGE_PROVIDER: dict[str, str] = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Anthropic",
    "openai/gpt-5.4": "OpenAI",
    "gemini/gemini-3.1-pro-preview": "Google",
}

# Subject model → provider (for exclusion logic)
_SUBJECT_PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("bedrock/us.anthropic.", "Anthropic"),
    ("bedrock/us.meta.", "Meta"),
    ("bedrock/amazon.", "Amazon"),
    ("bedrock/google.", "Google"),
    ("gemini/", "Google"),
    ("openai/gpt-", "OpenAI"),
    ("openai/gpt_", "OpenAI"),
    ("ai21/", "AI21"),
    ("xai/", "xAI"),
    ("dashscope/", "Alibaba"),
]


def _subject_provider(model_id: str) -> str:
    """Infer the provider family for a subject model."""
    for prefix, provider in _SUBJECT_PROVIDER_PREFIXES:
        if model_id.startswith(prefix):
            return provider
    return "Other"

# Short display names for judge models
_JUDGE_SHORT: dict[str, str] = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Claude Opus 4.6",
    "openai/gpt-5.4": "GPT-5.4",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}


def _judge_short(model_id: str) -> str:
    return _JUDGE_SHORT.get(model_id, model_id.split("/")[-1])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_judge_ratings(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load judge_ratings with parse_status='success', apply reverse-scoring.

    Returns DataFrame with corrected scores (analysis-ready), not raw scores.
    Columns include: behavioral_response_id, subject_model_id, prompt_id,
    run_number, judge_model_id, keying, score_RE..score_VB (corrected).
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT * FROM judge_ratings WHERE parse_status = 'success'",
        conn,
    )
    conn.close()

    if df.empty:
        return df

    # Apply reverse-scoring row-wise
    for i, factor in enumerate(FACTOR_ORDER):
        col = f"score_{factor}"
        r_mask = df["keying"].str[i] == "R"
        df.loc[r_mask, col] = 6 - df.loc[r_mask, col]

    return df


def load_all_judge_ratings(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load all judge_ratings rows (all parse_status values) for engineering checks."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM judge_ratings", conn)
    conn.close()
    return df


def load_instrument_factor_scores(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Compute per-model instrument factor scores from the 5-factor EFA solution.

    Runs a k=5 forced EFA on the exploration half (runs 1-15) to get the
    factor→items mapping, then computes mean scores per model per factor
    across all 30 runs.

    Returns DataFrame with columns: model_id, RE, DE, BO, GU, VB.
    Caches to analysis/output/factor_scores.csv.
    """
    factor_scores_path = OUTPUT_DIR / "factor_scores.csv"
    if factor_scores_path.exists():
        return pd.read_csv(factor_scores_path)

    print("  Computing instrument factor scores from 5-factor EFA...")

    # Load and prepare instrument data
    df_all = load_responses(db_path)
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)
    eligible = get_models_for_section(df_all, section=4)

    # Run k=5 EFA on exploration half
    from analysis.data_loader import compute_model_item_means
    from factor_analyzer import FactorAnalyzer

    efa_df, _ = split_half_data(df_success)
    obs_matrix, weights = build_pooled_matrix(efa_df, eligible, "direct")

    filled = obs_matrix.fillna(obs_matrix.mean())
    sqrt_w = np.sqrt(weights)
    wmean = np.average(filled.values, axis=0, weights=weights)
    weighted_centered = (filled.values - wmean) * sqrt_w[:, np.newaxis]
    weighted_df = pd.DataFrame(weighted_centered, columns=filled.columns)

    try:
        fa = FactorAnalyzer(n_factors=FORCED_N_FACTORS, rotation="oblimin", method="minres")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)
    except Exception:
        fa = FactorAnalyzer(n_factors=FORCED_N_FACTORS, rotation="oblimin", method="principal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)

    loadings_df = pd.DataFrame(
        fa.loadings_, index=obs_matrix.columns,
        columns=[f"Factor{i+1}" for i in range(FORCED_N_FACTORS)],
    )
    communalities_df = pd.DataFrame({
        "item_id": obs_matrix.columns, "communality": fa.get_communalities(),
    })

    # Item selection (primary ≥ 0.40, cross < 0.30)
    means_df = compute_model_item_means(df_success)
    item_report, retained, _ = select_items(loadings_df, means_df, communalities_df)

    # Build factor → items mapping
    factor_items: dict[str, list[str]] = {}
    for _, row in item_report[item_report["retained"]].iterrows():
        factor_items.setdefault(row["primary_factor"], []).append(row["item_id"])

    print(f"  Factor→items: {', '.join(f'{f}={len(v)}' for f, v in sorted(factor_items.items()))}")

    # Compute mean scores per model per factor (all 30 runs)
    ai_native = df_success[_is_ai_native(df_success)]
    rows = []
    for model_id in eligible:
        model_data = ai_native[ai_native["model_id"] == model_id]
        if len(model_data) == 0:
            continue
        row: dict[str, object] = {"model_id": model_id}
        for factor, items in sorted(factor_items.items()):
            code = _EFA_FACTOR_TO_CODE.get(factor, factor[:2])
            factor_data = model_data[model_data["item_id"].isin(items)]
            row[code] = float(factor_data["score"].mean()) if len(factor_data) > 0 else np.nan
        rows.append(row)

    scores = pd.DataFrame(rows)
    scores.to_csv(factor_scores_path, index=False)
    print(f"  Saved factor scores for {len(scores)} models → {factor_scores_path}")
    return scores


# ── Section 1: Engineering checks ────────────────────────────────────────────

def judge_engineering_checks(db_path: Path = DB_PATH) -> dict:
    """Coverage and parse error rates per judge model.

    Returns dict with:
      'status_table': DataFrame — judge × parse_status → count
      'coverage_table': DataFrame — judge × subject_provider → n_rated
      'total_samples': int
    """
    df = load_all_judge_ratings(db_path)
    if df.empty:
        return {"status_table": pd.DataFrame(), "coverage_table": pd.DataFrame(),
                "total_samples": 0}

    df["judge_short"] = df["judge_model_id"].map(_judge_short)

    status_table = (
        df.groupby(["judge_short", "parse_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"judge_short": "Judge"})
    )

    # Coverage: unique (subject_model, prompt, run) rated per judge
    success_df = df[df["parse_status"] == "success"].copy()
    coverage = (
        success_df.groupby("judge_model_id")["behavioral_response_id"]
        .nunique()
        .reset_index()
        .rename(columns={"behavioral_response_id": "n_rated", "judge_model_id": "judge"})
    )
    coverage["judge"] = coverage["judge"].map(_judge_short)

    return {
        "status_table": status_table,
        "coverage_table": coverage,
        "total_samples": df["behavioral_response_id"].nunique(),
    }


# ── Section 2: Inter-judge agreement ─────────────────────────────────────────

def inter_judge_agreement(df: pd.DataFrame) -> dict:
    """Compute pairwise Pearson r, Spearman ρ, and ICC(2,1) per factor.

    df: corrected judge ratings from load_judge_ratings().

    Returns dict with:
      'pairwise': DataFrame — factor × judge_pair → r, rho
      'icc': DataFrame — factor → ICC estimate, lower CI, upper CI
      'summary': DataFrame — factor, mean_r, mean_rho, icc, flag (r < threshold)
    """
    df = df.copy()
    df["judge_short"] = df["judge_model_id"].map(_judge_short)

    pairwise_rows = []
    icc_rows = []
    summary_rows = []

    for factor in FACTOR_ORDER:
        col = f"score_{factor}"

        # Pivot: behavioral_response_id × judge → score
        pivot = df.pivot_table(
            index="behavioral_response_id",
            columns="judge_short",
            values=col,
            aggfunc="mean",
        )
        # Drop samples where any judge is missing
        pivot = pivot.dropna()

        if len(pivot) < 5:
            summary_rows.append({
                "factor": factor,
                "factor_name": FACTOR_NAMES[factor],
                "n_samples": len(pivot),
                "mean_r": np.nan,
                "mean_rho": np.nan,
                "icc": np.nan,
                "flag": True,
            })
            continue

        judges = list(pivot.columns)
        rs, rhos = [], []

        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                x = pivot[judges[i]].values
                y = pivot[judges[j]].values
                r, _ = stats.pearsonr(x, y)
                rho, _ = stats.spearmanr(x, y)
                rs.append(r)
                rhos.append(rho)
                pairwise_rows.append({
                    "factor": factor,
                    "judge_a": judges[i],
                    "judge_b": judges[j],
                    "r": round(r, 3),
                    "rho": round(rho, 3),
                    "n": len(pivot),
                })

        mean_r = float(np.mean(rs)) if rs else np.nan
        mean_rho = float(np.mean(rhos)) if rhos else np.nan

        # ICC(2,1): two-way random, single measures
        icc_val, icc_lo, icc_hi = _compute_icc21(pivot.values)
        icc_rows.append({
            "factor": factor,
            "factor_name": FACTOR_NAMES[factor],
            "icc": round(icc_val, 3),
            "icc_lo": round(icc_lo, 3),
            "icc_hi": round(icc_hi, 3),
            "n_samples": len(pivot),
        })

        summary_rows.append({
            "factor": factor,
            "factor_name": FACTOR_NAMES[factor],
            "n_samples": len(pivot),
            "mean_r": round(mean_r, 3),
            "mean_rho": round(mean_rho, 3),
            "icc": round(icc_val, 3),
            "flag": mean_r < AGREEMENT_THRESHOLD,
        })

    return {
        "pairwise": pd.DataFrame(pairwise_rows),
        "icc": pd.DataFrame(icc_rows),
        "summary": pd.DataFrame(summary_rows),
    }


def _compute_icc21(data: np.ndarray) -> tuple[float, float, float]:
    """Compute ICC(2,1) — two-way random effects, single measures.

    data: (n_subjects × n_raters) array.
    Returns (icc, lower_95ci, upper_95ci).
    """
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan

    grand_mean = data.mean()
    ss_total = ((data - grand_mean) ** 2).sum()

    row_means = data.mean(axis=1, keepdims=True)
    col_means = data.mean(axis=0, keepdims=True)

    ss_between_subjects = k * ((row_means.squeeze() - grand_mean) ** 2).sum()
    ss_between_raters = n * ((col_means.squeeze() - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_between_subjects - ss_between_raters

    ms_subjects = ss_between_subjects / (n - 1)
    ms_raters = ss_between_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n)

    # 95% CI via F distribution approximation
    alpha = 0.05
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_lo = stats.f.ppf(alpha / 2, df1, df2)
    f_hi = stats.f.ppf(1 - alpha / 2, df1, df2)
    f_obs = ms_subjects / ms_error
    icc_lo = (f_obs / f_hi - 1) / (f_obs / f_hi + k - 1) if f_hi > 0 else np.nan
    icc_hi = (f_obs / f_lo - 1) / (f_obs / f_lo + k - 1) if f_lo > 0 else np.nan

    return float(icc), float(icc_lo), float(icc_hi)


# ── Section 3: Ensemble scores ────────────────────────────────────────────────

def _ensemble_agg(scores: pd.Series) -> float:
    """Mean if 2 judges, median if 3 (cross-model exclusion protocol)."""
    n = len(scores)
    if n <= 2:
        return float(scores.mean())
    return float(scores.median())


def compute_ensemble_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample per-factor ensemble score.

    Aggregation follows cross-model exclusion protocol:
      - 2 judges (same-provider judge excluded) → mean
      - 3 judges (no exclusion)                 → median

    df: corrected judge ratings from load_judge_ratings().

    Returns DataFrame: behavioral_response_id, subject_model_id, prompt_id,
    run_number, factor_code, ensemble_score, n_judges.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "behavioral_response_id", "subject_model_id", "prompt_id",
            "run_number", "factor_code", "ensemble_score", "n_judges",
        ])

    # Melt score columns to long format
    id_vars = ["behavioral_response_id", "subject_model_id", "prompt_id",
               "run_number", "judge_model_id"]
    score_vars = JUDGE_SCORE_COLS

    long = df[id_vars + score_vars].melt(
        id_vars=id_vars,
        value_vars=score_vars,
        var_name="score_col",
        value_name="score",
    )
    long = long.dropna(subset=["score"])
    long["factor_code"] = long["score_col"].str.replace("score_", "")

    group_cols = ["behavioral_response_id", "subject_model_id", "prompt_id",
                  "run_number", "factor_code"]
    ensemble = (
        long.groupby(group_cols)["score"]
        .agg(ensemble_score=_ensemble_agg, n_judges="count")
        .reset_index()
    )
    return ensemble


# ── Section 4: Predictive validity ───────────────────────────────────────────

def predictive_validity(
    instrument_scores: pd.DataFrame,
    ensemble_scores: pd.DataFrame,
) -> dict:
    """Correlate instrument factor scores with ensemble judge scores across models.

    instrument_scores: DataFrame with columns [model_id, RE, DE, BO, GU, VB]
                       (or similar — must have model_id and factor code columns)
    ensemble_scores:   from compute_ensemble_scores()

    Returns dict with:
      'per_factor': DataFrame — factor_code, r, rho, p_r, p_rho, n_models
      'convergent_discriminant': DataFrame — instrument_factor × judge_factor → r
      'summary': str description
    """
    # Aggregate ensemble scores to model level
    model_judge = (
        ensemble_scores
        .groupby(["subject_model_id", "factor_code"])["ensemble_score"]
        .mean()
        .reset_index()
        .pivot(index="subject_model_id", columns="factor_code", values="ensemble_score")
    )

    # Align instrument scores to model level
    inst = instrument_scores.copy()
    if "model_id" in inst.columns:
        inst = inst.set_index("model_id")

    common_models = inst.index.intersection(model_judge.index)
    if len(common_models) < 3:
        return {
            "per_factor": pd.DataFrame(),
            "convergent_discriminant": pd.DataFrame(),
            "summary": f"Insufficient model overlap: {len(common_models)} models in common.",
        }

    inst = inst.loc[common_models]
    model_judge = model_judge.loc[common_models]

    per_factor_rows = []
    for factor in FACTOR_ORDER:
        if factor not in inst.columns or factor not in model_judge.columns:
            continue
        x = inst[factor].values
        y = model_judge[factor].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            continue
        r, p_r = stats.pearsonr(x[mask], y[mask])
        rho, p_rho = stats.spearmanr(x[mask], y[mask])
        per_factor_rows.append({
            "factor_code": factor,
            "factor_name": FACTOR_NAMES[factor],
            "r": round(r, 3),
            "rho": round(rho, 3),
            "p_r": round(p_r, 4),
            "p_rho": round(p_rho, 4),
            "n_models": int(mask.sum()),
        })

    # Convergent / discriminant matrix: instrument_factor × judge_factor → r
    cd_data = {}
    for inst_factor in FACTOR_ORDER:
        if inst_factor not in inst.columns:
            continue
        cd_data[inst_factor] = {}
        for judge_factor in FACTOR_ORDER:
            if judge_factor not in model_judge.columns:
                cd_data[inst_factor][judge_factor] = np.nan
                continue
            x = inst[inst_factor].values
            y = model_judge[judge_factor].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                cd_data[inst_factor][judge_factor] = np.nan
            else:
                r, _ = stats.pearsonr(x[mask], y[mask])
                cd_data[inst_factor][judge_factor] = round(r, 3)

    cd_df = pd.DataFrame(cd_data).T
    cd_df.index.name = "instrument_factor"
    cd_df.columns.name = "judge_factor"

    return {
        "per_factor": pd.DataFrame(per_factor_rows),
        "convergent_discriminant": cd_df,
        "summary": f"{len(common_models)} models used for predictive validity analysis.",
    }


# ── Section 5: On-factor / off-factor predictive validity ───────────────���────

def on_off_factor_validity(
    instrument_scores: pd.DataFrame,
    df_judge: pd.DataFrame,
) -> pd.DataFrame:
    """Per-prompt-dimension × judge-factor predictive validity.

    For each behavioral prompt dimension (e.g. BO prompts), correlate
    instrument factor scores with model-level judge scores.  On-factor
    entries (prompt dim == judge factor) are flagged.

    Returns DataFrame: prompt_dim, judge_factor, r, p, n, on_factor.
    """
    inst = instrument_scores.copy()
    if "model_id" in inst.columns:
        inst = inst.set_index("model_id")

    conn = sqlite3.connect(str(DB_PATH))
    br = pd.read_sql_query(
        "SELECT id, dimension_code FROM behavioral_responses", conn,
    )
    conn.close()
    df = df_judge.merge(br, left_on="behavioral_response_id", right_on="id", how="left")

    rows = []
    for score_factor in FACTOR_ORDER:
        col = f"score_{score_factor}"
        model_dim = (
            df.groupby(["subject_model_id", "dimension_code"])[col]
            .mean()
            .reset_index()
        )
        for prompt_dim in sorted(df["dimension_code"].dropna().unique()):
            sub = model_dim[model_dim["dimension_code"] == prompt_dim].set_index(
                "subject_model_id"
            )
            common = sub.index.intersection(inst.index)
            if len(common) < 5:
                continue
            x = inst.loc[common, score_factor].values
            y = sub.loc[common, col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            rows.append({
                "prompt_dim": prompt_dim,
                "judge_factor": score_factor,
                "r": round(r, 3),
                "p": round(p, 3),
                "n": int(mask.sum()),
                "on_factor": prompt_dim == score_factor,
            })

    return pd.DataFrame(rows)


# ── Section 6: Keying effects ────────────────────────────────────────────────

def keying_effects(db_path: Path = DB_PATH) -> dict:
    """Analyse F-vs-R keying effects and judge × keying interactions.

    Returns dict with:
      'overall': DataFrame — factor, d (Cohen's d, F minus R-corrected)
      'by_judge': DataFrame — factor, judge, d
    """
    conn = sqlite3.connect(str(db_path))
    raw = pd.read_sql_query(
        "SELECT judge_model_id, keying, score_RE, score_DE, score_BO, "
        "score_GU, score_VB FROM judge_ratings WHERE parse_status = 'success'",
        conn,
    )
    conn.close()

    overall_rows = []
    judge_rows = []

    for i, factor in enumerate(FACTOR_ORDER):
        col = f"score_{factor}"
        f_scores = raw.loc[raw["keying"].str[i] == "F", col].dropna()
        r_scores = raw.loc[raw["keying"].str[i] == "R", col].dropna()
        r_corrected = 6 - r_scores

        d = _cohens_d(f_scores, r_corrected)
        overall_rows.append({
            "factor": factor, "factor_name": FACTOR_NAMES[factor],
            "n_F": len(f_scores), "mean_F": round(float(f_scores.mean()), 3),
            "n_R": len(r_scores), "mean_R_corr": round(float(r_corrected.mean()), 3),
            "d": round(d, 3),
        })

        for judge_id in sorted(raw["judge_model_id"].unique()):
            j = raw[raw["judge_model_id"] == judge_id]
            jf = j.loc[j["keying"].str[i] == "F", col].dropna()
            jr = j.loc[j["keying"].str[i] == "R", col].dropna()
            if len(jf) < 10 or len(jr) < 10:
                continue
            jr_corr = 6 - jr
            d_j = _cohens_d(jf, jr_corr)
            judge_rows.append({
                "factor": factor, "judge": _judge_short(judge_id), "d": round(d_j, 3),
            })

    return {
        "overall": pd.DataFrame(overall_rows),
        "by_judge": pd.DataFrame(judge_rows),
    }


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d (pooled SD)."""
    pooled_sd = float(np.sqrt((a.std() ** 2 + b.std() ** 2) / 2))
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


# ── Report generation ─────────────────────────────────────────────────────────

def generate_judge_report(db_path: Path = DB_PATH) -> str:
    """Generate full judge analysis report as a Markdown string."""
    lines = []
    lines.append("# LLM-as-Judge Analysis Report\n")

    # ── Section 1: Engineering ──
    lines.append("## 1. Engineering Checks\n")
    eng = judge_engineering_checks(db_path)

    lines.append(f"**Total unique behavioral samples with at least one rating:** "
                 f"{eng['total_samples']}\n")

    if not eng["status_table"].empty:
        lines.append("**Parse status by judge:**\n")
        lines.append(eng["status_table"].to_markdown(index=False))
        lines.append("")

    if not eng["coverage_table"].empty:
        lines.append("**Samples rated per judge:**\n")
        lines.append(eng["coverage_table"].to_markdown(index=False))
        lines.append("")

    # ── Load data ──
    df = load_judge_ratings(db_path)
    if df.empty:
        lines.append("\n*No successful judge ratings found. Remaining sections require data.*\n")
        return "\n".join(lines)

    n_judges = df["judge_model_id"].nunique()
    n_samples = df["behavioral_response_id"].nunique()
    lines.append(f"**Analysis basis:** {n_samples} samples × {n_judges} judges "
                 f"(after reverse-scoring)\n")

    # ── Section 2: Inter-judge agreement ──
    lines.append("## 2. Inter-Judge Agreement\n")
    agreement = inter_judge_agreement(df)

    if not agreement["summary"].empty:
        summary = agreement["summary"].copy()
        summary["flag"] = summary["flag"].map({True: "⚠ r < 0.65", False: ""})
        lines.append(summary.to_markdown(index=False))
        lines.append("")
        lines.append(
            f"*Threshold: mean_r ≥ {AGREEMENT_THRESHOLD} required for LLM-judge ratings "
            f"to be used as valid predictive evidence (r < threshold → human-rated only).*\n"
        )

    if not agreement["pairwise"].empty:
        lines.append("**Pairwise agreement:**\n")
        lines.append(agreement["pairwise"].to_markdown(index=False))
        lines.append("")

    if not agreement["icc"].empty:
        lines.append("**ICC(2,1) per factor:**\n")
        lines.append(agreement["icc"].to_markdown(index=False))
        lines.append("")

    # ── Section 3: Ensemble scores ──
    lines.append("## 3. Ensemble Scores\n")
    ensemble = compute_ensemble_scores(df)
    if not ensemble.empty:
        # Cross-model exclusion summary
        judge_counts = ensemble.groupby("n_judges").size()
        n2 = judge_counts.get(2, 0)
        n3 = judge_counts.get(3, 0)
        lines.append(
            f"**Cross-model exclusion:** {n2} sample×factor pairs rated by 2 judges (mean), "
            f"{n3} by 3 judges (median).\n"
        )

        model_factor_means = (
            ensemble.groupby(["subject_model_id", "factor_code"])["ensemble_score"]
            .mean()
            .unstack()
            .reindex(columns=FACTOR_ORDER)
            .round(3)
        )
        lines.append("**Mean ensemble score per model per factor:**\n")
        lines.append(model_factor_means.to_markdown())
        lines.append("")

    # ── Section 4: Predictive validity ──
    lines.append("## 4. Predictive Validity\n")
    try:
        instrument_scores = load_instrument_factor_scores(db_path)
        pv = predictive_validity(instrument_scores, ensemble)

        lines.append(f"*{pv['summary']}*\n")

        if not pv["per_factor"].empty:
            lines.append("**Per-factor correlations (instrument × judge):**\n")
            lines.append(pv["per_factor"].to_markdown(index=False))
            lines.append("")

        if not pv["convergent_discriminant"].empty:
            lines.append("**Convergent / discriminant matrix (instrument factor × judge factor → r):**\n")
            lines.append(pv["convergent_discriminant"].to_markdown())
            lines.append("")
            lines.append(
                "*Diagonal = convergent validity (within-factor). "
                "Off-diagonal = discriminant validity (cross-factor).*\n"
            )

        # ── Section 5: On-factor / off-factor ──
        lines.append("## 5. On-Factor vs Off-Factor Predictive Validity\n")
        lines.append(
            "Do prompts targeting a given factor show stronger instrument-judge "
            "convergence on that factor than off-factor prompts?\n"
        )
        onoff = on_off_factor_validity(instrument_scores, df)
        if not onoff.empty:
            # Pivot: rows = judge_factor, show on-factor r and mean off-factor r
            summary_rows = []
            for jf in FACTOR_ORDER:
                sub = onoff[onoff["judge_factor"] == jf]
                on = sub[sub["on_factor"]]
                off = sub[~sub["on_factor"]]
                on_r = on["r"].iloc[0] if len(on) > 0 else np.nan
                on_p = on["p"].iloc[0] if len(on) > 0 else np.nan
                off_mean_r = round(float(off["r"].mean()), 3) if len(off) > 0 else np.nan
                summary_rows.append({
                    "factor": jf,
                    "on_factor_r": on_r, "on_factor_p": on_p,
                    "off_factor_mean_r": off_mean_r,
                    "advantage": round(on_r - off_mean_r, 3) if not (np.isnan(on_r) or np.isnan(off_mean_r)) else np.nan,
                })
            summary_df = pd.DataFrame(summary_rows)
            lines.append(summary_df.to_markdown(index=False))
            lines.append("")

            lines.append("\n**Full on-factor / off-factor detail:**\n")
            onoff_display = onoff.copy()
            onoff_display["on_factor"] = onoff_display["on_factor"].map(
                {True: "**ON**", False: ""}
            )
            lines.append(onoff_display.to_markdown(index=False))
            lines.append("")

    except FileNotFoundError as exc:
        lines.append(f"*Instrument factor scores not available: {exc}*\n")
        lines.append("*Run analysis/factor_structure.py first to generate factor scores.*\n")

    # ── Section 6: Keying effects ──
    lines.append("## 6. Keying Effects\n")
    lines.append(
        "Comparison of forward-keyed (F) raw scores vs reverse-keyed (R) scores "
        "after correction (6 − raw). Cohen's d > 0 means F > R-corrected.\n"
    )
    ke = keying_effects(db_path)
    if not ke["overall"].empty:
        lines.append("**Overall keying effects:**\n")
        lines.append(ke["overall"].to_markdown(index=False))
        lines.append("")

    if not ke["by_judge"].empty:
        lines.append("**Keying × judge interaction (Cohen's d):**\n")
        pivot_judge = ke["by_judge"].pivot(index="factor", columns="judge", values="d")
        pivot_judge = pivot_judge.reindex(FACTOR_ORDER)
        lines.append(pivot_judge.to_markdown())
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_output_dirs()
    report = generate_judge_report()
    report_path = OUTPUT_DIR / "judge_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report[:5000])
    if len(report) > 5000:
        print(f"\n... (truncated — full report at {report_path})")
    print(f"\n[judge_analysis] Report saved to {report_path}")


if __name__ == "__main__":
    main()
