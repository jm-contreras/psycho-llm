"""Objective behavioral criteria from the stored behavioral samples.

The paper's predictive-validity criteria are human and LLM-judge *ratings* of
behavior. This module derives objective, text-computable measures from the
raw behavioral responses and asks three questions:

  1. Does self-report predict objective behavior? (instrument factor scores x
     objective measures at the model level, N = 25)
  2. Which raters weight surface features? (human / judge ratings x objective
     measures, model level and sample level)
  3. Does the Instrument-Judge Responsiveness correlation (r = .53) survive
     controlling for textual-surface covariates (length, markdown density,
     enthusiasm markers)? Partial correlation at the model level.

Objective measures per sample (visible output only; reasoning traces excluded):
  n_words          whitespace-tokenized word count
  md_density       markdown structure elements (headers, bullets, numbered
                   items, bold spans, code fences) per 100 words
  exclam_per_100w  exclamation marks per 100 words
  disclaimers      caveat/disclaimer markers per sample (count)
  offers           proactive-continuation offers per sample (count)
  refusal_markers  refusal/deflection markers per sample (count)

Marker lexicons are heuristic proxies; their validity is checked against the
human ratings of the matched construct (see report section 2).

Usage:
    python -m analysis.objective_behavior

Output:
    analysis/output/objective_behavior_report.md
    analysis/output/objective_measures_model_level.csv
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import OUTPUT_DIR, _load_group_map
from analysis.predictive_validity import (
    FACTORS, FACTOR_NAMES, RESPONSES_DB,
    load_human_ratings, load_instrument_scores, load_judge_ensemble,
    model_level_human_scores,
)

SEED = 20260705
N_BOOT = 10_000

# ── Marker lexicons (case-insensitive) ───────────────────────────────────────

DISCLAIMER_PATTERNS = [
    r"\bplease note\b", r"\bnote that\b", r"\bkeep in mind\b",
    r"\bit'?s (?:important|worth) (?:to note|noting|to remember)\b",
    r"\bdisclaimer\b", r"\bcaveat\b", r"\bas an ai\b",
    r"\bi(?:'m| am) an ai\b", r"\bi should (?:note|mention|clarify)\b",
    r"\bbe aware\b", r"\bjust so you know\b",
    r"\bthis is not (?:medical|legal|financial|professional) advice\b",
]

OFFER_PATTERNS = [
    r"\bwould you like\b", r"\blet me know\b", r"\bwant me to\b",
    r"\bhappy to\b", r"\bfeel free to\b", r"\bif you(?:'d| would) like\b",
    r"\bi can also\b", r"\bjust ask\b", r"\bhope (?:this|that) helps\b",
]

REFUSAL_PATTERNS = [
    r"\bi can(?:no|')t\b", r"\bi cannot\b", r"\bi(?:'m| am) not able to\b",
    r"\bi won'?t be able\b", r"\bi(?:'m| am) unable\b", r"\bi must decline\b",
    r"\bi don'?t feel comfortable\b", r"\bi(?:'m| am) not comfortable\b",
    r"\bconsult (?:a|your|with a) (?:professional|doctor|physician|lawyer|attorney|therapist|financial advisor)\b",
    r"\bseek professional\b", r"\bemergency services\b", r"\bcrisis (?:line|hotline)\b",
]

_DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), re.IGNORECASE)
_OFFER_RE = re.compile("|".join(OFFER_PATTERNS), re.IGNORECASE)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

_MD_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
_MD_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
_MD_FENCE_RE = re.compile(r"^```", re.MULTILINE)

MEASURES = ["n_words", "md_density", "exclam_per_100w",
            "disclaimers", "offers", "refusal_markers"]

# The construct each measure is a proxy for (used in the validation section).
MEASURE_CONSTRUCT = {
    "n_words": "VB", "disclaimers": "VB", "offers": "VB",
    "refusal_markers": "GU", "md_density": "RE", "exclam_per_100w": "RE",
}


def text_measures(text: str) -> dict:
    words = len(text.split())
    denom = max(words, 1)
    md_elements = (
        len(_MD_HEADER_RE.findall(text)) + len(_MD_BULLET_RE.findall(text))
        + len(_MD_NUMBERED_RE.findall(text)) + len(_MD_BOLD_RE.findall(text))
        + len(_MD_FENCE_RE.findall(text))
    )
    return {
        "n_words": words,
        "md_density": 100.0 * md_elements / denom,
        "exclam_per_100w": 100.0 * text.count("!") / denom,
        "disclaimers": len(_DISCLAIMER_RE.findall(text)),
        "offers": len(_OFFER_RE.findall(text)),
        "refusal_markers": len(_REFUSAL_RE.findall(text)),
    }


def load_behavioral_measures() -> pd.DataFrame:
    """Per-sample objective measures for all successful behavioral responses."""
    conn = sqlite3.connect(str(RESPONSES_DB))
    df = pd.read_sql_query(
        "SELECT id, model_id, prompt_id, raw_response FROM behavioral_responses "
        "WHERE status = 'success' AND raw_response IS NOT NULL",
        conn,
    )
    conn.close()
    group_map = _load_group_map()
    if group_map:
        df["model_id"] = df["model_id"].map(lambda x: group_map.get(x, x))
    feats = df["raw_response"].map(text_measures).apply(pd.Series)
    return pd.concat([df[["id", "model_id", "prompt_id"]], feats], axis=1)


def boot_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Pearson r, resampling paired rows."""
    n = len(x)
    rs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        xr, yr = x[idx], y[idx]
        if np.std(xr) == 0 or np.std(yr) == 0:
            continue
        rs.append(stats.pearsonr(xr, yr)[0])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def partial_corr(x: np.ndarray, y: np.ndarray, covars: np.ndarray) -> tuple[float, float, int]:
    """Partial Pearson correlation of x and y controlling for covars columns."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(covars).any(axis=1))
    x, y, c = x[mask], y[mask], covars[mask]
    design = np.column_stack([np.ones(len(x)), c])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r, _ = stats.pearsonr(rx, ry)
    # p-value with df adjusted for number of covariates
    n, k = len(x), c.shape[1]
    dof = n - 2 - k
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    return float(r), float(p), n


def corr_row(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> str:
    mask = ~(np.isnan(x) | np.isnan(y))
    r, p = stats.pearsonr(x[mask], y[mask])
    lo, hi = boot_ci(x[mask], y[mask], rng)
    return f"{r:+.3f} [{lo:+.2f}, {hi:+.2f}] (p={p:.4f})"


def main() -> None:
    rng = np.random.default_rng(SEED)

    samples = load_behavioral_measures()
    model_obj = samples.groupby("model_id")[MEASURES].mean()
    model_obj.to_csv(OUTPUT_DIR / "objective_measures_model_level.csv")

    inst = load_instrument_scores().set_index("model_id")
    judge = load_judge_ensemble()
    hr = load_human_ratings()
    human = model_level_human_scores(hr, "all")

    # Align all frames to the models present in the instrument scores
    models = sorted(set(inst.index) & set(model_obj.index))
    inst = inst.loc[models]
    obj = model_obj.loc[models]
    judge = judge.reindex(models)
    human = human.reindex(models)

    lines = [
        "# Objective Behavioral Criteria",
        "",
        f"Objective measures computed from {len(samples)} successful behavioral "
        f"samples ({samples['model_id'].nunique()} models), aggregated to "
        "model-level means. Pearson r with percentile bootstrap 95% CIs "
        f"({N_BOOT} resamples, seed={SEED}), N = {len(models)} models.",
        "",
        "## 1. Self-report factor scores x objective measures (model level)",
        "",
        "| Instrument factor | " + " | ".join(MEASURES) + " |",
        "|---|" + "---|" * len(MEASURES),
    ]
    for f in FACTORS:
        cells = []
        for m in MEASURES:
            r, p = stats.pearsonr(inst[f], obj[m])
            star = "**" if p < 0.05 else ""
            cells.append(f"{star}{r:+.2f}{star}")
        lines.append(f"| {FACTOR_NAMES[f]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Key preregistered-construct pairs (with bootstrap CIs)",
        "",
        "| Pair | r [95% CI] |",
        "|---|---|",
        f"| Verbosity (self-report) x words/response | {corr_row(inst['VB'].to_numpy(), obj['n_words'].to_numpy(), rng)} |",
        f"| Verbosity (self-report) x disclaimers | {corr_row(inst['VB'].to_numpy(), obj['disclaimers'].to_numpy(), rng)} |",
        f"| Verbosity (self-report) x offers | {corr_row(inst['VB'].to_numpy(), obj['offers'].to_numpy(), rng)} |",
        f"| Guardedness (self-report) x refusal markers | {corr_row(inst['GU'].to_numpy(), obj['refusal_markers'].to_numpy(), rng)} |",
        f"| Responsiveness (self-report) x markdown density | {corr_row(inst['RE'].to_numpy(), obj['md_density'].to_numpy(), rng)} |",
        f"| Boldness (self-report) x exclamations/100w | {corr_row(inst['BO'].to_numpy(), obj['exclam_per_100w'].to_numpy(), rng)} |",
        "",
        "## 2. Rater validation: do ratings track the objective measures?",
        "",
        "Model-level correlations of human / judge factor ratings with the "
        "objective proxy for the matched construct.",
        "",
        "| Pair | Human rating | Judge rating |",
        "|---|---|---|",
    ]
    for m in MEASURES:
        f = MEASURE_CONSTRUCT[m]
        h = corr_row(human[f].to_numpy(), obj[m].to_numpy(), rng)
        j = corr_row(judge[f].to_numpy(), obj[m].to_numpy(), rng)
        lines.append(f"| {FACTOR_NAMES[f]} rating x {m} | {h} | {j} |")

    # Sample-level (pooled) correlations of RE ratings with surface features
    ens = load_judge_ensemble_sample_level()
    hmean = (
        hr.groupby("behavioral_response_id")[["corrected_RE"]].mean()
        .rename(columns={"corrected_RE": "human_RE"})
    )
    samp = samples.set_index("id")
    j_samp = ens.join(samp[MEASURES], how="inner")
    h_samp = hmean.join(samp[MEASURES], how="inner")

    lines += [
        "",
        "### Sample-level (pooled) Responsiveness ratings x surface features",
        "",
        "| Feature | Judge RE (N={}) | Human RE (N={}) |".format(len(j_samp), len(h_samp)),
        "|---|---|---|",
    ]
    for m in ["n_words", "md_density", "exclam_per_100w"]:
        rj, pj = stats.pearsonr(j_samp["judge_RE"], j_samp[m])
        rh, ph = stats.pearsonr(h_samp["human_RE"], h_samp[m])
        lines.append(
            f"| {m} | {rj:+.3f} (p={pj:.4f}) | {rh:+.3f} (p={ph:.4f}) |"
        )

    # 3. Partial correlations: does Instrument-Judge RE survive surface controls?
    surface = obj[["n_words", "md_density", "exclam_per_100w"]].to_numpy()
    lines += [
        "",
        "## 3. Surface-controlled Responsiveness correlations (model level)",
        "",
        "Partial Pearson r controlling for mean words/response, markdown "
        "density, and exclamations/100w.",
        "",
        "| Pair | zero-order r | partial r | partial p | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, x, y in [
        ("Instrument x Judge (RE)", inst["RE"].to_numpy(), judge["RE"].to_numpy()),
        ("Instrument x Human (RE)", inst["RE"].to_numpy(), human["RE"].to_numpy()),
        ("Human x Judge (RE)", human["RE"].to_numpy(), judge["RE"].to_numpy()),
    ]:
        mask = ~(np.isnan(x) | np.isnan(y))
        r0, _ = stats.pearsonr(x[mask], y[mask])
        pr, pp, n = partial_corr(x, y, surface)
        lines.append(f"| {label} | {r0:+.3f} | {pr:+.3f} | {pp:.4f} | {n} |")

    out = OUTPUT_DIR / "objective_behavior_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def load_judge_ensemble_sample_level() -> pd.DataFrame:
    """Judge ensemble RE score per behavioral response id."""
    from analysis.judge_analysis import load_judge_ratings, compute_ensemble_scores
    df = load_judge_ratings(RESPONSES_DB)
    ens = compute_ensemble_scores(df)
    re_ens = ens[ens["factor_code"] == "RE"].set_index("behavioral_response_id")
    return re_ens[["ensemble_score"]].rename(columns={"ensemble_score": "judge_RE"})


if __name__ == "__main__":
    main()
