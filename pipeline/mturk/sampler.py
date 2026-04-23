"""Stratified sampling of behavioral responses for MTurk rating."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict

from pipeline.behavioral_loader import BEHAVIORAL_PROMPTS
from pipeline.config import load_model_registry
from pipeline.judge_prompt import FACTOR_ORDER, reverse_score
from pipeline.storage import DB_PATH
from pipeline.mturk.config import MTURK_SEED, SAMPLE_PATH
from pipeline.mturk.gold_standards import load_gold_items


def select_sample(n_target: int = 300, seed: int = MTURK_SEED) -> list[dict]:
    """Stratified selection of behavioral responses for human rating.

    Strategy:
      - Build model family map from registry (provider field).
      - Load behavioral responses via load_behavioral_samples_for_judging().
      - For each response, load judge scores, apply reverse_score, compute
        consensus score on the target dimension (matching prompt's dimension_code).
      - Bin into tertiles: low (1-2), medium (3), high (4-5).
      - Stratify by (model_family, dimension_code, score_bin).
      - Exclude gold item IDs if gold_items.json exists.
      - Minimum 1 per non-empty stratum; fill proportionally to n_target.
      - Save to data/mturk/sample.json.

    Returns list of dicts with: behavioral_response_id, model_id, prompt_id,
    dimension_code, model_family, consensus_score, score_bin, run_number.
    """
    import random
    rng = random.Random(seed)

    # Model family map: litellm_model_id -> provider string
    models = load_model_registry()
    family_map: dict[str, str] = {m["litellm_model_id"]: m["provider"] for m in models}

    # Load all behavioral responses
    from pipeline.storage import load_behavioral_samples_for_judging
    responses = load_behavioral_samples_for_judging()

    if not responses:
        print("No behavioral responses found.", file=sys.stderr)
        return []

    # Load judge scores
    judge_scores = _load_judge_scores()

    # Excluded IDs (gold items)
    gold_items = load_gold_items()
    excluded_ids = {g["behavioral_response_id"] for g in gold_items}

    # Build enriched response records with consensus scores
    enriched: list[dict] = []
    for resp in responses:
        rid = resp["id"]
        if rid in excluded_ids:
            continue

        dim_code = resp["dimension_code"]
        scores_for_response = judge_scores.get(rid, [])

        if not scores_for_response:
            consensus = None
            score_bin = "unscored"
        else:
            # Average corrected scores for the matching dimension factor
            factor_vals = [s.get(dim_code) for s in scores_for_response if s.get(dim_code) is not None]
            if not factor_vals:
                consensus = None
                score_bin = "unscored"
            else:
                consensus = sum(factor_vals) / len(factor_vals)
                score_bin = _score_bin(consensus)

        model_family = family_map.get(resp["model_id"], "unknown")
        enriched.append(
            {
                "behavioral_response_id": rid,
                "model_id": resp["model_id"],
                "prompt_id": resp["prompt_id"],
                "dimension_code": dim_code,
                "model_family": model_family,
                "consensus_score": consensus,
                "score_bin": score_bin,
                "run_number": resp["run_number"],
            }
        )

    print(f"Eligible responses (excl. gold): {len(enriched)}", file=sys.stderr)

    # Stratify
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for rec in enriched:
        key = (rec["model_family"], rec["dimension_code"], rec["score_bin"])
        strata[key].append(rec)

    selected = _stratified_sample(strata, n_target, rng)

    # Print strata summary
    _print_strata_summary(strata, selected)

    # Save
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_target": n_target,
        "seed": seed,
        "n_selected": len(selected),
        "n_gold_excluded": len(excluded_ids),
        "items": selected,
    }
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(selected)} sampled items to {SAMPLE_PATH}", file=sys.stderr)
    return selected


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_judge_scores() -> dict[int, list[dict]]:
    """Load judge_ratings and return corrected scores per behavioral_response_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT behavioral_response_id, keying,
                   score_RE, score_DE, score_BO, score_GU, score_VB
            FROM judge_ratings
            WHERE parse_status = 'success'
            """
        ).fetchall()
    finally:
        conn.close()

    by_response: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        raw = {f: row[f"score_{f}"] for f in FACTOR_ORDER}
        if any(v is None for v in raw.values()):
            continue
        corrected = reverse_score(raw, row["keying"])
        by_response[row["behavioral_response_id"]].append(corrected)

    return dict(by_response)


def _score_bin(score: float) -> str:
    """Bin a consensus score into low/medium/high tertile labels."""
    if score <= 2.0:
        return "low"
    elif score <= 3.0:
        return "medium"
    else:
        return "high"


def _stratified_sample(
    strata: dict[tuple, list[dict]],
    n_target: int,
    rng,
) -> list[dict]:
    """Sample n_target items from strata with minimum 1 per non-empty stratum."""
    non_empty = {k: v for k, v in strata.items() if v}
    n_strata = len(non_empty)

    if n_strata == 0:
        return []

    base = max(1, n_target // n_strata)
    selected: list[dict] = []
    remainder_pool: list[dict] = []

    sorted_keys = sorted(non_empty.keys())
    for key in sorted_keys:
        pool = non_empty[key][:]
        rng.shuffle(pool)
        take = min(base, len(pool))
        selected.extend(pool[:take])
        remainder_pool.extend(pool[take:])

    # Fill up to n_target from remainder
    remaining = n_target - len(selected)
    if remaining > 0 and remainder_pool:
        rng.shuffle(remainder_pool)
        selected.extend(remainder_pool[:remaining])

    return selected[:n_target]


def _print_strata_summary(
    strata: dict[tuple, list[dict]],
    selected: list[dict],
) -> None:
    """Print a summary table of strata sizes and selection counts."""
    selected_ids = {r["behavioral_response_id"] for r in selected}
    print(
        f"\n{'FAMILY':<20} {'DIM':<6} {'BIN':<10} {'TOTAL':>7} {'SELECTED':>9}",
        file=sys.stderr,
    )
    print("-" * 58, file=sys.stderr)

    for key in sorted(strata.keys()):
        family, dim, score_bin = key
        pool = strata[key]
        n_total = len(pool)
        n_sel = sum(1 for r in pool if r["behavioral_response_id"] in selected_ids)
        print(
            f"{family:<20} {dim:<6} {score_bin:<10} {n_total:>7} {n_sel:>9}",
            file=sys.stderr,
        )
    print(file=sys.stderr)
