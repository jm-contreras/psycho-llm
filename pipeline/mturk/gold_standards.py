"""Select and manage gold standard items for MTurk quality control."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from statistics import median

from pipeline.judge_prompt import FACTOR_ORDER, reverse_score
from pipeline.storage import DB_PATH, REPO_ROOT
from pipeline.mturk.config import MTURK_SEED, GOLD_ITEMS_PATH, GOLD_ACCURACY_THRESHOLD


def select_gold_items(n_gold: int = 60) -> list[dict]:
    """Select gold standard items where all judges agree (spread ≤ 1 on all 5 factors).

    Process:
      1. Load judge_ratings with parse_status='success'.
      2. Apply reverse_score() using each row's keying column.
      3. Group by behavioral_response_id; compute per-factor spread across judges.
      4. Retain items where all-factor spread ≤ 1.
      5. Ground truth = median corrected score per factor (rounded to int).
      6. Stratify by dimension_code; sample n_gold total seeded by MTURK_SEED.
      7. Save to data/mturk/gold_items.json and return list of dicts.

    Each returned dict has:
      behavioral_response_id, dimension_code, ground_truth (dict factor→int),
      n_judges, prompt_id, subject_model_id.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT jr.id, jr.behavioral_response_id, jr.subject_model_id, jr.prompt_id,
                   jr.keying, jr.score_RE, jr.score_DE, jr.score_BO, jr.score_GU, jr.score_VB,
                   br.dimension_code
            FROM judge_ratings jr
            JOIN behavioral_responses br ON jr.behavioral_response_id = br.id
            WHERE jr.parse_status = 'success'
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No judge_ratings with parse_status='success' found.", file=sys.stderr)
        return []

    # Group corrected scores by behavioral_response_id
    by_response: dict[int, list[dict]] = defaultdict(list)
    meta: dict[int, dict] = {}  # response_id -> {prompt_id, subject_model_id, dimension_code}

    for row in rows:
        rid = row["behavioral_response_id"]
        raw = {f: row[f"score_{f}"] for f in FACTOR_ORDER}
        # Skip rows with any NULL score
        if any(v is None for v in raw.values()):
            continue
        corrected = reverse_score(raw, row["keying"])
        by_response[rid].append(corrected)
        if rid not in meta:
            meta[rid] = {
                "prompt_id": row["prompt_id"],
                "subject_model_id": row["subject_model_id"],
                "dimension_code": row["dimension_code"],
            }

    # Filter: spread ≤ 1 on all 5 factors
    candidates: list[dict] = []
    for rid, score_list in by_response.items():
        if len(score_list) < 2:
            continue  # need at least 2 judges
        for factor in FACTOR_ORDER:
            vals = [s[factor] for s in score_list]
            if max(vals) - min(vals) > 1:
                break
        else:
            # All factors pass
            ground_truth = {
                factor: round(median([s[factor] for s in score_list]))
                for factor in FACTOR_ORDER
            }
            candidates.append(
                {
                    "behavioral_response_id": rid,
                    "dimension_code": meta[rid]["dimension_code"],
                    "prompt_id": meta[rid]["prompt_id"],
                    "subject_model_id": meta[rid]["subject_model_id"],
                    "ground_truth": ground_truth,
                    "n_judges": len(score_list),
                }
            )

    print(
        f"Gold candidates: {len(candidates)} responses pass spread ≤ 1 on all factors.",
        file=sys.stderr,
    )

    if len(candidates) <= n_gold:
        selected = candidates
    else:
        selected = _stratified_sample(candidates, n_gold, seed=MTURK_SEED)

    # Save
    GOLD_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved {len(selected)} gold items to {GOLD_ITEMS_PATH}", file=sys.stderr)
    return selected


def _stratified_sample(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by dimension_code and sample n total, minimum 1 per non-empty stratum."""
    import random
    rng = random.Random(seed)

    by_dim: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_dim[c["dimension_code"]].append(c)

    dims = sorted(by_dim.keys())
    n_dims = len(dims)
    base = max(1, n // n_dims)

    selected: list[dict] = []
    remainder_pool: list[dict] = []

    for dim in dims:
        pool = by_dim[dim][:]
        rng.shuffle(pool)
        take = min(base, len(pool))
        selected.extend(pool[:take])
        remainder_pool.extend(pool[take:])

    # Fill up to n from remainder
    remaining = n - len(selected)
    if remaining > 0 and remainder_pool:
        rng.shuffle(remainder_pool)
        selected.extend(remainder_pool[:remaining])

    return selected[:n]


def load_gold_items() -> list[dict]:
    """Load gold items from disk; return empty list if file not found."""
    if not GOLD_ITEMS_PATH.exists():
        return []
    with open(GOLD_ITEMS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Support both wrapped {"items": [...]} and flat list formats
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data


def check_worker_gold_performance(
    worker_ratings: list[dict],
    gold_items: list[dict],
) -> tuple[float, int, bool]:
    """Check a worker's accuracy on gold standard items.

    worker_ratings: list of dicts with behavioral_response_id and corrected_RE/DE/BO/GU/VB.
    gold_items: list of gold item dicts from select_gold_items().

    Returns (accuracy_pct, n_rated, pass_bool) where:
      accuracy_pct: fraction of per-factor ratings within ±1 of ground truth.
      n_rated: number of gold items the worker rated.
      pass_bool: True if accuracy_pct >= GOLD_ACCURACY_THRESHOLD.
    """
    gold_by_id = {g["behavioral_response_id"]: g for g in gold_items}

    total = 0
    correct = 0

    for wr in worker_ratings:
        rid = wr.get("behavioral_response_id")
        if rid not in gold_by_id:
            continue
        gt = gold_by_id[rid]["ground_truth"]
        for factor in FACTOR_ORDER:
            worker_val = wr.get(f"corrected_{factor}")
            if worker_val is None:
                continue
            total += 1
            if abs(worker_val - gt[factor]) <= 1:
                correct += 1

    if total == 0:
        return 0.0, 0, False

    n_rated = sum(
        1 for wr in worker_ratings if wr.get("behavioral_response_id") in gold_by_id
    )
    accuracy = correct / total
    passed = accuracy >= GOLD_ACCURACY_THRESHOLD
    return accuracy, n_rated, passed
