"""Post-study analysis for the Prolific rating pipeline.

Functions:
  collect_results()    — compute per-participant gold accuracy, flag low-quality
                          participants, export CSV, return summary dict.
  export_csv()         — write prolific_ratings to RESULTS_CSV_PATH.
  find_disagreements() — find behavioral_response_ids needing a 3rd rater.
  compute_icc()        — ICC(2,k) per factor.
  print_status()       — console summary: sessions, coverage, quality.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

from pipeline.judge_prompt import FACTOR_ORDER
from pipeline.mturk.collect import check_disagreement, icc_2_3
from pipeline.mturk.gold_standards import load_gold_items
from pipeline.prolific.config import (
    GOLD_ACCURACY_THRESHOLD,
    RESULTS_CSV_PATH,
)
from pipeline.prolific.models import get_db

# ── CSV schema ────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "id",
    "prolific_pid",
    "study_id",
    "session_id",
    "behavioral_response_id",
    "prompt_id",
    "keying",
    "is_gold",
    "item_position",
    "raw_RE", "raw_DE", "raw_BO", "raw_GU", "raw_VB",
    "corrected_RE", "corrected_DE", "corrected_BO", "corrected_GU", "corrected_VB",
    "response_time_seconds",
    "participant_flagged",
    "gold_accuracy",
    "timestamp",
]


# ── Public functions ──────────────────────────────────────────────────────────

def collect_results() -> dict:
    """Compute per-participant gold accuracy, flag low-quality participants, export CSV.

    Returns:
        {
            n_participants: int,
            n_ratings: int,
            n_flagged: int,
            mean_gold_accuracy: float | None,
        }
    """
    gold_items = load_gold_items()
    gold_by_id = {g["behavioral_response_id"]: g for g in gold_items}

    conn = get_db()
    try:
        # Load all ratings grouped by prolific_pid
        rating_rows = conn.execute(
            "SELECT * FROM prolific_ratings ORDER BY prolific_pid, item_position"
        ).fetchall()
        session_rows = conn.execute(
            "SELECT prolific_pid, gold_accuracy FROM prolific_sessions "
            "WHERE status = 'complete'"
        ).fetchall()
    finally:
        conn.close()

    # Per-participant: compute gold accuracy and determine flagged status
    by_pid: dict[str, list] = defaultdict(list)
    for row in rating_rows:
        by_pid[row["prolific_pid"]].append(dict(row))

    n_participants = len(by_pid)
    n_ratings = len(rating_rows)
    gold_accuracies: list[float] = []
    flagged_pids: set[str] = set()

    for pid, ratings in by_pid.items():
        gold_ratings = [r for r in ratings if r["is_gold"]]
        if not gold_ratings:
            continue

        total = correct = 0
        for r in gold_ratings:
            rid = r["behavioral_response_id"]
            if rid not in gold_by_id:
                continue
            gt = gold_by_id[rid]["ground_truth"]
            for factor in FACTOR_ORDER:
                val = r.get(f"corrected_{factor}")
                if val is None:
                    continue
                total += 1
                if abs(val - gt[factor]) <= 1:
                    correct += 1

        if total > 0:
            acc = correct / total
            gold_accuracies.append(acc)
            if acc < GOLD_ACCURACY_THRESHOLD:
                flagged_pids.add(pid)

    # Write participant_flagged flag back to DB
    if flagged_pids:
        conn = get_db()
        try:
            placeholders = ",".join("?" for _ in flagged_pids)
            conn.execute(
                f"UPDATE prolific_ratings SET participant_flagged = 1 "
                f"WHERE prolific_pid IN ({placeholders})",
                list(flagged_pids),
            )
            conn.commit()
        finally:
            conn.close()

    mean_gold = sum(gold_accuracies) / len(gold_accuracies) if gold_accuracies else None

    export_csv()

    return {
        "n_participants": n_participants,
        "n_ratings": n_ratings,
        "n_flagged": len(flagged_pids),
        "mean_gold_accuracy": mean_gold,
    }


def export_csv() -> None:
    """Write all prolific_ratings rows to RESULTS_CSV_PATH."""
    conn = get_db()
    try:
        # Fetch all columns that exist in the CSV schema
        rows = conn.execute(
            f"SELECT {', '.join(c for c in _CSV_COLUMNS if c != 'gold_accuracy')} "
            f"FROM prolific_ratings ORDER BY id"
        ).fetchall()
        # Also pull gold_accuracy from sessions table via join
        rows_with_ga = conn.execute(
            """
            SELECT pr.id, ps.gold_accuracy
            FROM prolific_ratings pr
            LEFT JOIN prolific_sessions ps ON pr.session_id = ps.session_id
            ORDER BY pr.id
            """
        ).fetchall()
    finally:
        conn.close()

    # Build gold_accuracy lookup by row id
    ga_by_id = {r["id"]: r["gold_accuracy"] for r in rows_with_ga}

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            d = dict(row)
            d["gold_accuracy"] = ga_by_id.get(d["id"])
            writer.writerow(d)

    print(f"Exported {len(rows)} rows to {RESULTS_CSV_PATH}", file=sys.stderr)


def find_disagreements() -> list[int]:
    """Return behavioral_response_ids with 2 unflagged ratings where spread >= DISAGREEMENT_THRESHOLD.

    Uses check_disagreement() from pipeline.mturk.collect.
    """
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT behavioral_response_id,
                   corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB
            FROM prolific_ratings
            WHERE participant_flagged = 0
            ORDER BY behavioral_response_id
            """
        ).fetchall()
    finally:
        conn.close()

    by_rid: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_rid[row["behavioral_response_id"]].append(dict(row))

    disagreement_ids: list[int] = []
    for rid, rater_list in by_rid.items():
        if len(rater_list) == 2 and check_disagreement(rater_list):
            disagreement_ids.append(rid)

    return disagreement_ids


def compute_icc() -> dict[str, float | None]:
    """Compute ICC(2,k) per factor across all unflagged prolific_ratings.

    Uses icc_2_3() from pipeline.mturk.collect.
    Returns dict mapping factor code -> ICC value (or None if insufficient data).
    """
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT behavioral_response_id,
                   corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB
            FROM prolific_ratings
            WHERE participant_flagged = 0
            """
        ).fetchall()
    finally:
        conn.close()

    by_rid: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_rid[row["behavioral_response_id"]].append(dict(row))

    multi = {rid: rlist for rid, rlist in by_rid.items() if len(rlist) >= 2}
    if not multi:
        print("No behavioral_response_ids with ≥ 2 raters for ICC.", file=sys.stderr)
        return {f: None for f in FACTOR_ORDER}

    result: dict[str, float | None] = {}
    for factor in FACTOR_ORDER:
        icc_val, ci_low, ci_high, n_items = icc_2_3(multi, f"corrected_{factor}")
        result[factor] = icc_val

    return result


def print_status() -> None:
    """Print a console summary: sessions by status, rating coverage, participant quality."""
    conn = get_db()
    try:
        session_counts = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM prolific_sessions GROUP BY status"
        ).fetchall()
        n_ratings = conn.execute(
            "SELECT COUNT(*) AS cnt FROM prolific_ratings"
        ).fetchone()["cnt"]
        coverage_rows = conn.execute(
            """
            SELECT behavioral_response_id, COUNT(*) AS cnt
            FROM prolific_ratings WHERE participant_flagged = 0
            GROUP BY behavioral_response_id
            """
        ).fetchall()
        participant_rows = conn.execute(
            """
            SELECT ps.prolific_pid,
                   COUNT(pr.id) AS n_ratings,
                   SUM(pr.is_gold) AS n_gold,
                   ps.gold_accuracy,
                   SUM(pr.participant_flagged) AS n_flagged
            FROM prolific_sessions ps
            LEFT JOIN prolific_ratings pr ON pr.prolific_pid = ps.prolific_pid
            GROUP BY ps.prolific_pid
            ORDER BY ps.gold_accuracy DESC NULLS LAST
            """
        ).fetchall()
    finally:
        conn.close()

    print("\n=== Prolific Survey Status ===\n")

    print("Sessions by status:")
    for r in session_counts:
        print(f"  {r['status']:<16} {r['cnt']:>6}")
    print()

    # Rating coverage
    coverage_buckets: dict[int, int] = defaultdict(int)
    for r in coverage_rows:
        cnt = r["cnt"]
        bucket = cnt if cnt <= 3 else 4
        coverage_buckets[bucket] += 1

    print(f"Total ratings collected: {n_ratings}")
    print("Coverage (# rated items with N unflagged ratings):")
    for k in sorted(coverage_buckets.keys()):
        label = f"{k}+" if k == 4 else str(k)
        print(f"  {label} ratings:  {coverage_buckets[k]} items")
    print()

    # Participant quality
    print(f"{'PARTICIPANT':<28} {'RATINGS':>8} {'GOLD':>6} {'GOLD ACC':>10} {'FLAGGED':>8}")
    print("-" * 66)
    for r in participant_rows:
        acc = f"{r['gold_accuracy']:.2%}" if r["gold_accuracy"] is not None else "N/A"
        print(
            f"{r['prolific_pid']:<28} {(r['n_ratings'] or 0):>8} "
            f"{(r['n_gold'] or 0):>6} {acc:>10} {(r['n_flagged'] or 0):>8}"
        )
    print()

    # ICC
    print("ICC(2,k) per factor:")
    icc_vals = compute_icc()
    for factor, icc in icc_vals.items():
        if icc is None:
            print(f"  {factor}: N/A")
        else:
            flag = " *" if icc >= 0.60 else " !"
            print(f"  {factor}: {icc:.3f}{flag}")
    print("* ICC >= 0.60 (target)   ! ICC < 0.60\n")
