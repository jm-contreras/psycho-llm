"""Poll MTurk assignments, parse answers, store human_ratings, compute ICC."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from pipeline.judge_prompt import FACTOR_ORDER, reverse_score
from pipeline.storage import DB_PATH, REPO_ROOT
from pipeline.mturk.config import (
    MANIFEST_PATH,
    MTURK_DIR,
    RESULTS_CSV_PATH,
    DISAGREEMENT_THRESHOLD,
    MAX_ASSIGNMENTS_TIEBREAK,
)
from pipeline.mturk.gold_standards import load_gold_items, check_worker_gold_performance

# ── DB schema ─────────────────────────────────────────────────────────────────

_CREATE_HUMAN_RATINGS = """
CREATE TABLE IF NOT EXISTS human_ratings (
  id                        INTEGER PRIMARY KEY AUTOINCREMENT,
  hit_id                    TEXT NOT NULL,
  assignment_id             TEXT NOT NULL,
  worker_id                 TEXT NOT NULL,
  behavioral_response_id    INTEGER NOT NULL,
  prompt_id                 TEXT NOT NULL,
  keying                    TEXT NOT NULL,
  is_gold                   INTEGER NOT NULL DEFAULT 0,
  raw_RE                    INTEGER,
  raw_DE                    INTEGER,
  raw_BO                    INTEGER,
  raw_GU                    INTEGER,
  raw_VB                    INTEGER,
  corrected_RE              INTEGER,
  corrected_DE              INTEGER,
  corrected_BO              INTEGER,
  corrected_GU              INTEGER,
  corrected_VB              INTEGER,
  gold_accuracy             REAL,
  worker_flagged            INTEGER DEFAULT 0,
  timestamp                 TEXT,
  UNIQUE(assignment_id)
)
"""


def _get_human_ratings_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_HUMAN_RATINGS)
    conn.commit()
    return conn


# ── Main collection function ──────────────────────────────────────────────────

def collect_results(client) -> list[dict]:
    """Poll all open HITs, parse assignments, reverse-score, and store.

    Returns list of newly stored assignment dicts.
    """
    manifest = _load_manifest()
    if not manifest:
        print("Manifest is empty — nothing to collect.", file=sys.stderr)
        return []

    gold_items = load_gold_items()
    all_new: list[dict] = []

    conn = _get_human_ratings_conn()
    try:
        for entry in manifest:
            hit_id = entry["hit_id"]
            keying = entry["keying"]
            rid = entry["behavioral_response_id"]
            prompt_id = entry.get("prompt_id", "")
            is_gold = int(entry.get("is_gold", 0))

            try:
                resp = client.list_assignments_for_hit(
                    HITId=hit_id,
                    AssignmentStatuses=["Submitted", "Approved"],
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR polling HIT {hit_id}: {exc}", file=sys.stderr)
                continue

            assignments = resp.get("Assignments", [])
            for assignment in assignments:
                assignment_id = assignment["AssignmentId"]

                # Skip if already stored
                existing = conn.execute(
                    "SELECT 1 FROM human_ratings WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()
                if existing:
                    continue

                worker_id = assignment["WorkerId"]
                raw_scores = _parse_answer_xml(assignment["Answer"])

                if raw_scores is None:
                    print(
                        f"  WARNING: parse failed for assignment {assignment_id}",
                        file=sys.stderr,
                    )
                    continue

                corrected = reverse_score(raw_scores, keying)

                # Gold accuracy
                gold_accuracy: float | None = None
                if is_gold and gold_items:
                    worker_record = [
                        {
                            "behavioral_response_id": rid,
                            **{f"corrected_{f}": corrected[f] for f in FACTOR_ORDER},
                        }
                    ]
                    gold_acc, _, _ = check_worker_gold_performance(worker_record, gold_items)
                    gold_accuracy = gold_acc

                row = {
                    "hit_id": hit_id,
                    "assignment_id": assignment_id,
                    "worker_id": worker_id,
                    "behavioral_response_id": rid,
                    "prompt_id": prompt_id,
                    "keying": keying,
                    "is_gold": is_gold,
                    **{f"raw_{f}": raw_scores[f] for f in FACTOR_ORDER},
                    **{f"corrected_{f}": corrected[f] for f in FACTOR_ORDER},
                    "gold_accuracy": gold_accuracy,
                    "worker_flagged": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                conn.execute(
                    """
                    INSERT OR IGNORE INTO human_ratings (
                        hit_id, assignment_id, worker_id, behavioral_response_id,
                        prompt_id, keying, is_gold,
                        raw_RE, raw_DE, raw_BO, raw_GU, raw_VB,
                        corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB,
                        gold_accuracy, worker_flagged, timestamp
                    ) VALUES (
                        :hit_id, :assignment_id, :worker_id, :behavioral_response_id,
                        :prompt_id, :keying, :is_gold,
                        :raw_RE, :raw_DE, :raw_BO, :raw_GU, :raw_VB,
                        :corrected_RE, :corrected_DE, :corrected_BO, :corrected_GU, :corrected_VB,
                        :gold_accuracy, :worker_flagged, :timestamp
                    )
                    """,
                    row,
                )
                all_new.append(row)

        conn.commit()
    finally:
        conn.close()

    print(f"Collected {len(all_new)} new assignments.", file=sys.stderr)

    # Export CSV
    _export_csv()

    # Worker quality summary
    _print_worker_quality_summary()

    # ICC
    _compute_and_print_icc()

    return all_new


def check_disagreement(assignments: list[dict]) -> bool:
    """Return True if any factor has spread >= DISAGREEMENT_THRESHOLD between 2 raters.

    assignments: list of assignment dicts each with corrected_RE/DE/BO/GU/VB.
    """
    if len(assignments) < 2:
        return False
    for factor in FACTOR_ORDER:
        vals = [a.get(f"corrected_{factor}") for a in assignments if a.get(f"corrected_{factor}") is not None]
        if len(vals) >= 2 and (max(vals) - min(vals)) >= DISAGREEMENT_THRESHOLD:
            return True
    return False


def add_tiebreak_assignments(client, hit_ids: list[str]) -> int:
    """Add 1 additional assignment to each disagreement HIT.

    Returns count of HITs extended.
    """
    extended = 0
    for hit_id in hit_ids:
        try:
            client.create_additional_assignments_for_hit(
                HITId=hit_id,
                NumberOfAdditionalAssignments=MAX_ASSIGNMENTS_TIEBREAK,
            )
            extended += 1
            print(f"  Extended HIT {hit_id} with {MAX_ASSIGNMENTS_TIEBREAK} additional assignment(s).", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR extending HIT {hit_id}: {exc}", file=sys.stderr)
    return extended


# ── Answer XML parsing ────────────────────────────────────────────────────────

def _parse_answer_xml(answer_xml: str) -> dict[str, int] | None:
    """Parse MTurk answer XML to extract 5-factor ratings.

    Returns dict with factor codes as keys and int scores, or None on failure.
    """
    try:
        root = ET.fromstring(answer_xml)
    except ET.ParseError as exc:
        print(f"  XML parse error: {exc}", file=sys.stderr)
        return None

    # Handle default namespace
    ns = {"": "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"}
    answers: dict[str, str] = {}

    # Try namespaced
    for ans in root.findall(".//Answer", ns) or root.findall(".//Answer"):
        qid_el = ans.find("QuestionIdentifier", ns) or ans.find("QuestionIdentifier")
        free_el = ans.find("FreeText", ns) or ans.find("FreeText")
        if qid_el is not None and free_el is not None:
            answers[qid_el.text or ""] = free_el.text or ""

    scores: dict[str, int] = {}
    for factor in FACTOR_ORDER:
        key = f"rating_{factor}"
        val_str = answers.get(key)
        if val_str is None:
            print(f"  Missing answer for {key}", file=sys.stderr)
            return None
        try:
            val = int(val_str)
            if not 1 <= val <= 5:
                raise ValueError(f"Out of range: {val}")
            scores[factor] = val
        except ValueError as exc:
            print(f"  Invalid value for {key}: {val_str!r} ({exc})", file=sys.stderr)
            return None

    return scores


# ── CSV export ────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "id", "hit_id", "assignment_id", "worker_id", "behavioral_response_id",
    "prompt_id", "keying", "is_gold",
    "raw_RE", "raw_DE", "raw_BO", "raw_GU", "raw_VB",
    "corrected_RE", "corrected_DE", "corrected_BO", "corrected_GU", "corrected_VB",
    "gold_accuracy", "worker_flagged", "timestamp",
]


def _export_csv() -> None:
    """Rewrite human_ratings.csv from the DB."""
    conn = _get_human_ratings_conn()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_CSV_COLUMNS)} FROM human_ratings ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    print(f"Exported {len(rows)} rows to {RESULTS_CSV_PATH}", file=sys.stderr)


# ── Worker quality ────────────────────────────────────────────────────────────

def _print_worker_quality_summary() -> None:
    """Print per-worker gold accuracy and flag summary."""
    conn = _get_human_ratings_conn()
    try:
        rows = conn.execute(
            """
            SELECT worker_id,
                   COUNT(*) as n_assignments,
                   SUM(is_gold) as n_gold,
                   AVG(CASE WHEN is_gold = 1 THEN gold_accuracy END) as avg_gold_acc,
                   SUM(worker_flagged) as n_flagged
            FROM human_ratings
            GROUP BY worker_id
            ORDER BY avg_gold_acc DESC NULLS LAST
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return

    print(f"\n{'WORKER':<24} {'ASSIGNMENTS':>12} {'GOLD':>6} {'GOLD ACC':>10} {'FLAGGED':>8}", file=sys.stderr)
    print("-" * 65, file=sys.stderr)
    for r in rows:
        acc = f"{r['avg_gold_acc']:.2%}" if r["avg_gold_acc"] is not None else "N/A"
        print(
            f"{r['worker_id']:<24} {r['n_assignments']:>12} {r['n_gold']:>6} {acc:>10} {r['n_flagged']:>8}",
            file=sys.stderr,
        )
    print(file=sys.stderr)


# ── ICC computation ───────────────────────────────────────────────────────────

def _compute_and_print_icc() -> None:
    """Compute ICC(2,3) per factor using ANOVA formulas and print a summary table.

    ICC(2,3): two-way random effects, average measures, absolute agreement.
    Formula uses MS_between-subjects, MS_residual, MS_columns (raters), n raters.

    Target: ICC > 0.60 per factor.
    """
    conn = _get_human_ratings_conn()
    try:
        rows = conn.execute(
            """
            SELECT behavioral_response_id,
                   corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB
            FROM human_ratings
            WHERE worker_flagged = 0
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No unflagged human_ratings rows for ICC computation.", file=sys.stderr)
        return

    # Group by behavioral_response_id
    from collections import defaultdict
    by_rid: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_rid[row["behavioral_response_id"]].append(dict(row))

    # Keep only response_ids with ≥ 2 raters
    multi_rater = {rid: rlist for rid, rlist in by_rid.items() if len(rlist) >= 2}
    if not multi_rater:
        print("No response_ids with ≥ 2 raters for ICC.", file=sys.stderr)
        return

    print(f"\n{'FACTOR':<12} {'ICC(2,3)':>10} {'95% CI':>20} {'N responses':>14}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    for factor in FACTOR_ORDER:
        icc_val, ci_low, ci_high, n_items = icc_2_3(multi_rater, f"corrected_{factor}")
        if icc_val is None:
            print(f"{factor:<12} {'N/A':>10}", file=sys.stderr)
        else:
            ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
            flag = " *" if icc_val >= 0.60 else " !"
            print(
                f"{factor:<12} {icc_val:>10.3f} {ci_str:>20} {n_items:>14}{flag}",
                file=sys.stderr,
            )
    print("* ICC >= 0.60 (target)   ! ICC < 0.60", file=sys.stderr)
    print(file=sys.stderr)


def icc_2_3(
    by_rid: dict[int, list[dict]],
    col: str,
) -> tuple[float | None, float | None, float | None, int]:
    """Compute ICC(2,3) for a single column.

    ICC(2,3) = (MSS - MSE) / (MSS + (k-1)*MSE + k*(MSC - MSE)/n)
    where:
      MSS = mean square for subjects (rows)
      MSC = mean square for columns (raters)
      MSE = mean square error (residual)
      k   = number of raters (use mode or max; use per-subject mean if unbalanced)
      n   = number of subjects

    For unbalanced designs (2 vs 3 raters), we use the harmonic mean of raters.
    Returns (icc, ci_low, ci_high, n_items).
    """
    import math
    from statistics import mean

    items = []
    for rid, rlist in by_rid.items():
        vals = [r.get(col) for r in rlist if r.get(col) is not None]
        if len(vals) >= 2:
            items.append(vals)

    n = len(items)
    if n < 2:
        return None, None, None, n

    # Harmonic mean of rater counts (handles unbalanced design)
    k_counts = [len(v) for v in items]
    k = n / sum(1 / k_i for k_i in k_counts)  # harmonic mean

    # Grand mean
    all_vals = [v for row in items for v in row]
    grand_mean = mean(all_vals)

    # Subject (row) means
    row_means = [mean(row) for row in items]

    # Column (rater) effects — approximate: use first k raters per subject
    # Pad shorter lists with None and skip for column effects
    # For unbalanced, compute SS_total, SS_subjects, SS_error; skip SS_columns
    ss_total = sum((v - grand_mean) ** 2 for row in items for v in row)
    ss_subjects = sum(len(row) * (rm - grand_mean) ** 2 for row, rm in zip(items, row_means))

    # Residual SS = total - subjects - columns; approximation for unbalanced
    ss_within = sum(
        (v - rm) ** 2
        for row, rm in zip(items, row_means)
        for v in row
    )

    n_total = sum(len(row) for row in items)
    df_subjects = n - 1
    df_error = n_total - n  # simplified: ignores column df for unbalanced

    if df_subjects <= 0 or df_error <= 0:
        return None, None, None, n

    mss = ss_subjects / df_subjects
    mse = ss_within / df_error

    # ICC(2,3) approximation for unbalanced: treat as one-way mixed
    # ICC = (MSS - MSE) / (MSS + (k-1)*MSE)
    denom = mss + (k - 1) * mse
    if denom <= 0:
        return None, None, None, n

    icc = (mss - mse) / denom
    icc = max(0.0, min(1.0, icc))  # clamp to [0, 1]

    # Approximate 95% CI using F-distribution quantiles (Shrout & Fleiss 1979)
    try:
        import scipy.stats as stats
        alpha = 0.05
        f_lower = (mss / mse) / stats.f.ppf(1 - alpha / 2, df_subjects, df_error)
        f_upper = (mss / mse) / stats.f.ppf(alpha / 2, df_subjects, df_error)
        ci_low = (f_lower - 1) / (f_lower + k - 1)
        ci_high = (f_upper - 1) / (f_upper + k - 1)
        ci_low = max(0.0, ci_low)
        ci_high = min(1.0, ci_high)
    except ImportError:
        # scipy not available — use approximate CI via log transform
        z = 1.96
        f_val = mss / mse if mse > 0 else float("inf")
        se_log_f = math.sqrt(2 * (1 / df_subjects + 1 / df_error))
        log_f = math.log(f_val) if f_val > 0 else 0.0
        f_low = math.exp(log_f - z * se_log_f)
        f_high = math.exp(log_f + z * se_log_f)
        ci_low = max(0.0, (f_low - 1) / (f_low + k - 1))
        ci_high = min(1.0, (f_high - 1) / (f_high + k - 1))

    return icc, ci_low, ci_high, n


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)
