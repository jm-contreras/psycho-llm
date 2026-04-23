"""Command-line interface for the MTurk behavioral rating pipeline."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    """Entry point for `python -m pipeline.mturk`."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.mturk",
        description="MTurk behavioral rating pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # qualify
    p_qualify = sub.add_parser("qualify", help="Create or retrieve the worker qualification type.")

    # gold
    p_gold = sub.add_parser("gold", help="Select gold standard items from judge_ratings.")
    p_gold.add_argument("--n-gold", type=int, default=60, help="Target number of gold items (default: 60).")

    # sample
    p_sample = sub.add_parser("sample", help="Select stratified sample for HIT submission.")
    p_sample.add_argument("--n-target", type=int, default=300, help="Target sample size (default: 300).")
    p_sample.add_argument("--seed", type=int, default=None, help="Random seed override.")

    # submit
    p_submit = sub.add_parser("submit", help="Create HITs on MTurk.")
    p_submit.add_argument("--dry-run", action="store_true", help="Print summary without submitting.")
    p_submit.add_argument("--batch-size", type=int, default=None, help="Submit only first N items.")

    # status
    p_status = sub.add_parser("status", help="Show HIT status summary from manifest.")

    # collect
    p_collect = sub.add_parser("collect", help="Poll assignments, parse, reverse-score, store.")

    # tiebreak
    p_tiebreak = sub.add_parser("tiebreak", help="Extend disagreement HITs with a 3rd rater.")

    # disqualify
    p_disqualify = sub.add_parser("disqualify", help="Revoke qualification for low-accuracy workers.")
    p_disqualify.add_argument("--dry-run", action="store_true", help="Print workers without revoking.")
    p_disqualify.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override gold accuracy threshold (0-1, default: config GOLD_ACCURACY_THRESHOLD).",
    )

    args = parser.parse_args()

    if args.command == "qualify":
        _cmd_qualify()

    elif args.command == "gold":
        _cmd_gold(n_gold=args.n_gold)

    elif args.command == "sample":
        from pipeline.mturk.config import MTURK_SEED
        seed = args.seed if args.seed is not None else MTURK_SEED
        _cmd_sample(n_target=args.n_target, seed=seed)

    elif args.command == "submit":
        _cmd_submit(dry_run=args.dry_run, batch_size=args.batch_size)

    elif args.command == "status":
        _cmd_status()

    elif args.command == "collect":
        _cmd_collect()

    elif args.command == "tiebreak":
        _cmd_tiebreak()

    elif args.command == "disqualify":
        _cmd_disqualify(dry_run=args.dry_run, threshold=args.threshold)


# ── Subcommand implementations ─────────────────────────────────────────────────

def _cmd_qualify() -> None:
    from pipeline.mturk.config import get_mturk_client, MTURK_SANDBOX
    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()
    from pipeline.mturk.qualification import get_or_create_qualification
    qt_id = get_or_create_qualification(client)
    print(f"QualificationTypeId: {qt_id}")


def _cmd_gold(n_gold: int) -> None:
    from pipeline.mturk.gold_standards import select_gold_items
    items = select_gold_items(n_gold=n_gold)
    print(f"Selected {len(items)} gold items.")


def _cmd_sample(n_target: int, seed: int) -> None:
    from pipeline.mturk.sampler import select_sample
    items = select_sample(n_target=n_target, seed=seed)
    print(f"Selected {len(items)} sample items.")


def _cmd_submit(dry_run: bool, batch_size: int | None) -> None:
    from pipeline.mturk.config import get_mturk_client, MTURK_SANDBOX
    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()
    from pipeline.mturk.submit import submit_hits
    entries = submit_hits(client, dry_run=dry_run, batch_size=batch_size)
    if not dry_run:
        print(f"Submitted {len(entries)} HITs.")


def _cmd_status() -> None:
    from pipeline.mturk.config import get_mturk_client, MANIFEST_PATH, MTURK_SANDBOX
    if not MANIFEST_PATH.exists():
        print("No manifest found. Run `submit` first.")
        return

    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} HITs\n")
    print(f"{'HIT_ID':<30} {'DIM':<6} {'GOLD':<6} {'SUBMITTED':>10} {'AVAIL':>7} {'PEND':>6} {'COMP':>6}")
    print("-" * 80)

    counts = {"available": 0, "pending": 0, "completed": 0}

    for entry in manifest:
        hit_id = entry["hit_id"]
        dim = entry.get("dimension_code", "?")
        is_gold = "Y" if entry.get("is_gold") else "N"
        submitted = entry.get("submitted_at", "")[:10]

        try:
            resp = client.get_hit(HITId=hit_id)
            hit = resp["HIT"]
            avail = hit.get("NumberOfAssignmentsAvailable", "?")
            pend = hit.get("NumberOfAssignmentsPending", "?")
            comp = hit.get("NumberOfAssignmentsCompleted", "?")
            counts["available"] += avail if isinstance(avail, int) else 0
            counts["pending"] += pend if isinstance(pend, int) else 0
            counts["completed"] += comp if isinstance(comp, int) else 0
        except Exception as exc:  # noqa: BLE001
            avail = pend = comp = "ERR"

        print(f"{hit_id:<30} {dim:<6} {is_gold:<6} {submitted:>10} {str(avail):>7} {str(pend):>6} {str(comp):>6}")

    print(f"\nTotals: available={counts['available']}, pending={counts['pending']}, completed={counts['completed']}")


def _cmd_collect() -> None:
    from pipeline.mturk.config import get_mturk_client, MTURK_SANDBOX
    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()
    from pipeline.mturk.collect import collect_results
    new_rows = collect_results(client)
    print(f"Stored {len(new_rows)} new assignment(s).")


def _cmd_tiebreak() -> None:
    from pipeline.mturk.config import get_mturk_client, MTURK_SANDBOX, MANIFEST_PATH
    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()

    # First collect latest results
    from pipeline.mturk.collect import collect_results, check_disagreement
    collect_results(client)

    # Load stored assignments grouped by HIT
    import sqlite3
    from pipeline.storage import DB_PATH
    from collections import defaultdict

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT hit_id, corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB "
            "FROM human_ratings WHERE worker_flagged = 0"
        ).fetchall()
    finally:
        conn.close()

    by_hit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hit[row["hit_id"]].append(dict(row))

    # Load manifest for n_assignments info
    if not MANIFEST_PATH.exists():
        print("No manifest found.", file=sys.stderr)
        return

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_by_hit = {e["hit_id"]: e for e in manifest}

    # Find HITs with exactly 2 raters that disagree
    disagreement_hit_ids: list[str] = []
    for hit_id, assignments in by_hit.items():
        if len(assignments) == 2 and check_disagreement(assignments):
            disagreement_hit_ids.append(hit_id)

    print(f"Found {len(disagreement_hit_ids)} HITs with disagreement.", file=sys.stderr)

    if not disagreement_hit_ids:
        print("No tiebreak assignments needed.")
        return

    from pipeline.mturk.collect import add_tiebreak_assignments
    extended = add_tiebreak_assignments(client, disagreement_hit_ids)
    print(f"Extended {extended} HITs with tiebreak assignment.")


def _cmd_disqualify(dry_run: bool, threshold: float | None) -> None:
    from pipeline.mturk.config import (
        get_mturk_client, MTURK_SANDBOX, GOLD_ACCURACY_THRESHOLD,
    )
    import sqlite3
    from pipeline.storage import DB_PATH
    from pipeline.mturk.gold_standards import load_gold_items, check_worker_gold_performance
    from pipeline.mturk.qualification import get_or_create_qualification

    thresh = threshold if threshold is not None else GOLD_ACCURACY_THRESHOLD

    print(f"Using {'SANDBOX' if MTURK_SANDBOX else 'PRODUCTION'} endpoint.", file=sys.stderr)
    client = get_mturk_client()

    gold_items = load_gold_items()
    if not gold_items:
        print("No gold items loaded — cannot assess worker quality.")
        return

    # Load all gold assignments per worker
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT worker_id, behavioral_response_id, "
            "corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB "
            "FROM human_ratings WHERE is_gold = 1"
        ).fetchall()
    finally:
        conn.close()

    from collections import defaultdict
    by_worker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_worker[row["worker_id"]].append(dict(row))

    qual_id = get_or_create_qualification(client)

    print(f"\n{'WORKER':<24} {'N GOLD':>8} {'ACCURACY':>10} {'ACTION':<12}")
    print("-" * 58)

    revoked = 0
    for worker_id, ratings in sorted(by_worker.items()):
        accuracy, n_rated, passed = check_worker_gold_performance(ratings, gold_items)
        action = "PASS" if passed else "REVOKE"
        print(f"{worker_id:<24} {n_rated:>8} {accuracy:>10.2%} {action:<12}")

        if not passed and not dry_run:
            try:
                client.disassociate_qualification_from_worker(
                    WorkerId=worker_id,
                    QualificationTypeId=qual_id,
                    Reason=(
                        f"Gold standard accuracy {accuracy:.0%} is below threshold "
                        f"{thresh:.0%}. You may re-take the qualification test after 24 hours."
                    ),
                )
                revoked += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR revoking qualification for {worker_id}: {exc}", file=sys.stderr)

    if dry_run:
        print("\ndry_run=True — no qualifications revoked.")
    else:
        print(f"\nRevoked qualification from {revoked} worker(s).")
