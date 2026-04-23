"""Command-line interface for the Prolific behavioral rating pipeline.

Subcommands:
  serve      Start the Flask survey server.
  status     Print session/rating/quality summary.
  collect    Compute gold accuracy, flag participants, export CSV.
  tiebreak   Find and mark disagreement items for a 3rd rater.

Usage:
  python -m pipeline.prolific serve [--port 5000] [--debug]
  python -m pipeline.prolific status
  python -m pipeline.prolific collect
  python -m pipeline.prolific tiebreak [--dry-run]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Entry point for `python -m pipeline.prolific`."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.prolific",
        description="Prolific behavioral rating pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the Flask survey server.")
    p_serve.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000).",
    )
    p_serve.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode (auto-reload, verbose errors).",
    )

    # status
    p_status = sub.add_parser("status", help="Print session/rating/quality summary.")

    # collect
    p_collect = sub.add_parser(
        "collect",
        help="Compute gold accuracy, flag low-quality participants, export CSV.",
    )

    # tiebreak
    p_tiebreak = sub.add_parser(
        "tiebreak",
        help="Find disagreement items and optionally mark them for a 3rd rater.",
    )
    p_tiebreak.add_argument(
        "--dry-run",
        action="store_true",
        help="Print disagreement item count without writing to DB.",
    )

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(port=args.port, debug=args.debug)

    elif args.command == "status":
        _cmd_status()

    elif args.command == "collect":
        _cmd_collect()

    elif args.command == "tiebreak":
        _cmd_tiebreak(dry_run=args.dry_run)


# ── Subcommand implementations ────────────────────────────────────────────────

def _cmd_serve(port: int, debug: bool) -> None:
    from pipeline.prolific.app import create_app

    app = create_app()
    print(f"Starting Prolific survey server on http://0.0.0.0:{port}/survey", file=sys.stderr)
    if debug:
        print("Debug mode: ON", file=sys.stderr)
    app.run(host="0.0.0.0", port=port, debug=debug)


def _cmd_status() -> None:
    from pipeline.prolific.collect import print_status

    print_status()


def _cmd_collect() -> None:
    from pipeline.prolific.collect import collect_results

    summary = collect_results()
    print(f"Participants:    {summary['n_participants']}")
    print(f"Total ratings:   {summary['n_ratings']}")
    print(f"Flagged:         {summary['n_flagged']}")
    if summary["mean_gold_accuracy"] is not None:
        print(f"Mean gold acc:   {summary['mean_gold_accuracy']:.2%}")
    else:
        print("Mean gold acc:   N/A (no gold ratings found)")


def _cmd_tiebreak(dry_run: bool) -> None:
    from pipeline.prolific.tiebreak import get_tiebreak_items, mark_tiebreak_items

    item_ids = get_tiebreak_items()
    print(f"Found {len(item_ids)} item(s) with rater disagreement.")

    if not item_ids:
        return

    if dry_run:
        print("--dry-run: not writing to DB.")
        for rid in item_ids:
            print(f"  behavioral_response_id={rid}")
    else:
        n = mark_tiebreak_items(item_ids)
        print(f"Marked {n} item(s) as needing tiebreak in prolific_tiebreak table.")
