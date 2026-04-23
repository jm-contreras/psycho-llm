"""
Error status check: surfaces models with parse errors, refusals, or API errors.

By default shows only models/types with at least one error. Use --all to see
everything. Use --by-item to drill down to individual failing items.

Usage:
    python -m pipeline.status
    python -m pipeline.status --all
    python -m pipeline.status --model "GLM"
    python -m pipeline.status --by-item
"""

from __future__ import annotations

import argparse
import sqlite3

from pipeline.config import load_model_registry
from pipeline.storage import DB_PATH


def _build_name_map() -> dict[str, tuple[str, str]]:
    models = load_model_registry()
    return {
        m["litellm_model_id"]: (m["model_name"], m.get("provider", ""))
        for m in models if m.get("litellm_model_id")
    }


def check_status(model_filter: str | None = None, show_all: bool = False, by_item: bool = False) -> None:
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        return

    name_map = _build_name_map()
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT
            model_id,
            item_type,
            status,
            COUNT(*) AS n
        FROM responses
        GROUP BY model_id, item_type, status
        ORDER BY model_id, item_type, status
    """).fetchall()

    # Pivot into {model_id: {item_type: {status: n}}}
    from collections import defaultdict
    data: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for model_id, item_type, status, n in rows:
        data[model_id][item_type or "unknown"][status] = n

    if model_filter:
        f = model_filter.lower()
        data = {
            k: v for k, v in data.items()
            if f in name_map.get(k, (k, ""))[0].lower()
            or f in name_map.get(k, (k, ""))[1].lower()
            or f in k.lower()
        }

    if not data:
        print("No rows found.")
        conn.close()
        return

    # Filter to only rows with errors unless --all
    def _has_errors(type_data: dict) -> bool:
        return any(type_data.get(s, 0) > 0 for s in ("parse_error", "refusal", "api_error"))

    name_w     = max((len(name_map.get(k, (k,))[0]) for k in data), default=10) + 2
    provider_w = max((len(name_map.get(k, ("", ""))[1]) for k in data), default=10) + 2
    type_w = 10

    print(f"\n{'model':<{name_w}} {'provider':<{provider_w}} {'type':<{type_w}} {'parse_err':>10} {'refusal':>10} {'api_err':>10} {'total':>8}")
    print("-" * (name_w + provider_w + type_w + 42))

    any_printed = False
    for model_id in sorted(data, key=lambda k: name_map.get(k, (k, ""))[0]):
        model_name, provider = name_map.get(model_id, (model_id, ""))
        first = True
        for item_type in ("direct", "scenario"):
            s = data[model_id].get(item_type)
            if s is None:
                continue
            if not show_all and not _has_errors(s):
                continue
            parse_err = s.get("parse_error", 0)
            refusal   = s.get("refusal",     0)
            api_err   = s.get("api_error",   0)
            total     = s.get("success", 0) + parse_err + refusal + api_err
            name_col     = model_name if first else ""
            provider_col = provider   if first else ""
            print(f"{name_col:<{name_w}} {provider_col:<{provider_w}} {item_type:<{type_w}} {parse_err:>10} {refusal:>10} {api_err:>10} {total:>8}")
            first = False
            any_printed = True

    if not any_printed:
        print("  No errors found.")

    # Per-item drill-down for failing items only
    if by_item:
        item_rows = conn.execute("""
            SELECT model_id, item_id, item_type, status, COUNT(*) AS n
            FROM responses
            WHERE status != 'success'
            GROUP BY model_id, item_id, item_type, status
            ORDER BY model_id, item_type, item_id, status
        """).fetchall()

        if model_filter:
            f = model_filter.lower()
            item_rows = [
                r for r in item_rows
                if f in name_map.get(r[0], (r[0], ""))[0].lower()
                or f in name_map.get(r[0], (r[0], ""))[1].lower()
                or f in r[0].lower()
            ]

        if item_rows:
            print(f"\n{'model':<{name_w}} {'provider':<{provider_w}} {'item':<10} {'type':<10} {'status':<14} {'n':>4}")
            print("-" * (name_w + provider_w + 42))
            for model_id, item_id, item_type, status, n in item_rows:
                model_name, provider = name_map.get(model_id, (model_id, ""))
                print(f"{model_name:<{name_w}} {provider:<{provider_w}} {item_id:<10} {item_type:<10} {status:<14} {n:>4}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Surface pipeline errors by model and item type.")
    parser.add_argument("--model", default=None, help="Substring filter on model name")
    parser.add_argument("--all", action="store_true", help="Show all models, not just those with errors")
    parser.add_argument("--by-item", action="store_true", help="Drill down to individual failing items")
    args = parser.parse_args()
    check_status(model_filter=args.model, show_all=args.all, by_item=args.by_item)


if __name__ == "__main__":
    main()
