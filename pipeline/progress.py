"""
Collection progress: how many (model, item, run) triples are complete.

Usage:
    python -m pipeline.progress                  # native item pool (default)
    python -m pipeline.progress --pool bfi       # BFI-44
    python -m pipeline.progress --n-runs 1       # for early testing
    python -m pipeline.progress --model "Claude"
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

from pipeline.config import load_model_registry
from pipeline.storage import DB_PATH

_TARGET_RUNS = 30
_TARGET_BEHAVIORAL_RUNS = 5


def _build_name_map() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Returns (name_map, group_map).

    name_map:  litellm_model_id -> (model_name, provider)
    group_map: litellm_model_id -> canonical_model_id (self if no group_as)
    """
    models = load_model_registry()
    name_map = {
        m["litellm_model_id"]: (m["model_name"], m.get("provider", ""))
        for m in models if m.get("litellm_model_id")
    }
    group_map = {
        m["litellm_model_id"]: m.get("group_as") or m["litellm_model_id"]
        for m in models if m.get("litellm_model_id")
    }
    return name_map, group_map


def show_progress(
    model_filter: str | None = None,
    n_runs: int | None = None,
    pool: str = "native",
) -> None:
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        return

    name_map, group_map = _build_name_map()

    if n_runs is None:
        n_runs = _TARGET_BEHAVIORAL_RUNS if pool == "behavioral" else _TARGET_RUNS

    if pool == "bfi":
        _show_bfi(name_map, group_map, model_filter, n_runs)
    elif pool == "behavioral":
        _show_behavioral(name_map, group_map, model_filter, n_runs)
    else:
        _show_native(name_map, group_map, model_filter, n_runs)


def _show_native(name_map: dict, group_map: dict, model_filter: str | None, n_runs: int) -> None:
    from pipeline.item_loader import load_items
    all_items = load_items()
    n_direct   = sum(1 for it in all_items if it["item_type"] == "direct")
    n_scenario = sum(1 for it in all_items if it["item_type"] == "scenario")
    exp_direct   = n_direct   * n_runs
    exp_scenario = n_scenario * n_runs
    exp_total    = exp_direct + exp_scenario

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT model_id, item_type, COUNT(*) AS n
        FROM responses
        WHERE status = 'success'
          AND item_id NOT LIKE 'BFI-%'
        GROUP BY model_id, item_type
    """).fetchall()
    conn.close()

    data: dict[str, dict[str, int]] = defaultdict(lambda: {"direct": 0, "scenario": 0})
    for model_id, item_type, n in rows:
        if item_type in ("direct", "scenario"):
            canonical = group_map.get(model_id, model_id)
            data[canonical][item_type] += n

    if model_filter:
        f = model_filter.lower()
        data = {
            k: v for k, v in data.items()
            if f in name_map.get(k, (k, ""))[0].lower()
            or f in name_map.get(k, ("", ""))[1].lower()
            or f in k.lower()
        }

    if not data:
        print("No successful responses found.")
        return

    name_w     = max((len(name_map.get(k, (k,))[0]) for k in data), default=10) + 2
    provider_w = max((len(name_map.get(k, ("", ""))[1]) for k in data), default=10) + 2
    col_w = max(len(f"{exp_direct:,}/{exp_direct:,}"), len("direct")) + 2

    header = (
        f"\n{'model':<{name_w}} {'provider':<{provider_w}}"
        f" {'direct':>{col_w}} {'scenario':>{col_w}} {'total':>{col_w}} {'done':>6}"
    )
    print(header)
    print("-" * len(header))

    grand_direct = grand_scenario = 0

    for model_id in sorted(data, key=lambda k: name_map.get(k, (k, ""))[0]):
        model_name, provider = name_map.get(model_id, (model_id, ""))
        d = data[model_id]["direct"]
        s = data[model_id]["scenario"]
        t = d + s
        grand_direct   += d
        grand_scenario += s
        pct = 100 * t / exp_total if exp_total else 0
        print(
            f"{model_name:<{name_w}} {provider:<{provider_w}}"
            f" {f'{d:,}/{exp_direct:,}':>{col_w}}"
            f" {f'{s:,}/{exp_scenario:,}':>{col_w}}"
            f" {f'{t:,}/{exp_total:,}':>{col_w}}"
            f" {f'{pct:.1f}%':>6}"
        )

    grand_total = grand_direct + grand_scenario
    n_models = len(data)
    exp_grand_direct   = exp_direct   * n_models
    exp_grand_scenario = exp_scenario * n_models
    exp_grand_total    = exp_total    * n_models
    pct = 100 * grand_total / exp_grand_total if exp_grand_total else 0
    sep = "-" * len(header)
    print(sep)
    print(
        f"{'TOTAL':<{name_w}} {'':<{provider_w}}"
        f" {f'{grand_direct:,}/{exp_grand_direct:,}':>{col_w}}"
        f" {f'{grand_scenario:,}/{exp_grand_scenario:,}':>{col_w}}"
        f" {f'{grand_total:,}/{exp_grand_total:,}':>{col_w}}"
        f" {f'{pct:.1f}%':>6}"
    )


def _show_bfi(name_map: dict, group_map: dict, model_filter: str | None, n_runs: int) -> None:
    from pipeline.bfi_items import load_bfi_items
    n_items = len(load_bfi_items())
    exp_total = n_items * n_runs

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT model_id, COUNT(*) AS n
        FROM responses
        WHERE status = 'success'
          AND item_id LIKE 'BFI-%'
        GROUP BY model_id
    """).fetchall()
    conn.close()

    data: dict[str, int] = defaultdict(int)
    for model_id, n in rows:
        canonical = group_map.get(model_id, model_id)
        data[canonical] += n

    if model_filter:
        f = model_filter.lower()
        data = {
            k: v for k, v in data.items()
            if f in name_map.get(k, (k, ""))[0].lower()
            or f in name_map.get(k, ("", ""))[1].lower()
            or f in k.lower()
        }

    if not data:
        print("No BFI responses found.")
        return

    name_w     = max((len(name_map.get(k, (k,))[0]) for k in data), default=10) + 2
    provider_w = max((len(name_map.get(k, ("", ""))[1]) for k in data), default=10) + 2
    col_w = max(len(f"{exp_total:,}/{exp_total:,}"), len("bfi-44")) + 2

    header = (
        f"\n{'model':<{name_w}} {'provider':<{provider_w}}"
        f" {'bfi-44':>{col_w}} {'done':>6}"
    )
    print(header)
    print("-" * len(header))

    grand = 0

    for model_id in sorted(data, key=lambda k: name_map.get(k, (k, ""))[0]):
        model_name, provider = name_map.get(model_id, (model_id, ""))
        n = data[model_id]
        grand += n
        pct = 100 * n / exp_total if exp_total else 0
        print(
            f"{model_name:<{name_w}} {provider:<{provider_w}}"
            f" {f'{n:,}/{exp_total:,}':>{col_w}}"
            f" {f'{pct:.1f}%':>6}"
        )

    n_models = len(data)
    exp_grand = exp_total * n_models
    pct = 100 * grand / exp_grand if exp_grand else 0
    sep = "-" * len(header)
    print(sep)
    print(
        f"{'TOTAL':<{name_w}} {'':<{provider_w}}"
        f" {f'{grand:,}/{exp_grand:,}':>{col_w}}"
        f" {f'{pct:.1f}%':>6}"
    )


def _show_behavioral(name_map: dict, group_map: dict, model_filter: str | None, n_runs: int) -> None:
    from pipeline.behavioral_loader import load_behavioral_prompts
    prompts = load_behavioral_prompts()
    n_single  = sum(1 for p in prompts if not p["is_two_turn"])
    n_two     = sum(1 for p in prompts if p["is_two_turn"])
    exp_single   = n_single * n_runs
    exp_two      = n_two    * n_runs
    exp_total    = exp_single + exp_two

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT model_id, is_two_turn, COUNT(*) AS n
        FROM behavioral_responses
        WHERE status = 'success'
        GROUP BY model_id, is_two_turn
    """).fetchall()
    conn.close()

    data: dict[str, dict[str, int]] = defaultdict(lambda: {"single": 0, "two_turn": 0})
    for model_id, is_two_turn, n in rows:
        canonical = group_map.get(model_id, model_id)
        key = "two_turn" if is_two_turn else "single"
        data[canonical][key] += n

    if model_filter:
        f = model_filter.lower()
        data = {
            k: v for k, v in data.items()
            if f in name_map.get(k, (k, ""))[0].lower()
            or f in name_map.get(k, ("", ""))[1].lower()
            or f in k.lower()
        }

    if not data:
        print("No behavioral responses found.")
        return

    name_w     = max((len(name_map.get(k, (k,))[0]) for k in data), default=10) + 2
    provider_w = max((len(name_map.get(k, ("", ""))[1]) for k in data), default=10) + 2
    col_w = max(len(f"{exp_total:,}/{exp_total:,}"), len("two-turn")) + 2

    header = (
        f"\n{'model':<{name_w}} {'provider':<{provider_w}}"
        f" {'single':>{col_w}} {'two-turn':>{col_w}} {'total':>{col_w}} {'done':>6}"
    )
    print(header)
    print("-" * len(header))

    grand_single = grand_two = 0

    for model_id in sorted(data, key=lambda k: name_map.get(k, (k, ""))[0]):
        model_name, provider = name_map.get(model_id, (model_id, ""))
        s = data[model_id]["single"]
        t = data[model_id]["two_turn"]
        total = s + t
        grand_single += s
        grand_two    += t
        pct = 100 * total / exp_total if exp_total else 0
        print(
            f"{model_name:<{name_w}} {provider:<{provider_w}}"
            f" {f'{s:,}/{exp_single:,}':>{col_w}}"
            f" {f'{t:,}/{exp_two:,}':>{col_w}}"
            f" {f'{total:,}/{exp_total:,}':>{col_w}}"
            f" {f'{pct:.1f}%':>6}"
        )

    grand_total = grand_single + grand_two
    n_models = len(data)
    exp_grand_single = exp_single * n_models
    exp_grand_two    = exp_two    * n_models
    exp_grand_total  = exp_total  * n_models
    pct = 100 * grand_total / exp_grand_total if exp_grand_total else 0
    sep = "-" * len(header)
    print(sep)
    print(
        f"{'TOTAL':<{name_w}} {'':<{provider_w}}"
        f" {f'{grand_single:,}/{exp_grand_single:,}':>{col_w}}"
        f" {f'{grand_two:,}/{exp_grand_two:,}':>{col_w}}"
        f" {f'{grand_total:,}/{exp_grand_total:,}':>{col_w}}"
        f" {f'{pct:.1f}%':>6}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show data collection progress.")
    parser.add_argument("--pool", choices=["native", "bfi", "behavioral"], default="native",
                        help="Item pool to report on: native (default), bfi, or behavioral.")
    parser.add_argument("--model", default=None, help="Substring filter on model name.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help=f"Expected runs per item (default: {_TARGET_RUNS} for native/bfi, {_TARGET_BEHAVIORAL_RUNS} for behavioral).")
    args = parser.parse_args()
    show_progress(model_filter=args.model, n_runs=args.n_runs, pool=args.pool)


if __name__ == "__main__":
    main()
