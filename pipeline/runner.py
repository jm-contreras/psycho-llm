"""
Entry point for data collection.

Usage:
  python -m pipeline.runner [options]

Options:
  --models "Model Name 1" "Model Name 2"   Filter by model_name (substring match, case-insensitive)
  --n-items N                              Number of items to run (first N by item_id sort; default: all)
  --n-runs N                               Number of runs per item (default: 2)
  --providers bedrock openai ...           Filter by api_provider (default: bedrock)
  --parallel                               Run API calls concurrently (async); respects RPM/TPM/TPD limits
  --dry-run                                Print plan without making API calls
  --debug                                  Verbose output: print prompt + full response for every call;
                                           also enables litellm HTTP-level logging

Examples:
  # Smoke test: 10 items × 2 runs, Bedrock only (11 models)
  python -m pipeline.runner --n-items 10 --n-runs 2 --providers bedrock

  # Test OpenAI models
  python -m pipeline.runner --n-items 1 --n-runs 1 --providers openai --debug

  # Parallel collection for OpenAI
  python -m pipeline.runner --n-runs 30 --providers openai --parallel

  # Test a specific Azure model (fill azure_api_base in registry first)
  python -m pipeline.runner --models "DeepSeek R1" --n-items 1 --n-runs 1 --debug

  # Test all Azure models
  python -m pipeline.runner --n-items 1 --n-runs 1 --providers azure --debug

  # Full collection
  python -m pipeline.runner --n-runs 30 --providers bedrock openai google xai azure alibaba xiaomi
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import time

import litellm

from pipeline import api_client, storage
from pipeline.config import (
    build_rate_limiters,
    build_tpm_budgets,
    filter_by_names,
    load_model_registry,
)
from pipeline.item_loader import load_items
from pipeline.rate_limiter import DailyLimitExhausted


def run(
    model_names: list[str] | None = None,
    n_items: int | None = None,
    n_runs: int = 2,
    providers: list[str] | None = None,
    item_types: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    if providers is None:
        providers = ["bedrock"]

    if debug:
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"

    # ── Load models and items ─────────────────────────────────────────────────
    models = load_model_registry(providers=providers)
    models = filter_by_names(models, model_names)
    budgets = build_tpm_budgets(models)

    if not models:
        print(
            f"[runner] No models found matching providers={providers}, names={model_names}."
        )
        sys.exit(1)

    all_items = load_items()
    if item_types is not None:
        all_items = [it for it in all_items if it["item_type"] in item_types]
    items = all_items[:n_items] if n_items is not None else all_items

    total_calls = len(models) * len(items) * n_runs
    print(
        f"[runner] {len(models)} model(s) × {len(items)} item(s) × {n_runs} run(s)"
        f" = {total_calls} total calls"
    )
    if dry_run:
        print("[runner] --dry-run: no API calls will be made.")
        _print_plan(models, items, n_runs)
        return

    # ── Main collection loop ──────────────────────────────────────────────────
    item_ids = [it["item_id"] for it in items]
    completed = 0
    skipped = 0

    for model in models:
        mid = model["litellm_model_id"]
        model_label = model["model_name"]
        print(f"\n[{model_label}]")

        for item in items:
            for run_num in range(1, n_runs + 1):
                if storage.already_completed(mid, item["item_id"], run_num):
                    skipped += 1
                    continue

                budget = budgets.get(model.get("resource_group"))
                result = api_client.call_model(model, item, run_num, debug=debug, budget=budget)
                storage.store(result)

                status = result["status"]
                score = result.get("parsed_score")
                lp = result.get("logprob_score")

                # Compact per-call log
                lp_str = f" lp={lp:.3f}" if lp is not None else ""
                print(
                    f"  {item['item_id']} run={run_num} "
                    f"→ {status} score={score}{lp_str}"
                )
                if debug and status != "success":
                    print(f"    error_message : {result.get('error_message')}")
                    print(f"    raw_response  : {result.get('raw_response')!r}")
                completed += 1

    print(f"\n[runner] Done. {completed} new calls, {skipped} skipped (already complete).")

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary(models, items, n_runs)


# ── Async parallel runner ────────────────────────────────────────────────────

async def async_run(
    model_names: list[str] | None = None,
    n_items: int | None = None,
    n_runs: int = 2,
    providers: list[str] | None = None,
    item_types: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    """
    Async/parallel version of run().  All models run concurrently; within each
    model, requests are paced by the AsyncRateLimiter (RPM/TPM/TPD).
    """
    if providers is None:
        providers = ["bedrock"]

    # ── Load models and items ────────────────────────────────────────────────
    models = load_model_registry(providers=providers)
    models = filter_by_names(models, model_names)

    if not models:
        print(f"[runner] No models found matching providers={providers}, names={model_names}.")
        sys.exit(1)

    all_items = load_items()
    if item_types is not None:
        all_items = [it for it in all_items if it["item_type"] in item_types]
    items = all_items[:n_items] if n_items is not None else all_items

    # ── Build rate limiters ──────────────────────────────────────────────────
    limiters = build_rate_limiters(models)

    # ── Pre-load completed set for fast skip checks ──────────────────────────
    # Build group_map: litellm_model_id -> canonical_model_id.
    # Models with group_as set share a completion pool — a run done by any
    # model in the group counts as done for all others (avoids duplicate work).
    all_registry = load_model_registry()
    group_map = {
        m["litellm_model_id"]: m.get("group_as") or m["litellm_model_id"]
        for m in all_registry if m.get("litellm_model_id")
    }
    group_completed = storage.load_group_completed_set(group_map)

    # ── Build per-model work queues ──────────────────────────────────────────
    work: dict[str, list[tuple[dict, dict, int]]] = {}  # mid -> [(model, item, run)]
    skipped = 0
    for model in models:
        mid = model["litellm_model_id"]
        canonical = group_map.get(mid, mid)
        done_for_group = group_completed.get(canonical, set())
        work[mid] = []
        for item in items:
            for run_num in range(1, n_runs + 1):
                if (item["item_id"], run_num) in done_for_group:
                    skipped += 1
                else:
                    work[mid].append((model, item, run_num))

    total_pending = sum(len(q) for q in work.values())
    total_calls = len(models) * len(items) * n_runs
    print(
        f"[runner] {len(models)} model(s) × {len(items)} item(s) × {n_runs} run(s)"
        f" = {total_calls} total ({total_pending} pending, {skipped} already done)"
    )
    if dry_run:
        print("[runner] --dry-run: no API calls will be made.")
        _print_plan(models, items, n_runs)
        return

    # ── Shared progress counters ─────────────────────────────────────────────
    counters: dict[str, int] = {"completed": 0, "errors": 0, "tpd_stopped": 0}
    t0 = time.monotonic()

    async def _process_model(mid: str, queue: list) -> None:
        """Process all pending calls for one model, respecting rate limits."""
        limiter = limiters[mid]
        model_label = queue[0][0]["model_name"] if queue else mid
        rpm = queue[0][0].get("requests_per_minute", 60) if queue else 60

        # Semaphore caps in-flight requests to avoid overwhelming connections.
        # Target: enough concurrency to saturate RPM given typical latency (~30s
        # for reasoning models), without startup bursting. rpm*30//60 = rpm/2.
        max_concurrent = min(max(rpm * 30 // 60, 2), 200)
        sem = asyncio.Semaphore(max_concurrent)

        tpd_exhausted = False

        async def _do_call(model: dict, item: dict, run_num: int) -> None:
            nonlocal tpd_exhausted
            if tpd_exhausted:
                return
            async with sem:
                if tpd_exhausted:
                    return
                try:
                    result = await api_client.async_call_model(
                        model, item, run_num, limiter=limiter, debug=debug,
                    )
                except DailyLimitExhausted as exc:
                    tpd_exhausted = True
                    counters["tpd_stopped"] += 1
                    print(
                        f"\n[{model_label}] Daily token limit reached "
                        f"({exc.used:,}/{exc.limit:,}). Stopping this model."
                    )
                    return

                storage.store(result)

                status = result["status"]
                if status == "success":
                    counters["completed"] += 1
                else:
                    counters["errors"] += 1

                # Periodic progress (every 50 successful calls)
                done = counters["completed"]
                if (done > 0 and done % 50 == 0) or done + counters["errors"] == total_pending:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    tpd_info = ""
                    remaining = limiter.tpd_remaining
                    if remaining is not None:
                        tpd_info = f"  tpd_remaining={remaining:,}"
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"  {ts} [{model_label}] {done}/{total_pending} "
                        f"({rate:.1f} calls/s){tpd_info}"
                    )

                if debug and status != "success":
                    print(
                        f"  [{model_label}] {item['item_id']} run={run_num} "
                        f"→ {status}: {result.get('error_message')}"
                    )

        tasks = [_do_call(m, it, rn) for m, it, rn in queue]
        await asyncio.gather(*tasks)

    # ── Launch all models concurrently ───────────────────────────────────────
    print(f"[runner] Starting parallel collection...")
    model_tasks = [
        _process_model(mid, queue)
        for mid, queue in work.items()
        if queue  # skip models with no pending work
    ]
    await asyncio.gather(*model_tasks)

    elapsed = time.monotonic() - t0
    done = counters["completed"] + counters["errors"]
    print(
        f"\n[runner] Done. {counters['completed']} success, {counters['errors']} errors, "
        f"{skipped} skipped. {elapsed:.1f}s elapsed ({done / elapsed:.1f} calls/s)."
    )
    if counters["tpd_stopped"]:
        print(
            f"[runner] {counters['tpd_stopped']} model(s) stopped early due to daily token limits. "
            f"Re-run to continue."
        )

    _print_summary(models, items, n_runs)

    # Close litellm's async HTTP client pool so SSL teardown completes before
    # the event loop exits (prevents "RuntimeError: Event loop is closed" noise).
    await litellm.close_litellm_async_clients()


def _print_plan(models: list[dict], items: list[dict], n_runs: int) -> None:
    print(f"\nModels ({len(models)}):")
    for m in models:
        print(f"  {m['model_name']}  [{m['api_provider']}]  {m['litellm_model_id']}")
    print(f"\nFirst 5 items (of {len(items)}):")
    for it in items[:5]:
        print(f"  {it['item_id']}  {it['item_type']}  {it['text'][:70]!r}")


def _print_summary(models: list[dict], items: list[dict], n_runs: int) -> None:
    direct_ids = [it["item_id"] for it in items if it["item_type"] == "direct"]
    scenario_ids = [it["item_id"] for it in items if it["item_type"] == "scenario"]
    type_groups = [("direct", direct_ids), ("scenario", scenario_ids)]

    header = f"{'Model':<30} {'Type':<10} {'Collected':>12} {'Success':>8} {'Refusal':>8} {'ParseErr':>9} {'APIErr':>7} {'LogProbs':>9}"
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    for model in models:
        mid = model["litellm_model_id"]
        first = True
        for type_label, ids in type_groups:
            if not ids:
                continue
            counts = storage.count_by_status(mid, ids, n_runs)
            success = counts.get("success", 0)
            refusal = counts.get("refusal", 0)
            parse_err = counts.get("parse_error", 0)
            api_err = counts.get("api_error", 0)
            lp_count = counts.get("logprob_available", 0)
            total = success + refusal + parse_err + api_err
            expected = len(ids) * n_runs

            name_col = model["model_name"] if first else ""
            lp_str = "YES" if lp_count > 0 else "no"
            collected_str = f"{total}/{expected}"
            print(
                f"{name_col:<30} {type_label:<10} {collected_str:>12} "
                f"{success:>8} {refusal:>8} {parse_err:>9} {api_err:>7} {lp_str:>9}"
            )
            first = False

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM psychometric data collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Model name(s) to include (substring match). Default: all.",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        metavar="N",
        help="Number of items to run (first N sorted by item_id). Default: all.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=2,
        metavar="N",
        help="Number of runs per item. Default: 2.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["bedrock"],
        metavar="PROVIDER",
        help="API providers to include. Default: bedrock.",
    )
    parser.add_argument(
        "--item-types",
        nargs="+",
        choices=["direct", "scenario"],
        default=None,
        metavar="TYPE",
        help="Filter items by type: direct, scenario, or both. Default: all.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run API calls concurrently (async). Respects RPM/TPM/TPD limits from registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making any API calls.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompt + full response for every call; enable litellm HTTP logging.",
    )

    args = parser.parse_args()

    kwargs = dict(
        model_names=args.models,
        n_items=args.n_items,
        n_runs=args.n_runs,
        providers=args.providers,
        item_types=args.item_types,
        dry_run=args.dry_run,
        debug=args.debug,
    )

    if args.parallel:
        asyncio.run(async_run(**kwargs))
    else:
        run(**kwargs)


if __name__ == "__main__":
    main()
