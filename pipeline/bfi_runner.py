"""
Entry point for BFI-44 data collection.

Administers all 44 BFI items to each model. Uses the same infrastructure as
runner.py (rate limiting, retry, logprob extraction, storage) — only the item
source changes.

Usage:
  python -m pipeline.bfi_runner [options]

Options:
  --models "Model Name 1" "Model Name 2"   Filter by model_name (substring match, case-insensitive)
  --n-items N                              Number of items to run (first N by item_id sort; default: all 44)
  --n-runs N                               Number of runs per item (default: 2)
  --providers bedrock openai ...           Filter by api_provider (default: bedrock)
  --parallel                               Run API calls concurrently (async); respects RPM/TPM/TPD limits
  --dry-run                                Print plan without making API calls
  --debug                                  Verbose output: print prompt + full response for every call

Examples:
  # Smoke test: 5 items × 2 runs, Bedrock only
  python -m pipeline.bfi_runner --n-items 5 --n-runs 2 --providers bedrock --debug

  # Full collection (all providers except google and gpt-5.4-mini which use batch scripts)
  python -m pipeline.bfi_runner --n-runs 30 \\
    --providers bedrock openai xai azure alibaba xiaomi ai21 \\
    --parallel

  # Single model test
  python -m pipeline.bfi_runner --models "Claude Sonnet 4.6" --n-items 5 --n-runs 2 --debug

Note: Gemini Pro and GPT-5.4 Mini use dedicated batch scripts:
  python -m pipeline.bfi_batch_gemini
  python -m pipeline.bfi_batch_openai
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import litellm

from pipeline import api_client, storage
from pipeline.bfi_items import load_bfi_items
from pipeline.config import (
    build_rate_limiters,
    build_tpm_budgets,
    filter_by_names,
    load_model_registry,
)
from pipeline.rate_limiter import DailyLimitExhausted


def run(
    model_names: list[str] | None = None,
    n_items: int | None = None,
    n_runs: int = 2,
    providers: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    if providers is None:
        providers = ["bedrock"]

    if debug:
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"

    models = load_model_registry(providers=providers)
    models = filter_by_names(models, model_names)
    budgets = build_tpm_budgets(models)

    if not models:
        print(
            f"[bfi_runner] No models found matching providers={providers}, names={model_names}."
        )
        sys.exit(1)

    all_items = load_bfi_items()
    items = all_items[:n_items] if n_items is not None else all_items

    total_calls = len(models) * len(items) * n_runs
    print(
        f"[bfi_runner] {len(models)} model(s) × {len(items)} item(s) × {n_runs} run(s)"
        f" = {total_calls} total calls"
    )
    if dry_run:
        print("[bfi_runner] --dry-run: no API calls will be made.")
        _print_plan(models, items, n_runs)
        return

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

                lp_str = f" lp={lp:.3f}" if lp is not None else ""
                print(
                    f"  {item['item_id']} run={run_num} "
                    f"→ {status} score={score}{lp_str}"
                )
                if debug and status != "success":
                    print(f"    error_message : {result.get('error_message')}")
                    print(f"    raw_response  : {result.get('raw_response')!r}")
                completed += 1

    print(f"\n[bfi_runner] Done. {completed} new calls, {skipped} skipped (already complete).")

    _print_summary(models, items, n_runs)


async def async_run(
    model_names: list[str] | None = None,
    n_items: int | None = None,
    n_runs: int = 2,
    providers: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    if providers is None:
        providers = ["bedrock"]

    models = load_model_registry(providers=providers)
    models = filter_by_names(models, model_names)

    if not models:
        print(f"[bfi_runner] No models found matching providers={providers}, names={model_names}.")
        sys.exit(1)

    all_items = load_bfi_items()
    items = all_items[:n_items] if n_items is not None else all_items

    limiters = build_rate_limiters(models)
    completed_set = storage.load_completed_set()

    work: dict[str, list[tuple[dict, dict, int]]] = {}
    skipped = 0
    for model in models:
        mid = model["litellm_model_id"]
        work[mid] = []
        for item in items:
            for run_num in range(1, n_runs + 1):
                if (mid, item["item_id"], run_num) in completed_set:
                    skipped += 1
                else:
                    work[mid].append((model, item, run_num))

    total_pending = sum(len(q) for q in work.values())
    total_calls = len(models) * len(items) * n_runs
    print(
        f"[bfi_runner] {len(models)} model(s) × {len(items)} item(s) × {n_runs} run(s)"
        f" = {total_calls} total ({total_pending} pending, {skipped} already done)"
    )
    if dry_run:
        print("[bfi_runner] --dry-run: no API calls will be made.")
        _print_plan(models, items, n_runs)
        return

    counters: dict[str, int] = {"completed": 0, "errors": 0, "tpd_stopped": 0}
    t0 = time.monotonic()

    async def _process_model(mid: str, queue: list) -> None:
        limiter = limiters[mid]
        model_label = queue[0][0]["model_name"] if queue else mid
        rpm = queue[0][0].get("requests_per_minute", 60) if queue else 60

        max_concurrent = min(max(rpm, 2), 200)
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

                done = counters["completed"]
                if done % 50 == 0 or done + counters["errors"] == total_pending:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    tpd_info = ""
                    remaining = limiter.tpd_remaining
                    if remaining is not None:
                        tpd_info = f"  tpd_remaining={remaining:,}"
                    print(
                        f"  [{model_label}] {done}/{total_pending} "
                        f"({rate:.1f} calls/s){tpd_info}"
                    )

                if debug and status != "success":
                    print(
                        f"  [{model_label}] {item['item_id']} run={run_num} "
                        f"→ {status}: {result.get('error_message')}"
                    )

        tasks = [_do_call(m, it, rn) for m, it, rn in queue]
        await asyncio.gather(*tasks)

    print(f"[bfi_runner] Starting parallel collection...")
    model_tasks = [
        _process_model(mid, queue)
        for mid, queue in work.items()
        if queue
    ]
    await asyncio.gather(*model_tasks)

    elapsed = time.monotonic() - t0
    done = counters["completed"] + counters["errors"]
    print(
        f"\n[bfi_runner] Done. {counters['completed']} success, {counters['errors']} errors, "
        f"{skipped} skipped. {elapsed:.1f}s elapsed ({done / elapsed:.1f} calls/s)."
    )
    if counters["tpd_stopped"]:
        print(
            f"[bfi_runner] {counters['tpd_stopped']} model(s) stopped early due to daily token limits. "
            f"Re-run to continue."
        )

    _print_summary(models, items, n_runs)

    await litellm.close_litellm_async_clients()


def _print_plan(models: list[dict], items: list[dict], n_runs: int) -> None:
    print(f"\nModels ({len(models)}):")
    for m in models:
        print(f"  {m['model_name']}  [{m['api_provider']}]  {m['litellm_model_id']}")
    print(f"\nFirst 5 items (of {len(items)}):")
    for it in items[:5]:
        print(f"  {it['item_id']}  {it['dimension_code']}  {it['text'][:70]!r}")


def _print_summary(models: list[dict], items: list[dict], n_runs: int) -> None:
    item_ids = [it["item_id"] for it in items]

    header = f"{'Model':<30} {'Collected':>12} {'Success':>8} {'Refusal':>8} {'ParseErr':>9} {'APIErr':>7} {'LogProbs':>9}"
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    for model in models:
        mid = model["litellm_model_id"]
        counts = storage.count_by_status(mid, item_ids, n_runs)
        success = counts.get("success", 0)
        refusal = counts.get("refusal", 0)
        parse_err = counts.get("parse_error", 0)
        api_err = counts.get("api_error", 0)
        lp_count = counts.get("logprob_available", 0)
        total = success + refusal + parse_err + api_err
        expected = len(item_ids) * n_runs

        lp_str = "YES" if lp_count > 0 else "no"
        collected_str = f"{total}/{expected}"
        print(
            f"{model['model_name']:<30} {collected_str:>12} "
            f"{success:>8} {refusal:>8} {parse_err:>9} {api_err:>7} {lp_str:>9}"
        )

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BFI-44 data collection across LLM models.",
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
        help="Number of items to run (first N sorted by item_id). Default: all 44.",
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
        dry_run=args.dry_run,
        debug=args.debug,
    )

    if args.parallel:
        asyncio.run(async_run(**kwargs))
    else:
        run(**kwargs)


if __name__ == "__main__":
    main()
