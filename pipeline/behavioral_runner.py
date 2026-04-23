"""
Behavioral data collection for Phase 3 predictive validity.

Collects free-text responses to 48 open-ended prompts from all 25 models.
No scoring at collection time — responses are rated later by LLM-as-judge
ensemble and human raters (A2I).

Usage:
  python -m pipeline.behavioral_runner [options]

Options:
  --models "Model Name 1" "Model Name 2"  Filter by model_name (substring match)
  --n-prompts N                           First N prompts sorted by prompt_id (default: all)
  --n-runs N                              Runs per prompt (default: 5)
  --providers bedrock openai ...          Filter by api_provider (default: bedrock)
  --prompt-types single two-turn          Filter by prompt type (default: all)
  --parallel                              Run concurrently (async); respects RPM/TPM/TPD limits
  --dry-run                               Print plan without making API calls
  --debug                                 Print prompt + full response for every call

Examples:
  # Smoke test: 1 prompt × 1 run, single model
  python -m pipeline.behavioral_runner --models "Claude Sonnet 4.6" --n-prompts 1 --n-runs 1

  # Test only two-turn prompts
  python -m pipeline.behavioral_runner --models "Claude Sonnet 4.6" --prompt-types two-turn --n-runs 1

  # Full collection, all providers
  python -m pipeline.behavioral_runner --n-runs 5 --providers bedrock openai google xai azure alibaba xiaomi ai21 deepseek --parallel

  # Gemini models only (use batch_gemini_behavioral.py instead for cheaper rates)
  python -m pipeline.behavioral_runner --n-runs 5 --providers google --parallel
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import time
from datetime import timezone

import litellm

from pipeline import storage
from pipeline.behavioral_loader import build_messages, load_behavioral_prompts
from pipeline.config import (
    build_rate_limiters,
    filter_by_names,
    load_model_registry,
)
from pipeline.rate_limiter import AsyncRateLimiter, DailyLimitExhausted
from pipeline.reasoning_params import get_provider_kwargs

litellm.suppress_debug_info = True
litellm.drop_params = True

_MAX_TOKENS = 2048
_TEMPERATURE = 1.0
_TIMEOUT = 180  # seconds; behavioral responses can be longer than item responses


def _filter_prompts(
    n_prompts: int | None = None,
    prompt_types: list[str] | None = None,
) -> list[dict]:
    """Load prompts, apply type filter, then n_prompts slice (in that order)."""
    prompts = load_behavioral_prompts()  # already sorted by prompt_id
    if prompt_types is not None:
        want_two_turn = "two-turn" in prompt_types
        want_single = "single" in prompt_types
        prompts = [
            p for p in prompts
            if (p["is_two_turn"] and want_two_turn) or (not p["is_two_turn"] and want_single)
        ]
    if n_prompts is not None:
        prompts = prompts[:n_prompts]
    return prompts


# ── Sync runner ──────────────────────────────────────────────────────────────

def run(
    model_names: list[str] | None = None,
    n_prompts: int | None = None,
    n_runs: int = 5,
    providers: list[str] | None = None,
    prompt_types: list[str] | None = None,
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

    if not models:
        print(f"[behavioral_runner] No models found (providers={providers}, names={model_names}).")
        sys.exit(1)

    prompts = _filter_prompts(n_prompts=n_prompts, prompt_types=prompt_types)
    total_calls = len(models) * len(prompts) * n_runs
    print(
        f"[behavioral_runner] {len(models)} model(s) × {len(prompts)} prompt(s) × {n_runs} run(s)"
        f" = {total_calls} total calls"
    )

    if dry_run:
        print("[behavioral_runner] --dry-run: no API calls will be made.")
        _print_plan(models, prompts, n_runs)
        return

    completed = skipped = 0

    for model in models:
        mid = model["litellm_model_id"]
        print(f"\n[{model['model_name']}]")

        for prompt in prompts:
            for run_num in range(1, n_runs + 1):
                if storage.already_completed_behavioral(mid, prompt["prompt_id"], run_num):
                    skipped += 1
                    continue

                result = _call_model_sync(model, prompt, run_num, debug=debug)
                storage.store_behavioral(result)

                status = result["status"]
                print(f"  {prompt['prompt_id']} run={run_num} → {status}")
                if debug:
                    if status != "success":
                        print(f"    error: {result.get('error_message')}")
                    print(f"    response: {result.get('raw_response')!r}")
                completed += 1

    print(f"\n[behavioral_runner] Done. {completed} new calls, {skipped} skipped.")
    _print_summary(models, prompts, n_runs)


# ── Async parallel runner ────────────────────────────────────────────────────

async def async_run(
    model_names: list[str] | None = None,
    n_prompts: int | None = None,
    n_runs: int = 5,
    providers: list[str] | None = None,
    prompt_types: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    if providers is None:
        providers = ["bedrock"]

    models = load_model_registry(providers=providers)
    models = filter_by_names(models, model_names)

    if not models:
        print(f"[behavioral_runner] No models found (providers={providers}, names={model_names}).")
        sys.exit(1)

    prompts = _filter_prompts(n_prompts=n_prompts, prompt_types=prompt_types)
    limiters = build_rate_limiters(models)

    # Pre-load completed set for fast skip checks.
    completed_set = storage.load_completed_behavioral_set()

    work: dict[str, list[tuple[dict, dict, int]]] = {}
    skipped = 0
    for model in models:
        mid = model["litellm_model_id"]
        work[mid] = []
        for prompt in prompts:
            for run_num in range(1, n_runs + 1):
                if (mid, prompt["prompt_id"], run_num) in completed_set:
                    skipped += 1
                else:
                    work[mid].append((model, prompt, run_num))

    total_pending = sum(len(q) for q in work.values())
    total_calls = len(models) * len(prompts) * n_runs
    print(
        f"[behavioral_runner] {len(models)} model(s) × {len(prompts)} prompt(s) × {n_runs} run(s)"
        f" = {total_calls} total ({total_pending} pending, {skipped} already done)"
    )

    if dry_run:
        print("[behavioral_runner] --dry-run: no API calls will be made.")
        _print_plan(models, prompts, n_runs)
        return

    counters: dict[str, int] = {"completed": 0, "errors": 0, "tpd_stopped": 0}
    t0 = time.monotonic()

    async def _process_model(mid: str, queue: list) -> None:
        limiter = limiters[mid]
        model_label = queue[0][0]["model_name"] if queue else mid
        rpm = queue[0][0].get("requests_per_minute", 60) if queue else 60
        max_concurrent = min(max(rpm * 30 // 60, 2), 200)
        sem = asyncio.Semaphore(max_concurrent)
        tpd_exhausted = False
        model_pending = len(queue)
        model_done = 0

        async def _do_call(model: dict, prompt: dict, run_num: int) -> None:
            nonlocal tpd_exhausted, model_done
            if tpd_exhausted:
                return
            async with sem:
                if tpd_exhausted:
                    return
                try:
                    result = await _call_model_async(
                        model, prompt, run_num, limiter=limiter, debug=debug
                    )
                except DailyLimitExhausted as exc:
                    tpd_exhausted = True
                    counters["tpd_stopped"] += 1
                    print(
                        f"\n[{model_label}] Daily token limit reached "
                        f"({exc.used:,}/{exc.limit:,}). Stopping this model."
                    )
                    return

                storage.store_behavioral(result)

                if result["status"] == "success":
                    counters["completed"] += 1
                    model_done += 1
                else:
                    counters["errors"] += 1

                if (model_done > 0 and model_done % 50 == 0) or model_done + counters["errors"] == model_pending:
                    elapsed = time.monotonic() - t0
                    rate = model_done / elapsed if elapsed > 0 else 0
                    tpd_info = ""
                    remaining = limiter.tpd_remaining
                    if remaining is not None:
                        tpd_info = f"  tpd_remaining={remaining:,}"
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"  {ts} [{model_label}] {model_done}/{model_pending} "
                        f"({rate:.1f} calls/s){tpd_info}"
                    )

                if debug:
                    if result["status"] != "success":
                        print(
                            f"  [{model_label}] {prompt['prompt_id']} run={run_num} "
                            f"→ {result['status']}: {result.get('error_message')}"
                        )
                    print(
                        f"  [{model_label}] {prompt['prompt_id']} run={run_num} "
                        f"response: {result.get('raw_response')!r}"
                    )

        tasks = [_do_call(m, p, rn) for m, p, rn in queue]
        await asyncio.gather(*tasks)

    print("[behavioral_runner] Starting parallel collection...")
    model_tasks = [
        _process_model(mid, queue)
        for mid, queue in work.items()
        if queue
    ]
    await asyncio.gather(*model_tasks)

    elapsed = time.monotonic() - t0
    done = counters["completed"] + counters["errors"]
    print(
        f"\n[behavioral_runner] Done. {counters['completed']} success, {counters['errors']} errors, "
        f"{skipped} skipped. {elapsed:.1f}s elapsed ({done / elapsed:.1f} calls/s)."
    )
    if counters["tpd_stopped"]:
        print(
            f"[behavioral_runner] {counters['tpd_stopped']} model(s) stopped early due to daily "
            f"token limits. Re-run to continue."
        )

    _print_summary(models, prompts, n_runs)
    await litellm.close_litellm_async_clients()


# ── Core call functions ───────────────────────────────────────────────────────

def _call_model_sync(
    model: dict,
    prompt: dict,
    run_number: int,
    debug: bool = False,
) -> dict:
    """Make a single synchronous behavioral API call. Returns a storage-ready row."""
    litellm_id = model["litellm_model_id"]
    messages = build_messages(prompt)

    if debug:
        print(f"\n  [debug] model={litellm_id}  prompt={prompt['prompt_id']}  run={run_number}")
        for msg in messages:
            print(f"  [debug] {msg['role'].upper()}: {msg['content']}")

    kwargs: dict = dict(
        model=litellm_id,
        messages=messages,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    kwargs.update(get_provider_kwargs(model, behavioral=True))

    rpm = model.get("requests_per_minute")

    max_attempts = 3
    last_error: str | None = None
    raw_text: str | None = None
    reasoning: str = ""

    for attempt in range(max_attempts):
        try:
            if rpm:
                time.sleep(60.0 / rpm)
            response = litellm.completion(**kwargs)
            msg = response.choices[0].message
            raw_text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not raw_text:
                raw_text = reasoning
            last_error = None
            break
        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            time.sleep(2 ** (attempt + 2))
        except litellm.exceptions.APIError as exc:
            last_error = f"APIError: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)

    return _build_row(model, prompt, run_number, raw_text, reasoning, last_error)


async def _call_model_async(
    model: dict,
    prompt: dict,
    run_number: int,
    limiter: AsyncRateLimiter,
    debug: bool = False,
) -> dict:
    """Make a single async behavioral API call. Returns a storage-ready row."""
    litellm_id = model["litellm_model_id"]
    messages = build_messages(prompt)

    # Estimate input tokens for rate limiter pre-reservation.
    _input_est = sum(len(m["content"]) for m in messages) // 4
    estimated_tokens = _input_est + _MAX_TOKENS // 4  # rough output estimate

    kwargs: dict = dict(
        model=litellm_id,
        messages=messages,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    kwargs.update(get_provider_kwargs(model, behavioral=True))

    max_attempts = 3
    last_error: str | None = None
    raw_text: str | None = None
    reasoning: str = ""

    await limiter.acquire(estimated_tokens)

    for attempt in range(max_attempts):
        try:
            response = await litellm.acompletion(timeout=_TIMEOUT, **kwargs)
            actual_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
            if actual_tokens:
                await limiter.record(actual_tokens, estimated_tokens)
            msg = response.choices[0].message
            raw_text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not raw_text:
                raw_text = reasoning
            last_error = None
            break
        except DailyLimitExhausted:
            raise
        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            await asyncio.sleep(2 ** (attempt + 2))
        except litellm.exceptions.APIError as exc:
            last_error = f"APIError: {exc}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)

    return _build_row(model, prompt, run_number, raw_text, reasoning, last_error)


def _build_row(
    model: dict,
    prompt: dict,
    run_number: int,
    raw_text: str | None,
    reasoning: str,
    error: str | None,
) -> dict:
    row: dict = {
        "model_id": model["litellm_model_id"],
        "prompt_id": prompt["prompt_id"],
        "dimension": prompt["dimension"],
        "dimension_code": prompt["dimension_code"],
        "is_two_turn": 1 if prompt["is_two_turn"] else 0,
        "run_number": run_number,
        "raw_response": None,
        "reasoning_content": None,
        "status": "api_error",
        "error_message": error,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    }
    if error:
        return row
    if not raw_text:
        row["status"] = "api_error"
        row["error_message"] = "empty response: API returned success but no content"
        return row
    row["raw_response"] = raw_text
    row["reasoning_content"] = reasoning or None
    row["status"] = "success"
    row["error_message"] = None
    return row


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_plan(models: list[dict], prompts: list[dict], n_runs: int) -> None:
    print(f"\nModels ({len(models)}):")
    for m in models:
        print(f"  {m['model_name']}  [{m['api_provider']}]  {m['litellm_model_id']}")
    print(f"\nPrompts ({len(prompts)}):")
    for p in prompts:
        if p["is_two_turn"]:
            print(f"  {p['prompt_id']}  [two-turn]")
            print(f"    Turn 1 user:      {p['turn1_user']}")
            print(f"    Turn 1 assistant: {p['turn1_assistant']}")
            print(f"    Turn 2 user:      {p['turn2_user']}")
        else:
            print(f"  {p['prompt_id']}  {p['text']}")


def _print_summary(models: list[dict], prompts: list[dict], n_runs: int) -> None:
    conn_fn = storage._get_conn  # reuse internal connection helper for summary query
    prompt_ids = [p["prompt_id"] for p in prompts]
    if not prompt_ids:
        return

    header = (
        f"{'Model':<30} {'Collected':>12} {'Success':>8} {'APIErr':>7}"
    )
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    conn = conn_fn()
    try:
        for model in models:
            mid = model["litellm_model_id"]
            placeholders = ",".join("?" for _ in prompt_ids)
            rows = conn.execute(
                f"""
                SELECT status, COUNT(*) as cnt
                FROM behavioral_responses
                WHERE model_id=? AND prompt_id IN ({placeholders}) AND run_number <= ?
                GROUP BY status
                """,
                [mid, *prompt_ids, n_runs],
            ).fetchall()
            counts = {r["status"]: r["cnt"] for r in rows}
            success = counts.get("success", 0)
            api_err = counts.get("api_error", 0)
            total = success + api_err
            expected = len(prompt_ids) * n_runs
            print(
                f"{model['model_name']:<30} {total:>5}/{expected:<5} "
                f"{success:>8} {api_err:>7}"
            )
    finally:
        conn.close()
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral data collection for Phase 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Model name(s) to include (substring match). Default: all.")
    parser.add_argument("--n-prompts", type=int, default=None, metavar="N",
                        help="Number of prompts to run (first N sorted by prompt_id). Default: all.")
    parser.add_argument("--n-runs", type=int, default=5, metavar="N",
                        help="Number of runs per prompt. Default: 5.")
    parser.add_argument("--providers", nargs="+", default=["bedrock"], metavar="PROVIDER",
                        help="API providers to include. Default: bedrock.")
    parser.add_argument("--prompt-types", nargs="+", choices=["single", "two-turn"], default=None,
                        metavar="TYPE",
                        help="Filter by prompt type: single, two-turn, or both. Default: all.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run API calls concurrently (async).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making any API calls.")
    parser.add_argument("--debug", action="store_true",
                        help="Print prompt + full response for every call.")

    args = parser.parse_args()
    kwargs = dict(
        model_names=args.models,
        n_prompts=args.n_prompts,
        n_runs=args.n_runs,
        providers=args.providers,
        prompt_types=args.prompt_types,
        dry_run=args.dry_run,
        debug=args.debug,
    )

    if args.parallel:
        asyncio.run(async_run(**kwargs))
    else:
        run(**kwargs)


if __name__ == "__main__":
    main()
