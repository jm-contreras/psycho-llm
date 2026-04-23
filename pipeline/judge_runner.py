"""
LLM-as-judge rating pipeline for Phase 3 behavioral samples.

Loads behavioral_responses (v2 prompt IDs), assigns judge models with cross-model
exclusion based on the subject model's provider, and collects structured 5-factor
ratings from the judge ensemble using randomized F/R statement keying.

Gemini 3.1 Pro is NOT run here — use pipeline/batch_gemini_judge.py instead
(required due to Gemini's RPD=250 real-time limit).

Usage:
  python -m pipeline.judge_runner [options]

Options:
  --judges "Claude Opus 4.6" "GPT-5.4"
                         Judge models by name. Default: Claude Opus 4.6 + GPT-5.4.
  --models "Name"        Filter subject models (substring match). Default: all.
  --n-samples N          First N behavioral samples (ordered by model_id, prompt_id,
                         run_number). Default: all.
  --providers p1 p2 ...  Filter subject models by api_provider. Default: all.
  --parallel             Run concurrently (async). Recommended for full runs.
  --dry-run              Print plan without making API calls.
  --debug                Print prompt + full response for every call.

Examples:
  # Smoke test: 1 sample, one judge
  python -m pipeline.judge_runner --judges "Claude Opus 4.6" --n-samples 1

  # Pilot: 50 samples, both judges, parallel
  python -m pipeline.judge_runner --n-samples 50 --parallel

  # Full run
  python -m pipeline.judge_runner --parallel
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import time
from datetime import timezone
from pathlib import Path

import litellm
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

from pipeline import storage
from pipeline.behavioral_loader import load_behavioral_prompts
from pipeline.config import build_rate_limiters, filter_by_names, load_model_registry
from pipeline.judge_prompt import (
    FACTOR_ORDER,
    build_judge_messages,
    parse_judge_response,
    sample_keying,
)
from pipeline.rate_limiter import AsyncRateLimiter, DailyLimitExhausted
from pipeline.reasoning_params import get_provider_kwargs

litellm.suppress_debug_info = True
litellm.drop_params = True

# ── Constants ─────────────────────────────────────────────────────────────────

# Real-time judge models (Gemini handled separately via batch_gemini_judge.py)
DEFAULT_JUDGE_NAMES = ["Claude Opus 4.6", "GPT-5.4"]

_MAX_TOKENS = 512   # Judge responses are short JSON blobs; extra headroom for Gemini preamble
_TEMPERATURE = 0.0  # Deterministic ratings
_TIMEOUT = 60       # seconds

# Cross-model exclusion: judge model_name -> provider of subject models to skip
_JUDGE_EXCLUSIONS: dict[str, str] = {
    "Claude Opus 4.6": "Anthropic",
    "GPT-5.4":         "OpenAI",
    "Gemini 3.1 Pro":  "Google",
}


# ── Exclusion helpers ─────────────────────────────────────────────────────────

def _build_provider_map(registry: list[dict]) -> dict[str, str]:
    """Return litellm_model_id -> provider string (e.g. 'Anthropic', 'Google')."""
    return {m["litellm_model_id"]: m.get("provider", "") for m in registry}


def _should_exclude(subject_provider: str, judge_model_name: str) -> bool:
    """True if this judge should not rate samples from this subject model provider."""
    return _JUDGE_EXCLUSIONS.get(judge_model_name, "") == subject_provider


# ── Work queue construction ───────────────────────────────────────────────────

def _build_work_queues(
    samples: list[dict],
    judges: list[dict],
    completed_set: set[tuple[int, str]],
) -> dict[str, list[tuple[dict, dict]]]:
    """Build per-judge work queues with cross-model exclusion and skip-completed applied.

    Returns: {judge_litellm_id: [(sample_dict, judge_dict), ...]}
    """
    work: dict[str, list[tuple[dict, dict]]] = {j["litellm_model_id"]: [] for j in judges}

    for sample in samples:
        subject_provider = sample.get("subject_provider", "")
        for judge in judges:
            if _should_exclude(subject_provider, judge["model_name"]):
                continue
            key = (sample["id"], judge["litellm_model_id"])
            if key in completed_set:
                continue
            work[judge["litellm_model_id"]].append((sample, judge))

    return work


# ── Sync runner ───────────────────────────────────────────────────────────────

def run(
    judge_names: list[str] | None = None,
    model_names: list[str] | None = None,
    n_samples: int | None = None,
    providers: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    judge_names = judge_names or DEFAULT_JUDGE_NAMES
    full_registry = load_model_registry(providers=None)
    provider_map = _build_provider_map(full_registry)

    all_judges = load_model_registry(providers=["bedrock", "openai"])
    judges = filter_by_names(all_judges, judge_names)
    if not judges:
        print(f"[judge_runner] No judge models found matching {judge_names}.")
        sys.exit(1)

    subject_model_ids = _get_subject_model_ids(full_registry, model_names, providers)
    samples = storage.load_behavioral_samples_for_judging(
        model_ids=subject_model_ids,
        n_samples=n_samples,
    )
    for s in samples:
        s["subject_provider"] = provider_map.get(s["model_id"], "")

    total = len(samples) * len(judges)
    print(
        f"[judge_runner] {len(samples)} sample(s) × {len(judges)} judge(s)"
        f" = up to {total} calls (cross-model exclusion reduces this)"
    )

    if dry_run:
        _print_plan(judges, samples)
        return

    prompt_lookup = {p["prompt_id"]: p for p in load_behavioral_prompts()}
    completed = skipped = 0

    for sample in samples:
        for judge in judges:
            if _should_exclude(sample.get("subject_provider", ""), judge["model_name"]):
                skipped += 1
                continue
            if storage.already_judged(sample["id"], judge["litellm_model_id"]):
                skipped += 1
                continue

            result = _call_judge_sync(sample, judge, prompt_lookup, debug=debug)
            storage.store_judge_rating(result)

            status = result["parse_status"]
            print(f"  {sample['prompt_id']} run={sample['run_number']} "
                  f"[{judge['model_name']}] → {status}")
            if debug and status != "success":
                print(f"    error: {result.get('error_message')}")
            completed += 1

    print(f"\n[judge_runner] Done. {completed} new calls, {skipped} skipped.")


# ── Async runner ──────────────────────────────────────────────────────────────

async def async_run(
    judge_names: list[str] | None = None,
    model_names: list[str] | None = None,
    n_samples: int | None = None,
    providers: list[str] | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    judge_names = judge_names or DEFAULT_JUDGE_NAMES
    full_registry = load_model_registry(providers=None)
    provider_map = _build_provider_map(full_registry)

    all_judges = load_model_registry(providers=["bedrock", "openai"])
    judges = filter_by_names(all_judges, judge_names)
    if not judges:
        print(f"[judge_runner] No judge models found matching {judge_names}.")
        sys.exit(1)

    subject_model_ids = _get_subject_model_ids(full_registry, model_names, providers)
    samples = storage.load_behavioral_samples_for_judging(
        model_ids=subject_model_ids,
        n_samples=n_samples,
    )
    for s in samples:
        s["subject_provider"] = provider_map.get(s["model_id"], "")

    limiters = build_rate_limiters(judges)
    completed_set = storage.load_completed_judge_set()
    work = _build_work_queues(samples, judges, completed_set)

    total_pending = sum(len(q) for q in work.values())
    total_possible = sum(
        1 for s in samples for j in judges
        if not _should_exclude(s.get("subject_provider", ""), j["model_name"])
    )
    skipped = total_possible - total_pending

    print(
        f"[judge_runner] {len(samples)} sample(s) × {len(judges)} judge(s)"
        f" = {total_possible} eligible ({total_pending} pending, {skipped} already done)"
    )

    if dry_run:
        _print_plan(judges, samples)
        return

    prompt_lookup = {p["prompt_id"]: p for p in load_behavioral_prompts()}
    counters: dict[str, int] = {"completed": 0, "errors": 0, "tpd_stopped": 0}
    t0 = time.monotonic()

    async def _process_judge(judge_litellm_id: str, queue: list) -> None:
        if not queue:
            return
        limiter = limiters[judge_litellm_id]
        judge_entry = queue[0][1]
        judge_label = judge_entry["model_name"]
        # Cap concurrency at 10 for judge calls — responses are tiny (30-token JSON),
        # so throughput doesn't benefit from high concurrency, and a low cap limits
        # how many in-flight calls can complete after a DailyLimitExhausted is raised.
        sem = asyncio.Semaphore(10)
        tpd_exhausted = False
        model_pending = len(queue)
        model_done = 0

        async def _do_call(sample: dict, judge: dict) -> None:
            nonlocal tpd_exhausted, model_done
            if tpd_exhausted:
                return
            async with sem:
                if tpd_exhausted:
                    return
                try:
                    result = await _call_judge_async(
                        sample, judge, prompt_lookup, limiter=limiter, debug=debug
                    )
                except DailyLimitExhausted as exc:
                    tpd_exhausted = True
                    counters["tpd_stopped"] += 1
                    print(
                        f"\n[{judge_label}] Daily token limit reached "
                        f"({exc.used:,}/{exc.limit:,}). Stopping this judge."
                    )
                    return

                storage.store_judge_rating(result)

                if result["parse_status"] == "success":
                    counters["completed"] += 1
                    model_done += 1
                else:
                    counters["errors"] += 1

                if (model_done > 0 and model_done % 50 == 0) or \
                        model_done + counters["errors"] == model_pending:
                    elapsed = time.monotonic() - t0
                    rate = model_done / elapsed if elapsed > 0 else 0
                    remaining = limiter.tpd_remaining
                    tpd_info = f"  tpd_remaining={remaining:,}" if remaining is not None else ""
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"  {ts} [{judge_label}] {model_done}/{model_pending} "
                        f"({rate:.1f} calls/s){tpd_info}"
                    )

                if debug:
                    status = result["parse_status"]
                    if status != "success":
                        print(
                            f"  [{judge_label}] {sample['prompt_id']} run={sample['run_number']}"
                            f" → {status}: {result.get('error_message')}"
                        )

        tasks = [_do_call(s, j) for s, j in queue]
        await asyncio.gather(*tasks)

    print("[judge_runner] Starting parallel judging...")
    judge_tasks = [_process_judge(jid, queue) for jid, queue in work.items() if queue]
    await asyncio.gather(*judge_tasks)

    elapsed = time.monotonic() - t0
    done = counters["completed"] + counters["errors"]
    print(
        f"\n[judge_runner] Done. {counters['completed']} success, {counters['errors']} errors, "
        f"{skipped} skipped. {elapsed:.1f}s elapsed "
        f"({done / elapsed:.1f} calls/s)." if elapsed > 0 else
        f"\n[judge_runner] Done. {counters['completed']} success, {counters['errors']} errors, "
        f"{skipped} skipped."
    )
    if counters["tpd_stopped"]:
        print(
            f"[judge_runner] {counters['tpd_stopped']} judge(s) stopped early due to daily "
            "token limits. Re-run to continue."
        )

    await litellm.close_litellm_async_clients()


# ── Core call functions ───────────────────────────────────────────────────────

def _call_judge_sync(
    sample: dict,
    judge: dict,
    prompt_lookup: dict[str, dict],
    debug: bool = False,
) -> dict:
    """Make a single synchronous judge API call. Returns a storage-ready row."""
    keying = sample_keying()
    prompt = prompt_lookup.get(sample["prompt_id"])
    if prompt is None:
        return _build_judge_row(sample, judge, keying, None,
                                f"prompt_id {sample['prompt_id']!r} not in behavioral_loader")

    messages = build_judge_messages(
        prompt,
        sample["raw_response"],
        bool(sample["is_two_turn"]),
        keying=keying,
    )

    if debug:
        print(f"\n  [debug] judge={judge['model_name']}  "
              f"sample={sample['prompt_id']} run={sample['run_number']}")
        for msg in messages:
            print(f"  [debug] {msg['role'].upper()}: {msg['content'][:300]}")

    kwargs: dict = dict(
        model=judge["litellm_model_id"],
        messages=messages,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    kwargs.update(get_provider_kwargs(judge, behavioral=False))
    # Judges don't need log-probs; clear any extra_body injected for scoring
    kwargs.pop("extra_body", None)

    rpm = judge.get("requests_per_minute")
    max_attempts = 3
    raw_text: str | None = None
    last_error: str | None = None

    for attempt in range(max_attempts):
        try:
            if rpm:
                time.sleep(60.0 / rpm)
            response = litellm.completion(**kwargs)
            raw_text = response.choices[0].message.content or ""
            last_error = None
            break
        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            time.sleep(2 ** (attempt + 2))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)

    return _build_judge_row(sample, judge, keying, raw_text, last_error)


async def _call_judge_async(
    sample: dict,
    judge: dict,
    prompt_lookup: dict[str, dict],
    limiter: AsyncRateLimiter,
    debug: bool = False,
) -> dict:
    """Make a single async judge API call. Returns a storage-ready row."""
    keying = sample_keying()
    prompt = prompt_lookup.get(sample["prompt_id"])
    if prompt is None:
        return _build_judge_row(sample, judge, keying, None,
                                f"prompt_id {sample['prompt_id']!r} not in behavioral_loader")

    messages = build_judge_messages(
        prompt,
        sample["raw_response"],
        bool(sample["is_two_turn"]),
        keying=keying,
    )

    estimated_tokens = sum(len(m["content"]) for m in messages) // 4 + _MAX_TOKENS // 4

    kwargs: dict = dict(
        model=judge["litellm_model_id"],
        messages=messages,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    kwargs.update(get_provider_kwargs(judge, behavioral=False))
    kwargs.pop("extra_body", None)

    await limiter.acquire(estimated_tokens)

    max_attempts = 3
    raw_text: str | None = None
    last_error: str | None = None

    for attempt in range(max_attempts):
        try:
            response = await litellm.acompletion(timeout=_TIMEOUT, **kwargs)
            actual_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
            if actual_tokens:
                await limiter.record(actual_tokens, estimated_tokens)
            raw_text = response.choices[0].message.content or ""
            last_error = None
            break
        except DailyLimitExhausted:
            raise
        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            await asyncio.sleep(2 ** (attempt + 2))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)

    return _build_judge_row(sample, judge, keying, raw_text, last_error)


def _build_judge_row(
    sample: dict,
    judge: dict,
    keying: str,
    raw_text: str | None,
    error: str | None,
) -> dict:
    """Construct the storage-ready dict from a completed judge call."""
    row: dict = {
        "behavioral_response_id": sample["id"],
        "subject_model_id": sample["model_id"],
        "prompt_id": sample["prompt_id"],
        "run_number": sample["run_number"],
        "judge_model_id": judge["litellm_model_id"],
        "keying": keying,
        "raw_response": raw_text,
        "error_message": error,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    }
    for col in storage.JUDGE_SCORE_COLUMNS:
        row[col] = None

    if error or not raw_text:
        row["parse_status"] = "api_error"
        if not error:
            row["error_message"] = "empty response"
        return row

    scores, parse_error = parse_judge_response(raw_text)
    if parse_error:
        row["parse_status"] = "parse_error"
        row["error_message"] = parse_error
        return row

    for factor in FACTOR_ORDER:
        row[f"score_{factor}"] = scores[factor]
    row["parse_status"] = "success"
    return row


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_subject_model_ids(
    full_registry: list[dict],
    model_names: list[str] | None,
    providers: list[str] | None,
) -> list[str] | None:
    """Return litellm_model_ids to filter samples by, or None for all."""
    if not model_names and not providers:
        return None
    models = full_registry
    if providers:
        models = [m for m in models if m.get("api_provider") in providers]
    if model_names:
        models = filter_by_names(models, model_names)
    return [m["litellm_model_id"] for m in models] if models else []


def _print_plan(judges: list[dict], samples: list[dict]) -> None:
    print(f"\nJudges ({len(judges)}):")
    for j in judges:
        print(f"  {j['model_name']}  [{j['api_provider']}]  {j['litellm_model_id']}")
    print(f"\nSamples: {len(samples)} behavioral responses")

    from collections import Counter
    by_prompt = Counter(s["prompt_id"] for s in samples)
    by_model = Counter(s["model_id"] for s in samples)
    print(f"  Unique prompts: {len(by_prompt)}")
    print(f"  Unique subject models: {len(by_model)}")

    excluded = sum(
        1 for s in samples for j in judges
        if _should_exclude(s.get("subject_provider", ""), j["model_name"])
    )
    total_possible = len(samples) * len(judges)
    print(f"\n  Total (samples × judges): {total_possible}")
    print(f"  Excluded by cross-model rule: {excluded}")
    print(f"  Eligible calls: {total_possible - excluded}")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(judges: list[dict], samples: list[dict]) -> None:
    prompt_ids = list({s["prompt_id"] for s in samples})
    subject_model_ids = list({s["model_id"] for s in samples})
    if not prompt_ids or not subject_model_ids:
        return

    import sqlite3
    conn = sqlite3.connect(str(storage.DB_PATH))
    conn.row_factory = sqlite3.Row

    header = f"{'Judge':<30} {'Success':>8} {'ParseErr':>9} {'APIErr':>7}"
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    subj_placeholders = ",".join("?" for _ in subject_model_ids)
    for judge in judges:
        rows = conn.execute(
            f"""
            SELECT parse_status, COUNT(*) as cnt
            FROM judge_ratings
            WHERE judge_model_id=?
              AND subject_model_id IN ({subj_placeholders})
            GROUP BY parse_status
            """,
            [judge["litellm_model_id"], *subject_model_ids],
        ).fetchall()
        counts = {r["parse_status"]: r["cnt"] for r in rows}
        print(
            f"{judge['model_name']:<30}"
            f"{counts.get('success', 0):>8}"
            f"{counts.get('parse_error', 0):>9}"
            f"{counts.get('api_error', 0):>7}"
        )

    conn.close()
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge rating pipeline for Phase 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--judges", nargs="+", metavar="NAME",
                        default=None,
                        help=f"Judge model name(s). Default: {DEFAULT_JUDGE_NAMES}")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Filter subject models by name (substring). Default: all.")
    parser.add_argument("--n-samples", type=int, default=None, metavar="N",
                        help="Number of behavioral samples to judge. Default: all.")
    parser.add_argument("--providers", nargs="+", metavar="PROVIDER",
                        help="Filter subject models by api_provider. Default: all.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run API calls concurrently (async).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making any API calls.")
    parser.add_argument("--debug", action="store_true",
                        help="Print prompt + full response for every call.")

    args = parser.parse_args()
    kwargs = dict(
        judge_names=args.judges,
        model_names=args.models,
        n_samples=args.n_samples,
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
