"""
OpenAI Batch API judge runner for Phase 3 LLM-as-judge rating.

Uses OpenAI Batch API (50% cheaper than standard, 24h completion window).
Required because GPT-5.4 RPM limits make real-time rating of ~989 pending
samples impractical.

Rates behavioral_responses samples on the 5-factor rating scale using randomized
F/R statement keying (same design as the real-time judge_runner.py).

Keying is pre-generated at submission time and encoded in each request's custom_id
as "{behavioral_response_id}_{keying}" (e.g. "4521_FFRFF"). This makes keying
recoverable at collect time without a separate manifest lookup.

Setup:
  OPENAI_API_KEY must be set in .env.

Usage:
  # Dry run: show pending count without submitting
  python -m pipeline.batch_openai_judge --dry-run

  # Submit, poll, and collect in one shot
  python -m pipeline.batch_openai_judge

  # Resume ingestion from an already-completed job (if polling was interrupted)
  python -m pipeline.batch_openai_judge --ingest batch_abc123
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

from pipeline import storage
from pipeline.behavioral_loader import load_behavioral_prompts
from pipeline.config import load_model_registry
from pipeline.judge_prompt import (
    FACTOR_ORDER,
    FEW_SHOT_EXAMPLES,
    build_judge_messages,
    parse_judge_response,
    sample_keying,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_OPENAI_MODEL_ID = "gpt-5.4"
_LITELLM_MODEL_ID = "openai/gpt-5.4"
_MAX_TOKENS = 512
_TEMPERATURE = 0.0

_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_openai_judge_manifest.json"
_INPUT_PATH = REPO_ROOT / "data" / "raw" / "_batch_openai_judge_input.jsonl"

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


# ── Exclusion ─────────────────────────────────────────────────────────────────

def _build_provider_map(registry: list[dict]) -> dict[str, str]:
    return {m["litellm_model_id"]: m.get("provider", "") for m in registry}


# ── Batch input construction ──────────────────────────────────────────────────

def _make_custom_id(behavioral_response_id: int, keying: str) -> str:
    return f"{behavioral_response_id}_{keying}"


def _parse_custom_id(custom_id: str) -> tuple[int, str]:
    """Parse custom_id back into (behavioral_response_id, keying)."""
    parts = custom_id.split("_", 1)
    return int(parts[0]), parts[1]


def build_batch_input(
    samples: list[dict],
    prompt_lookup: dict[str, dict],
    completed_set: set[tuple[int, str]],
) -> tuple[str, dict]:
    """Build JSONL string and manifest for all pending (sample, GPT-5.4) pairs.

    Returns (jsonl_string, manifest_dict).
    manifest keys: custom_id -> {behavioral_response_id, subject_model_id, prompt_id,
                                  run_number, keying}
    """
    lines = []
    manifest = {}

    for sample in samples:
        key = (sample["id"], _LITELLM_MODEL_ID)
        if key in completed_set:
            continue

        keying = sample_keying()
        custom_id = _make_custom_id(sample["id"], keying)

        prompt = prompt_lookup.get(sample["prompt_id"])
        if prompt is None:
            continue

        # build_judge_messages() already returns OpenAI-compatible format
        # (role + string content), no conversion needed.
        messages = build_judge_messages(
            prompt,
            sample["raw_response"],
            bool(sample["is_two_turn"]),
            keying=keying,
        )

        lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": _OPENAI_MODEL_ID,
                "messages": messages,
                "max_completion_tokens": _MAX_TOKENS,
                "temperature": _TEMPERATURE,
            },
        }))
        manifest[custom_id] = {
            "behavioral_response_id": sample["id"],
            "subject_model_id": sample["model_id"],
            "prompt_id": sample["prompt_id"],
            "run_number": sample["run_number"],
            "keying": keying,
        }

    return "\n".join(lines), manifest


# ── OpenAI Batch API ──────────────────────────────────────────────────────────

def _client():
    import openai
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def submit_job(jsonl: str) -> object:
    """Upload JSONL via Files API and submit batch job. Returns batch object."""
    client = _client()

    _INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INPUT_PATH.write_text(jsonl, encoding="utf-8")

    print(f"  Uploading {_INPUT_PATH.stat().st_size:,} bytes via Files API...")
    with open(_INPUT_PATH, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File: {uploaded.id}")

    print(f"  Submitting batch job (model={_OPENAI_MODEL_ID})...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "psycho-llm-judge-gpt54"},
    )
    print(f"  Batch: {batch.id}  status={batch.status}")
    return batch


def poll_job(batch) -> object:
    """Poll until terminal state. Returns final batch object."""
    client = _client()
    print("\nPolling (every 60s)...")
    while batch.status not in TERMINAL_STATES:
        time.sleep(60)
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] {batch.status}"
            f"  completed={counts.completed} failed={counts.failed} total={counts.total}"
        )

    if batch.status == "completed":
        counts = batch.request_counts
        print(f"\nBatch completed: {counts.completed} succeeded, {counts.failed} failed.")
    else:
        raise RuntimeError(f"Batch ended with status={batch.status}")

    return batch


def download_results(batch) -> list[dict]:
    """Download and parse output + error JSONL from completed batch."""
    client = _client()
    rows = []

    for file_id, label in [
        (batch.output_file_id, "output"),
        (batch.error_file_id, "error"),
    ]:
        if not file_id:
            continue
        print(f"  Downloading {label} file ({file_id})...")
        raw = client.files.content(file_id)
        text = raw.text if hasattr(raw, "text") else raw.content.decode("utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"  {len(rows)} result lines downloaded")
    return rows


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest(rows: list[dict], manifest: dict) -> None:
    """Parse result rows and write to judge_ratings table."""
    success = parse_error = api_error = 0

    for row in rows:
        custom_id = row.get("custom_id", "")
        meta = manifest.get(custom_id)
        if meta is None:
            print(f"  WARNING: no manifest entry for custom_id {custom_id!r}, skipping")
            continue

        # Recover keying from custom_id (canonical source; manifest is backup)
        try:
            brid, keying = _parse_custom_id(custom_id)
        except (ValueError, IndexError):
            keying = meta.get("keying", "FFFFF")
            brid = meta["behavioral_response_id"]

        base_row = {
            "behavioral_response_id": brid,
            "subject_model_id": meta["subject_model_id"],
            "prompt_id": meta["prompt_id"],
            "run_number": meta["run_number"],
            "judge_model_id": _LITELLM_MODEL_ID,
            "keying": keying,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for col in storage.JUDGE_SCORE_COLUMNS:
            base_row[col] = None

        # Request-level error
        if row.get("error"):
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"Batch API error: {row['error']}",
            })
            api_error += 1
            continue

        try:
            response_body = row["response"]["body"]
        except (KeyError, TypeError) as exc:
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"Malformed response: {exc}",
            })
            api_error += 1
            continue

        # HTTP-level or body-level error
        status_code = row["response"].get("status_code", 200)
        body_error = response_body.get("error") if isinstance(response_body, dict) else None
        if status_code != 200 or body_error:
            msg = body_error.get("message", str(body_error)) if body_error else f"HTTP {status_code}"
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"API error ({status_code}): {msg}",
            })
            api_error += 1
            continue

        try:
            raw_text = response_body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"Could not extract response text: {exc}",
            })
            api_error += 1
            continue

        scores, parse_err = parse_judge_response(raw_text)
        if parse_err:
            storage.store_judge_rating({
                **base_row,
                "raw_response": raw_text,
                "parse_status": "parse_error",
                "error_message": parse_err,
            })
            parse_error += 1
            continue

        for factor in FACTOR_ORDER:
            base_row[f"score_{factor}"] = scores[factor]
        storage.store_judge_rating({
            **base_row,
            "raw_response": raw_text,
            "parse_status": "success",
            "error_message": None,
        })
        success += 1

    print(f"\nIngested: {success} success, {parse_error} parse_error, {api_error} api_error")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI batch judge runner for Phase 3 LLM-as-judge (GPT-5.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-samples", type=int, default=None, metavar="N",
                        help="Limit to first N samples (for testing). Default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="BATCH_ID",
                        help="Skip submission; ingest from completed batch (e.g. batch_abc123).")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in .env")

    if not FEW_SHOT_EXAMPLES:
        raise SystemExit(
            "FEW_SHOT_EXAMPLES in pipeline/judge_prompt.py is empty.\n"
            "Author and validate examples first: python -m pipeline.judge_prompt_validate"
        )

    # Load all samples, filter to non-OpenAI subject models
    full_registry = load_model_registry(providers=None)
    provider_map = _build_provider_map(full_registry)

    all_samples = storage.load_behavioral_samples_for_judging(n_samples=args.n_samples)
    samples = [
        s for s in all_samples
        if provider_map.get(s["model_id"], "") != "OpenAI"
    ]

    prompt_lookup = {p["prompt_id"]: p for p in load_behavioral_prompts()}
    completed_set = storage.load_completed_judge_set()

    pending_count = sum(
        1 for s in samples
        if (s["id"], _LITELLM_MODEL_ID) not in completed_set
    )

    print(
        f"[batch_openai_judge] {len(samples)} eligible samples "
        f"({pending_count} pending) for {_LITELLM_MODEL_ID}"
    )

    if args.dry_run or pending_count == 0:
        return

    if args.ingest:
        print(f"\nLoading manifest from {_MANIFEST_PATH}")
        manifest = json.loads(_MANIFEST_PATH.read_text())
        client = _client()
        batch = client.batches.retrieve(args.ingest)
        print(f"Batch status: {batch.status}")
        if batch.status != "completed":
            raise SystemExit(f"Batch not completed (status={batch.status}), cannot ingest.")
        rows = download_results(batch)
        ingest(rows, manifest)
        return

    jsonl, manifest = build_batch_input(samples, prompt_lookup, completed_set)
    print(f"  Built {len(manifest)} requests")

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"  Manifest saved to {_MANIFEST_PATH}")

    print("\nSubmitting batch job...")
    batch = submit_job(jsonl)

    # Save batch ID to manifest for easy recovery
    manifest_data = json.loads(_MANIFEST_PATH.read_text())
    manifest_data["_batch_id"] = batch.id
    _MANIFEST_PATH.write_text(json.dumps(manifest_data))

    batch = poll_job(batch)
    rows = download_results(batch)
    ingest(rows, manifest)


if __name__ == "__main__":
    main()
