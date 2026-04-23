"""
Gemini batch judge runner for Phase 3 LLM-as-judge rating.

Uses Google AI Batch API (50% cheaper than standard, ~24h turnaround).
Required because Gemini 3.1 Pro has rpd=250 real-time requests/day, which
is insufficient for the ~2,500 non-Google samples to rate.

Rates behavioral_responses samples on the 5-factor rating scale using randomized
F/R statement keying (same design as the real-time judge_runner.py).

Keying is pre-generated at submission time and encoded in each request's custom_id
as "{behavioral_response_id}_{keying}" (e.g. "4521_FFRFF"). This makes keying
recoverable at collect time without a separate manifest lookup.

Setup:
  GEMINI_API_KEY must be set in .env

Usage:
  # Dry run: show pending count without submitting
  python -m pipeline.batch_gemini_judge --dry-run

  # Submit, poll, and collect in one shot
  python -m pipeline.batch_gemini_judge

  # Resume ingestion from an already-completed job (if polling was interrupted)
  python -m pipeline.batch_gemini_judge --ingest batches/123456789
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
    _build_system_prompt_no_examples,
    build_judge_messages,
    parse_judge_response,
    sample_keying,
    FEW_SHOT_EXAMPLES,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_GEMINI_MODEL_ID = "gemini-3.1-pro-preview"
_LITELLM_MODEL_ID = "gemini/gemini-3.1-pro-preview"
_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_gemini_judge_manifest.json"
_INPUT_PATH = REPO_ROOT / "data" / "raw" / "_batch_gemini_judge_input.jsonl"
_DISPLAY_NAME = "psycho-llm-judge-pro"

_MAX_TOKENS = 4096
_TEMPERATURE = 0.0

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


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


def _build_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert litellm-style messages to Gemini batch API contents format.

    The system prompt is prepended as the first user turn (Gemini batch API
    does not support a dedicated system field in the batch request format).
    """
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    # Gemini requires alternating roles. If the first two are both "user"
    # (system + user), merge them.
    if len(contents) >= 2 and contents[0]["role"] == "user" and contents[1]["role"] == "user":
        merged_text = (
            contents[0]["parts"][0]["text"]
            + "\n\n"
            + contents[1]["parts"][0]["text"]
        )
        contents = [{"role": "user", "parts": [{"text": merged_text}]}] + contents[2:]
    return contents


def build_batch_input(
    samples: list[dict],
    prompt_lookup: dict[str, dict],
    completed_set: set[tuple[int, str]],
) -> tuple[str, dict]:
    """Build JSONL string and manifest for all pending (sample, Gemini) pairs.

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

        messages = build_judge_messages(
            prompt,
            sample["raw_response"],
            bool(sample["is_two_turn"]),
            keying=keying,
        )
        contents = _build_gemini_contents(messages)

        lines.append(json.dumps({
            "key": custom_id,
            "request": {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": _MAX_TOKENS,
                    "temperature": _TEMPERATURE,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                },
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


# ── Google AI Batch API ───────────────────────────────────────────────────────

def _client():
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def submit_job(jsonl: str) -> object:
    """Upload JSONL via Files API and submit batch job. Returns job object."""
    from google.genai import types

    client = _client()
    _INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INPUT_PATH.write_text(jsonl)

    print(f"  Uploading {_INPUT_PATH.stat().st_size:,} bytes via Files API...")
    uploaded = client.files.upload(
        file=_INPUT_PATH,
        config=types.UploadFileConfig(display_name=_DISPLAY_NAME, mime_type="jsonl"),
    )
    print(f"  File: {uploaded.name}")

    print(f"  Submitting batch job (model={_GEMINI_MODEL_ID})...")
    job = client.batches.create(
        model=_GEMINI_MODEL_ID,
        src=uploaded.name,
        config={"display_name": _DISPLAY_NAME},
    )
    print(f"  Job: {job.name}  state={job.state.name}")
    return job


def poll_job(job) -> object:
    """Poll until complete. Returns final job object."""
    client = _client()
    print("\nPolling (every 60s)...")
    while job.state.name not in COMPLETED_STATES:
        time.sleep(60)
        job = client.batches.get(name=job.name)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {job.state.name}")

    if job.state.name == "JOB_STATE_SUCCEEDED":
        print("Job succeeded.")
    else:
        raise RuntimeError(f"Batch job ended with state={job.state.name}")
    return job


def download_results(job) -> list[dict]:
    """Download and parse output JSONL from completed job."""
    client = _client()
    file_name = job.dest.file_name
    print(f"  Downloading results from {file_name}...")
    raw = client.files.download(file=file_name)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    rows = []
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
        custom_id = row.get("key", "")
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

        if "error" in row:
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"Batch API error: {row['error']}",
            })
            api_error += 1
            continue

        try:
            raw_text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
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
        description="Gemini batch judge runner for Phase 3 LLM-as-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-samples", type=int, default=None, metavar="N",
                        help="Limit to first N samples (for testing). Default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="JOB_NAME",
                        help="Skip submission; ingest from completed job (e.g. batches/123).")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set in .env")

    if not FEW_SHOT_EXAMPLES:
        raise SystemExit(
            "FEW_SHOT_EXAMPLES in pipeline/judge_prompt.py is empty.\n"
            "Author and validate examples first: python -m pipeline.judge_prompt_validate"
        )

    # Load all samples, filter to non-Google subject models
    full_registry = load_model_registry(providers=None)
    provider_map = _build_provider_map(full_registry)

    all_samples = storage.load_behavioral_samples_for_judging(n_samples=args.n_samples)
    samples = [
        s for s in all_samples
        if provider_map.get(s["model_id"], "") != "Google"
    ]

    prompt_lookup = {p["prompt_id"]: p for p in load_behavioral_prompts()}
    completed_set = storage.load_completed_judge_set()

    pending_count = sum(
        1 for s in samples
        if (s["id"], _LITELLM_MODEL_ID) not in completed_set
    )

    print(
        f"[batch_gemini_judge] {len(samples)} eligible samples "
        f"({pending_count} pending) for {_LITELLM_MODEL_ID}"
    )

    if args.dry_run or pending_count == 0:
        return

    if args.ingest:
        print(f"\nLoading manifest from {_MANIFEST_PATH}")
        manifest = json.loads(_MANIFEST_PATH.read_text())
        client = _client()
        job = client.batches.get(name=args.ingest)
        print(f"Job state: {job.state.name}")
        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"Job not succeeded (state={job.state.name}), cannot ingest.")
        rows = download_results(job)
        ingest(rows, manifest)
        return

    jsonl, manifest = build_batch_input(samples, prompt_lookup, completed_set)
    print(f"  Built {len(manifest)} requests")

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"  Manifest saved to {_MANIFEST_PATH}")

    print("\nSubmitting batch job...")
    job = submit_job(jsonl)
    job = poll_job(job)
    rows = download_results(job)
    ingest(rows, manifest)


if __name__ == "__main__":
    main()
