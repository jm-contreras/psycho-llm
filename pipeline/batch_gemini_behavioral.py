"""
Gemini batch collection for Phase 3 behavioral prompts.

Uses Google AI Batch API (50% cheaper than standard, ~24h turnaround).
Collects free-text responses — no score parsing.

Setup:
  GEMINI_API_KEY must be set in .env

Usage:
  # Show pending count, do nothing
  python -m pipeline.batch_gemini_behavioral --dry-run

  # Collect with Gemini 3.1 Pro (default)
  python -m pipeline.batch_gemini_behavioral

  # Collect with Gemini 3.1 Flash Lite
  python -m pipeline.batch_gemini_behavioral --model flash-lite

  # Single-turn prompts only
  python -m pipeline.batch_gemini_behavioral --prompt-types single

  # Two-turn prompts only
  python -m pipeline.batch_gemini_behavioral --prompt-types two-turn

  # Ingest from an already-completed job (if polling was interrupted)
  python -m pipeline.batch_gemini_behavioral --model flash-lite --ingest batches/123456789
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
from pipeline.behavioral_loader import build_messages, load_behavioral_prompts

# ── Model registry ────────────────────────────────────────────────────────────

_MODELS = {
    "pro": {
        "litellm_id": "gemini/gemini-3.1-pro-preview",
        "gemini_id": "gemini-3.1-pro-preview",
        "short": "pro",
    },
    "flash-lite": {
        "litellm_id": "gemini/gemini-3.1-flash-lite-preview",
        "gemini_id": "gemini-3.1-flash-lite-preview",
        "short": "flash-lite",
    },
}

N_RUNS = 5

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _manifest_path(short: str) -> Path:
    return REPO_ROOT / "data" / "raw" / f"batch_gemini_behavioral_{short}_manifest.json"


# ── Build batch input ─────────────────────────────────────────────────────────

def _request_key(prompt_id: str, run_num: int) -> str:
    return f"{prompt_id}__r{run_num:02d}"


def _build_request(prompt: dict) -> dict:
    """Format a behavioral prompt as a GenerateContentRequest dict."""
    messages = build_messages(prompt)
    # No system prompt for behavioral prompts.
    # Map to Gemini contents array: user/model roles (Gemini uses "model" not "assistant").
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    return {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 1.0,
        },
    }


def build_batch_input(
    prompts: list[dict],
    completed_set: set,
    model_litellm_id: str,
    n_runs: int = N_RUNS,
) -> tuple[str, dict]:
    """Build JSONL string and manifest for all pending (prompt, run) pairs."""
    lines = []
    manifest = {}

    for prompt in prompts:
        for run_num in range(1, n_runs + 1):
            if (model_litellm_id, prompt["prompt_id"], run_num) in completed_set:
                continue
            key = _request_key(prompt["prompt_id"], run_num)
            lines.append(json.dumps({
                "key": key,
                "request": _build_request(prompt),
            }))
            manifest[key] = {
                "prompt_id": prompt["prompt_id"],
                "dimension": prompt["dimension"],
                "dimension_code": prompt["dimension_code"],
                "is_two_turn": prompt["is_two_turn"],
                "run_number": run_num,
            }

    return "\n".join(lines), manifest


# ── Google AI Batch API ───────────────────────────────────────────────────────

def _client():
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def submit_job(jsonl: str, gemini_model_id: str, display_name: str) -> object:
    """Upload JSONL via Files API and submit batch job. Returns job object."""
    from google.genai import types

    client = _client()
    tmp_path = REPO_ROOT / "data" / "raw" / "_batch_gemini_behavioral_input.jsonl"
    tmp_path.write_text(jsonl)

    print(f"  Uploading {tmp_path.stat().st_size:,} bytes via Files API...")
    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"  File: {uploaded.name}")

    print(f"  Submitting batch job (model={gemini_model_id})...")
    job = client.batches.create(
        model=gemini_model_id,
        src=uploaded.name,
        config={"display_name": display_name},
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


# ── Ingest results ────────────────────────────────────────────────────────────

def ingest(rows: list[dict], manifest: dict, model_litellm_id: str) -> None:
    """Parse result rows and write to behavioral_responses table."""
    success = api_error = 0

    for row in rows:
        key = row.get("key", "")
        meta = manifest.get(key)
        if meta is None:
            print(f"  WARNING: no manifest entry for key {key!r}, skipping")
            continue

        if "error" in row:
            _store_error(meta, model_litellm_id, f"Batch API error: {row['error']}")
            api_error += 1
            continue

        try:
            raw_text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            _store_error(meta, model_litellm_id, f"Could not extract response text: {exc}")
            api_error += 1
            continue

        storage.store_behavioral({
            "model_id": model_litellm_id,
            "prompt_id": meta["prompt_id"],
            "dimension": meta["dimension"],
            "dimension_code": meta["dimension_code"],
            "is_two_turn": 1 if meta["is_two_turn"] else 0,
            "run_number": meta["run_number"],
            "raw_response": raw_text,
            "reasoning_content": None,
            "status": "success",
            "error_message": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        success += 1

    print(f"\nIngested: {success} success, {api_error} api_error")


def _store_error(meta: dict, model_litellm_id: str, error_msg: str) -> None:
    storage.store_behavioral({
        "model_id": model_litellm_id,
        "prompt_id": meta["prompt_id"],
        "dimension": meta["dimension"],
        "dimension_code": meta["dimension_code"],
        "is_two_turn": 1 if meta["is_two_turn"] else 0,
        "run_number": meta["run_number"],
        "raw_response": None,
        "reasoning_content": None,
        "status": "api_error",
        "error_message": error_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini batch behavioral collection (Phase 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="pro",
        help="Gemini model to use. Default: pro.",
    )
    parser.add_argument("--prompt-types", nargs="+", choices=["single", "two-turn"], default=None,
                        metavar="TYPE",
                        help="Filter by prompt type: single, two-turn, or both. Default: all.")
    parser.add_argument("--n-prompts", type=int, default=None, metavar="N",
                        help="Limit to first N prompts (for testing). Default: all.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS, metavar="N",
                        help=f"Runs per prompt. Default: {N_RUNS}.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="JOB_NAME",
                        help="Skip submission; ingest from completed job (e.g. batches/123).")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set in .env")

    model_cfg = _MODELS[args.model]
    litellm_id = model_cfg["litellm_id"]
    gemini_id = model_cfg["gemini_id"]
    short = model_cfg["short"]
    manifest_path = _manifest_path(short)
    display_name = f"psycho-llm-behavioral-{short}"

    prompts = load_behavioral_prompts()
    if args.prompt_types is not None:
        want_two_turn = "two-turn" in args.prompt_types
        want_single   = "single"   in args.prompt_types
        prompts = [
            p for p in prompts
            if (p["is_two_turn"] and want_two_turn) or (not p["is_two_turn"] and want_single)
        ]
    if args.n_prompts is not None:
        prompts = prompts[:args.n_prompts]
    completed_set = storage.load_completed_behavioral_set()
    jsonl, manifest = build_batch_input(prompts, completed_set, litellm_id, n_runs=args.n_runs)

    print(f"[batch_gemini_behavioral] {len(manifest)} pending calls for {litellm_id}")

    if args.dry_run or not manifest:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    if args.ingest:
        print(f"\nLoading manifest from {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        client = _client()
        job = client.batches.get(name=args.ingest)
        print(f"Job state: {job.state.name}")
        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"Job not succeeded (state={job.state.name}), cannot ingest.")
        rows = download_results(job)
        ingest(rows, manifest, litellm_id)
        return

    print("\nSubmitting batch job...")
    job = submit_job(jsonl, gemini_id, display_name)
    job = poll_job(job)
    rows = download_results(job)
    ingest(rows, manifest, litellm_id)


if __name__ == "__main__":
    main()
