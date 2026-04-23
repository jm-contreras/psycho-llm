"""
BFI-44 Gemini 3.1 Pro batch collection via Google AI Batch API.

50% cheaper than standard API calls, 24h turnaround target.

Setup (one-time):
  GEMINI_API_KEY must be set in .env — no other credentials needed.

  Install dependency:
    pip install google-genai

Usage:
  # Show pending count, do nothing
  python -m pipeline.bfi_batch_gemini --dry-run

  # Submit batch job and block until results are ingested
  python -m pipeline.bfi_batch_gemini

  # Ingest from an already-completed job (if polling was interrupted)
  python -m pipeline.bfi_batch_gemini --ingest batches/123456789
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
from pipeline.api_client import (
    _BFI_SYSTEM,
    _BFI_USER,
    _parse_text_score,
)
from pipeline.bfi_items import load_bfi_items

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_LITELLM_ID = "gemini/gemini-3.1-pro-preview"
GEMINI_MODEL = "gemini-3.1-pro-preview"
N_RUNS = 30

MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_bfi_gemini_manifest.json"

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# ── Build batch input ─────────────────────────────────────────────────────────

def _request_key(item_id: str, run_num: int) -> str:
    return f"{item_id}__r{run_num:02d}"


def _build_request(item: dict) -> dict:
    """Format a single BFI item as a GenerateContentRequest dict."""
    return {
        "systemInstruction": {"parts": [{"text": _BFI_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": _BFI_USER.format(text=item["text"])}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def build_batch_input(
    items: list[dict], completed_set: set, n_runs: int = N_RUNS
) -> tuple[str, dict]:
    """
    Build JSONL string and manifest for all pending (item, run) pairs.

    Returns:
        jsonl:    newline-joined request lines
        manifest: dict mapping request key → metadata
    """
    lines = []
    manifest = {}

    for item in items:
        for run_num in range(1, n_runs + 1):
            if (MODEL_LITELLM_ID, item["item_id"], run_num) in completed_set:
                continue

            key = _request_key(item["item_id"], run_num)
            lines.append(json.dumps({
                "key": key,
                "request": _build_request(item),
            }))
            manifest[key] = {
                "item_id": item["item_id"],
                "dimension": item["dimension"],
                "item_type": item["item_type"],
                "keying": item.get("keying"),
                "run_number": run_num,
                "option_order": None,
                "shuffled_options": None,
            }

    return "\n".join(lines), manifest


# ── Google AI Batch API ───────────────────────────────────────────────────────

def _client():
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def submit_job(jsonl: str, display_name: str = "psycho-llm-bfi-batch") -> object:
    """Upload JSONL via Files API and submit batch job. Returns job object."""
    from google import genai
    from google.genai import types

    client = _client()

    tmp_path = REPO_ROOT / "data" / "raw" / "_batch_bfi_gemini_input.jsonl"
    tmp_path.write_text(jsonl)

    print(f"  Uploading {tmp_path.stat().st_size:,} bytes via Files API...")
    uploaded = client.files.upload(
        file=tmp_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"  File: {uploaded.name}")

    print(f"  Submitting batch job (model={GEMINI_MODEL})...")
    job = client.batches.create(
        model=GEMINI_MODEL,
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
        print(f"\nJob succeeded.")
    else:
        raise RuntimeError(f"Batch job ended with state={job.state.name}")

    return job


def download_results(job) -> list[dict]:
    """Download and parse output JSONL from completed job. Returns list of result dicts."""
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

def ingest(rows: list[dict], manifest: dict) -> None:
    """Parse result rows and write to DB."""
    items_by_id = {it["item_id"]: it for it in load_bfi_items()}

    success = parse_error = api_error = 0

    for row in rows:
        key = row.get("key", "")
        meta = manifest.get(key)
        if meta is None:
            print(f"  WARNING: no manifest entry for key {key!r}, skipping")
            continue

        if "error" in row:
            _store_error(meta, f"Batch API error: {row['error']}")
            api_error += 1
            continue

        try:
            raw_text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            _store_error(meta, f"Could not extract response text: {exc}")
            api_error += 1
            continue

        item = items_by_id.get(meta["item_id"], {"item_type": meta["item_type"], "options": []})
        parsed_score, text_method = _parse_text_score(raw_text, item, None)

        storage.store({
            "model_id": MODEL_LITELLM_ID,
            "item_id": meta["item_id"],
            "dimension": meta["dimension"],
            "item_type": meta["item_type"],
            "keying": meta["keying"],
            "run_number": meta["run_number"],
            "text_scoring_method": text_method,
            "raw_response": raw_text,
            "reasoning_content": None,
            "parsed_score": parsed_score,
            "logprob_score": None,
            "logprob_token_logprob": None,
            "logprob_vector": None,
            "logprob_match_token": None,
            "logprob_available": 0,
            "option_order": None,
            "status": "success" if parsed_score is not None else "parse_error",
            "error_message": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if parsed_score is not None:
            success += 1
        else:
            parse_error += 1

    print(f"\nIngested: {success} success, {parse_error} parse_error, {api_error} api_error")


def _store_error(meta: dict, error_msg: str) -> None:
    storage.store({
        "model_id": MODEL_LITELLM_ID,
        "item_id": meta["item_id"],
        "dimension": meta["dimension"],
        "item_type": meta["item_type"],
        "keying": meta["keying"],
        "run_number": meta["run_number"],
        "text_scoring_method": None,
        "raw_response": None,
        "reasoning_content": None,
        "parsed_score": None,
        "logprob_score": None,
        "logprob_token_logprob": None,
        "logprob_vector": None,
        "logprob_match_token": None,
        "logprob_available": 0,
        "option_order": None,
        "status": "api_error",
        "error_message": error_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BFI-44 Gemini 3.1 Pro batch collection via Google AI Batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-items", type=int, default=None, metavar="N",
                        help="Limit to first N items (for testing). Default: all 44.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS, metavar="N",
                        help=f"Runs per item. Default: {N_RUNS}.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="JOB_NAME",
                        help="Skip submission; ingest from completed job (e.g. batches/123).")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set in .env")

    items = load_bfi_items()
    if args.n_items is not None:
        items = items[: args.n_items]

    completed_set = storage.load_completed_set()
    jsonl, manifest = build_batch_input(items, completed_set, n_runs=args.n_runs)

    print(f"[bfi_batch_gemini] {len(manifest)} pending calls for {MODEL_LITELLM_ID}")

    if args.dry_run or not manifest:
        return

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest))

    if args.ingest:
        print(f"\nLoading manifest from {MANIFEST_PATH}")
        manifest = json.loads(MANIFEST_PATH.read_text())
        client = _client()
        job = client.batches.get(name=args.ingest)
        print(f"Job state: {job.state.name}")
        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"Job not succeeded (state={job.state.name}), cannot ingest.")
        rows = download_results(job)
        ingest(rows, manifest)
        return

    print("\nSubmitting batch job...")
    job = submit_job(jsonl)

    job = poll_job(job)
    rows = download_results(job)
    ingest(rows, manifest)


if __name__ == "__main__":
    main()
