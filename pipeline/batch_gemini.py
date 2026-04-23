"""
Gemini 3.1 Pro batch collection via Google AI Batch API.

50% cheaper than standard API calls, 24h turnaround target.

Setup (one-time):
  GEMINI_API_KEY must be set in .env — no other credentials needed.

  Install dependency:
    pip install google-genai

Usage:
  # Show pending count, do nothing
  python -m pipeline.batch_gemini --dry-run

  # Submit batch job and block until results are ingested
  python -m pipeline.batch_gemini

  # Ingest from an already-completed job (if polling was interrupted)
  python -m pipeline.batch_gemini --ingest batches/123456789
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

from pipeline import storage
from pipeline.api_client import (
    _DIRECT_SYSTEM,
    _DIRECT_USER,
    _SCENARIO_SYSTEM,
    _SCENARIO_USER,
    _parse_text_score,
)
from pipeline.item_loader import load_items

# ── Model configs ─────────────────────────────────────────────────────────────

_MODELS = {
    "pro": {
        "litellm_id": "gemini/gemini-3.1-pro-preview",
        "gemini_id": "gemini-3.1-pro-preview",
    },
    "flash-lite": {
        "litellm_id": "gemini/gemini-3.1-flash-lite-preview",
        "gemini_id": "gemini-3.1-flash-lite-preview",
    },
}

# Defaults (used when called without --model; preserved for backward compatibility)
MODEL_LITELLM_ID = _MODELS["pro"]["litellm_id"]
GEMINI_MODEL = _MODELS["pro"]["gemini_id"]

N_RUNS = 30

# Local manifest — maps request key → metadata; no cloud storage needed.
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_gemini_manifest.json"


def _manifest_path_for(model_short: str) -> Path:
    if model_short == "pro":
        return MANIFEST_PATH  # preserve existing filename for pro
    return REPO_ROOT / "data" / "raw" / f"batch_gemini_manifest_{model_short}.json"

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# ── Build batch input ─────────────────────────────────────────────────────────

def _request_key(item_id: str, run_num: int) -> str:
    return f"{item_id}__r{run_num:02d}"


def _build_request(item: dict, shuffled_options: list | None) -> dict:
    """Format a single item as a GenerateContentRequest dict."""
    if item["item_type"] == "direct":
        system_text = _DIRECT_SYSTEM
        user_text = _DIRECT_USER.format(text=item["text"])
    else:
        opts = shuffled_options or item["options"]
        system_text = _SCENARIO_SYSTEM
        user_text = _SCENARIO_USER.format(
            context=item["text"],
            opt_a=opts[0]["text"],
            opt_b=opts[1]["text"],
            opt_c=opts[2]["text"],
            opt_d=opts[3]["text"],
        )
    return {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def build_batch_input(
    items: list[dict], completed_set: set, n_runs: int = N_RUNS,
    model_litellm_id: str = MODEL_LITELLM_ID,
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
            if (model_litellm_id, item["item_id"], run_num) in completed_set:
                continue

            shuffled_options = None
            option_order = None
            if item["item_type"] == "scenario":
                shuffled_options = random.sample(item["options"], len(item["options"]))
                option_order = ",".join(o["label"] for o in shuffled_options)

            key = _request_key(item["item_id"], run_num)
            lines.append(json.dumps({
                "key": key,
                "request": _build_request(item, shuffled_options),
            }))
            manifest[key] = {
                "item_id": item["item_id"],
                "dimension": item["dimension"],
                "item_type": item["item_type"],
                "keying": item.get("keying"),
                "run_number": run_num,
                "option_order": option_order,
                "shuffled_options": [
                    {"label": o["label"], "text": o["text"], "score": o["score"]}
                    for o in shuffled_options
                ] if shuffled_options else None,
            }

    return "\n".join(lines), manifest


# ── Google AI Batch API ───────────────────────────────────────────────────────

def _client():
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def submit_job(
    jsonl: str,
    gemini_model_id: str = GEMINI_MODEL,
    display_name: str = "psycho-llm-batch",
) -> object:
    """Upload JSONL via Files API and submit batch job. Returns job object."""
    from google import genai
    from google.genai import types

    client = _client()

    # Write to a temp file for upload
    tmp_path = REPO_ROOT / "data" / "raw" / "_batch_gemini_input.jsonl"
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

def ingest(rows: list[dict], manifest: dict, model_litellm_id: str = MODEL_LITELLM_ID) -> None:
    """Parse result rows and write to DB."""
    items_by_id = {it["item_id"]: it for it in load_items()}

    success = parse_error = api_error = 0

    for row in rows:
        key = row.get("key", "")
        meta = manifest.get(key)
        if meta is None:
            print(f"  WARNING: no manifest entry for key {key!r}, skipping")
            continue

        # API-level error
        if "error" in row:
            _store_error(meta, f"Batch API error: {row['error']}", model_litellm_id)
            api_error += 1
            continue

        try:
            raw_text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            _store_error(meta, f"Could not extract response text: {exc}", model_litellm_id)
            api_error += 1
            continue

        shuffled_options = meta.get("shuffled_options") or None
        item = items_by_id.get(meta["item_id"], {"item_type": meta["item_type"], "options": []})
        parsed_score, text_method = _parse_text_score(raw_text, item, shuffled_options)

        storage.store({
            "model_id": model_litellm_id,
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
            "option_order": meta.get("option_order"),
            "status": "success" if parsed_score is not None else "parse_error",
            "error_message": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if parsed_score is not None:
            success += 1
        else:
            parse_error += 1

    print(f"\nIngested: {success} success, {parse_error} parse_error, {api_error} api_error")


def _store_error(meta: dict, error_msg: str, model_litellm_id: str = MODEL_LITELLM_ID) -> None:
    storage.store({
        "model_id": model_litellm_id,
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
        "option_order": meta.get("option_order"),
        "status": "api_error",
        "error_message": error_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini batch item collection via Google AI Batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="pro",
        help="Gemini model to use. Default: pro.",
    )
    parser.add_argument("--item-types", nargs="+", choices=["direct", "scenario"], default=None,
                        help="Filter by item type. Default: all.")
    parser.add_argument("--n-items", type=int, default=None, metavar="N",
                        help="Limit to first N items (for testing). Default: all.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS, metavar="N",
                        help=f"Runs per item. Default: {N_RUNS}.")
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
    manifest_path = _manifest_path_for(args.model)
    display_name = f"psycho-llm-batch-{args.model}"

    items = load_items()
    if args.item_types is not None:
        items = [it for it in items if it["item_type"] in args.item_types]
    if args.n_items is not None:
        items = items[: args.n_items]

    completed_set = storage.load_completed_set()
    jsonl, manifest = build_batch_input(items, completed_set, n_runs=args.n_runs,
                                        model_litellm_id=litellm_id)

    print(f"[batch_gemini] {len(manifest)} pending calls for {litellm_id}")

    if args.dry_run or not manifest:
        return

    # Save manifest locally so --ingest can reuse it without resubmitting
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
