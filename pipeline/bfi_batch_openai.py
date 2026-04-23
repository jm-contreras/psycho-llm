"""
BFI-44 GPT-5.4 Mini batch collection via OpenAI Batch API.

50% cheaper than standard API calls, 24h completion window.
Logprobs are supported and extracted.

Setup:
  OPENAI_API_KEY must be set in .env.

Usage:
  # Show pending count, do nothing
  python -m pipeline.bfi_batch_openai --dry-run

  # Submit batch job and block until results are ingested
  python -m pipeline.bfi_batch_openai

  # Ingest from an already-completed job (if polling was interrupted)
  python -m pipeline.bfi_batch_openai --ingest batch_abc123
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
    _extract_logprob_score,
    _parse_text_score,
)
from pipeline.bfi_items import load_bfi_items

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_LITELLM_ID = "openai/gpt-5.4-mini"
MODEL_PROVIDER_ID = "gpt-5.4-mini"
N_RUNS = 30

MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_bfi_openai_mini_manifest.json"

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}

# ── Build batch input ─────────────────────────────────────────────────────────

def _request_key(item_id: str, run_num: int) -> str:
    return f"{item_id}__r{run_num:02d}"


def _build_request(item: dict) -> dict:
    """Format a single BFI item as an OpenAI Chat Completions request body."""
    return {
        "model": MODEL_PROVIDER_ID,
        "messages": [
            {"role": "system", "content": _BFI_SYSTEM},
            {"role": "user", "content": _BFI_USER.format(text=item["text"])},
        ],
        "max_completion_tokens": 512,
        "response_format": {"type": "json_object"},
        "logprobs": True,
        "top_logprobs": 5,
    }


def build_batch_input(
    items: list[dict], completed_set: set, n_runs: int = N_RUNS
) -> tuple[str, dict]:
    """
    Build JSONL string and manifest for all pending (item, run) pairs.

    Returns:
        jsonl:    newline-joined request lines
        manifest: dict mapping custom_id → metadata
    """
    lines = []
    manifest = {}

    for item in items:
        for run_num in range(1, n_runs + 1):
            if (MODEL_LITELLM_ID, item["item_id"], run_num) in completed_set:
                continue

            key = _request_key(item["item_id"], run_num)
            lines.append(json.dumps({
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _build_request(item),
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


# ── OpenAI Batch API ──────────────────────────────────────────────────────────

def _client():
    import openai
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def submit_job(jsonl: str, description: str = "psycho-llm-bfi-mini") -> object:
    """Upload JSONL via Files API and submit batch job. Returns batch object."""
    client = _client()

    tmp_path = REPO_ROOT / "data" / "raw" / "_batch_bfi_openai_mini_input.jsonl"
    tmp_path.write_text(jsonl)

    print(f"  Uploading {tmp_path.stat().st_size:,} bytes via Files API...")
    with open(tmp_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  File: {uploaded.id}")

    print(f"  Submitting batch job (model={MODEL_PROVIDER_ID})...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
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
    """Download and parse output + error JSONL from completed batch. Returns list of result dicts."""
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


# ── Ingest results ────────────────────────────────────────────────────────────

class _TopLogprob:
    """Thin wrapper so dict-format top_logprobs entries work with _extract_logprob_score."""
    def __init__(self, d: dict):
        self.token = d["token"]
        self.logprob = d["logprob"]


class _LogprobToken:
    """Thin wrapper so dict-format logprob content entries work with _extract_logprob_score."""
    def __init__(self, d: dict):
        self.token = d["token"]
        self.logprob = d["logprob"]
        self.top_logprobs = [_TopLogprob(t) for t in d.get("top_logprobs", [])]


def ingest(rows: list[dict], manifest: dict) -> None:
    """Parse result rows and write to DB."""
    items_by_id = {it["item_id"]: it for it in load_bfi_items()}

    success = parse_error = api_error = 0

    for row in rows:
        key = row.get("custom_id", "")
        meta = manifest.get(key)
        if meta is None:
            print(f"  WARNING: no manifest entry for custom_id {key!r}, skipping")
            continue

        if row.get("error"):
            _store_error(meta, f"Batch API error: {row['error']}")
            api_error += 1
            continue

        try:
            response_body = row["response"]["body"]
        except (KeyError, TypeError) as exc:
            _store_error(meta, f"Malformed response: {exc}")
            api_error += 1
            continue

        status_code = row["response"].get("status_code", 200)
        body_error = response_body.get("error") if isinstance(response_body, dict) else None
        if status_code != 200 or body_error:
            msg = body_error.get("message", str(body_error)) if body_error else f"HTTP {status_code}"
            _store_error(meta, f"API error ({status_code}): {msg}")
            api_error += 1
            continue

        try:
            choice = response_body["choices"][0]
            raw_text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            _store_error(meta, f"Could not extract response text: {exc}")
            api_error += 1
            continue

        item = items_by_id.get(meta["item_id"], {"item_type": meta["item_type"], "options": []})
        parsed_score, text_method = _parse_text_score(raw_text, item, None)

        logprob_data = None
        try:
            lp_content = choice.get("logprobs", {}).get("content") or []
            if lp_content:
                logprob_data = [_LogprobToken(t) for t in lp_content]
        except (KeyError, TypeError, AttributeError):
            pass

        logprob_score, logprob_token_logprob, logprob_vector, logprob_match_token, logprob_available = (
            _extract_logprob_score(item, logprob_data, parsed_score, None)
        )

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
            "logprob_score": logprob_score,
            "logprob_token_logprob": logprob_token_logprob,
            "logprob_vector": logprob_vector,
            "logprob_match_token": logprob_match_token,
            "logprob_available": logprob_available,
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
        description="BFI-44 GPT-5.4 Mini batch collection via OpenAI Batch API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-items", type=int, default=None, metavar="N",
                        help="Limit to first N items (for testing). Default: all 44.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS, metavar="N",
                        help=f"Runs per item. Default: {N_RUNS}.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="BATCH_ID",
                        help="Skip submission; ingest from completed batch (e.g. batch_abc123).")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in .env")

    items = load_bfi_items()
    if args.n_items is not None:
        items = items[: args.n_items]

    completed_set = storage.load_completed_set()
    jsonl, manifest = build_batch_input(items, completed_set, n_runs=args.n_runs)

    print(f"[bfi_batch_openai] {len(manifest)} pending calls for {MODEL_LITELLM_ID}")

    if args.dry_run or not manifest:
        return

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest))

    if args.ingest:
        print(f"\nLoading manifest from {MANIFEST_PATH}")
        manifest = json.loads(MANIFEST_PATH.read_text())
        client = _client()
        batch = client.batches.retrieve(args.ingest)
        print(f"Batch status: {batch.status}")
        if batch.status != "completed":
            raise SystemExit(f"Batch not completed (status={batch.status}), cannot ingest.")
        rows = download_results(batch)
        ingest(rows, manifest)
        return

    print("\nSubmitting batch job...")
    batch = submit_job(jsonl)

    batch = poll_job(batch)
    rows = download_results(batch)
    ingest(rows, manifest)


if __name__ == "__main__":
    main()
