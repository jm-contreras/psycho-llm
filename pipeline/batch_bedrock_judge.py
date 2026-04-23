"""
Bedrock Batch Inference judge runner for Phase 3 LLM-as-judge rating.

Uses AWS Bedrock Batch Inference (50% cheaper than standard, ~24h turnaround).
Required because Claude Opus 4.6 has strict RPM limits that make real-time
rating of ~768+ pending samples impractical.

Rates behavioral_responses samples on the 5-factor rating scale using randomized
F/R statement keying (same design as the real-time judge_runner.py).

Keying is pre-generated at submission time and encoded in each record's recordId
as "{behavioral_response_id}_{keying}" (e.g. "4521_FFRFF"). This makes keying
recoverable at collect time without a separate manifest lookup.

Setup:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION must be set in .env.
  BEDROCK_BATCH_S3_BUCKET must be set (an S3 bucket in the same region used for batch I/O).

Usage:
  # Dry run: show pending count without submitting
  python -m pipeline.batch_bedrock_judge --dry-run

  # Submit, poll, and collect in one shot
  python -m pipeline.batch_bedrock_judge

  # Resume ingestion from an already-completed job (if polling was interrupted)
  python -m pipeline.batch_bedrock_judge --ingest arn:aws:bedrock:us-east-1:...:model-invocation-job/...
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

_BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
_LITELLM_MODEL_ID = "bedrock/us.anthropic.claude-opus-4-6-v1"
_MAX_TOKENS = 512
_TEMPERATURE = 0.0

_DEFAULT_S3_BUCKET = None  # set via BEDROCK_BATCH_S3_BUCKET env var
_S3_INPUT_PREFIX = "batch-judge/input"
_S3_OUTPUT_PREFIX = "batch-judge/output"

_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "batch_bedrock_judge_manifest.json"
_INPUT_PATH = REPO_ROOT / "data" / "raw" / "_batch_bedrock_judge_input.jsonl"

TERMINAL_STATES = {"Completed", "Failed", "Stopped", "Expired"}


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


def _convert_messages_to_bedrock(messages: list[dict]) -> tuple[list[dict], str]:
    """Convert litellm-style messages to Bedrock/Anthropic native format.

    Extracts the system message into a separate top-level system field and
    converts remaining messages so content is a list of content blocks.

    Returns (converted_messages, system_text).
    """
    system_text = ""
    converted = []
    for msg in messages:
        role = msg["role"]
        content_str = msg["content"]
        if role == "system":
            system_text = content_str
        else:
            converted.append({
                "role": role,
                "content": [{"type": "text", "text": content_str}],
            })
    return converted, system_text


def build_batch_input(
    samples: list[dict],
    prompt_lookup: dict[str, dict],
    completed_set: set[tuple[int, str]],
) -> tuple[str, dict]:
    """Build JSONL string and manifest for all pending (sample, Bedrock judge) pairs.

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
        converted_messages, system_text = _convert_messages_to_bedrock(messages)

        model_input: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "messages": converted_messages,
        }
        if system_text:
            model_input["system"] = [{"type": "text", "text": system_text}]

        lines.append(json.dumps({
            "recordId": custom_id,
            "modelInput": model_input,
        }))
        manifest[custom_id] = {
            "behavioral_response_id": sample["id"],
            "subject_model_id": sample["model_id"],
            "prompt_id": sample["prompt_id"],
            "run_number": sample["run_number"],
            "keying": keying,
        }

    return "\n".join(lines), manifest


# ── AWS Bedrock Batch API ─────────────────────────────────────────────────────

def _boto3_session():
    """Create a boto3 session using .env credentials (not the default CLI profile)."""
    import boto3
    return boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _s3_client():
    return _boto3_session().client("s3")


def _bedrock_client():
    return _boto3_session().client("bedrock")


def _s3_bucket() -> str:
    bucket = os.environ.get("BEDROCK_BATCH_S3_BUCKET") or _DEFAULT_S3_BUCKET
    if not bucket:
        raise SystemExit(
            "BEDROCK_BATCH_S3_BUCKET not set. Provide an S3 bucket for Bedrock batch I/O."
        )
    return bucket


def _role_arn() -> str:
    arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN", "")
    if not arn:
        raise SystemExit(
            "BEDROCK_BATCH_ROLE_ARN not set in .env.\n"
            "This IAM role must allow Bedrock to read/write your S3 bucket.\n"
            "See: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html"
        )
    return arn


def submit_job(jsonl: str) -> str:
    """Upload JSONL to S3 and submit Bedrock batch job. Returns job ARN."""
    bucket = _s3_bucket()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    input_filename = f"bedrock_judge_input_{timestamp}.jsonl"
    input_key = f"{_S3_INPUT_PREFIX}/{input_filename}"
    output_prefix = f"{_S3_OUTPUT_PREFIX}/{timestamp}"

    _INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INPUT_PATH.write_text(jsonl, encoding="utf-8")

    s3 = _s3_client()
    print(f"  Uploading {_INPUT_PATH.stat().st_size:,} bytes to s3://{bucket}/{input_key} ...")
    with open(_INPUT_PATH, "rb") as f:
        s3.upload_fileobj(f, bucket, input_key)
    print(f"  Uploaded: s3://{bucket}/{input_key}")

    job_name = f"psycho-llm-judge-opus-{timestamp}"
    bedrock = _bedrock_client()
    print(f"  Submitting batch job (model={_BEDROCK_MODEL_ID}, name={job_name})...")
    response = bedrock.create_model_invocation_job(
        modelId=_BEDROCK_MODEL_ID,
        jobName=job_name,
        roleArn=_role_arn(),
        inputDataConfig={
            "s3InputDataConfig": {
                "s3Uri": f"s3://{bucket}/{input_key}",
                "s3InputFormat": "JSONL",
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{bucket}/{output_prefix}/",
            }
        },
    )
    job_arn = response["jobArn"]
    print(f"  Job ARN: {job_arn}")
    return job_arn


def poll_job(job_arn: str) -> dict:
    """Poll until terminal state. Returns final job description dict."""
    bedrock = _bedrock_client()
    print("\nPolling (every 60s)...")
    while True:
        response = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = response["status"]
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status}")
        if status in TERMINAL_STATES:
            break
        time.sleep(60)

    if status == "Completed":
        print("Job completed successfully.")
    else:
        raise RuntimeError(f"Batch job ended with status={status}")
    return response


def download_results(job_response: dict) -> list[dict]:
    """Download and parse output JSONL from completed Bedrock batch job."""
    s3 = _s3_client()
    bucket = _s3_bucket()

    # Bedrock writes output to the configured output S3 URI prefix.
    # The exact key is: {output_prefix}/{job_id}/{input_filename}.out
    output_s3_uri = job_response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    # Strip s3://{bucket}/ prefix to get the prefix path
    output_prefix = output_s3_uri.replace(f"s3://{bucket}/", "").rstrip("/")

    print(f"  Listing output objects under s3://{bucket}/{output_prefix}/ ...")
    paginator = s3.get_paginator("list_objects_v2")
    output_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".out") or key.endswith(".jsonl"):
                output_keys.append(key)

    if not output_keys:
        raise RuntimeError(
            f"No output files found under s3://{bucket}/{output_prefix}/. "
            "Check that the job completed and output was written."
        )

    rows = []
    for key in output_keys:
        print(f"  Downloading s3://{bucket}/{key} ...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        text = obj["Body"].read().decode("utf-8")
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
        custom_id = row.get("recordId", "")
        meta = manifest.get(custom_id)
        if meta is None:
            print(f"  WARNING: no manifest entry for recordId {custom_id!r}, skipping")
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

        model_output = row.get("modelOutput", {})

        # Check for error in modelOutput
        if "error" in model_output:
            storage.store_judge_rating({
                **base_row,
                "raw_response": None,
                "parse_status": "api_error",
                "error_message": f"Batch API error: {model_output['error']}",
            })
            api_error += 1
            continue

        # Extract text from Anthropic response format
        try:
            content_blocks = model_output["content"]
            raw_text = next(
                block["text"] for block in content_blocks if block.get("type") == "text"
            )
        except (KeyError, IndexError, TypeError, StopIteration) as exc:
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
        description="Bedrock batch judge runner for Phase 3 LLM-as-judge (Claude Opus 4.6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-samples", type=int, default=None, metavar="N",
                        help="Limit to first N samples (for testing). Default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count only; do not submit.")
    parser.add_argument("--ingest", metavar="JOB_ARN",
                        help="Skip submission; ingest from completed job ARN.")
    args = parser.parse_args()

    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        print("WARNING: AWS_ACCESS_KEY_ID not set in .env — relying on IAM role or AWS_PROFILE")

    if not FEW_SHOT_EXAMPLES:
        raise SystemExit(
            "FEW_SHOT_EXAMPLES in pipeline/judge_prompt.py is empty.\n"
            "Author and validate examples first: python -m pipeline.judge_prompt_validate"
        )

    # Load all samples, filter to non-Anthropic subject models
    full_registry = load_model_registry(providers=None)
    provider_map = _build_provider_map(full_registry)

    all_samples = storage.load_behavioral_samples_for_judging(n_samples=args.n_samples)
    samples = [
        s for s in all_samples
        if provider_map.get(s["model_id"], "") != "Anthropic"
    ]

    prompt_lookup = {p["prompt_id"]: p for p in load_behavioral_prompts()}
    completed_set = storage.load_completed_judge_set()

    pending_count = sum(
        1 for s in samples
        if (s["id"], _LITELLM_MODEL_ID) not in completed_set
    )

    print(
        f"[batch_bedrock_judge] {len(samples)} eligible samples "
        f"({pending_count} pending) for {_LITELLM_MODEL_ID}"
    )

    if args.dry_run or pending_count == 0:
        return

    if args.ingest:
        print(f"\nLoading manifest from {_MANIFEST_PATH}")
        manifest = json.loads(_MANIFEST_PATH.read_text())
        bedrock = _bedrock_client()
        job_response = bedrock.get_model_invocation_job(jobIdentifier=args.ingest)
        status = job_response["status"]
        print(f"Job status: {status}")
        if status != "Completed":
            raise SystemExit(f"Job not completed (status={status}), cannot ingest.")
        rows = download_results(job_response)
        ingest(rows, manifest)
        return

    jsonl, manifest = build_batch_input(samples, prompt_lookup, completed_set)
    print(f"  Built {len(manifest)} requests")

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"  Manifest saved to {_MANIFEST_PATH}")

    print("\nSubmitting batch job...")
    job_arn = submit_job(jsonl)

    # Save job ARN to manifest for easy recovery
    manifest_data = json.loads(_MANIFEST_PATH.read_text())
    manifest_data["_job_arn"] = job_arn
    _MANIFEST_PATH.write_text(json.dumps(manifest_data))

    job_response = poll_job(job_arn)
    rows = download_results(job_response)
    ingest(rows, manifest)


if __name__ == "__main__":
    main()
