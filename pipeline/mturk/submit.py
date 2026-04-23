"""Create MTurk HITs for the behavioral rating sample."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone

from pipeline.behavioral_loader import BEHAVIORAL_PROMPTS
from pipeline.judge_prompt import FACTOR_ORDER
from pipeline.storage import DB_PATH
from pipeline.mturk.config import (
    MTURK_SEED,
    MANIFEST_PATH,
    SAMPLE_PATH,
    GOLD_ITEMS_PATH,
    REWARD_PER_HIT,
    ASSIGNMENT_DURATION_SECONDS,
    HIT_LIFETIME_SECONDS,
    AUTO_APPROVAL_DELAY_SECONDS,
    MAX_ASSIGNMENTS_INITIAL,
    GOLD_RATE,
    QUALIFICATION_THRESHOLD,
)
from pipeline.mturk.gold_standards import load_gold_items
from pipeline.mturk.hit_template import generate_keying, render_hit_html
from pipeline.mturk.qualification import get_or_create_qualification

_PROMPT_BY_ID: dict[str, dict] = {p["prompt_id"]: p for p in BEHAVIORAL_PROMPTS}


def submit_hits(
    client,
    dry_run: bool = False,
    batch_size: int | None = None,
) -> list[dict]:
    """Create HITs for all sample + gold items.

    Args:
        client:     boto3 MTurk client from get_mturk_client().
        dry_run:    If True, print summary and cost estimate, make no API calls.
        batch_size: If set, only submit the first N items.

    Returns list of manifest entries (hit_id, response_id, keying, is_gold, ...).
    """
    import random
    rng = random.Random(MTURK_SEED)

    # Load sample
    if not SAMPLE_PATH.exists():
        print("ERROR: sample.json not found. Run `python -m pipeline.mturk sample` first.", file=sys.stderr)
        return []
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        sample_data = json.load(f)
    sample_items = sample_data.get("items", [])

    # Load monitoring gold items only (qualification items are NOT submitted as HITs)
    all_gold_items = load_gold_items()
    monitoring_gold = [g for g in all_gold_items if g.get("purpose", "monitoring") == "monitoring"]

    # Resample monitoring gold items to hit the target GOLD_RATE.
    # Each sample item counts as 1 HIT; we need enough gold copies so that
    # n_gold / (n_sample + n_gold) ≈ GOLD_RATE.
    # Solving: n_gold = n_sample * GOLD_RATE / (1 - GOLD_RATE)
    n_sample = len(sample_items)
    n_gold_target = round(n_sample * GOLD_RATE / (1.0 - GOLD_RATE))
    # Resample monitoring gold with replacement (seeded) to reach the target count
    gold_copies: list[dict] = []
    if monitoring_gold:
        gold_rng = random.Random(MTURK_SEED)
        for _ in range(n_gold_target):
            gold_copies.append(gold_rng.choice(monitoring_gold))

    # Build merged item list; add copy_index to distinguish resampled duplicates
    all_items: list[dict] = []
    for item in sample_items:
        all_items.append({**item, "is_gold": False, "gold_copy_index": None})

    # Track how many times each behavioral_response_id has appeared as gold
    gold_copy_counter: dict[int, int] = {}
    for g in gold_copies:
        rid = g["behavioral_response_id"]
        gold_copy_counter[rid] = gold_copy_counter.get(rid, 0) + 1
        copy_idx = gold_copy_counter[rid]
        all_items.append(
            {
                "behavioral_response_id": rid,
                "prompt_id": g.get("prompt_id", ""),
                "dimension_code": g.get("dimension_code", ""),
                "model_id": g.get("model_id") or g.get("subject_model_id", ""),
                "is_gold": True,
                "score_bin": "gold",
                "model_family": "",
                "consensus_score": None,
                "run_number": None,
                "gold_copy_index": copy_idx,  # 1-indexed; >1 means a duplicate
            }
        )

    # Shuffle merged list
    rng.shuffle(all_items)

    # Apply batch_size cap
    if batch_size is not None:
        all_items = all_items[:batch_size]

    n_hits = len(all_items)
    n_gold_actual = sum(1 for it in all_items if it.get("is_gold"))
    gold_rate_actual = n_gold_actual / n_hits if n_hits > 0 else 0.0
    total_cost = n_hits * MAX_ASSIGNMENTS_INITIAL * float(REWARD_PER_HIT) * 1.20
    print(
        f"HITs to submit: {n_hits}  "
        f"(sample={n_sample}, gold={n_gold_actual} [{gold_rate_actual:.1%}], "
        f"target={GOLD_RATE:.1%}, batch_cap={batch_size})",
        file=sys.stderr,
    )
    print(
        f"Estimated cost: ${total_cost:.2f} "
        f"({n_hits} × {MAX_ASSIGNMENTS_INITIAL} raters × ${REWARD_PER_HIT} × 1.20 fee)",
        file=sys.stderr,
    )

    if dry_run:
        print("dry_run=True — no HITs submitted.", file=sys.stderr)
        return []

    # Get qualification type
    qual_id = get_or_create_qualification(client)

    # Load existing manifest to avoid re-submitting.
    # For regular (non-gold) items, deduplicate by behavioral_response_id.
    # For gold items, deduplicate by (behavioral_response_id, gold_copy_index) so that
    # resampled copies of the same gold response each get their own HIT.
    existing_manifest = _load_manifest()
    already_submitted_regular: set[int] = set()
    already_submitted_gold: set[tuple] = set()
    for entry in existing_manifest:
        if entry.get("is_gold"):
            already_submitted_gold.add(
                (entry["behavioral_response_id"], entry.get("gold_copy_index"))
            )
        else:
            already_submitted_regular.add(entry["behavioral_response_id"])

    new_entries: list[dict] = []

    for item in all_items:
        rid = item["behavioral_response_id"]
        is_gold = bool(item.get("is_gold", False))
        gold_copy_index = item.get("gold_copy_index")

        if is_gold:
            dedup_key = (rid, gold_copy_index)
            if dedup_key in already_submitted_gold:
                print(f"  Skipping gold {rid} copy={gold_copy_index} (already in manifest)", file=sys.stderr)
                continue
        else:
            if rid in already_submitted_regular:
                print(f"  Skipping {rid} (already in manifest)", file=sys.stderr)
                continue

        # Load response text and prompt data from DB
        response_text, prompt_id = _load_response(rid)
        if response_text is None:
            print(f"  WARNING: response {rid} not found in DB, skipping.", file=sys.stderr)
            continue

        prompt_data = _PROMPT_BY_ID.get(prompt_id or item.get("prompt_id", ""))
        if prompt_data is None:
            print(f"  WARNING: prompt {prompt_id!r} not found, skipping response {rid}.", file=sys.stderr)
            continue

        keying = generate_keying(rid)

        html = render_hit_html(
            response_text=response_text,
            prompt_data=prompt_data,
            keying=keying,
            response_id=rid,
            is_gold=is_gold,
        )

        annotation = json.dumps(
            {
                "response_id": rid,
                "keying": keying,
                "is_gold": is_gold,
                "gold_copy_index": gold_copy_index,
                "prompt_id": prompt_id or item.get("prompt_id"),
                "dimension_code": item.get("dimension_code"),
            }
        )

        try:
            resp = client.create_hit(
                Title="Rate an AI Assistant's Response (5 quick ratings)",
                Description=(
                    "Read a short conversation and rate the AI assistant's response on 5 behavioral "
                    "dimensions using a 1-5 scale. Takes 3-5 minutes."
                ),
                Keywords="AI, rating, behavior, NLP, research",
                Reward=REWARD_PER_HIT,
                MaxAssignments=MAX_ASSIGNMENTS_INITIAL,
                LifetimeInSeconds=HIT_LIFETIME_SECONDS,
                AssignmentDurationInSeconds=ASSIGNMENT_DURATION_SECONDS,
                AutoApprovalDelayInSeconds=AUTO_APPROVAL_DELAY_SECONDS,
                Question=html,
                RequesterAnnotation=annotation,
                QualificationRequirements=[
                    {
                        "QualificationTypeId": qual_id,
                        "Comparator": "GreaterThanOrEqualTo",
                        "IntegerValues": [QUALIFICATION_THRESHOLD],
                        "RequiredToPreview": True,
                    }
                ],
            )
            hit_id = resp["HIT"]["HITId"]
            entry = {
                "hit_id": hit_id,
                "behavioral_response_id": rid,
                "prompt_id": prompt_id or item.get("prompt_id"),
                "dimension_code": item.get("dimension_code"),
                "model_id": item.get("model_id"),
                "keying": keying,
                "is_gold": is_gold,
                "gold_copy_index": gold_copy_index,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "status": "open",
            }
            new_entries.append(entry)
            print(f"  Created HIT {hit_id} for response {rid}", file=sys.stderr)

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR creating HIT for response {rid}: {exc}", file=sys.stderr)

    # Append to manifest
    _append_manifest(new_entries)
    print(f"\nSubmitted {len(new_entries)} HITs. Manifest: {MANIFEST_PATH}", file=sys.stderr)
    return new_entries


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_response(behavioral_response_id: int) -> tuple[str | None, str | None]:
    """Return (raw_response, prompt_id) for a behavioral_response_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT raw_response, prompt_id FROM behavioral_responses WHERE id = ?",
            (behavioral_response_id,),
        ).fetchone()
        if row:
            return row["raw_response"], row["prompt_id"]
        return None, None
    finally:
        conn.close()


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _append_manifest(new_entries: list[dict]) -> None:
    """Append new entries to the manifest JSON file (append-safe)."""
    existing = _load_manifest()
    existing.extend(new_entries)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
