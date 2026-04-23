"""Dynamic item assignment for the Prolific rating pipeline.

Selects sample items (prioritising under-rated items) and one monitoring gold
item per session, then builds a randomised presentation order.
"""

from __future__ import annotations

import json
import random
import sqlite3

from pipeline.behavioral_loader import BEHAVIORAL_PROMPTS
from pipeline.mturk.gold_standards import load_gold_items
from pipeline.mturk.hit_template import generate_keying
from pipeline.storage import DB_PATH
from pipeline.prolific.config import (
    GOLD_ITEMS_PATH,
    PROLIFIC_SEED,
    SAMPLE_ITEMS_PER_SESSION,
    SAMPLE_PATH,
    TARGET_RATINGS_PER_ITEM,
)
from pipeline.prolific.models import get_db, get_item_rating_counts


# ── Public: session assignment ────────────────────────────────────────────────

def assign_items_for_session(
    prolific_pid: str,
    study_id: str,
) -> tuple[list[dict], list[dict]]:
    """Select sample and gold items for a new participant session.

    Priority for sample items:
      1. Items with 0 unflagged ratings (most needed)
      2. Items with exactly 1 unflagged rating (one more needed)
      3. Items with 2+ ratings (least needed)

    Gold selection is deterministic: seeded by (PROLIFIC_SEED, prolific_pid).

    Returns:
        (sample_items, gold_items) where each element is a list of enriched
        item dicts (see _enrich_item for keys).
    """
    # Load the sample pool (pre-filtered behavioral_response_ids)
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample pool not found at {SAMPLE_PATH}. "
            "Run the MTurk sample-selection step first."
        )
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        raw_pool: list[dict] = json.load(f)
        # Support both flat list and {"items": [...]} format
        if isinstance(raw_pool, dict) and "items" in raw_pool:
            raw_pool = raw_pool["items"]

    # Get current rating counts and in-progress assignments to exclude
    rating_counts = get_item_rating_counts()
    in_progress_ids = _get_in_progress_assigned_ids()

    # Identify under-rated items (need at least one more rating) across the full pool,
    # ignoring the in-progress lock. If an under-rated item is locked by a ghost session
    # (Prolific shows no active participants) it would never get assigned otherwise.
    under_rated_ids = {
        item["behavioral_response_id"]
        for item in raw_pool
        if rating_counts.get(item["behavioral_response_id"], 0) < TARGET_RATINGS_PER_ITEM
    }
    # Under-rated items bypass the in-progress lock so ghost sessions don't block them.
    # For all other items, respect the lock to avoid double-assignment.
    available = [
        item for item in raw_pool
        if item["behavioral_response_id"] not in in_progress_ids
        or item["behavioral_response_id"] in under_rated_ids
    ]

    # Sort by rating count (ascending) so under-rated items come first;
    # within each count tier shuffle deterministically per participant
    rng = random.Random(f"{PROLIFIC_SEED}:{prolific_pid}:sample")
    tier0 = [i for i in available if rating_counts.get(i["behavioral_response_id"], 0) == 0]
    tier1 = [i for i in available if rating_counts.get(i["behavioral_response_id"], 0) == 1]
    tier2 = [i for i in available if rating_counts.get(i["behavioral_response_id"], 0) >= 2]

    rng.shuffle(tier0)
    rng.shuffle(tier1)
    rng.shuffle(tier2)

    ordered = tier0 + tier1 + tier2

    # Select items ensuring no two items share the same prompt_id within a session
    seen_prompts: set[str] = set()
    selected: list[dict] = []
    for item in ordered:
        pid = item.get("prompt_id", "")
        if pid not in seen_prompts:
            selected.append(item)
            seen_prompts.add(pid)
        if len(selected) >= SAMPLE_ITEMS_PER_SESSION:
            break

    sample_items = [_enrich_item(item) for item in selected]

    # Select 1 monitoring gold item seeded by (PROLIFIC_SEED, prolific_pid)
    gold_pool = load_gold_items()
    gold_rng = random.Random(f"{PROLIFIC_SEED}:{prolific_pid}:gold")
    gold_rng.shuffle(gold_pool)
    gold_item = gold_pool[0] if gold_pool else None
    gold_items = [_enrich_item(gold_item, is_gold=True)] if gold_item else []

    return sample_items, gold_items


def build_session_order(
    sample_items: list[dict],
    gold_items: list[dict],
    prolific_pid: str,
) -> list[dict]:
    """Merge sample + gold items, placing gold at a random interior position.

    The gold item is never placed first or last. Seeded by (PROLIFIC_SEED, prolific_pid).

    Returns a list of item dicts, each containing:
      behavioral_response_id, prompt_id, dimension_code, is_gold, keying
    """
    rng = random.Random(f"{PROLIFIC_SEED}:{prolific_pid}:order")

    # Start with shuffled sample items
    items = list(sample_items)
    rng.shuffle(items)

    # Insert each gold item at a random interior position (1 .. len-1)
    for gold in gold_items:
        if len(items) >= 2:
            pos = rng.randint(1, len(items) - 1)
        else:
            pos = len(items)
        items.insert(pos, gold)

    # Return slim dicts with only the fields the view layer needs
    return [
        {
            "behavioral_response_id": item["behavioral_response_id"],
            "prompt_id": item["prompt_id"],
            "dimension_code": item["dimension_code"],
            "is_gold": item.get("is_gold", False),
            "keying": item["keying"],
        }
        for item in items
    ]


def get_training_items() -> list[dict]:
    """Return the first 2 gold items enriched with response_text, prompt_data, and feedback.

    Loads qualification gold items from GOLD_ITEMS_PATH. Each returned dict has:
      behavioral_response_id, prompt_id, dimension_code, keying,
      response_text, prompt_data, ground_truth, feedback (None for non-gold items).
    """
    gold_items = load_gold_items()
    training_raw = gold_items[:2]

    result = []
    for item in training_raw:
        rid = item["behavioral_response_id"]
        raw_response, prompt_id = _load_response(rid)

        # Fall back to the item's own prompt_id if the DB lookup fails
        pid = prompt_id or item.get("prompt_id", "")
        prompt_data = _prompt_by_id(pid)

        result.append({
            "behavioral_response_id": rid,
            "prompt_id": pid,
            "dimension_code": item.get("dimension_code", ""),
            "keying": generate_keying(rid),
            "response_text": raw_response or "",
            "prompt_data": prompt_data,
            "ground_truth": item.get("ground_truth"),
            "feedback": item.get("feedback"),
        })

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _enrich_item(item: dict, is_gold: bool = False) -> dict:
    """Add keying and is_gold flag to a pool item dict."""
    rid = item["behavioral_response_id"]
    return {
        **item,
        "is_gold": is_gold,
        "keying": generate_keying(rid),
    }


def _get_in_progress_assigned_ids() -> set[int]:
    """Return behavioral_response_ids assigned in non-expired in-progress sessions.

    Sessions older than SESSION_TIMEOUT_MINUTES are treated as abandoned and
    do not block item assignment.
    """
    from pipeline.prolific.config import SESSION_TIMEOUT_MINUTES
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    cutoff_iso = cutoff.isoformat()

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT items_assigned, gold_items_assigned "
            "FROM prolific_sessions "
            "WHERE status = 'in_progress' AND started_at > ?",
            (cutoff_iso,),
        ).fetchall()
    finally:
        conn.close()

    assigned: set[int] = set()
    for row in rows:
        for col in ("items_assigned", "gold_items_assigned"):
            raw = row[col]
            if not raw:
                continue
            try:
                items = json.loads(raw)
                if not isinstance(items, list):
                    continue
                for item in items:
                    # items_assigned stores dicts with behavioral_response_id key
                    if isinstance(item, dict):
                        rid = item.get("behavioral_response_id")
                        if isinstance(rid, int):
                            assigned.add(rid)
                    elif isinstance(item, int):
                        assigned.add(item)
            except (json.JSONDecodeError, TypeError):
                pass
    return assigned


def _load_response(behavioral_response_id: int) -> tuple[str | None, str | None]:
    """Load (raw_response, prompt_id) from behavioral_responses for a given id."""
    conn = sqlite3.connect(str(DB_PATH))
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


def _prompt_by_id(prompt_id: str) -> dict:
    """Return the BEHAVIORAL_PROMPTS entry for prompt_id, or an empty dict."""
    prompt_map = {p["prompt_id"]: p for p in BEHAVIORAL_PROMPTS}
    return prompt_map.get(prompt_id, {})
