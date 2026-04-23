"""Mark behavioral_response_ids that need a 3rd rater (tiebreak).

Functions:
  get_tiebreak_items()      — find items with rater disagreement.
  mark_tiebreak_items(ids)  — write them to prolific_tiebreak table.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pipeline.prolific.collect import find_disagreements
from pipeline.prolific.models import get_db

# ── Schema (created on first use) ─────────────────────────────────────────────

_CREATE_TIEBREAK = """
CREATE TABLE IF NOT EXISTS prolific_tiebreak (
  behavioral_response_id  INTEGER PRIMARY KEY,
  marked_at               TEXT NOT NULL
)
"""


def _ensure_tiebreak_table() -> None:
    conn = get_db()
    try:
        conn.execute(_CREATE_TIEBREAK)
        conn.commit()
    finally:
        conn.close()


# ── Public ────────────────────────────────────────────────────────────────────

def get_tiebreak_items() -> list[int]:
    """Return behavioral_response_ids that need a 3rd rater.

    Delegates to find_disagreements() which checks:
      - exactly 2 unflagged ratings
      - spread >= DISAGREEMENT_THRESHOLD on any factor
    """
    return find_disagreements()


def mark_tiebreak_items(item_ids: list[int]) -> int:
    """Insert item_ids into prolific_tiebreak so assignment.py gives them priority.

    Uses INSERT OR IGNORE — already-marked items are silently skipped.
    Returns the count of newly inserted rows.
    """
    if not item_ids:
        return 0

    _ensure_tiebreak_table()
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    inserted = 0
    try:
        for rid in item_ids:
            cur = conn.execute(
                "INSERT OR IGNORE INTO prolific_tiebreak (behavioral_response_id, marked_at) "
                "VALUES (?, ?)",
                (rid, now),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()

    return inserted
