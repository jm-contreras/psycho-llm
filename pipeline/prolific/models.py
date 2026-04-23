"""SQLite schema and data-access functions for the Prolific rating pipeline.

Uses a separate DB at PROLIFIC_DB_PATH (data/prolific/prolific.db) with WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from pipeline.prolific.config import PROLIFIC_DB_PATH


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS prolific_sessions (
  id                      INTEGER PRIMARY KEY AUTOINCREMENT,
  prolific_pid            TEXT NOT NULL,
  study_id                TEXT NOT NULL,
  session_id              TEXT NOT NULL,
  status                  TEXT NOT NULL DEFAULT 'in_progress',
  items_assigned          TEXT,
  gold_items_assigned     TEXT,
  training_completed      INTEGER DEFAULT 0,
  items_completed         INTEGER DEFAULT 0,
  gold_accuracy           REAL,
  completion_code         TEXT,
  started_at              TEXT,
  completed_at            TEXT,
  UNIQUE(prolific_pid, study_id)
)
"""

_CREATE_RATINGS = """
CREATE TABLE IF NOT EXISTS prolific_ratings (
  id                        INTEGER PRIMARY KEY AUTOINCREMENT,
  prolific_pid              TEXT NOT NULL,
  study_id                  TEXT NOT NULL,
  session_id                TEXT NOT NULL,
  behavioral_response_id    INTEGER NOT NULL,
  prompt_id                 TEXT NOT NULL,
  keying                    TEXT NOT NULL,
  is_gold                   INTEGER NOT NULL DEFAULT 0,
  item_position             INTEGER NOT NULL,
  raw_RE                    INTEGER,
  raw_DE                    INTEGER,
  raw_BO                    INTEGER,
  raw_GU                    INTEGER,
  raw_VB                    INTEGER,
  corrected_RE              INTEGER,
  corrected_DE              INTEGER,
  corrected_BO              INTEGER,
  corrected_GU              INTEGER,
  corrected_VB              INTEGER,
  response_time_seconds     REAL,
  participant_flagged        INTEGER DEFAULT 0,
  gold_accuracy             REAL,
  timestamp                 TEXT,
  UNIQUE(prolific_pid, behavioral_response_id)
)
"""


def get_db() -> sqlite3.Connection:
    """Open (or create) the Prolific SQLite DB, ensure tables exist, and enable WAL mode."""
    PROLIFIC_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PROLIFIC_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_SESSIONS)
    conn.execute(_CREATE_RATINGS)
    conn.commit()
    return conn


# ── Session helpers ───────────────────────────────────────────────────────────

def create_session(
    prolific_pid: str,
    study_id: str,
    session_id: str,
    items: list[int],
    gold_items: list[int],
) -> int:
    """Insert a new session row and return the auto-increment row id.

    items and gold_items are lists of behavioral_response_ids serialised as JSON.
    Raises sqlite3.IntegrityError on duplicate (prolific_pid, study_id).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO prolific_sessions
              (prolific_pid, study_id, session_id, status,
               items_assigned, gold_items_assigned, started_at)
            VALUES (?, ?, ?, 'in_progress', ?, ?, ?)
            """,
            (
                prolific_pid,
                study_id,
                session_id,
                json.dumps(items),
                json.dumps(gold_items),
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_session(prolific_pid: str, study_id: str) -> dict[str, Any] | None:
    """Return the session row as a dict, or None if not found."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM prolific_sessions WHERE prolific_pid=? AND study_id=?",
            (prolific_pid, study_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_session_progress(
    prolific_pid: str,
    study_id: str,
    items_completed: int,
    training_completed: int | None = None,
) -> None:
    """Update items_completed (and optionally training_completed) for a session."""
    conn = get_db()
    try:
        if training_completed is not None:
            conn.execute(
                """
                UPDATE prolific_sessions
                SET items_completed=?, training_completed=?
                WHERE prolific_pid=? AND study_id=?
                """,
                (items_completed, training_completed, prolific_pid, study_id),
            )
        else:
            conn.execute(
                """
                UPDATE prolific_sessions
                SET items_completed=?
                WHERE prolific_pid=? AND study_id=?
                """,
                (items_completed, prolific_pid, study_id),
            )
        conn.commit()
    finally:
        conn.close()


def complete_session(
    prolific_pid: str,
    study_id: str,
    completion_code: str,
    gold_accuracy: float,
) -> None:
    """Mark a session as complete with its completion code and gold accuracy."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            """
            UPDATE prolific_sessions
            SET status='complete', completion_code=?, gold_accuracy=?, completed_at=?
            WHERE prolific_pid=? AND study_id=?
            """,
            (completion_code, gold_accuracy, now, prolific_pid, study_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Rating helpers ────────────────────────────────────────────────────────────

def record_rating(
    prolific_pid: str,
    study_id: str,
    session_id: str,
    behavioral_response_id: int,
    prompt_id: str,
    keying: str,
    is_gold: int,
    item_position: int,
    raw_scores: dict[str, int],
    corrected_scores: dict[str, int],
    response_time_seconds: float,
) -> None:
    """Insert a rating row.

    INSERT OR IGNORE — duplicate (prolific_pid, behavioral_response_id) submissions
    are silently skipped.

    raw_scores and corrected_scores are dicts keyed by factor code (RE/DE/BO/GU/VB).
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO prolific_ratings (
              prolific_pid, study_id, session_id,
              behavioral_response_id, prompt_id, keying, is_gold, item_position,
              raw_RE, raw_DE, raw_BO, raw_GU, raw_VB,
              corrected_RE, corrected_DE, corrected_BO, corrected_GU, corrected_VB,
              response_time_seconds, timestamp
            ) VALUES (
              ?, ?, ?,
              ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?,
              ?, ?
            )
            """,
            (
                prolific_pid, study_id, session_id,
                behavioral_response_id, prompt_id, keying, is_gold, item_position,
                raw_scores.get("RE"), raw_scores.get("DE"),
                raw_scores.get("BO"), raw_scores.get("GU"), raw_scores.get("VB"),
                corrected_scores.get("RE"), corrected_scores.get("DE"),
                corrected_scores.get("BO"), corrected_scores.get("GU"), corrected_scores.get("VB"),
                response_time_seconds, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_item_rating_counts() -> dict[int, int]:
    """Return a dict mapping behavioral_response_id -> count of unflagged ratings.

    Excludes ratings from manually excluded participants (rejected/returned on
    Prolific) so that the assignment priority queue reflects only usable data.
    """
    from pipeline.prolific.config import EXCLUDED_PROLIFIC_PIDS
    conn = get_db()
    try:
        if EXCLUDED_PROLIFIC_PIDS:
            placeholders = ",".join("?" * len(EXCLUDED_PROLIFIC_PIDS))
            rows = conn.execute(
                f"""
                SELECT behavioral_response_id, COUNT(*) AS cnt
                FROM prolific_ratings
                WHERE participant_flagged = 0
                  AND prolific_pid NOT IN ({placeholders})
                GROUP BY behavioral_response_id
                """,
                tuple(EXCLUDED_PROLIFIC_PIDS),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT behavioral_response_id, COUNT(*) AS cnt
                FROM prolific_ratings
                WHERE participant_flagged = 0
                GROUP BY behavioral_response_id
                """
            ).fetchall()
        return {row["behavioral_response_id"]: row["cnt"] for row in rows}
    finally:
        conn.close()


def get_ratings_for_participant(prolific_pid: str) -> list[dict[str, Any]]:
    """Return all rating rows for a given participant as a list of dicts."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM prolific_ratings WHERE prolific_pid=? ORDER BY item_position",
            (prolific_pid,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
