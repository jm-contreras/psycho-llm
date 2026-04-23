"""
Persist response rows to SQLite (authoritative) and CSV (for easy inspection).

Both outputs go to data/raw/responses.db and data/raw/responses.csv.

Idempotency: already_completed(model_id, item_id, run_number) returns True if a
'success' row already exists for that triple. Failed rows are re-attempted and
overwritten (via INSERT OR REPLACE) on next success.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

DB_PATH = RAW_DIR / "responses.db"
CSV_PATH = RAW_DIR / "responses.csv"

COLUMNS = [
    "model_id",
    "item_id",
    "dimension",
    "item_type",
    "keying",
    "run_number",
    "text_scoring_method",
    "raw_response",
    "reasoning_content",
    "parsed_score",
    "logprob_score",
    "logprob_token_logprob",
    "logprob_vector",
    "logprob_match_token",
    "logprob_available",
    "option_order",
    "status",
    "error_message",
    "timestamp",
]

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS responses (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  {",".join(f"  {c} TEXT" if c not in ("run_number", "logprob_available") else f"  {c} INTEGER" for c in COLUMNS)},
  parsed_score_num    REAL GENERATED ALWAYS AS (CAST(parsed_score AS REAL)) VIRTUAL,
  logprob_score_num   REAL GENERATED ALWAYS AS (CAST(logprob_score AS REAL)) VIRTUAL,
  UNIQUE(model_id, item_id, run_number)
)
"""

# Simpler version without generated columns for broader SQLite compatibility
_CREATE_TABLE_SIMPLE = """
CREATE TABLE IF NOT EXISTS responses (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id            TEXT NOT NULL,
  item_id             TEXT NOT NULL,
  dimension           TEXT,
  item_type           TEXT,
  keying              TEXT,
  run_number          INTEGER,
  text_scoring_method TEXT,
  raw_response        TEXT,
  reasoning_content   TEXT,
  parsed_score        REAL,
  logprob_score       REAL,
  logprob_available   INTEGER,
  status              TEXT,
  error_message       TEXT,
  timestamp           TEXT,
  UNIQUE(model_id, item_id, run_number)
)
"""


_CREATE_BEHAVIORAL_TABLE = """
CREATE TABLE IF NOT EXISTS behavioral_responses (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id          TEXT NOT NULL,
  prompt_id         TEXT NOT NULL,
  dimension         TEXT NOT NULL,
  dimension_code    TEXT NOT NULL,
  is_two_turn       INTEGER NOT NULL DEFAULT 0,
  run_number        INTEGER NOT NULL,
  raw_response      TEXT,
  reasoning_content TEXT,
  status            TEXT NOT NULL,
  error_message     TEXT,
  timestamp         TEXT,
  UNIQUE(model_id, prompt_id, run_number)
)
"""

BEHAVIORAL_COLUMNS = [
    "model_id",
    "prompt_id",
    "dimension",
    "dimension_code",
    "is_two_turn",
    "run_number",
    "raw_response",
    "reasoning_content",
    "status",
    "error_message",
    "timestamp",
]


def _get_conn() -> sqlite3.Connection:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE_SIMPLE)
    # Add new columns to existing DBs without breaking old ones
    for col, typedef in [("logprob_token_logprob", "REAL"), ("logprob_vector", "TEXT"), ("logprob_match_token", "TEXT"), ("option_order", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE responses ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.execute(_CREATE_BEHAVIORAL_TABLE)
    conn.execute(_CREATE_JUDGE_TABLE)
    conn.commit()
    return conn


def already_completed(model_id: str, item_id: str, run_number: int) -> bool:
    """Return True if a 'success' row exists for this (model_id, item_id, run_number)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM responses WHERE model_id=? AND item_id=? AND run_number=? AND status='success'",
            (model_id, item_id, run_number),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def load_completed_set() -> set[tuple[str, str, int]]:
    """
    Load all successful (model_id, item_id, run_number) triples into a set.
    Used by the async runner to batch-check completions at startup instead
    of hitting SQLite per call.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT model_id, item_id, run_number FROM responses WHERE status='success'"
        ).fetchall()
        return {(r["model_id"], r["item_id"], r["run_number"]) for r in rows}
    finally:
        conn.close()


def load_group_completed_set(group_map: dict[str, str]) -> dict[str, set[tuple[str, int]]]:
    """
    Return a mapping of canonical_model_id -> set of (item_id, run_number) that
    are already complete by ANY model in the same group.

    group_map: litellm_model_id -> canonical_model_id (from registry group_as field).
    Models with no group_as map to themselves.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT model_id, item_id, run_number FROM responses WHERE status='success'"
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, set[tuple[str, int]]] = {}
    for r in rows:
        canonical = group_map.get(r["model_id"], r["model_id"])
        if canonical not in result:
            result[canonical] = set()
        result[canonical].add((r["item_id"], r["run_number"]))
    return result


def store(row: dict) -> None:
    """
    Insert or replace a response row in SQLite and append to CSV.

    Uses INSERT OR REPLACE so failed rows from previous runs are overwritten
    when a successful retry comes in.
    """
    # Coerce numeric fields
    row = dict(row)
    for field in ("parsed_score", "logprob_score"):
        v = row.get(field)
        if v is not None:
            try:
                row[field] = float(v)
            except (TypeError, ValueError):
                row[field] = None
    for field in ("run_number", "logprob_available"):
        v = row.get(field)
        if v is not None:
            try:
                row[field] = int(v)
            except (TypeError, ValueError):
                row[field] = None

    conn = _get_conn()
    try:
        placeholders = ", ".join("?" for _ in COLUMNS)
        cols = ", ".join(COLUMNS)
        values = [row.get(c) for c in COLUMNS]
        conn.execute(
            f"INSERT OR REPLACE INTO responses ({cols}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()

    _append_csv(row)


def _append_csv(row: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def rename_model_ids(mapping: dict[str, str]) -> dict[str, int]:
    """
    Rename model_ids in the DB (and rebuild the CSV) for models that changed
    access route. Typical use: models moved from Bedrock to Azure.

    mapping: {old_litellm_model_id: new_litellm_model_id}
    Returns: {old_id: rows_updated} for each entry in mapping.

    Example:
        from pipeline.storage import rename_model_ids
        rename_model_ids({
            "bedrock/openai.gpt-oss-120b-1:0":                        "azure/gpt-oss-120b",
            "bedrock/deepseek.v3.2":                                   "azure/deepseek-v3.2",
            "bedrock/converse/us.deepseek.r1-v1:0":                   "azure/deepseek-r1",
            "bedrock/moonshotai.kimi-k2.5":                           "azure/kimi-k2.5",
            "bedrock/converse/mistral.mistral-large-3-675b-instruct": "azure/mistral-large-3",
            "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0":      "azure/llama-4-maverick",
        })
    """
    conn = _get_conn()
    updated: dict[str, int] = {}
    try:
        for old_id, new_id in mapping.items():
            conn.execute(
                "UPDATE responses SET model_id=? WHERE model_id=?",
                (new_id, old_id),
            )
            updated[old_id] = conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    _rebuild_csv()
    return updated


def delete_scenario_responses(statuses: list[str] | None = None) -> int:
    """
    Delete scenario item rows from the DB and rebuild the CSV.
    Pass statuses to restrict deletion (e.g. ['success']). Default: all statuses.

    Returns: number of rows deleted.

    Example:
        from pipeline.storage import delete_scenario_responses
        delete_scenario_responses()                 # all scenario rows
        delete_scenario_responses(['success'])       # only successful ones
    """
    conn = _get_conn()
    try:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            conn.execute(
                f"DELETE FROM responses WHERE item_type='scenario' AND status IN ({placeholders})",
                statuses,
            )
        else:
            conn.execute("DELETE FROM responses WHERE item_type='scenario'")
        deleted = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    _rebuild_csv()
    return deleted


def delete_errors(statuses: list[str] | None = None) -> int:
    """
    Delete rows with non-success statuses from the DB and rebuild the CSV.
    Defaults to deleting 'api_error' and 'parse_error' rows; pass statuses
    explicitly to include 'refusal' or any other value.

    Returns: number of rows deleted.

    Example:
        from pipeline.storage import delete_errors
        delete_errors()                                    # api_error + parse_error
        delete_errors(["api_error", "parse_error", "refusal"])  # all failures
    """
    if statuses is None:
        statuses = ["api_error", "parse_error"]
    conn = _get_conn()
    try:
        placeholders = ",".join("?" for _ in statuses)
        conn.execute(
            f"DELETE FROM responses WHERE status IN ({placeholders})",
            statuses,
        )
        deleted = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    _rebuild_csv()
    return deleted


def _rebuild_csv() -> None:
    """Rewrite the CSV from the current DB state (used after any DB mutation)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(COLUMNS)} FROM responses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def migrate_scenario_letter_answers() -> dict[str, int]:
    """
    One-time migration: rewrite raw_response for scenario rows that used the old
    letter-based format ({"answer": "a"}) to the new ordinal format ({"answer": 1}).

    This normalises existing smoke-test rows so they are consistent with the prompt
    change in api_client.py (scenario items now request 1–4 instead of a–d).

    Only raw_response is changed. parsed_score is already the option's numeric score
    and requires no update. logprob_* columns for old letter-format rows remain as-is
    (they will be superseded when those rows are re-collected).

    Returns: {"updated": N, "skipped": N} counts.
    """
    import json, re as _re

    _LETTER_TO_ORDINAL = {"a": 1, "b": 2, "c": 3, "d": 4}
    _LETTER_RE = _re.compile(r'\{[^{}]*"answer"[^{}]*\}')

    conn = _get_conn()
    updated = skipped = 0
    try:
        # Only touch rows where JSON parsing succeeded ('structured'). Rows parsed by
        # regex have raw_response set to the full model output (e.g. a reasoning trace),
        # which may contain {"answer": ...} patterns in the thinking text that must not
        # be modified.
        rows = conn.execute(
            "SELECT id, raw_response FROM responses "
            "WHERE item_type='scenario' AND status='success' AND text_scoring_method='structured'"
        ).fetchall()

        for row in rows:
            raw = row["raw_response"] or ""
            # Find the last {"answer": ...} object in the response
            matches = list(_LETTER_RE.finditer(raw))
            if not matches:
                skipped += 1
                continue
            last_match = matches[-1]
            try:
                obj = json.loads(last_match.group())
            except json.JSONDecodeError:
                skipped += 1
                continue

            answer = obj.get("answer")
            # Already an integer ordinal — nothing to do
            if isinstance(answer, int) and 1 <= answer <= 4:
                skipped += 1
                continue
            # Letter format — convert
            if isinstance(answer, str):
                letter = answer.strip().lower()
                ordinal = _LETTER_TO_ORDINAL.get(letter)
                if ordinal is None:
                    skipped += 1
                    continue
                new_raw = raw[: last_match.start()] + json.dumps({"answer": ordinal}) + raw[last_match.end() :]
                conn.execute("UPDATE responses SET raw_response=? WHERE id=?", (new_raw, row["id"]))
                updated += 1
            else:
                skipped += 1

        conn.commit()
    finally:
        conn.close()

    _rebuild_csv()
    return {"updated": updated, "skipped": skipped}


def flag_suspect_regex_parses(
    model_ids: list[str] | None = None,
    max_response_len: int = 500,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Retroactively flag bad regex-parsed rows as parse_error so they will be
    re-collected on the next runner invocation.

    A row is suspect if it meets ANY of:
      1. text_scoring_method='regex' AND length(raw_response) > max_response_len
         (long responses — model ignored the JSON instruction and rambled; the
         regex digit almost certainly came from the scale description, not the answer)
      2. text_scoring_method='regex' AND model_id is in model_ids
         (model-level flag for models whose short regex rows are also known-bad,
         e.g. Nova Pro where the first-digit-from-scale issue affects shorter responses)

    Only rows currently marked status='success' are touched; already-failed rows
    are left alone.

    Args:
        model_ids:        litellm_model_ids to flag ALL regex rows for, regardless
                          of response length. Pass None to skip model-level flagging.
        max_response_len: Flag regex rows longer than this many chars. Default 500.
        dry_run:          If True, print counts without writing to DB.

    Returns:
        {"length_flagged": N, "model_flagged": N, "total": N}

    Example — flag everything for Nova Pro + MiniMax, plus long rows everywhere:
        from pipeline.storage import flag_suspect_regex_parses
        flag_suspect_regex_parses(
            model_ids=[
                "bedrock/amazon.nova-pro-v1:0",
                "bedrock/converse/minimax.minimax-m2.5",
            ]
        )
    """
    conn = _get_conn()
    length_flagged = model_flagged = 0
    try:
        # 1. Long-response regex rows (all models)
        rows = conn.execute(
            "SELECT id FROM responses "
            "WHERE status='success' AND text_scoring_method='regex' "
            "AND length(raw_response) > ?",
            (max_response_len,),
        ).fetchall()
        length_ids = [r["id"] for r in rows]
        length_flagged = len(length_ids)

        # 2. All regex rows for specified models
        model_ids_extra: list[int] = []
        if model_ids:
            placeholders = ",".join("?" for _ in model_ids)
            rows2 = conn.execute(
                f"SELECT id FROM responses "
                f"WHERE status='success' AND text_scoring_method='regex' "
                f"AND model_id IN ({placeholders})",
                model_ids,
            ).fetchall()
            model_ids_extra = [r["id"] for r in rows2]
            model_flagged = len(model_ids_extra)

        all_ids = list(set(length_ids) | set(model_ids_extra))
        total = len(all_ids)

        print(f"flag_suspect_regex_parses: {length_flagged} long-response rows, "
              f"{model_flagged} model-targeted rows, {total} unique rows to flag")
        if dry_run:
            print("(dry_run=True — no changes written)")
            return {"length_flagged": length_flagged, "model_flagged": model_flagged, "total": total}

        if all_ids:
            placeholders = ",".join("?" for _ in all_ids)
            conn.execute(
                f"UPDATE responses SET status='parse_error', "
                f"error_message='Suspect regex parse: flagged by flag_suspect_regex_parses()' "
                f"WHERE id IN ({placeholders})",
                all_ids,
            )
            conn.commit()
    finally:
        conn.close()

    if all_ids:
        _rebuild_csv()
    return {"length_flagged": length_flagged, "model_flagged": model_flagged, "total": total}


def show_errors(
    model_id: str | None = None,
    statuses: list[str] | None = None,
    limit: int = 20,
    truncate: int = 200,
) -> None:
    """
    Print error rows from the DB for quick diagnosis.

    Args:
        model_id:  Filter to a specific model (substring match). None = all models.
        statuses:  Which statuses to show. Defaults to ['api_error', 'parse_error'].
        limit:     Max rows to print (most recent first).
        truncate:  Truncate error_message and raw_response to this many chars.

    Example:
        from pipeline.storage import show_errors
        show_errors()                              # last 20 errors, all models
        show_errors("gpt")                         # filter to GPT-OSS
        show_errors(statuses=["parse_error"])      # only parse errors
        show_errors(limit=5, truncate=400)         # wider output
    """
    if statuses is None:
        statuses = ["api_error", "parse_error"]

    conn = _get_conn()
    try:
        where_clauses = [f"status IN ({','.join('?' for _ in statuses)})"]
        params: list = list(statuses)
        if model_id:
            where_clauses.append("model_id LIKE ?")
            params.append(f"%{model_id}%")
        params.append(limit)
        rows = conn.execute(
            f"SELECT model_id, item_id, run_number, status, error_message, raw_response "
            f"FROM responses WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No matching error rows found.")
        return

    def trunc(s, n):
        if s is None:
            return "(null)"
        s = str(s).replace("\n", " ")
        return s[:n] + "…" if len(s) > n else s

    # Group by error message
    from collections import Counter
    counts: Counter = Counter()
    examples: dict = {}
    for r in rows:
        detail = r["error_message"] or r["raw_response"] or "(null)"
        key = trunc(detail, truncate)
        counts[key] += 1
        if key not in examples:
            examples[key] = (r["model_id"], r["status"])

    print(f"\n{'COUNT':>6}  {'STATUS':<12}  ERROR / RESPONSE")
    print("-" * 100)
    for msg, count in counts.most_common():
        model_id, status = examples[msg]
        print(f"{count:>6}  {status:<12}  {msg}")
    print(f"\n({sum(counts.values())} total rows, {len(counts)} unique messages)")


def count_by_status(model_id: str, item_ids: list[str], n_runs: int) -> dict:
    """
    Return status counts for a given model over the specified items and run range.
    Used by runner.py to build the summary table.
    """
    if not item_ids:
        return {}
    conn = _get_conn()
    try:
        placeholders = ",".join("?" for _ in item_ids)
        rows = conn.execute(
            f"""
            SELECT status, COUNT(*) as cnt
            FROM responses
            WHERE model_id=?
              AND item_id IN ({placeholders})
              AND run_number <= ?
            GROUP BY status
            """,
            [model_id, *item_ids, n_runs],
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}

        # Also count logprob_available = 1
        lp_row = conn.execute(
            f"""
            SELECT COUNT(*) as cnt
            FROM responses
            WHERE model_id=?
              AND item_id IN ({placeholders})
              AND run_number <= ?
              AND logprob_available = 1
            """,
            [model_id, *item_ids, n_runs],
        ).fetchone()
        counts["logprob_available"] = lp_row["cnt"] if lp_row else 0
        return counts
    finally:
        conn.close()


# ── Behavioral responses ──────────────────────────────────────────────────────

def store_behavioral(row: dict) -> None:
    """Insert or replace a behavioral response row in SQLite."""
    row = dict(row)
    for field in ("run_number", "is_two_turn"):
        v = row.get(field)
        if v is not None:
            try:
                row[field] = int(v)
            except (TypeError, ValueError):
                row[field] = None

    conn = _get_conn()
    try:
        placeholders = ", ".join("?" for _ in BEHAVIORAL_COLUMNS)
        cols = ", ".join(BEHAVIORAL_COLUMNS)
        values = [row.get(c) for c in BEHAVIORAL_COLUMNS]
        conn.execute(
            f"INSERT OR REPLACE INTO behavioral_responses ({cols}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def already_completed_behavioral(model_id: str, prompt_id: str, run_number: int) -> bool:
    """Return True if a 'success' row exists for this (model_id, prompt_id, run_number)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM behavioral_responses "
            "WHERE model_id=? AND prompt_id=? AND run_number=? AND status='success'",
            (model_id, prompt_id, run_number),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def load_completed_behavioral_set() -> set[tuple[str, str, int]]:
    """Load all successful (model_id, prompt_id, run_number) triples into a set."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT model_id, prompt_id, run_number FROM behavioral_responses WHERE status='success'"
        ).fetchall()
        return {(r["model_id"], r["prompt_id"], r["run_number"]) for r in rows}
    finally:
        conn.close()


# ── Judge ratings ─────────────────────────────────────────────────────────────

_CREATE_JUDGE_TABLE = """
CREATE TABLE IF NOT EXISTS judge_ratings (
  id                        INTEGER PRIMARY KEY AUTOINCREMENT,
  behavioral_response_id    INTEGER NOT NULL,
  subject_model_id          TEXT NOT NULL,
  prompt_id                 TEXT NOT NULL,
  run_number                INTEGER NOT NULL,
  judge_model_id            TEXT NOT NULL,
  keying                    TEXT NOT NULL,
  score_RE                  INTEGER,
  score_DE                  INTEGER,
  score_BO                  INTEGER,
  score_GU                  INTEGER,
  score_VB                  INTEGER,
  raw_response              TEXT,
  parse_status              TEXT NOT NULL,
  error_message             TEXT,
  timestamp                 TEXT,
  UNIQUE(behavioral_response_id, judge_model_id)
)
"""

JUDGE_SCORE_COLUMNS = ["score_RE", "score_DE", "score_BO", "score_GU", "score_VB"]

JUDGE_COLUMNS = [
    "behavioral_response_id",
    "subject_model_id",
    "prompt_id",
    "run_number",
    "judge_model_id",
    "keying",
    *JUDGE_SCORE_COLUMNS,
    "raw_response",
    "parse_status",
    "error_message",
    "timestamp",
]


def store_judge_rating(row: dict) -> None:
    """Insert or replace a judge rating row in SQLite.

    Uses INSERT OR REPLACE so parse_error rows are overwritten on retry success.
    """
    row = dict(row)
    for field in ("behavioral_response_id", "run_number", *JUDGE_SCORE_COLUMNS):
        v = row.get(field)
        if v is not None:
            try:
                row[field] = int(v)
            except (TypeError, ValueError):
                row[field] = None

    conn = _get_conn()
    try:
        placeholders = ", ".join("?" for _ in JUDGE_COLUMNS)
        cols = ", ".join(JUDGE_COLUMNS)
        values = [row.get(c) for c in JUDGE_COLUMNS]
        conn.execute(
            f"INSERT OR REPLACE INTO judge_ratings ({cols}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def already_judged(behavioral_response_id: int, judge_model_id: str) -> bool:
    """Return True if a 'success' parse_status row exists for this (sample, judge) pair."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM judge_ratings "
            "WHERE behavioral_response_id=? AND judge_model_id=? AND parse_status='success'",
            (behavioral_response_id, judge_model_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def load_completed_judge_set() -> set[tuple[int, str]]:
    """Load all (behavioral_response_id, judge_model_id) pairs with parse_status='success'."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT behavioral_response_id, judge_model_id "
            "FROM judge_ratings WHERE parse_status='success'"
        ).fetchall()
        return {(r["behavioral_response_id"], r["judge_model_id"]) for r in rows}
    finally:
        conn.close()


def load_behavioral_samples_for_judging(
    model_ids: list[str] | None = None,
    prompt_ids: list[str] | None = None,
    n_samples: int | None = None,
) -> list[dict]:
    """Load successful behavioral_responses rows for judge rating.

    Returns list of dicts with keys: id, model_id, prompt_id, run_number,
    dimension, dimension_code, is_two_turn, raw_response.
    Ordered by (model_id, prompt_id, run_number) for reproducible n_samples slicing.
    """
    conn = _get_conn()
    try:
        where = ["status = 'success'"]
        params: list = []
        if model_ids:
            placeholders = ",".join("?" for _ in model_ids)
            where.append(f"model_id IN ({placeholders})")
            params.extend(model_ids)
        if prompt_ids:
            placeholders = ",".join("?" for _ in prompt_ids)
            where.append(f"prompt_id IN ({placeholders})")
            params.extend(prompt_ids)
        where_clause = " AND ".join(where)
        limit_clause = f" LIMIT {n_samples}" if n_samples is not None else ""
        rows = conn.execute(
            f"""
            SELECT id, model_id, prompt_id, run_number,
                   dimension, dimension_code, is_two_turn, raw_response
            FROM behavioral_responses
            WHERE {where_clause}
            ORDER BY model_id, prompt_id, run_number
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
