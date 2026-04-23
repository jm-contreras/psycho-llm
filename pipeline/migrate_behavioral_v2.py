"""
One-time migration: rename v1 behavioral prompt IDs/dimensions to v2 factor names.

v1 (7 factors) → v2 (5 factors):
  - DI (Discernment) → RE (Responsiveness)
  - DE (Deference) → DE (Deference) [unchanged]
  - OR (Originality) → BO (Boldness)
  - OP (Openness) → GU (Guardedness)
  - EL (Elaboration) → VB (Verbosity)
  - DR (Directness) → DELETED
  - PR (Proportionality) → DELETED

Deletes all rows for dropped factors (DR, PR).
Exports the migrated table to data/raw/behavioral_responses.csv.

Run with:
    python -m pipeline.migrate_behavioral_v2
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "data" / "raw" / "responses.db"
BEHAVIORAL_CSV_PATH = REPO_ROOT / "data" / "raw" / "behavioral_responses.csv"

# v1 prompt_id -> (v2 prompt_id, v2 dimension, v2 dimension_code)
RENAME_MAP: dict[str, tuple[str, str, str]] = {
    # Discernment → Responsiveness
    "DI-BP01": ("RE-BP01", "Responsiveness", "RE"),
    "DI-BP02": ("RE-BP02", "Responsiveness", "RE"),
    "DI-BP03": ("RE-BP03", "Responsiveness", "RE"),
    "DI-BP04": ("RE-BP04", "Responsiveness", "RE"),
    # Originality → Boldness
    "OR-BP01": ("BO-BP01", "Boldness", "BO"),
    "OR-BP02": ("BO-BP02", "Boldness", "BO"),
    "OR-BP03": ("BO-BP03", "Boldness", "BO"),
    "OR-BP04": ("BO-BP04", "Boldness", "BO"),
    # Openness → Guardedness
    "OP-BP01": ("GU-BP01", "Guardedness", "GU"),
    "OP-BP02": ("GU-BP02", "Guardedness", "GU"),
    "OP-BP03": ("GU-BP03", "Guardedness", "GU"),
    "OP-BP04": ("GU-BP04", "Guardedness", "GU"),
    # Elaboration → Verbosity
    "EL-BP01": ("VB-BP01", "Verbosity", "VB"),
    "EL-BP02": ("VB-BP02", "Verbosity", "VB"),
    "EL-BP03": ("VB-BP03", "Verbosity", "VB"),
    "EL-BP04": ("VB-BP04", "Verbosity", "VB"),
}

# DE-BP01–04 stay the same ID and dimension — no rename needed.
# DR-BP01–04 (Directness) and PR-BP01–04 (Proportionality) are deleted.

# All valid v2 prompt_ids after migration
V2_KEPT_IDS: set[str] = (
    {v2_id for v2_id, _, _ in RENAME_MAP.values()}
    | {"DE-BP01", "DE-BP02", "DE-BP03", "DE-BP04"}
)

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


def migrate() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Count rows before migration
    rows_before = conn.execute("SELECT COUNT(*) FROM behavioral_responses").fetchone()[0]
    print(f"Rows before migration: {rows_before:,}")

    # Step 1: Rename the 16 prompts that change IDs
    renamed = 0
    for v1_id, (v2_id, v2_dim, v2_code) in RENAME_MAP.items():
        conn.execute(
            """
            UPDATE behavioral_responses
            SET prompt_id = ?, dimension = ?, dimension_code = ?
            WHERE prompt_id = ?
            """,
            (v2_id, v2_dim, v2_code, v1_id),
        )
        n = conn.execute("SELECT changes()").fetchone()[0]
        renamed += n
        if n > 0:
            print(f"  Renamed {v1_id} -> {v2_id} ({v2_dim}): {n} rows")

    conn.commit()
    print(f"\nTotal rows renamed: {renamed:,}")

    # Step 2: Delete rows for dropped factors (Directness, Proportionality)
    conn.execute(
        "DELETE FROM behavioral_responses WHERE dimension_code IN ('DR', 'PR')"
    )
    deleted = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    print(f"Rows deleted (Directness + Proportionality): {deleted:,}")

    # Count rows after migration
    rows_after = conn.execute("SELECT COUNT(*) FROM behavioral_responses").fetchone()[0]
    print(f"Rows after migration: {rows_after:,}")

    # Step 3: Verify
    print("\nDistinct prompt_ids remaining:")
    distinct_ids = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT prompt_id FROM behavioral_responses ORDER BY prompt_id"
        )
    ]
    for pid in distinct_ids:
        print(f"  {pid}")
    print(f"\nTotal distinct prompt_ids: {len(distinct_ids)}")

    print("\nDimension counts:")
    for r in conn.execute(
        "SELECT dimension, dimension_code, COUNT(*) as cnt "
        "FROM behavioral_responses GROUP BY dimension ORDER BY dimension"
    ):
        print(f"  {r[1]:4s} {r[0]:20s} {r[2]:,} rows")

    # Step 4: Export to CSV
    rows = conn.execute(
        f"SELECT {', '.join(BEHAVIORAL_COLUMNS)} FROM behavioral_responses ORDER BY id"
    ).fetchall()
    conn.close()

    BEHAVIORAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEHAVIORAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BEHAVIORAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"\nExported {len(rows):,} rows to {BEHAVIORAL_CSV_PATH}")

    # Assertions
    assert len(distinct_ids) == 20, (
        f"Expected 20 distinct prompt_ids, got {len(distinct_ids)}"
    )
    assert set(distinct_ids) == V2_KEPT_IDS, (
        f"Unexpected prompt_ids: {set(distinct_ids) - V2_KEPT_IDS}"
    )
    print("\nAll assertions passed. ✓")


if __name__ == "__main__":
    migrate()
