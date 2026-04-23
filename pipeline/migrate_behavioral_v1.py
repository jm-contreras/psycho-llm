"""
One-time migration: rename v0.10 behavioral prompt IDs/dimensions to v1 factor names.

Migrates 22 reused prompts by updating prompt_id, dimension, and dimension_code.
Deletes all rows for v0.10 prompts not reused in v1.
Exports the migrated table to data/raw/behavioral_responses.csv.

Run with:
    python -m pipeline.migrate_behavioral_v1
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "data" / "raw" / "responses.db"
BEHAVIORAL_CSV_PATH = REPO_ROOT / "data" / "raw" / "behavioral_responses.csv"

# v0 prompt_id -> (v1 prompt_id, v1 dimension, v1 dimension_code)
RENAME_MAP: dict[str, tuple[str, str, str]] = {
    "CR-BP01": ("PR-BP01", "Proportionality", "PR"),
    "CR-BP02": ("PR-BP02", "Proportionality", "PR"),
    "CR-BP03": ("PR-BP03", "Proportionality", "PR"),
    "CR-BP04": ("PR-BP04", "Proportionality", "PR"),
    "RS-BP01": ("OP-BP01", "Openness", "OP"),
    "RS-BP02": ("OP-BP02", "Openness", "OP"),
    "RS-BP03": ("OP-BP03", "Openness", "OP"),
    "RS-BP04": ("OP-BP04", "Openness", "OP"),
    "CC-BP01": ("OR-BP01", "Originality", "OR"),
    "CC-BP02": ("OR-BP02", "Originality", "OR"),
    "CC-BP03": ("OR-BP03", "Originality", "OR"),
    "CC-BP04": ("OR-BP04", "Originality", "OR"),
    "SA-BP01": ("DE-BP01", "Deference", "DE"),
    "SA-BP02": ("DE-BP02", "Deference", "DE"),
    "SA-BP04": ("DE-BP04", "Deference", "DE"),
    "VE-BP01": ("EL-BP01", "Elaboration", "EL"),
    "VE-BP02": ("EL-BP02", "Elaboration", "EL"),
    "PI-BP01": ("EL-BP03", "Elaboration", "EL"),
    "WR-BP01": ("DR-BP01", "Directness", "DR"),
    "WR-BP02": ("DR-BP02", "Directness", "DR"),
    "WR-BP03": ("DR-BP03", "Directness", "DR"),
    "WR-BP04": ("DR-BP04", "Directness", "DR"),
}

# The set of valid v1 prompt_ids after renaming (22 reused prompts)
V1_KEPT_IDS: set[str] = {v1_id for v1_id, _, _ in RENAME_MAP.values()}

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

    # Step 1: Rename the 22 reused prompts
    renamed = 0
    for v0_id, (v1_id, v1_dim, v1_code) in RENAME_MAP.items():
        conn.execute(
            """
            UPDATE behavioral_responses
            SET prompt_id = ?, dimension = ?, dimension_code = ?
            WHERE prompt_id = ?
            """,
            (v1_id, v1_dim, v1_code, v0_id),
        )
        n = conn.execute("SELECT changes()").fetchone()[0]
        renamed += n
        if n > 0:
            print(f"  Renamed {v0_id} -> {v1_id} ({v1_dim}): {n} rows")

    conn.commit()
    print(f"\nTotal rows renamed: {renamed:,}")

    # Step 2: Delete rows for v0 prompts not reused in v1
    # After the renames above, any prompt_id NOT in V1_KEPT_IDS is a v0-only prompt
    placeholders = ",".join("?" for _ in V1_KEPT_IDS)
    conn.execute(
        f"DELETE FROM behavioral_responses WHERE prompt_id NOT IN ({placeholders})",
        list(V1_KEPT_IDS),
    )
    deleted = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    print(f"Rows deleted (v0-only prompts): {deleted:,}")

    # Count rows after migration
    rows_after = conn.execute("SELECT COUNT(*) FROM behavioral_responses").fetchone()[0]
    print(f"Rows after migration: {rows_after:,}")

    # Step 3: Verify
    print("\nDistinct prompt_ids remaining:")
    distinct_ids = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT prompt_id FROM behavioral_responses ORDER BY prompt_id"
        )
    ]
    for pid in distinct_ids:
        print(f"  {pid}")
    print(f"\nTotal distinct prompt_ids: {len(distinct_ids)}")

    print("\nDimension counts:")
    for r in conn.execute(
        "SELECT dimension, COUNT(*) as cnt FROM behavioral_responses GROUP BY dimension ORDER BY dimension"
    ):
        print(f"  {r[0]}: {r[1]:,}")

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
    assert len(distinct_ids) == 22, f"Expected 22 distinct prompt_ids, got {len(distinct_ids)}"
    assert set(distinct_ids) == V1_KEPT_IDS, (
        f"Unexpected prompt_ids: {set(distinct_ids) - V1_KEPT_IDS}"
    )
    print("\nAll assertions passed.")


if __name__ == "__main__":
    migrate()
