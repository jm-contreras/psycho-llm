"""Build the OSF data archive from a local raw-data tree.

Prolific participant IDs (prolific_pid) are replaced with a stable 12-char
hash: sha256(SALT + pid).hexdigest()[:12]. study_id and session_id are
dropped. All other data is copied as-is.

Usage:
    python scripts/anonymize_and_archive.py \\
        --src    /absolute/path/to/data      \\
        --out    /tmp/psycho-llm-data        \\
        --archive /tmp/psycho-llm-data-v1.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import tarfile
from pathlib import Path

SALT = "psycho-llm-osf-v1"


def anon(pid: str) -> str:
    return hashlib.sha256((SALT + pid).encode()).hexdigest()[:12]


def anon_prolific_db(src_db: Path, dst_db: Path) -> None:
    """Copy the DB and replace prolific_pid / drop session_id+study_id."""
    shutil.copy2(src_db, dst_db)
    con = sqlite3.connect(dst_db)
    cur = con.cursor()

    for table in ("prolific_sessions", "prolific_ratings"):
        cur.execute(f"SELECT DISTINCT prolific_pid FROM {table}")
        mapping = {pid: anon(pid) for (pid,) in cur.fetchall()}
        for pid, hashed in mapping.items():
            cur.execute(
                f"UPDATE {table} SET prolific_pid = ? WHERE prolific_pid = ?",
                (hashed, pid),
            )
        cur.execute(f"UPDATE {table} SET session_id = '', study_id = ''")

    con.commit()
    con.close()


def anon_prolific_csv(src_csv: Path, dst_csv: Path) -> None:
    import csv

    with src_csv.open() as fin, dst_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = [c for c in reader.fieldnames if c not in {"session_id", "study_id"}]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if "prolific_pid" in row:
                row["prolific_pid"] = anon(row["prolific_pid"])
            for drop in ("session_id", "study_id"):
                row.pop(drop, None)
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="source data/ directory")
    ap.add_argument("--out", type=Path, required=True, help="destination tree")
    ap.add_argument("--archive", type=Path, required=True, help="tar.gz output")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    # raw/ — no PII, copy entire tree
    shutil.copytree(args.src / "raw", args.out / "raw")

    # mturk/ — no PII
    shutil.copytree(args.src / "mturk", args.out / "mturk")

    # prolific/ — anonymize
    (args.out / "prolific").mkdir()
    anon_prolific_db(args.src / "prolific" / "prolific.db",
                     args.out / "prolific" / "prolific.db")
    anon_prolific_csv(args.src / "prolific" / "prolific_ratings.csv",
                      args.out / "prolific" / "prolific_ratings.csv")

    # DATA_README
    (args.out / "DATA_README.md").write_text(
        "# psycho-llm data archive v1\n\n"
        "Rater IDs are anonymized via sha256(SALT + prolific_pid)[:12] with "
        f"SALT = '{SALT}'. study_id and session_id are removed.\n\n"
        "See scripts/anonymize_and_archive.py in the companion GitHub repo.\n"
    )

    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(args.out, arcname=args.out.name)

    print(f"Wrote {args.archive}")


if __name__ == "__main__":
    main()
