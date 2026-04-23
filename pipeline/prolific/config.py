"""Configuration constants for the Prolific survey pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.storage import REPO_ROOT

# ── Randomisation ─────────────────────────────────────────────────────────────

PROLIFIC_SEED: int = 20260318

# ── Survey / session design ───────────────────────────────────────────────────

SAMPLE_ITEMS_PER_SESSION: int = 6   # Non-gold items per session
GOLD_ITEMS_PER_SESSION: int = 1     # Monitoring gold items per session
ITEMS_PER_SESSION: int = 7          # Total rated items (SAMPLE + GOLD)
TRAINING_ITEMS: int = 2             # Fixed training items shown before rated items

# ── Manually excluded participants (rejected/returned on Prolific) ────────────
# Raw Prolific PIDs are not distributed; the anonymized OSF data archive uses
# sha256(salt + pid)[:12] as stable rater IDs. The values below are the
# anonymized IDs of the 8 participants excluded after manual review. Salt:
# "psycho-llm-osf-v1" (see data anonymization script in the OSF bundle).
EXCLUDED_PROLIFIC_PIDS: frozenset[str] = frozenset({
    "ca0b05c20ff3",  # returned
    "6f523c30c8cf",  # rejected
    "bf53ae6e33a4",  # rejected
    "22658e748c45",  # rejected
    "de38314720fd",  # rejected
    "5401bb5bf3ed",  # rejected — responded too quickly, failed attention checks
    "413aff8a004b",  # rejected
    "0a56fec18ea4",  # rejected
})

# ── Quality control ───────────────────────────────────────────────────────────

GOLD_RATE: float = GOLD_ITEMS_PER_SESSION / ITEMS_PER_SESSION  # fraction of items that are gold
GOLD_ACCURACY_THRESHOLD: float = 0.80   # Participant pass threshold (fraction within ±1)
DISAGREEMENT_THRESHOLD: int = 2          # Spread ≥ 2 on any factor triggers tiebreak
TARGET_RATINGS_PER_ITEM: int = 2         # Minimum ratings to collect per sample item

# ── Pay & timing ──────────────────────────────────────────────────────────────

SESSION_TIMEOUT_MINUTES: int = 60
PAY_RATE_PER_HOUR_USD: float = 14.00

# ── Prolific API ──────────────────────────────────────────────────────────────

PROLIFIC_API_TOKEN: str | None = os.environ.get("PROLIFIC_API_TOKEN")
PROLIFIC_COMPLETION_CODE: str | None = os.environ.get("PROLIFIC_COMPLETION_CODE")
PROLIFIC_API_BASE: str = "https://api.prolific.com/api/v1"

# ── Flask ─────────────────────────────────────────────────────────────────────

FLASK_SECRET_KEY: str = os.environ.get(
    "PROLIFIC_FLASK_SECRET",
    os.environ.get("FLASK_SECRET_KEY", "dev-insecure-key-change-before-production"),
)

# ── Data paths ────────────────────────────────────────────────────────────────

PROLIFIC_DIR: Path = REPO_ROOT / "data" / "prolific"
PROLIFIC_DB_PATH: Path = PROLIFIC_DIR / "prolific.db"
RESULTS_CSV_PATH: Path = PROLIFIC_DIR / "prolific_ratings.csv"

# Reuse MTurk sample and gold items — same behavioral_response pool
SAMPLE_PATH: Path = REPO_ROOT / "data" / "mturk" / "sample.json"
GOLD_ITEMS_PATH: Path = REPO_ROOT / "data" / "mturk" / "gold_items.json"

# Ensure data/prolific/ exists on import
PROLIFIC_DIR.mkdir(parents=True, exist_ok=True)

# ── Behavioral prompts (dict keyed by prompt_id for O(1) lookup) ──────────────

from pipeline.behavioral_loader import BEHAVIORAL_PROMPTS as _PROMPT_LIST  # noqa: E402

BEHAVIORAL_PROMPTS: dict[str, dict] = {p["prompt_id"]: p for p in _PROMPT_LIST}
