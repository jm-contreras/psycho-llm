"""MTurk pipeline configuration and AWS client factory."""

from __future__ import annotations

import os
from pathlib import Path

import boto3

from pipeline.storage import REPO_ROOT

# ── Endpoints ─────────────────────────────────────────────────────────────────

MTURK_SANDBOX = os.environ.get("MTURK_SANDBOX", "true").lower() != "false"

SANDBOX_ENDPOINT = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
PRODUCTION_ENDPOINT = "https://mturk-requester.us-east-1.amazonaws.com"

MTURK_ENDPOINT = SANDBOX_ENDPOINT if MTURK_SANDBOX else PRODUCTION_ENDPOINT

# ── Requester info ─────────────────────────────────────────────────────────────

REQUESTER_NAME = "Juan Manuel Contreras, Ph.D."
REQUESTER_EMAIL = "jm.contreras.phd@gmail.com"

# ── Randomisation ─────────────────────────────────────────────────────────────

MTURK_SEED = 20260327

# ── Assignment design ──────────────────────────────────────────────────────────

MAX_ASSIGNMENTS_INITIAL = 2     # Start with 2 raters per HIT
MAX_ASSIGNMENTS_TIEBREAK = 1    # Add 3rd rater on disagreement
DISAGREEMENT_THRESHOLD = 2      # Spread ≥ 2 on any factor triggers tiebreak

# ── Pay & timing ──────────────────────────────────────────────────────────────

REWARD_PER_HIT = "1.00"
ASSIGNMENT_DURATION_SECONDS = 600          # 10 minutes
HIT_LIFETIME_SECONDS = 7 * 24 * 3600      # 1 week
AUTO_APPROVAL_DELAY_SECONDS = 3 * 24 * 3600  # 3 days

# ── Gold standard ─────────────────────────────────────────────────────────────

GOLD_RATE = 0.15                 # ~15% of HITs are gold items
GOLD_ACCURACY_THRESHOLD = 0.60  # Pass if ≥ 60% of gold ratings are within ±1

# ── Qualification ──────────────────────────────────────────────────────────────

QUALIFICATION_THRESHOLD = 80    # Minimum score (out of 100) to earn qualification

# ── Data paths ─────────────────────────────────────────────────────────────────

MTURK_DIR = REPO_ROOT / "data" / "mturk"
MANIFEST_PATH = MTURK_DIR / "hit_manifest.json"
GOLD_ITEMS_PATH = MTURK_DIR / "gold_items.json"
SAMPLE_PATH = MTURK_DIR / "sample.json"
RESULTS_CSV_PATH = MTURK_DIR / "human_ratings.csv"

# Ensure data/mturk/ exists on import
MTURK_DIR.mkdir(parents=True, exist_ok=True)


# ── Client factory ─────────────────────────────────────────────────────────────

AWS_PROFILE = os.environ.get("AWS_PROFILE", "psycho-llm")


def get_mturk_client():
    """Return a boto3 MTurk client pointed at sandbox or production."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(
        "mturk",
        endpoint_url=MTURK_ENDPOINT,
        region_name="us-east-1",
    )
