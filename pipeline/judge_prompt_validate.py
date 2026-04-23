"""
CLI entry point for validating few-shot examples against judge models.

Usage:
  python -m pipeline.judge_prompt_validate

Sends each FEW_SHOT_EXAMPLE in judge_prompt.py to all 3 judge models (cold, no few-shot
context) and compares their ratings against the hand-crafted ground-truth ratings.

Iterate on FEW_SHOT_EXAMPLES until all deltas are ≤ 1 per factor per judge, then the
examples are ready for production use in judge_runner.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

from pipeline.config import filter_by_names, load_model_registry
from pipeline.judge_prompt import FEW_SHOT_EXAMPLES, validate_few_shot_examples

_JUDGE_NAMES = ["Claude Opus 4.6", "GPT-5.4", "Gemini 3.1 Pro"]


def main() -> None:
    if not FEW_SHOT_EXAMPLES:
        print(
            "FEW_SHOT_EXAMPLES is empty.\n"
            "Edit pipeline/judge_prompt.py and populate FEW_SHOT_EXAMPLES first.\n"
            "See the TODO comment in that file for authoring instructions."
        )
        return

    all_models = load_model_registry(providers=None)
    judges = filter_by_names(all_models, _JUDGE_NAMES)

    if not judges:
        print(f"No judge models found matching {_JUDGE_NAMES}. Check model registry.")
        return

    print(f"Judges: {[j['model_name'] for j in judges]}\n")
    validate_few_shot_examples(judges)


if __name__ == "__main__":
    main()
