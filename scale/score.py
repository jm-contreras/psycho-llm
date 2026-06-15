"""Score raw 1-5 Likert responses on the AI-Native Behavioral Instrument (Scale v1).

Standard library only -- no third-party dependencies. See README.md for the scoring
rationale; the procedure mirrors the analysis code used in the paper.

Usage:
    from score import load_items, score_respondent

    items = load_items()                      # reads scale_v1_items.csv next to this file
    responses = {"RE-01": 4, "RE-02": 2, ...}  # item_code -> integer 1..5
    scores = score_respondent(responses, items)  # -> {factor_name: float}

Run `python score.py` for a worked example.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_CSV = os.path.join(_HERE, "scale_v1_items.csv")
NORMS_CSV = os.path.join(_HERE, "reference_norms.csv")

FACTORS = ["Responsiveness", "Deference", "Guardedness", "Boldness", "Verbosity"]


def load_items(path: str = ITEMS_CSV) -> list[dict]:
    """Load scale_v1_items.csv into a list of dicts with typed loading values."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["primary_loading"] = float(r["primary_loading"])
    return rows


def score_respondent(
    responses: dict[str, int],
    items: list[dict] | None = None,
    *,
    require_complete: bool = False,
) -> dict[str, float]:
    """Compute the 5 factor scores for one respondent.

    Procedure (see README.md "Scoring"):
      1. Reverse-key items with keying == '-':  r -> 6 - r.
      2. Align to the empirical factor:  multiply by sign(primary_loading).
      3. Unit-weighted mean of the aligned scores within each factor.

    Args:
        responses: mapping of item_code -> integer 1..5.
        items:     parsed scale items; loaded from scale_v1_items.csv if None.
        require_complete: if True, raise when any scale item is missing a response.

    Returns:
        {factor_name: mean_aligned_score}. Higher = more of that factor.
    """
    if items is None:
        items = load_items()

    aligned: dict[str, list[float]] = defaultdict(list)
    missing: list[str] = []

    for item in items:
        code = item["item_code"]
        if code not in responses:
            missing.append(code)
            continue
        raw = responses[code]
        if not (isinstance(raw, int) and 1 <= raw <= 5):
            raise ValueError(f"Response for {code} must be an integer 1-5, got {raw!r}")

        score = 6 - raw if item["keying"] == "-" else raw          # 1. reverse-key
        if item["primary_loading"] < 0:                            # 2. align to factor
            score = 6 - score
        aligned[item["factor"]].append(score)

    if missing and require_complete:
        raise ValueError(f"Missing responses for {len(missing)} items: {missing}")

    return {
        factor: sum(vals) / len(vals)
        for factor in FACTORS
        if (vals := aligned.get(factor))
    }


def load_norms(path: str = NORMS_CSV) -> dict[str, dict[str, float]]:
    """Load reference_norms.csv -> {factor: {'mean': .., 'sd': ..}} (25-model reference set)."""
    with open(path, newline="") as f:
        return {
            r["factor"]: {"mean": float(r["mean"]), "sd": float(r["sd"])}
            for r in csv.DictReader(f)
        }


def zscore_respondent(
    scores: dict[str, float],
    norms: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Convert raw factor scores to z-scores against the 25-model reference norms.

    z > 0 means the respondent is higher on that factor than the average of the
    25 models in the paper's reference set; z = +1 is one between-model SD above it.
    """
    if norms is None:
        norms = load_norms()
    return {
        factor: (value - norms[factor]["mean"]) / norms[factor]["sd"]
        for factor, value in scores.items()
        if factor in norms
    }


if __name__ == "__main__":
    items = load_items()

    # Worked example: a synthetic respondent who, on every item, answers in the direction
    # of HIGH Responsiveness and otherwise neutral. We build responses programmatically so
    # the example stays valid if items change.
    #
    # For each Responsiveness item we pick the raw answer (5 or 1) that maps to a HIGH
    # aligned score; all other items get a neutral 3.
    responses: dict[str, int] = {}
    for item in items:
        if item["factor"] == "Responsiveness":
            # aligned-high means: after reverse-keying and loading-sign alignment, score = 5.
            reverse = item["keying"] == "-"
            flip = item["primary_loading"] < 0
            # want aligned == 5; invert the two transforms to find the raw answer.
            target = 5
            pre_align = (6 - target) if flip else target
            raw = (6 - pre_align) if reverse else pre_align
            responses[item["item_code"]] = raw
        else:
            responses[item["item_code"]] = 3

    scores = score_respondent(responses, items)
    z = zscore_respondent(scores)
    print("Factor scores for a 'high Responsiveness, otherwise neutral' respondent:\n")
    print(f"  {'Factor':<15}{'raw (1-5)':>11}{'z vs 25-model ref':>20}")
    for factor in FACTORS:
        print(f"  {factor:<15}{scores[factor]:>11.2f}{z[factor]:>20.2f}")
    print("\n(Raw Responsiveness should be ~5.0 and the rest ~3.0; the z-column places")
    print(" each score relative to the 25-model reference set in reference_norms.csv.)")
