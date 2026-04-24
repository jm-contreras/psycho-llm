"""
Parse llm_native_item_pool_v0.2.md into a list of item dicts.

Each dict has:
  item_id       e.g. "SA-D01" or "SA-S01"
  dimension     e.g. "Social Alignment"
  dimension_code e.g. "SA"
  item_type     "direct" | "scenario"
  keying        "+" | "-"  (direct items only; None for scenario)
  text          Statement text (direct) or scenario context (scenario)
  options       list of {"label": "a", "text": "...", "score": 4}  (scenario only; [] for direct)

Items are returned sorted by item_id (alphabetical), which is the canonical ordering
used by runner.py for reproducible n_items selection.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ITEM_POOL_PATH = REPO_ROOT / "items" / "llm_native_item_pool_v0.2.md"

# Regex patterns
_DIRECT_ROW = re.compile(
    r"^\|\s*([A-Z]+-D\d+)\s*\|\s*(.*?)\s*\|\s*([+\-−])\s*\|"
)
_SCENARIO_HEADER = re.compile(
    r"^\*\*([A-Z]+-S\d+)\.\*\*\s*(.*)"
)
_OPTION_ROW = re.compile(
    r"^\|\s*([a-d])\s*\|\s*(.*?)\s*\|\s*(\d)\s*\|"
)
_DIMENSION_HEADING = re.compile(
    r"^##\s+\d+\.\s+(.+)"
)


def _normalise_keying(raw: str) -> str:
    """Normalise +/−/- to '+' or '-'."""
    return "+" if raw.strip() == "+" else "-"


def _extract_dimension_code(item_id: str) -> str:
    return item_id.split("-")[0]


def load_items(path: Path | None = None) -> list[dict]:
    """
    Parse the item pool markdown and return all items sorted by item_id.
    """
    if path is None:
        path = ITEM_POOL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Item pool not found at {ITEM_POOL_PATH}.")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    items: list[dict] = []
    current_dimension: str = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track current dimension from ## headings
        dim_match = _DIMENSION_HEADING.match(line)
        if dim_match:
            name = dim_match.group(1).strip()
            # Skip the merged dimension note
            if "Sensitivity to Criticism" not in name:
                current_dimension = name
            i += 1
            continue

        # Direct Likert item row
        direct_match = _DIRECT_ROW.match(line)
        if direct_match and current_dimension:
            item_id = direct_match.group(1).strip()
            item_text = direct_match.group(2).strip()
            keying = _normalise_keying(direct_match.group(3))
            items.append({
                "item_id": item_id,
                "dimension": current_dimension,
                "dimension_code": _extract_dimension_code(item_id),
                "item_type": "direct",
                "keying": keying,
                "text": item_text,
                "options": [],
            })
            i += 1
            continue

        # Scenario item header
        scenario_match = _SCENARIO_HEADER.match(line)
        if scenario_match and current_dimension:
            item_id = scenario_match.group(1).strip()
            scenario_text = scenario_match.group(2).strip()

            # Collect options from the table that follows
            options: list[dict] = []
            j = i + 1
            while j < len(lines):
                opt_match = _OPTION_ROW.match(lines[j])
                if opt_match:
                    options.append({
                        "label": opt_match.group(1).strip(),
                        "text": opt_match.group(2).strip(),
                        "score": int(opt_match.group(3)),
                    })
                elif lines[j].startswith("**") and re.match(r"^\*\*[A-Z]+-S\d+", lines[j]):
                    # Next scenario — stop
                    break
                elif lines[j].startswith("##") or lines[j].startswith("---"):
                    # New section — stop
                    break
                j += 1

            items.append({
                "item_id": item_id,
                "dimension": current_dimension,
                "dimension_code": _extract_dimension_code(item_id),
                "item_type": "scenario",
                "keying": None,
                "text": scenario_text,
                "options": options,
            })
            i = j
            continue

        i += 1

    # Sort by item_id — canonical ordering for n_items slicing
    items.sort(key=lambda x: x["item_id"])
    return items


def get_items(
    n_items: int | None = None,
    item_type: str | None = None,
    dimension_codes: list[str] | None = None,
) -> list[dict]:
    """
    Convenience wrapper: load items with optional filtering.

    Args:
        n_items:         Return only the first N items (after sort and filter).
        item_type:       "direct", "scenario", or None for both.
        dimension_codes: List of dimension codes to include, e.g. ["SA", "CA"].
    """
    items = load_items()
    if item_type:
        items = [it for it in items if it["item_type"] == item_type]
    if dimension_codes:
        items = [it for it in items if it["dimension_code"] in dimension_codes]
    if n_items is not None:
        items = items[:n_items]
    return items


if __name__ == "__main__":
    items = load_items()
    direct = [it for it in items if it["item_type"] == "direct"]
    scenario = [it for it in items if it["item_type"] == "scenario"]
    print(f"Loaded {len(items)} items total: {len(direct)} direct, {len(scenario)} scenario")
    print("\nFirst 5 items:")
    for it in items[:5]:
        print(f"  {it['item_id']}  [{it['dimension_code']}]  {it['item_type']}  "
              f"keying={it['keying']}  text={it['text'][:60]!r}")
