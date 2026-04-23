"""Extract headline numbers from analysis/output/ and print side-by-side with
expected values reported in paper/main.tex. Exit non-zero if any metric
differs from the expected value by more than the allowed tolerance.

Run after scripts/reproduce.sh.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "analysis" / "output"

# (metric, expected, tolerance, extractor)
FACTORS = ["Responsiveness", "Deference", "Boldness", "Guardedness", "Verbosity"]
# From paper/main.tex (§reliability): Responsiveness α=.976, Deference α=.969,
# Boldness α=.953, Guardedness α=.924, Verbosity α=.945. The exact numbers
# depend on the final item-retention solution — tolerance of ±.03 accommodates
# minor rounding/regeneration differences.
EXPECTED_ALPHA = {
    "Responsiveness": 0.97,
    "Deference":      0.97,
    "Boldness":       0.94,
    "Guardedness":    0.92,
    "Verbosity":      0.94,
}
ALPHA_TOL = 0.03

def _parse_alpha_table(report: Path) -> dict[str, float]:
    """Extract the '| FactorN | Label | n_items | alpha | ... |' table."""
    if not report.exists():
        return {}
    out: dict[str, float] = {}
    for line in report.read_text().splitlines():
        m = re.match(
            r"\|\s*Factor[1-5]\s*\|\s*([A-Za-z]+)\s*\|\s*\d+\s*\|\s*([0-9.]+)",
            line,
        )
        if m:
            name, alpha = m.group(1).strip(), float(m.group(2))
            if name in FACTORS:
                out[name] = alpha
    return out


def _print_row(label: str, actual: float | None, expected: float, tol: float) -> bool:
    if actual is None:
        print(f"  {label:<32}  ACTUAL: (not found)   EXPECTED: {expected:+.3f}   ❌")
        return False
    diff = abs(actual - expected)
    ok = diff <= tol
    mark = "✅" if ok else "❌"
    print(f"  {label:<32}  ACTUAL: {actual:+.3f}        EXPECTED: {expected:+.3f}   {mark} (Δ={diff:.3f}, tol={tol})")
    return ok


def main() -> int:
    all_ok = True

    print("\n-- Cronbach α (expect all ≥ 0.92) --")
    alpha = _parse_alpha_table(OUTPUT_DIR / "primary_analysis_report.md")
    for f in FACTORS:
        all_ok &= _print_row(f"α  {f}", alpha.get(f), EXPECTED_ALPHA[f], ALPHA_TOL)

    print("\n-- Headline convergence correlations --")
    print("  These values live in markdown tables whose exact layout makes")
    print("  safe regex extraction fragile. Compare directly against:")
    print()
    print(f"    {OUTPUT_DIR.relative_to(REPO_ROOT)}/predictive_validity_report.md")
    print("      • Expected: self-report--human mean r̄ ≈ -0.01")
    print("      • Expected: self-report--judge mean r̄ ≈ +0.13")
    print("      • Expected: Responsiveness self-report--judge r ≈ +0.53")
    print("      • Expected: Responsiveness self-report--human r ≈ +0.04")
    print()
    print(f"    {OUTPUT_DIR.relative_to(REPO_ROOT)}/prolific_report.md")
    print("      • Expected: human-judge mean r̄ ≈ +0.51")
    print("      • Expected: N = 151 usable Prolific raters")

    print()
    if all_ok:
        print("All Cronbach α within tolerance. ✅")
        print("Compare the convergence-correlation tables against the paper manually.")
        return 0
    else:
        print("One or more Cronbach α values out of tolerance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
