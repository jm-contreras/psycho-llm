"""Formal tests of the instrument/judge/human dissociation.

Under a single common factor driving instrument scores (I), human ratings (H),
and judge ratings (J) with nonnegative loadings a, b, c:
    r_IJ = ab, r_IH = ac, r_HJ = bc  =>  r_IJ * r_HJ = a c b^2 <= a c = r_IH.
So the product r_IJ * r_HJ is a lower bound on r_IH. This module tests the
observed violation of that bound per factor:

  1. Percentile bootstrap (resampling models) of d = r_IH - (r_IJ * r_HJ).
     d < 0 with CI excluding 0 rejects the single-common-factor account.
  2. Steiger's z for dependent correlations sharing one variable
     (H0: r_IH = r_IJ, given r_HJ), per Steiger (1980) / Williams (1959).

Usage:
    python -m analysis.dissociation_test

Output:
    analysis/output/dissociation_test.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import OUTPUT_DIR
from analysis.predictive_validity import (
    FACTORS, FACTOR_NAMES,
    load_human_ratings, load_judge_ensemble, load_instrument_scores,
    model_level_human_scores,
)

N_BOOT = 10_000
SEED = 20260705


def steiger_z(r_ih: float, r_ij: float, r_hj: float, n: int) -> tuple[float, float]:
    """Steiger's (1980) z for H0: rho_IH = rho_IJ with shared variable I."""
    rm2 = (r_ih ** 2 + r_ij ** 2) / 2
    f = (1 - r_hj) / (2 * (1 - rm2))
    f = min(f, 1.0)
    h = (1 - f * rm2) / (1 - rm2)
    z_ih = np.arctanh(r_ih)
    z_ij = np.arctanh(r_ij)
    z = (z_ih - z_ij) * np.sqrt((n - 3) / (2 * (1 - r_hj) * h))
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def main() -> None:
    rng = np.random.default_rng(SEED)

    inst = load_instrument_scores().set_index("model_id")[FACTORS]
    judge = load_judge_ensemble()
    human = model_level_human_scores(load_human_ratings(), "all")

    models = sorted(set(inst.index) & set(judge.index) & set(human.index))
    inst, judge, human = inst.loc[models], judge.loc[models], human.loc[models]
    n = len(models)

    lines = [
        "# Single-Common-Factor Bound Test and Steiger Tests",
        "",
        f"N = {n} models. Under one common factor with nonnegative loadings, "
        "r_IH >= r_IJ * r_HJ. d = r_IH - r_IJ*r_HJ; percentile bootstrap over "
        f"models ({N_BOOT:,} resamples, seed={SEED}). Steiger's z tests "
        "H0: r_IH = r_IJ (shared variable I).",
        "",
        "| Factor | r_IH | r_IJ | r_HJ | bound r_IJ*r_HJ | d | d 95% CI | Steiger z | p |",
        "|---|---:|---:|---:|---:|---:|:---|---:|---:|",
    ]

    for f in FACTORS:
        x = inst[f].to_numpy()   # I
        h = human[f].to_numpy()  # H
        j = judge[f].to_numpy()  # J

        def _d(ix):
            xi, hi_, ji = x[ix], h[ix], j[ix]
            if min(np.std(xi), np.std(hi_), np.std(ji)) == 0:
                return np.nan
            r_ih = stats.pearsonr(xi, hi_)[0]
            r_ij = stats.pearsonr(xi, ji)[0]
            r_hj = stats.pearsonr(hi_, ji)[0]
            return r_ih - r_ij * r_hj

        idx = np.arange(n)
        r_ih = stats.pearsonr(x, h)[0]
        r_ij = stats.pearsonr(x, j)[0]
        r_hj = stats.pearsonr(h, j)[0]
        d_obs = r_ih - r_ij * r_hj

        boots = np.array([_d(rng.integers(0, n, n)) for _ in range(N_BOOT)])
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])

        z, p = steiger_z(r_ih, r_ij, r_hj, n)
        lines.append(
            f"| {FACTOR_NAMES[f]} | {r_ih:+.3f} | {r_ij:+.3f} | {r_hj:+.3f} "
            f"| {r_ij * r_hj:+.3f} | {d_obs:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
            f"| {z:+.2f} | {p:.4f} |"
        )

    out = OUTPUT_DIR / "dissociation_test.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
