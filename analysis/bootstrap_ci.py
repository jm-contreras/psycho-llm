"""Bootstrap 95% CIs for all model-level correlations (N=25).

Addresses reviewer concern (Gemini peer review) that N=25 CIs are wide
and should be reported transparently.

Computes bootstrapped CIs for:
  1. Predictive validity: instrument × human (convergent per factor, all prompts and on-target)
  2. Instrument × judge (convergent per factor)
  3. Human × judge (convergent per factor)
  4. MTMM off-diagonal means (discriminant)

Output: analysis/output/bootstrap_cis.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.predictive_validity import (
    FACTORS, FACTOR_NAMES,
    load_human_ratings, load_judge_ensemble, load_instrument_scores,
    model_level_human_scores,
)
from analysis.data_loader import OUTPUT_DIR

N_BOOT = 10000
RNG = np.random.default_rng(seed=20260416)


def boot_r_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05):
    """Percentile bootstrap CI for Pearson r.

    Resamples (x_i, y_i) pairs with replacement. Returns (r, lo, hi, n).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, n
    r_obs = stats.pearsonr(x, y)[0]
    idx = np.arange(n)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        xs, ys = x[s], y[s]
        if xs.std() == 0 or ys.std() == 0:
            rs[i] = np.nan
        else:
            rs[i] = stats.pearsonr(xs, ys)[0]
    rs = rs[~np.isnan(rs)]
    lo, hi = np.percentile(rs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return r_obs, lo, hi, n


def boot_mean_r_ci(
    inst: pd.DataFrame, other: pd.DataFrame, factors: list[str],
    n_boot: int = N_BOOT, alpha: float = 0.05,
):
    """Bootstrap CI for the mean diagonal (convergent) correlation across factors.

    Resamples models (rows), recomputes per-factor rs, averages, repeats.
    """
    common = inst.index.intersection(other.index)
    inst = inst.loc[common]
    other = other.loc[common]
    n = len(common)
    idx = np.arange(n)

    def _mean_diag(ix_subset):
        rs = []
        for f in factors:
            x = inst[f].values[ix_subset].astype(float)
            y = other[f].values[ix_subset].astype(float)
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 4 or x[m].std() == 0 or y[m].std() == 0:
                continue
            rs.append(stats.pearsonr(x[m], y[m])[0])
        return np.mean(rs) if rs else np.nan

    obs = _mean_diag(idx)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        boots[i] = _mean_diag(s)
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return obs, lo, hi, n


def boot_mean_offdiag_ci(
    inst: pd.DataFrame, other: pd.DataFrame, factors: list[str],
    n_boot: int = N_BOOT, alpha: float = 0.05,
):
    """Bootstrap CI for the mean off-diagonal (discriminant) correlation."""
    common = inst.index.intersection(other.index)
    inst = inst.loc[common]
    other = other.loc[common]
    n = len(common)
    idx = np.arange(n)

    def _mean_off(ix_subset):
        rs = []
        for fi in factors:
            for fj in factors:
                if fi == fj:
                    continue
                x = inst[fi].values[ix_subset].astype(float)
                y = other[fj].values[ix_subset].astype(float)
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() < 4 or x[m].std() == 0 or y[m].std() == 0:
                    continue
                rs.append(stats.pearsonr(x[m], y[m])[0])
        return np.mean(rs) if rs else np.nan

    obs = _mean_off(idx)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        boots[i] = _mean_off(s)
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return obs, lo, hi, n


def factor_table(inst: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-factor convergent r with bootstrap 95% CI."""
    common = inst.index.intersection(other.index)
    inst = inst.loc[common]
    other = other.loc[common]
    rows = []
    for f in FACTORS:
        x = inst[f].values.astype(float)
        y = other[f].values.astype(float)
        r, lo, hi, n = boot_r_ci(x, y)
        rows.append({
            "factor": f, "name": FACTOR_NAMES[f],
            "r": round(r, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n,
        })
    df = pd.DataFrame(rows)
    df["comparison"] = label
    return df


def main():
    print("Loading data...")
    inst_raw = load_instrument_scores()
    inst = inst_raw.set_index("model_id")[FACTORS]
    hr = load_human_ratings()
    judge = load_judge_ensemble()
    human_all = model_level_human_scores(hr, "all")
    human_on = model_level_human_scores(hr, "on_target")

    print(f"Instrument: {inst.shape}, Human-all: {human_all.shape}, "
          f"Human-on: {human_on.shape}, Judge: {judge.shape}")

    print("Computing per-factor CIs...")
    pf_inst_human = factor_table(inst, human_all, "instrument × human (all prompts)")
    pf_inst_human_on = factor_table(inst, human_on, "instrument × human (on-target)")
    pf_inst_judge = factor_table(inst, judge, "instrument × judge")
    pf_human_judge = factor_table(human_all, judge, "human × judge")

    print("Computing mean diagonal CIs...")
    mean_ih_r, mean_ih_lo, mean_ih_hi, n_ih = boot_mean_r_ci(inst, human_all, FACTORS)
    mean_iho_r, mean_iho_lo, mean_iho_hi, _ = boot_mean_r_ci(inst, human_on, FACTORS)
    mean_ij_r, mean_ij_lo, mean_ij_hi, _ = boot_mean_r_ci(inst, judge, FACTORS)
    mean_hj_r, mean_hj_lo, mean_hj_hi, _ = boot_mean_r_ci(human_all, judge, FACTORS)

    print("Computing mean off-diagonal CIs...")
    off_ih_r, off_ih_lo, off_ih_hi, _ = boot_mean_offdiag_ci(inst, human_all, FACTORS)
    off_iho_r, off_iho_lo, off_iho_hi, _ = boot_mean_offdiag_ci(inst, human_on, FACTORS)
    off_ij_r, off_ij_lo, off_ij_hi, _ = boot_mean_offdiag_ci(inst, judge, FACTORS)
    off_hj_r, off_hj_lo, off_hj_hi, _ = boot_mean_offdiag_ci(human_all, judge, FACTORS)

    # Write report
    out = OUTPUT_DIR / "bootstrap_cis.md"
    with out.open("w") as f:
        f.write("# Bootstrap 95% CIs for Model-Level Correlations\n\n")
        f.write(f"Percentile bootstrap, {N_BOOT:,} resamples, seed=20260416. "
                f"N={n_ih} models.\n\n")

        f.write("## Per-factor convergent correlations (diagonal)\n\n")
        for table, label in [
            (pf_inst_human, "Instrument × Human (all prompts)"),
            (pf_inst_human_on, "Instrument × Human (on-target prompts)"),
            (pf_inst_judge, "Instrument × Judge ensemble"),
            (pf_human_judge, "Human × Judge (behavioral agreement)"),
        ]:
            f.write(f"### {label}\n\n")
            f.write("| Factor | r | 95% CI |\n|---|---:|:---|\n")
            for _, row in table.iterrows():
                f.write(f"| {row['name']} | {row['r']:+.2f} | "
                        f"[{row['lo']:+.2f}, {row['hi']:+.2f}] |\n")
            f.write("\n")

        f.write("## Mean convergent & discriminant correlations\n\n")
        f.write("| Comparison | Mean convergent r | 95% CI | "
                "Mean discriminant r | 95% CI |\n")
        f.write("|---|---:|:---|---:|:---|\n")
        for label, conv, conv_lo, conv_hi, disc, disc_lo, disc_hi in [
            ("Instrument × Human (all)", mean_ih_r, mean_ih_lo, mean_ih_hi,
             off_ih_r, off_ih_lo, off_ih_hi),
            ("Instrument × Human (on-target)", mean_iho_r, mean_iho_lo, mean_iho_hi,
             off_iho_r, off_iho_lo, off_iho_hi),
            ("Instrument × Judge", mean_ij_r, mean_ij_lo, mean_ij_hi,
             off_ij_r, off_ij_lo, off_ij_hi),
            ("Human × Judge", mean_hj_r, mean_hj_lo, mean_hj_hi,
             off_hj_r, off_hj_lo, off_hj_hi),
        ]:
            f.write(f"| {label} | {conv:+.2f} | "
                    f"[{conv_lo:+.2f}, {conv_hi:+.2f}] | "
                    f"{disc:+.2f} | [{disc_lo:+.2f}, {disc_hi:+.2f}] |\n")

    print(f"\nWrote {out}")
    print(f"\nKey findings:")
    print(f"  Instrument × Human (all):   mean r = {mean_ih_r:+.2f} "
          f"[{mean_ih_lo:+.2f}, {mean_ih_hi:+.2f}]")
    print(f"  Instrument × Human (on):    mean r = {mean_iho_r:+.2f} "
          f"[{mean_iho_lo:+.2f}, {mean_iho_hi:+.2f}]")
    print(f"  Instrument × Judge:         mean r = {mean_ij_r:+.2f} "
          f"[{mean_ij_lo:+.2f}, {mean_ij_hi:+.2f}]")
    print(f"  Human × Judge:              mean r = {mean_hj_r:+.2f} "
          f"[{mean_hj_lo:+.2f}, {mean_hj_hi:+.2f}]")


if __name__ == "__main__":
    main()
