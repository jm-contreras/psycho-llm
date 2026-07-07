"""Leave-one-family-out jackknife for the model-level validity correlations.

The 25 model configurations cluster into 17 developer families (registry
`provider` field); within-family models share training lineages, so the
effective N is below 25. This check drops one family at a time and recomputes
the three model-level correlation columns of the predictive-validity table
(Instrument x Human all-prompts, Instrument x Judge, Human x Judge) per
factor, reporting the observed r and the min/max across the 17 leave-outs.

Usage:
    python -m analysis.family_jackknife

Output:
    analysis/output/family_jackknife.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data_loader import OUTPUT_DIR
from analysis.predictive_validity import (
    FACTORS, FACTOR_NAMES,
    load_human_ratings, load_judge_ensemble, load_instrument_scores,
    model_level_human_scores,
)

REGISTRY = Path(__file__).parent.parent / "model_registry.json"


def load_family_map() -> dict[str, str]:
    reg = json.loads(REGISTRY.read_text())
    fam = {m["litellm_model_id"]: m["provider"] for m in reg["bedrock_models"]}
    return fam


def main() -> None:
    inst = load_instrument_scores().set_index("model_id")[FACTORS]
    judge = load_judge_ensemble()
    human = model_level_human_scores(load_human_ratings(), "all")

    models = sorted(set(inst.index) & set(judge.index) & set(human.index))
    inst, judge, human = inst.loc[models], judge.loc[models], human.loc[models]

    fam_map = load_family_map()
    families = sorted({fam_map.get(m, m) for m in models})
    fam_of = np.array([fam_map.get(m, m) for m in models])

    comparisons = [
        ("Instrument x Human", inst, human),
        ("Instrument x Judge", inst, judge),
        ("Human x Judge", human, judge),
    ]

    lines = [
        "# Leave-One-Family-Out Jackknife",
        "",
        f"N = {len(models)} models in {len(families)} developer families. "
        "Each row: observed model-level Pearson r and the [min, max] across "
        "the 17 correlations obtained by dropping one family at a time. "
        "`sign flip` marks cells where any leave-out changes the sign of r "
        "for |r_obs| >= .20.",
        "",
        "| Factor | " + " | ".join(c[0] for c in comparisons) + " |",
        "|---|" + "---|" * len(comparisons),
    ]

    for f in FACTORS:
        cells = []
        for _, a, b in comparisons:
            x, y = a[f].to_numpy(), b[f].to_numpy()
            r_obs = stats.pearsonr(x, y)[0]
            jack = []
            for fam in families:
                keep = fam_of != fam
                if keep.sum() < 4:
                    continue
                jack.append(stats.pearsonr(x[keep], y[keep])[0])
            lo, hi = min(jack), max(jack)
            flip = " (sign flip)" if abs(r_obs) >= 0.20 and (lo < 0 < hi) else ""
            cells.append(f"{r_obs:+.2f} [{lo:+.2f}, {hi:+.2f}]{flip}")
        lines.append(f"| {FACTOR_NAMES[f]} | " + " | ".join(cells) + " |")

    out = OUTPUT_DIR / "family_jackknife.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
