"""Figure 3: per-model Big Five (OCEAN) profile across 25 models.

Mirrors hero_profile_all_smalls layout (5 panels, horizontal bars, alphabetical-
by-family row order, pool-relative z-scoring across the 25 models). Uses the
forward-only Extraversion scale (E_fwd) to sidestep acquiescence on reverse E
items (see bfi_analysis.py §1b).

Usage:
    python -m analysis.make_ocean_profile
Output:
    analysis/output/plots/ocean_profile_all_smalls.png
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.bfi_analysis import bfi_dimension_scores
from analysis.data_loader import (
    filter_success,
    load_responses,
    recode_reverse_items,
)
from analysis.make_hero_profile import MODEL_FAMILY_ORDER
from analysis.profile_utils import ROOT, R1_ALIASES, display_name, z_score


OCEAN_DIMS = ["O", "C", "E_fwd", "A", "N"]

OCEAN_FULL_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E_fwd": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

OCEAN_HIGH_POLE = {
    "O": "more open",
    "C": "more conscientious",
    "E_fwd": "more extraverted",
    "A": "more agreeable",
    "N": "more neurotic",
}

# Semantic palette, distinct from AI-native factor colors.
OCEAN_COLORS = {
    "O": "#8E7CC3",   # violet  — imagination/openness
    "C": "#3D5A80",   # deep blue — order/discipline
    "E_fwd": "#E8A03C",  # amber — warmth/energy
    "A": "#D97A95",   # rose — prosocial warmth
    "N": "#B5533C",   # rust — distress
}


def _prepare() -> tuple[list[str], list[str], np.ndarray]:
    df_all = load_responses()
    df_success = recode_reverse_items(filter_success(df_all))
    scores, _ = bfi_dimension_scores(df_success)

    scores = scores.copy()
    scores["model_id"] = scores["model_id"].replace(R1_ALIASES)
    # Collapse any duplicates introduced by alias remap.
    scores = scores.groupby("model_id", as_index=False)[OCEAN_DIMS].mean()

    ids = [m for m in MODEL_FAMILY_ORDER if m in set(scores["model_id"])]
    scores = scores.set_index("model_id").loc[ids].reset_index()

    z = z_score(scores, OCEAN_DIMS).set_index("model_id").loc[ids]
    row_labels = [display_name(m) for m in ids]
    return ids, row_labels, z[OCEAN_DIMS].values


def plot_ocean(output_path: str) -> None:
    ids, row_labels, arr = _prepare()
    n_models, n_d = arr.shape

    fig, axes = plt.subplots(
        1, n_d,
        figsize=(2.4 * n_d, max(6.0, 0.32 * n_models + 1.6)),
        sharey=True,
    )

    vmax = float(np.nanmax(np.abs(arr)))
    xlim = (-vmax * 1.15, vmax * 1.15)
    y_positions = np.arange(n_models)[::-1]

    for i, d in enumerate(OCEAN_DIMS):
        ax = axes[i]
        for j, y in enumerate(y_positions):
            if j % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="0.96", zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.7, zorder=1)

        vals = arr[:, i]
        color = OCEAN_COLORS[d]
        ax.barh(y_positions, vals, color=color, edgecolor="0.25",
                linewidth=0.4, height=0.72, zorder=2)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, n_models - 0.3)
        ax.set_title(OCEAN_FULL_NAMES[d], fontsize=12, fontweight="bold", pad=18)
        ax.text(
            0.5, 1.005, f"{OCEAN_HIGH_POLE[d]} →",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="0.35",
        )
        ax.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(row_labels, fontsize=9)

    fig.supxlabel("z-score (pool-relative, 25-model pool)", fontsize=10, y=0.02)
    fig.suptitle(
        "Big Five (BFI-44) profiles across 25 models",
        fontsize=12, y=1.01,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved OCEAN profile → {output_path}")


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "analysis", "output", "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_ocean(os.path.join(out_dir, "ocean_profile_all_smalls.png"))
