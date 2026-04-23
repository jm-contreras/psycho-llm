"""Figure 2: self-report vs. human-rating profile, dumbbell per (model, factor).

Layout mirrors hero_profile_all_smalls: 5 panels (one per factor), models as
rows, shared alphabetical-by-family order. In each cell:
    - left endpoint = self-report z-score
    - right endpoint = human-rating z-score
    - connecting segment = disagreement between methods

Z-scoring is applied independently within each source (across the 25 models),
so both endpoints live in comparable pool-relative space.

Usage:
    python -m analysis.make_method_convergence
Output:
    analysis/output/plots/method_convergence.png
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.profile_utils import (
    FACTOR_FULL_NAMES,
    FACTOR_HIGH_POLE,
    FACTORS,
    ROOT,
    display_name,
    load_human_profile,
    load_instrument_profile,
    z_score,
)
from analysis.make_hero_profile import FACTOR_COLORS, MODEL_FAMILY_ORDER


SELF_MARKER = "s"  # robot/machine
HUMAN_MARKER = "o"  # human


def _prepare() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    inst = load_instrument_profile().set_index("model_id")
    hum = load_human_profile().set_index("model_id")

    ids = [m for m in MODEL_FAMILY_ORDER if m in inst.index and m in hum.index]
    inst = inst.loc[ids]
    hum = hum.loc[ids]

    inst_z = z_score(inst.reset_index(), FACTORS).set_index("model_id").loc[ids]
    hum_z = z_score(hum.reset_index(), FACTORS).set_index("model_id").loc[ids]

    row_labels = [display_name(m) for m in ids]
    return ids, row_labels, inst_z[FACTORS].values, hum_z[FACTORS].values


def plot_dumbbell(output_path: str) -> None:
    ids, row_labels, inst_arr, hum_arr = _prepare()
    n_models, n_f = inst_arr.shape

    fig, axes = plt.subplots(
        1, n_f,
        figsize=(2.4 * n_f, max(6.0, 0.32 * n_models + 1.6)),
        sharey=True,
    )

    vmax = float(np.nanmax(np.abs(np.concatenate([inst_arr.ravel(), hum_arr.ravel()]))))
    xlim = (-vmax * 1.15, vmax * 1.15)
    y_positions = np.arange(n_models)[::-1]

    for i, f in enumerate(FACTORS):
        ax = axes[i]
        for j, y in enumerate(y_positions):
            if j % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="0.96", zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.7, zorder=1)

        x_self = inst_arr[:, i]
        x_hum = hum_arr[:, i]
        color = FACTOR_COLORS[f]

        # Connecting segments
        for k, y in enumerate(y_positions):
            if np.isnan(x_self[k]) or np.isnan(x_hum[k]):
                continue
            ax.plot([x_self[k], x_hum[k]], [y, y],
                    color=color, alpha=0.45, linewidth=1.4, zorder=2)

        # Endpoints: self-report = filled square (machine), human = hollow circle
        ax.scatter(x_self, y_positions, marker=SELF_MARKER,
                   s=30, color=color, edgecolor="0.2", linewidth=0.5,
                   zorder=3)
        ax.scatter(x_hum, y_positions, marker=HUMAN_MARKER,
                   s=32, facecolor="white",
                   edgecolor=color, linewidth=1.3, zorder=4)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, n_models - 0.3)
        ax.set_title(FACTOR_FULL_NAMES[f], fontsize=12, fontweight="bold", pad=18)
        ax.text(
            0.5, 1.005, f"{FACTOR_HIGH_POLE[f]} →",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="0.35",
        )
        ax.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(row_labels, fontsize=9)

    # Legend as figure-level shared element.
    legend_handles = [
        plt.Line2D([], [], marker=SELF_MARKER, linestyle="", color="0.3",
                   markersize=7, label="Self-report"),
        plt.Line2D([], [], marker=HUMAN_MARKER, linestyle="",
                   markerfacecolor="white", markeredgecolor="0.3",
                   markeredgewidth=1.3, markersize=7, label="Human raters"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=10)

    fig.supxlabel("z-score (pool-relative, 25-model pool)", fontsize=10, y=0.02)
    fig.suptitle(
        "Self-report vs. human-rated profiles across five AI-native factors",
        fontsize=12, y=1.03,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved method convergence (dumbbell) → {output_path}")


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "analysis", "output", "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_dumbbell(os.path.join(out_dir, "method_convergence.png"))
