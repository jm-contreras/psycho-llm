"""Figure 1 (replacement): model × factor profile of self-report z-scores.

Layouts:
  --layout bars      : grouped bar chart, one cluster per model (the "personality" view).
  --layout smalls    : 5 small-multiple panels, one per factor, shared model order.
  --layout panels    : one panel per factor, models as rows (dot version).
  --layout combined  : one row per model, all 5 factors on shared x-axis.
  --layout all       : render every layout.

Usage:
    python -m analysis.make_hero_profile --subset popular --layout bars
    python -m analysis.make_hero_profile --subset all --layout smalls

Outputs:
    analysis/output/plots/hero_profile_{subset}_{layout}.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.profile_utils import (
    FACTOR_FULL_NAMES,
    FACTOR_HIGH_POLE,
    FACTORS,
    POPULAR_MODEL_IDS,
    ROOT,
    display_name,
    load_instrument_profile,
    z_score,
)


# Semantic palette: each factor's color evokes its construct.
#   RE (Responsiveness) — warm coral: engaged, helpful, warm
#   DE (Deference)      — soft blue:  yielding, cool, compliant
#   BO (Boldness)       — crimson:    assertive, confident, high-energy
#   GU (Guardedness)    — slate gray: cautious, withholding, neutral
#   VB (Verbosity)      — leaf green: expansive, flowing, growth
FACTOR_COLORS = {
    "RE": "#E07A5F",  # warm coral
    "DE": "#7FA9C9",  # soft blue
    "BO": "#C14953",  # crimson
    "GU": "#6B7280",  # slate gray
    "VB": "#6DA34D",  # leaf green
}


def _prepare(subset: str):
    inst = load_instrument_profile()
    inst_z = z_score(inst, FACTORS).set_index("model_id")

    if subset == "popular":
        missing = [m for m in POPULAR_MODEL_IDS if m not in inst_z.index]
        if missing:
            raise ValueError(f"Popular subset missing from profile: {missing}")
        display = inst_z.loc[POPULAR_MODEL_IDS].copy()
    else:
        display = inst_z.copy()

    display = display.sort_values("RE", ascending=False)
    row_labels = [display_name(m) for m in display.index]
    return display, row_labels


def plot_panels(subset: str, output_path: str) -> None:
    display, row_labels = _prepare(subset)
    n_models, n_factors = len(display), len(FACTORS)

    fig, axes = plt.subplots(
        1, n_factors,
        figsize=(2.1 * n_factors, max(3.5, 0.28 * n_models + 1.4)),
        sharey=True,
    )
    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    xlim = (-vmax * 1.1, vmax * 1.1)
    y_positions = np.arange(n_models)[::-1]
    cmap = plt.get_cmap("RdBu_r")

    for i, f in enumerate(FACTORS):
        ax = axes[i]
        vals = display[f].values
        colors = cmap(0.5 + 0.5 * vals / vmax)
        ax.axvline(0, color="0.7", linewidth=0.8, zorder=1)
        ax.scatter(vals, y_positions, c=colors, s=60,
                   edgecolor="0.25", linewidth=0.5, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.8, n_models - 0.2)
        ax.set_title(FACTOR_FULL_NAMES[f], fontsize=11)
        ax.set_xlabel(f"↑ {FACTOR_HIGH_POLE[f]}", fontsize=8, color="0.35")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", linestyle=":", color="0.85", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(row_labels, fontsize=8)
    fig.supxlabel("z-score (pool-relative, 25-model pool)", fontsize=10, y=0.02)
    fig.suptitle("Model profiles across five AI-native factors (self-report)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (panels, {subset}) → {output_path}")


def plot_combined(subset: str, output_path: str) -> None:
    """One row per model; factors plotted as colored dots on a shared x-axis.

    Vertical alignment across rows lets the reader track each factor column.
    Factor identity is encoded by color + a top x-axis legend band.
    """
    display, row_labels = _prepare(subset)
    n_models = len(display)

    fig, ax = plt.subplots(
        figsize=(8.5, max(3.5, 0.38 * n_models + 1.5)),
    )
    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    xlim = (-vmax * 1.1, vmax * 1.1)
    y_positions = np.arange(n_models)[::-1]

    ax.axvline(0, color="0.7", linewidth=0.8, zorder=1)

    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="0.96", zorder=0)

    # Vertical jitter per factor so overlapping dots separate cleanly.
    n_f = len(FACTORS)
    offsets = np.linspace(-0.18, 0.18, n_f)
    for f, dy in zip(FACTORS, offsets):
        vals = display[f].values
        ax.scatter(vals, y_positions + dy, c=FACTOR_COLORS[f], s=45,
                   edgecolor="0.25", linewidth=0.4, zorder=3,
                   label=FACTOR_FULL_NAMES[f])

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, n_models - 0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("z-score (pool-relative, 25-model pool)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.04 + 2.0 / max(n_models, 1)),
        ncol=5, frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.2,
    )
    title_pad = 32 + int(180 / max(n_models, 1))
    ax.set_title("Model profiles across five AI-native factors (self-report)",
                 fontsize=12, pad=title_pad)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (combined, {subset}) → {output_path}")


# Family ordering for alphabetical-by-family model sort (used by `smalls` layout).
# Earlier entries = top of plot.
MODEL_FAMILY_ORDER = [
    # Anthropic
    "bedrock/us.anthropic.claude-opus-4-6-v1",
    "bedrock/us.anthropic.claude-sonnet-4-6",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    # OpenAI
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini-2026-03-17",
    "openai/gpt-5.4-nano",
    "openai/gpt-oss-120b",
    # Google
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
    "bedrock/google.gemma-3-27b-it",
    # xAI
    "xai/grok-4.20-beta-0309-non-reasoning",
    # Meta
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8",
    # Mistral
    "openai/mistral-large-3",
    # DeepSeek
    "deepseek/deepseek-reasoner",
    "openai/deepseek-v3.2",
    # Alibaba
    "dashscope/qwen3.5-plus",
    # Moonshot
    "bedrock/converse/moonshotai.kimi-k2.5",
    # Zhipu
    "bedrock/converse/zai.glm-5",
    # MiniMax
    "bedrock/converse/minimax.minimax-m2.5",
    # Xiaomi
    "openai/mimo-v2-pro",
    # Cohere
    "openai/cohere-command-a",
    # Amazon
    "bedrock/amazon.nova-pro-v1:0",
    # Microsoft
    "openai/phi-4",
    # AI21
    "ai21/jamba-large-1.7",
    # NVIDIA
    "bedrock/converse/nvidia.nemotron-super-3-120b",
]


def _prepare_family_order(subset: str):
    """Like _prepare, but rows follow MODEL_FAMILY_ORDER (families grouped)."""
    inst = load_instrument_profile()
    inst_z = z_score(inst, FACTORS).set_index("model_id")

    if subset == "popular":
        missing = [m for m in POPULAR_MODEL_IDS if m not in inst_z.index]
        if missing:
            raise ValueError(f"Popular subset missing from profile: {missing}")
        ids = [m for m in MODEL_FAMILY_ORDER if m in POPULAR_MODEL_IDS]
    else:
        ids = [m for m in MODEL_FAMILY_ORDER if m in inst_z.index]
    display = inst_z.loc[ids].copy()
    row_labels = [display_name(m) for m in display.index]
    return display, row_labels


def plot_bars(subset: str, output_path: str) -> None:
    """Grouped horizontal bar chart: one cluster per model, five thin bars per cluster.

    Reader sees each model's "personality" as a small fingerprint.
    """
    display, row_labels = _prepare_family_order(subset)
    n_models, n_f = len(display), len(FACTORS)

    fig, ax = plt.subplots(
        figsize=(8.0, max(4.0, 0.55 * n_models + 1.4)),
    )

    bar_thickness = 0.15
    # Within-cluster vertical offsets: top = first factor.
    offsets = np.linspace(
        (n_f - 1) / 2 * bar_thickness,
        -(n_f - 1) / 2 * bar_thickness,
        n_f,
    )

    cluster_centers = np.arange(n_models)[::-1]  # top = first in order

    ax.axvline(0, color="0.5", linewidth=0.8, zorder=2)

    for i, y in enumerate(cluster_centers):
        if i % 2 == 0:
            cluster_half = (n_f * bar_thickness) / 2 + 0.08
            ax.axhspan(y - cluster_half, y + cluster_half, color="0.96", zorder=0)

    for f, dy in zip(FACTORS, offsets):
        vals = display[f].values
        ax.barh(
            cluster_centers + dy, vals,
            height=bar_thickness * 0.9,
            color=FACTOR_COLORS[f],
            edgecolor="none",
            label=FACTOR_FULL_NAMES[f],
            zorder=3,
        )

    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    ax.set_xlim(-vmax * 1.15, vmax * 1.15)
    cluster_half = (n_f * bar_thickness) / 2 + 0.18
    ax.set_ylim(cluster_centers.min() - cluster_half,
                cluster_centers.max() + cluster_half)
    ax.set_yticks(cluster_centers)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("z-score (pool-relative, 25-model pool)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Place legend just above the axis; use figure-level title above legend.
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=5, frameon=False, fontsize=9,
        handletextpad=0.3, columnspacing=1.2,
    )
    fig.suptitle(
        "Model profiles across five AI-native factors (self-report)",
        fontsize=12, y=1.0,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (bars, {subset}) → {output_path}")


def plot_smalls(subset: str, output_path: str) -> None:
    """5 small multiples; each panel = one factor; shared family-grouped model order."""
    display, row_labels = _prepare_family_order(subset)
    n_models, n_f = len(display), len(FACTORS)

    fig, axes = plt.subplots(
        1, n_f,
        figsize=(2.2 * n_f, max(4.0, 0.32 * n_models + 1.2)),
        sharey=True,
    )
    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    xlim = (-vmax * 1.15, vmax * 1.15)
    y_positions = np.arange(n_models)[::-1]

    for i, f in enumerate(FACTORS):
        ax = axes[i]
        vals = display[f].values
        for j, y in enumerate(y_positions):
            if j % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="0.96", zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.7, zorder=1)
        ax.barh(y_positions, vals, height=0.72,
                color=FACTOR_COLORS[f], edgecolor="none", zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, n_models - 0.3)
        # Two-line panel header: factor name, then direction hint.
        ax.set_title(FACTOR_FULL_NAMES[f], fontsize=12, fontweight="bold",
                     pad=18)
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
    fig.supxlabel("z-score (pool-relative, 25-model pool)", fontsize=10, y=0.02)
    fig.suptitle(
        "Model profiles across five AI-native factors (self-report)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (smalls, {subset}) → {output_path}")


def _prepare_sorted_by_RE(subset: str):
    """Like _prepare_family_order, but sort by Responsiveness descending."""
    inst = load_instrument_profile()
    inst_z = z_score(inst, FACTORS).set_index("model_id")

    if subset == "popular":
        missing = [m for m in POPULAR_MODEL_IDS if m not in inst_z.index]
        if missing:
            raise ValueError(f"Popular subset missing from profile: {missing}")
        display = inst_z.loc[POPULAR_MODEL_IDS].copy()
    else:
        display = inst_z.copy()
    display = display.sort_values("RE", ascending=False)
    row_labels = [display_name(m) for m in display.index]
    return display, row_labels


def plot_bars_vertical(subset: str, output_path: str) -> None:
    """Grouped vertical bar chart: one cluster per model (x), 5 factors per cluster.

    Hero view: models sorted by Responsiveness descending so that similar
    profiles cluster visually left-to-right.
    """
    display, row_labels = _prepare_sorted_by_RE(subset)
    n_models, n_f = len(display), len(FACTORS)

    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.62 * n_models + 1.4), 5.2),
    )

    bar_width = 0.15
    offsets = np.linspace(
        -(n_f - 1) / 2 * bar_width,
        (n_f - 1) / 2 * bar_width,
        n_f,
    )
    cluster_centers = np.arange(n_models)

    ax.axhline(0, color="0.5", linewidth=0.8, zorder=2)

    for i, x in enumerate(cluster_centers):
        if i % 2 == 0:
            cluster_half = (n_f * bar_width) / 2 + 0.08
            ax.axvspan(x - cluster_half, x + cluster_half, color="0.96", zorder=0)

    for f, dx in zip(FACTORS, offsets):
        vals = display[f].values
        ax.bar(
            cluster_centers + dx, vals,
            width=bar_width * 0.9,
            color=FACTOR_COLORS[f],
            edgecolor="none",
            label=FACTOR_FULL_NAMES[f],
            zorder=3,
        )

    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    ax.set_ylim(-vmax * 1.15, vmax * 1.15)
    cluster_half = (n_f * bar_width) / 2 + 0.18
    ax.set_xlim(cluster_centers.min() - cluster_half,
                cluster_centers.max() + cluster_half)
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("z-score (pool-relative, 25-model pool)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=5, frameon=False, fontsize=9,
        handletextpad=0.3, columnspacing=1.2,
    )
    fig.suptitle(
        "Model profiles across five AI-native factors (self-report)",
        fontsize=12, y=1.0,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (bars_vertical, {subset}) → {output_path}")


def plot_smalls_vertical(subset: str, output_path: str) -> None:
    """5 small multiples stacked vertically; each panel = one factor; x = model."""
    display, row_labels = _prepare_family_order(subset)
    n_models, n_f = len(display), len(FACTORS)

    fig, axes = plt.subplots(
        n_f, 1,
        figsize=(max(6.5, 0.42 * n_models + 1.2), 1.4 * n_f + 1.2),
        sharex=True,
    )
    vmax = float(np.nanmax(np.abs(display[FACTORS].values)))
    ylim = (-vmax * 1.15, vmax * 1.15)
    x_positions = np.arange(n_models)

    for i, f in enumerate(FACTORS):
        ax = axes[i]
        vals = display[f].values
        for j, x in enumerate(x_positions):
            if j % 2 == 0:
                ax.axvspan(x - 0.5, x + 0.5, color="0.96", zorder=0)
        ax.axhline(0, color="0.5", linewidth=0.7, zorder=1)
        ax.bar(x_positions, vals, width=0.72,
               color=FACTOR_COLORS[f], edgecolor="none", zorder=3)
        ax.set_ylim(*ylim)
        ax.set_ylabel(FACTOR_FULL_NAMES[f], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(row_labels, rotation=45, ha="right", fontsize=9)
    fig.supylabel("z-score (pool-relative, 25-model pool)", fontsize=10)
    fig.suptitle(
        "Model profiles across five AI-native factors (self-report)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (smalls_vertical, {subset}) → {output_path}")


def plot_parallel(subset: str, output_path: str) -> None:
    """Parallel-coordinates plot: x = factor, y = z-score, one line per model.

    For `all`, fade all models to light grey and highlight the popular-9 on top
    to avoid the spaghetti-plot problem.
    """
    display, row_labels = _prepare_family_order(subset)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x_positions = np.arange(len(FACTORS))

    ax.axhline(0, color="0.7", linewidth=0.8, zorder=1)
    for x in x_positions:
        ax.axvline(x, color="0.92", linewidth=0.6, zorder=0)

    # Color palette: distinct colors per model for popular; grey+highlight for all.
    if subset == "popular":
        palette = plt.get_cmap("tab10").colors
        for i, (model_id, row) in enumerate(display.iterrows()):
            vals = row[FACTORS].values.astype(float)
            color = palette[i % len(palette)]
            ax.plot(x_positions, vals, color=color, linewidth=1.8,
                    marker="o", markersize=6, label=display_name(model_id),
                    alpha=0.9, zorder=3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=False)
    else:
        highlight_ids = [m for m in POPULAR_MODEL_IDS if m in display.index]
        palette = plt.get_cmap("tab10").colors
        # Background: all non-highlighted models in light grey.
        for model_id, row in display.iterrows():
            if model_id in highlight_ids:
                continue
            vals = row[FACTORS].values.astype(float)
            ax.plot(x_positions, vals, color="0.75", linewidth=0.8,
                    alpha=0.5, zorder=2)
        # Foreground: popular-9 in color.
        for i, mid in enumerate(highlight_ids):
            vals = display.loc[mid, FACTORS].values.astype(float)
            color = palette[i % len(palette)]
            ax.plot(x_positions, vals, color=color, linewidth=1.8,
                    marker="o", markersize=5, label=display_name(mid),
                    alpha=0.95, zorder=4)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=False,
                  title="Highlighted", title_fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([FACTOR_FULL_NAMES[f] for f in FACTORS], fontsize=10)
    ax.set_ylabel("z-score (pool-relative, 25-model pool)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Model profiles across five AI-native factors (self-report)",
                 fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hero profile (parallel, {subset}) → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["all", "popular"], default="all")
    parser.add_argument(
        "--layout",
        choices=["bars", "bars_vertical", "smalls", "smalls_vertical",
                 "parallel", "panels", "combined", "all"],
        default="all",
    )
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "analysis", "output", "plots")
    os.makedirs(out_dir, exist_ok=True)

    layouts = (["bars", "bars_vertical", "smalls", "smalls_vertical",
                "parallel", "panels", "combined"]
               if args.layout == "all" else [args.layout])
    renderers = {
        "bars": plot_bars,
        "bars_vertical": plot_bars_vertical,
        "smalls": plot_smalls,
        "smalls_vertical": plot_smalls_vertical,
        "parallel": plot_parallel,
        "panels": plot_panels,
        "combined": plot_combined,
    }
    for layout in layouts:
        out = os.path.join(out_dir, f"hero_profile_{args.subset}_{layout}.png")
        renderers[layout](args.subset, out)


if __name__ == "__main__":
    main()
