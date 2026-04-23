"""Figure 4: group-level self-report profiles across model metadata splits.

For each (split, factor) cell, we compare group mean z-scores and test the
between-group difference via permutation (shuffle the model→group assignment
10,000 times; p = fraction of |permuted stat| ≥ observed). Holm correction is
applied *within each split*.

Splits (curated, not from registry):
    - Origin region: US / China / EU / Other
    - Backbone: Transformer / Mamba-Transformer hybrid
    - Routing: Dense / MoE
    - Parameter tier: Small / Medium / Large / Unknown

Groups with n < 3 are excluded from a split. Non-significant (Holm-adjusted)
cells are drawn in a muted gray so reader sees the null result.

Usage:
    python -m analysis.make_metadata_aggregation
Output:
    analysis/output/plots/metadata_aggregation.png
    analysis/output/metadata_permutation_results.csv
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.make_hero_profile import FACTOR_COLORS
from analysis.profile_utils import (
    FACTOR_FULL_NAMES,
    FACTORS,
    ROOT,
    load_instrument_profile,
    z_score,
)


# ---------------------------------------------------------------------------
# Metadata assignments (curated from public info)
# ---------------------------------------------------------------------------
# Keys are canonical model_ids (post-R1 alias).
ORIGIN = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "US",
    "bedrock/us.anthropic.claude-sonnet-4-6": "US",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0": "US",
    "openai/gpt-5.4": "US",
    "openai/gpt-5.4-mini-2026-03-17": "US",
    "openai/gpt-5.4-nano": "US",
    "openai/gpt-oss-120b": "US",
    "gemini/gemini-3.1-pro-preview": "US",
    "gemini/gemini-3.1-flash-lite-preview": "US",
    "bedrock/google.gemma-3-27b-it": "US",
    "xai/grok-4.20-beta-0309-non-reasoning": "US",
    "openai/deepseek-v3.2": "China",
    "deepseek/deepseek-reasoner": "China",
    "dashscope/qwen3.5-plus": "China",
    "bedrock/converse/moonshotai.kimi-k2.5": "China",
    "bedrock/converse/zai.glm-5": "China",
    "bedrock/converse/minimax.minimax-m2.5": "China",
    "openai/mimo-v2-pro": "China",
    "openai/mistral-large-3": "EU",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8": "US",
    "openai/cohere-command-a": "Canada",
    "bedrock/amazon.nova-pro-v1:0": "US",
    "openai/phi-4": "US",
    "ai21/jamba-large-1.7": "Israel",
    "bedrock/converse/nvidia.nemotron-super-3-120b": "US",
}

# Backbone architecture family.
BACKBONE = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Transformer",
    "bedrock/us.anthropic.claude-sonnet-4-6": "Transformer",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0": "Transformer",
    "openai/gpt-5.4": "Transformer",
    "openai/gpt-5.4-mini-2026-03-17": "Transformer",
    "openai/gpt-5.4-nano": "Transformer",
    "openai/gpt-oss-120b": "Transformer",
    "gemini/gemini-3.1-pro-preview": "Transformer",
    "gemini/gemini-3.1-flash-lite-preview": "Transformer",
    "bedrock/google.gemma-3-27b-it": "Transformer",
    "xai/grok-4.20-beta-0309-non-reasoning": "Transformer",
    "openai/deepseek-v3.2": "Transformer",
    "deepseek/deepseek-reasoner": "Transformer",
    "dashscope/qwen3.5-plus": "Transformer",
    "bedrock/converse/moonshotai.kimi-k2.5": "Transformer",
    "bedrock/converse/zai.glm-5": "Transformer",
    "bedrock/converse/minimax.minimax-m2.5": "Transformer",
    "openai/mimo-v2-pro": "Transformer",
    "openai/mistral-large-3": "Transformer",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8": "Transformer",
    "openai/cohere-command-a": "Transformer",
    "bedrock/amazon.nova-pro-v1:0": "Transformer",
    "openai/phi-4": "Transformer",
    "ai21/jamba-large-1.7": "Mamba-Transformer",
    "bedrock/converse/nvidia.nemotron-super-3-120b": "Transformer",
}

# Dense vs. MoE routing (best-effort from public documentation; models where
# the architecture is ambiguous or closed are marked "Unknown" and excluded).
ROUTING = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Unknown",
    "bedrock/us.anthropic.claude-sonnet-4-6": "Unknown",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0": "Unknown",
    "openai/gpt-5.4": "Unknown",
    "openai/gpt-5.4-mini-2026-03-17": "Unknown",
    "openai/gpt-5.4-nano": "Unknown",
    "openai/gpt-oss-120b": "MoE",
    "gemini/gemini-3.1-pro-preview": "Unknown",
    "gemini/gemini-3.1-flash-lite-preview": "Unknown",
    "bedrock/google.gemma-3-27b-it": "Dense",
    "xai/grok-4.20-beta-0309-non-reasoning": "Unknown",
    "openai/deepseek-v3.2": "MoE",
    "deepseek/deepseek-reasoner": "MoE",
    "dashscope/qwen3.5-plus": "MoE",
    "bedrock/converse/moonshotai.kimi-k2.5": "MoE",
    "bedrock/converse/zai.glm-5": "MoE",
    "bedrock/converse/minimax.minimax-m2.5": "MoE",
    "openai/mimo-v2-pro": "Unknown",
    "openai/mistral-large-3": "Dense",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8": "MoE",
    "openai/cohere-command-a": "Dense",
    "bedrock/amazon.nova-pro-v1:0": "Unknown",
    "openai/phi-4": "Dense",
    "ai21/jamba-large-1.7": "MoE",
    "bedrock/converse/nvidia.nemotron-super-3-120b": "Dense",
}

# Size tier (approximate; "Frontier" = closed flagship, exact params unknown).
SIZE_TIER = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": "Frontier",
    "bedrock/us.anthropic.claude-sonnet-4-6": "Frontier",
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0": "Frontier",
    "openai/gpt-5.4": "Frontier",
    "openai/gpt-5.4-mini-2026-03-17": "Frontier",
    "openai/gpt-5.4-nano": "Frontier",
    "openai/gpt-oss-120b": "Large (70B+)",
    "gemini/gemini-3.1-pro-preview": "Frontier",
    "gemini/gemini-3.1-flash-lite-preview": "Frontier",
    "bedrock/google.gemma-3-27b-it": "Medium (15–70B)",
    "xai/grok-4.20-beta-0309-non-reasoning": "Frontier",
    "openai/deepseek-v3.2": "Large (70B+)",
    "deepseek/deepseek-reasoner": "Large (70B+)",
    "dashscope/qwen3.5-plus": "Large (70B+)",
    "bedrock/converse/moonshotai.kimi-k2.5": "Large (70B+)",
    "bedrock/converse/zai.glm-5": "Large (70B+)",
    "bedrock/converse/minimax.minimax-m2.5": "Large (70B+)",
    "openai/mimo-v2-pro": "Frontier",
    "openai/mistral-large-3": "Large (70B+)",
    "openai/Llama-4-Maverick-17B-128E-Instruct-FP8": "Large (70B+)",
    "openai/cohere-command-a": "Large (70B+)",
    "bedrock/amazon.nova-pro-v1:0": "Frontier",
    "openai/phi-4": "Medium (15–70B)",
    "ai21/jamba-large-1.7": "Large (70B+)",
    "bedrock/converse/nvidia.nemotron-super-3-120b": "Large (70B+)",
}

SPLITS = [
    ("Origin", ORIGIN, ["US", "China", "EU", "Canada", "Israel"]),
    ("Backbone", BACKBONE, ["Transformer", "Mamba-Transformer"]),
    ("Routing", ROUTING, ["Dense", "MoE"]),
    ("Size tier", SIZE_TIER, ["Frontier", "Large (70B+)", "Medium (15–70B)"]),
]

MIN_GROUP_N = 3
N_PERMUTATIONS = 10_000
RNG = np.random.default_rng(42)


def _group_stat(values: np.ndarray, labels: np.ndarray, groups: list[str]) -> float:
    """Between-group spread: max(group mean) − min(group mean)."""
    means = [values[labels == g].mean() for g in groups]
    return float(max(means) - min(means))


def _permutation_p(values: np.ndarray, labels: np.ndarray, groups: list[str]) -> float:
    observed = _group_stat(values, labels, groups)
    perm_stats = np.empty(N_PERMUTATIONS)
    shuffled = labels.copy()
    for i in range(N_PERMUTATIONS):
        RNG.shuffle(shuffled)
        perm_stats[i] = _group_stat(values, shuffled, groups)
    return float((perm_stats >= observed).mean())


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (n - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def _compute_split(
    split_name: str,
    mapping: dict[str, str],
    preferred_order: list[str],
    z_by_model: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (per-group means & CIs × factor, ordered groups with n≥3)."""
    df = z_by_model.copy()
    df["group"] = df.index.map(mapping)
    df = df[df["group"].notna() & (df["group"] != "Unknown")]

    counts = df["group"].value_counts()
    kept = [g for g in preferred_order if counts.get(g, 0) >= MIN_GROUP_N]
    # Append any remaining in-group with n≥3 not listed in preferred_order.
    for g in counts.index:
        if g not in kept and counts[g] >= MIN_GROUP_N:
            kept.append(g)
    df = df[df["group"].isin(kept)]

    rows = []
    for f in FACTORS:
        vals = df[f].values
        labels = df["group"].values
        p_raw = _permutation_p(vals, labels, kept)
        for g in kept:
            gvals = df.loc[df["group"] == g, f].values
            mean = gvals.mean()
            # Bootstrap 95% CI
            if len(gvals) >= 2:
                idx = RNG.integers(0, len(gvals), size=(2000, len(gvals)))
                boot = gvals[idx].mean(axis=1)
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            else:
                ci_lo = ci_hi = mean
            rows.append({
                "split": split_name,
                "factor": f,
                "group": g,
                "n": int(len(gvals)),
                "mean_z": mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_raw": p_raw,
            })

    out = pd.DataFrame(rows)
    # Holm within split, across factors (5 tests per split).
    p_per_factor = out[["factor", "p_raw"]].drop_duplicates().sort_values("factor")
    p_adj = _holm(p_per_factor["p_raw"].values)
    p_map = dict(zip(p_per_factor["factor"].values, p_adj))
    out["p_holm"] = out["factor"].map(p_map)
    out["significant"] = out["p_holm"] < 0.05
    return out, kept


def _plot(results: dict[str, tuple[pd.DataFrame, list[str]]], output_path: str) -> None:
    split_names = [s[0] for s in SPLITS]
    n_splits = len(split_names)
    n_f = len(FACTORS)

    fig, axes = plt.subplots(
        n_splits, n_f,
        figsize=(2.5 * n_f, 2.0 * n_splits + 0.8),
        sharex=True,
    )

    all_means = []
    for df, _ in results.values():
        all_means.extend(df["ci_lo"].tolist() + df["ci_hi"].tolist())
    vmax = float(np.nanmax(np.abs(all_means))) * 1.15

    for si, split_name in enumerate(split_names):
        df, groups = results[split_name]
        y_positions = np.arange(len(groups))[::-1]
        for fi, f in enumerate(FACTORS):
            ax = axes[si, fi] if n_splits > 1 else axes[fi]
            sub = df[df["factor"] == f].set_index("group").loc[groups]
            ax.axvline(0, color="0.5", linewidth=0.7, zorder=1)

            means = sub["mean_z"].values
            cis_lo = sub["ci_lo"].values
            cis_hi = sub["ci_hi"].values
            sig = sub["significant"].iloc[0]  # scalar per (split, factor)

            color = FACTOR_COLORS[f] if sig else "0.75"
            err = np.array([means - cis_lo, cis_hi - means])
            ax.errorbar(
                means, y_positions, xerr=err, fmt="o",
                color=color, ecolor=color, capsize=3, markersize=6,
                elinewidth=1.3, zorder=3,
            )
            ax.set_xlim(-vmax, vmax)
            ax.set_ylim(-0.7, len(groups) - 0.3)

            if si == 0:
                ax.set_title(FACTOR_FULL_NAMES[f], fontsize=10,
                             fontweight="bold", pad=8)
            if fi == 0:
                labels = [f"{g} (n={int(sub.loc[g, 'n'])})" for g in groups]
                ax.set_yticks(y_positions)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_ylabel(split_name, fontsize=10, fontweight="bold",
                              labelpad=12)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(axis="x", labelsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.supxlabel("z-score (pool-relative, 25-model pool)", fontsize=9, y=0.02)
    fig.suptitle(
        "Group-level profiles by metadata split (Holm-adjusted permutation tests)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metadata aggregation → {output_path}")


def main():
    inst = load_instrument_profile().set_index("model_id")
    z = z_score(inst.reset_index(), FACTORS).set_index("model_id")

    results = {}
    all_rows = []
    for split_name, mapping, preferred in SPLITS:
        df, groups = _compute_split(split_name, mapping, preferred, z)
        results[split_name] = (df, groups)
        all_rows.append(df)

    out_plot = os.path.join(ROOT, "analysis", "output", "plots", "metadata_aggregation.png")
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    _plot(results, out_plot)

    all_df = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(ROOT, "analysis", "output", "metadata_permutation_results.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"Saved permutation results → {csv_path}")

    # Console summary
    print("\nSurviving effects (Holm-adjusted p < .05):")
    sig = all_df[all_df["significant"]].drop_duplicates(subset=["split", "factor"])
    if sig.empty:
        print("  (none)")
    else:
        for _, r in sig.iterrows():
            print(f"  {r['split']:>10} × {r['factor']}: p_holm={r['p_holm']:.3f}")


if __name__ == "__main__":
    main()
