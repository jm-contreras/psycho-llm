"""Section 3: Dimension-level coherence — inter-item correlations, Cronbach's alpha."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import get_short_model_name


def inter_item_correlation_matrix(
    score_matrix: pd.DataFrame,
    item_ids: list[str],
) -> pd.DataFrame:
    """Pairwise Pearson correlations among items within a set.

    score_matrix: models×items pivot. Returns correlation DataFrame.
    """
    sub = score_matrix[[c for c in item_ids if c in score_matrix.columns]]
    return sub.corr(min_periods=3)


def plot_inter_item_heatmap(
    corr_matrix: pd.DataFrame,
    dimension: str,
    output_path: str,
) -> None:
    """Save a heatmap PNG of the inter-item correlation matrix."""
    n = len(corr_matrix)
    figsize = max(6, n * 0.4)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=n <= 25,
        fmt=".2f" if n <= 25 else "",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Inter-item correlations: {dimension}", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cronbachs_alpha(
    score_matrix: pd.DataFrame,
    item_ids: list[str],
) -> float:
    """Compute Cronbach's alpha for a set of items.

    Uses models×items matrix (rows=models, cols=items).
    Drops rows with any NaN in the item subset.
    """
    sub = score_matrix[[c for c in item_ids if c in score_matrix.columns]].dropna()
    k = sub.shape[1]
    n = sub.shape[0]

    if k < 2 or n < 2:
        return np.nan

    item_vars = sub.var(axis=0, ddof=1)
    total_var = sub.sum(axis=1).var(ddof=1)

    if total_var == 0:
        return np.nan

    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def dimension_reliability_table(
    score_matrix: pd.DataFrame,
    means_df: pd.DataFrame,
    item_type: str = "direct",
) -> pd.DataFrame:
    """Cronbach's alpha and mean inter-item r per dimension.

    Computes separately for the specified item_type (default: direct).
    Flags dimensions with alpha < 0.60.
    """
    type_items = means_df[means_df["item_type"] == item_type]
    dims = sorted(type_items["dimension"].unique())

    rows = []
    for dim in dims:
        dim_items = sorted(
            type_items[type_items["dimension"] == dim]["item_id"].unique()
        )
        present = [c for c in dim_items if c in score_matrix.columns]
        if len(present) < 2:
            continue

        alpha = cronbachs_alpha(score_matrix, present)
        corr = inter_item_correlation_matrix(score_matrix, present)
        # Mean of off-diagonal correlations
        mask = ~np.eye(len(corr), dtype=bool)
        mean_r = corr.values[mask].mean() if mask.sum() > 0 else np.nan

        rows.append({
            "dimension": dim,
            "n_items": len(present),
            "alpha": alpha,
            "mean_inter_item_r": mean_r,
            "flagged_low_alpha": alpha < 0.60 if not np.isnan(alpha) else True,
        })

    return pd.DataFrame(rows)


def run_dimension_coherence(
    score_matrix_direct: pd.DataFrame,
    score_matrix_scenario: pd.DataFrame,
    means_df: pd.DataFrame,
    plots_dir: str,
) -> dict:
    """Run all Section 3 analyses.

    Returns dict with 'reliability_direct', 'reliability_scenario', 'plot_paths'.
    """
    rel_direct = dimension_reliability_table(score_matrix_direct, means_df, "direct")
    rel_scenario = dimension_reliability_table(
        score_matrix_scenario, means_df, "scenario"
    )

    plot_paths = []
    type_items = means_df[means_df["item_type"] == "direct"]

    for dim in sorted(type_items["dimension"].unique()):
        dim_items = sorted(
            type_items[type_items["dimension"] == dim]["item_id"].unique()
        )
        present = [c for c in dim_items if c in score_matrix_direct.columns]
        if len(present) < 2:
            continue

        corr = inter_item_correlation_matrix(score_matrix_direct, present)
        safe_dim = dim.replace("/", "-").replace(" ", "_")
        path = f"{plots_dir}/heatmap_{safe_dim}.png"
        plot_inter_item_heatmap(corr, dim, path)
        plot_paths.append(path)

    return {
        "reliability_direct": rel_direct,
        "reliability_scenario": rel_scenario,
        "plot_paths": plot_paths,
    }
