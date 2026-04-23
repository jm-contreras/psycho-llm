"""Section 2: Item-level quality — variance, refusal rates, item-total correlations."""

import pandas as pd
import numpy as np


def item_variance_across_models(means_df: pd.DataFrame) -> pd.DataFrame:
    """Per-item statistics of model means.

    Flag items where the range of model means < 0.5 (zero-variance candidates
    for dropping, per preregistration).
    """
    stats = (
        means_df.groupby(["item_id", "dimension", "item_type"])["mean_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    stats.columns = [
        "item_id", "dimension", "item_type",
        "mean_of_means", "std_of_means", "min_mean", "max_mean", "n_models",
    ]
    stats["range_of_means"] = stats["max_mean"] - stats["min_mean"]
    stats["flagged_zero_var"] = stats["range_of_means"] < 0.5
    stats = stats.sort_values("item_id").reset_index(drop=True)
    return stats


def item_refusal_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-item refusal rate across all models. Flag >10%.

    df should be the full (all-status) DataFrame.
    """
    # Only consider models with real data (more than 1 row)
    model_counts = df.groupby("model_id").size()
    real_models = model_counts[model_counts > 1].index
    df_real = df[df["model_id"].isin(real_models)]

    item_totals = df_real.groupby("item_id").size().rename("total")
    item_refusals = (
        df_real[df_real["status"] == "refusal"]
        .groupby("item_id")
        .size()
        .rename("n_refusal")
    )

    result = pd.DataFrame({"total": item_totals}).join(item_refusals, how="left")
    result["n_refusal"] = result["n_refusal"].fillna(0).astype(int)
    result["refusal_rate"] = result["n_refusal"] / result["total"]
    result["flagged_refusal"] = result["refusal_rate"] > 0.10
    result = result.reset_index().sort_values("item_id")
    return result


def item_total_correlations(
    score_matrix: pd.DataFrame,
    item_ids: list[str],
) -> pd.DataFrame:
    """Corrected item-total correlation within a set of items.

    For each item, correlate it with the sum of all OTHER items in the set.
    score_matrix: models×items pivot table.
    Returns DataFrame: item_id, item_total_r, flagged (r < 0.30).
    """
    # Subset to the items of interest, drop models with all-NaN
    sub = score_matrix[
        [c for c in item_ids if c in score_matrix.columns]
    ].dropna(how="all")

    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return pd.DataFrame(
            {"item_id": item_ids, "item_total_r": np.nan, "flagged_low_r": False}
        )

    results = []
    for item in sub.columns:
        others = sub.drop(columns=[item])
        total = others.mean(axis=1)  # mean of other items (handles NaN better than sum)
        valid = sub[item].notna() & total.notna()
        if valid.sum() < 3:
            r = np.nan
        else:
            r = sub.loc[valid, item].corr(total[valid])
        results.append({"item_id": item, "item_total_r": r})

    result = pd.DataFrame(results)
    result["flagged_low_r"] = result["item_total_r"] < 0.30
    return result


def all_item_total_correlations(
    score_matrix: pd.DataFrame,
    means_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run item-total correlations for every dimension, separately by item_type.

    Returns concatenated DataFrame with dimension column added.
    """
    all_results = []

    for item_type in ["direct", "scenario"]:
        type_items = means_df[means_df["item_type"] == item_type]
        dims = type_items["dimension"].unique()

        for dim in sorted(dims):
            dim_items = sorted(
                type_items[type_items["dimension"] == dim]["item_id"].unique()
            )
            if len(dim_items) < 2:
                continue
            r_df = item_total_correlations(score_matrix, dim_items)
            r_df["dimension"] = dim
            r_df["item_type"] = item_type
            all_results.append(r_df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def run_item_quality(
    df: pd.DataFrame,
    means_df: pd.DataFrame,
    score_matrix_direct: pd.DataFrame,
    score_matrix_scenario: pd.DataFrame,
) -> dict:
    """Run all Section 2 checks.

    Returns dict with keys: 'item_variance', 'item_refusals', 'item_total_r'.
    """
    variance = item_variance_across_models(means_df)
    refusals = item_refusal_rates(df)

    # Item-total correlations on the appropriate matrix per item type
    r_direct = all_item_total_correlations(score_matrix_direct, means_df)
    r_scenario = all_item_total_correlations(score_matrix_scenario, means_df)
    r_all = pd.concat([r_direct, r_scenario], ignore_index=True)

    return {
        "item_variance": variance,
        "item_refusals": refusals,
        "item_total_r": r_all,
    }
