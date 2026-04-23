"""Section 4: Preliminary factor structure — ICC, EFA, parallel analysis (direct items only)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import get_short_model_name


# ---------------------------------------------------------------------------
# Pooled observation matrix
# ---------------------------------------------------------------------------

def build_pooled_matrix(
    df_success: pd.DataFrame,
    eligible_models: list[str],
    item_type: str = "direct",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the pooled observation matrix for EFA.

    Each row = one run from one model for a set of items.
    Weighted so each model contributes equally (weight = 1/n_runs_for_model).

    Args:
        df_success: Success-only DataFrame with 'score' column (after recode).
        eligible_models: Models to include (≥200 direct items).
        item_type: 'direct' or 'scenario'.

    Returns:
        (obs_matrix, weights) — obs_matrix is runs×items DataFrame,
        weights is 1-D array of per-row weights.
    """
    sub = df_success[
        (df_success["item_type"] == item_type)
        & df_success["model_id"].isin(eligible_models)
    ].copy()

    # Pivot: rows = (model_id, run_number), columns = item_id, values = score
    matrix = sub.pivot_table(
        index=["model_id", "run_number"],
        columns="item_id",
        values="score",
        aggfunc="first",
    )
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)

    # Drop rows with excessive missingness (>20% of items missing)
    n_items = matrix.shape[1]
    threshold = 0.80 * n_items
    matrix = matrix.dropna(thresh=int(threshold))

    # Compute weights: 1 / n_runs for each model
    run_counts = matrix.groupby(level="model_id").size()
    weight_series = matrix.index.get_level_values("model_id").map(
        lambda m: 1.0 / run_counts[m]
    )
    weights = weight_series.values.astype(float)

    # Flatten index for downstream use
    matrix = matrix.reset_index(drop=True)

    return matrix, weights


# ---------------------------------------------------------------------------
# ICC analysis
# ---------------------------------------------------------------------------

def compute_icc(
    df_success: pd.DataFrame,
    eligible_models: list[str],
    item_type: str = "direct",
) -> pd.DataFrame:
    """Compute ICC(1) per item across models.

    ICC(1) = (MSb - MSw) / (MSb + (k-1)*MSw)
    where MSb = between-model mean square, MSw = within-model mean square,
    k = harmonic mean of runs per model.

    Only uses models with multiple runs (>1 observation per item).
    """
    sub = df_success[
        (df_success["item_type"] == item_type)
        & df_success["model_id"].isin(eligible_models)
    ]

    # Only models with >1 run
    run_counts = sub.groupby("model_id")["run_number"].nunique()
    multi_run = run_counts[run_counts > 1].index
    sub = sub[sub["model_id"].isin(multi_run)]

    items = sorted(sub["item_id"].unique())
    results = []

    for item in items:
        item_data = sub[sub["item_id"] == item][["model_id", "score"]].dropna()
        groups = item_data.groupby("model_id")["score"]

        n_groups = groups.ngroups
        if n_groups < 2:
            results.append({"item_id": item, "icc": np.nan})
            continue

        # Group sizes and grand mean
        ns = groups.count()
        grand_mean = item_data["score"].mean()
        n_total = len(item_data)

        # Between-group SS
        group_means = groups.mean()
        ss_between = (ns * (group_means - grand_mean) ** 2).sum()
        df_between = n_groups - 1

        # Within-group SS
        ss_within = groups.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
        df_within = n_total - n_groups

        if df_between == 0 or df_within == 0:
            results.append({"item_id": item, "icc": np.nan})
            continue

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Harmonic mean of group sizes
        k = n_groups / (1.0 / ns).sum()

        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        results.append({"item_id": item, "icc": icc})

    result = pd.DataFrame(results)
    result["classification"] = pd.cut(
        result["icc"],
        bins=[-np.inf, 0.0, 0.40, 0.60, 0.75, 1.0],
        labels=["negative", "poor", "fair", "good", "excellent"],
    )
    return result


def plot_icc_distribution(icc_df: pd.DataFrame, output_path: str) -> None:
    """Histogram of ICC values across items."""
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = icc_df["icc"].dropna()
    ax.hist(valid, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0.40, color="orange", linestyle="--", label="Fair threshold (0.40)")
    ax.axvline(0.60, color="green", linestyle="--", label="Good threshold (0.60)")
    ax.set_xlabel("ICC(1)")
    ax.set_ylabel("Number of items")
    ax.set_title("ICC(1) distribution across items")
    ax.legend()
    median_icc = valid.median()
    ax.annotate(
        f"Median = {median_icc:.3f}",
        xy=(median_icc, 0), xytext=(median_icc + 0.05, ax.get_ylim()[1] * 0.8),
        arrowprops=dict(arrowstyle="->"), fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallel analysis
# ---------------------------------------------------------------------------

def parallel_analysis(
    obs_matrix: pd.DataFrame,
    weights: np.ndarray,
    n_iterations: int = 1000,
    percentile: int = 95,
) -> dict:
    """Horn's parallel analysis for factor retention.

    Computes eigenvalues from the weighted correlation matrix of the real data,
    then compares to eigenvalues from random data with same dimensions.
    """
    # Weighted correlation matrix
    real_corr = _weighted_corr(obs_matrix, weights)
    real_eigenvalues = np.sort(np.linalg.eigvalsh(real_corr))[::-1]

    n, p = obs_matrix.shape
    random_eigenvalues = np.zeros((n_iterations, p))

    rng = np.random.default_rng(42)
    for i in range(n_iterations):
        random_data = rng.standard_normal((n, p))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]

    random_95 = np.percentile(random_eigenvalues, percentile, axis=0)

    # Number of factors where real > random
    n_factors = int(np.sum(real_eigenvalues > random_95))

    return {
        "real_eigenvalues": real_eigenvalues,
        "random_eigenvalues_95": random_95,
        "n_factors_suggested": n_factors,
    }


def plot_scree(
    real_eigenvalues: np.ndarray,
    random_eigenvalues: np.ndarray,
    n_factors: int,
    output_path: str,
) -> None:
    """Scree plot with parallel analysis overlay."""
    n_plot = min(30, len(real_eigenvalues))
    x = np.arange(1, n_plot + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, real_eigenvalues[:n_plot], "bo-", label="Actual eigenvalues")
    ax.plot(x, random_eigenvalues[:n_plot], "r--", label="95th percentile (random)")
    ax.axvline(n_factors + 0.5, color="gray", linestyle=":", alpha=0.5,
               label=f"Suggested factors: {n_factors}")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree plot with parallel analysis")
    ax.legend()
    ax.set_xticks(x)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EFA
# ---------------------------------------------------------------------------

def run_efa(
    obs_matrix: pd.DataFrame,
    weights: np.ndarray,
    n_factors: int,
    rotation: str = "oblimin",
) -> dict:
    """Run EFA using principal axis factoring with oblimin rotation.

    Uses the weighted correlation matrix. Falls back to PCA if PAF fails.
    """
    from factor_analyzer import FactorAnalyzer

    # Prepare weighted data: replicate-weight approach.
    # Apply sqrt(weight) to each row so that the covariance matrix of the
    # transformed data equals the weighted covariance matrix.
    filled = obs_matrix.fillna(obs_matrix.mean())
    sqrt_w = np.sqrt(weights)
    weighted_data = filled.values * sqrt_w[:, np.newaxis]
    # Center the weighted data
    wmean = np.average(filled.values, axis=0, weights=weights)
    weighted_data_centered = (filled.values - wmean) * sqrt_w[:, np.newaxis]

    # Use the weighted centered data as input to factor_analyzer
    weighted_df = pd.DataFrame(weighted_data_centered, columns=filled.columns)

    method_used = "minres"  # minimum residual (default, most robust)
    try:
        fa = FactorAnalyzer(
            n_factors=n_factors,
            rotation=rotation,
            method="minres",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa.fit(weighted_df)
        loadings = fa.loadings_
        eigenvalues = fa.get_eigenvalues()
    except Exception as e:
        # Fall back to principal (PCA-based)
        method_used = "principal_fallback"
        try:
            fa = FactorAnalyzer(
                n_factors=n_factors,
                rotation=rotation,
                method="principal",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fa.fit(weighted_df)
            loadings = fa.loadings_
            eigenvalues = fa.get_eigenvalues()
        except Exception as e2:
            return {
                "error": f"EFA (minres) failed: {e}; principal fallback also failed: {e2}",
                "method_used": "failed",
            }

    loadings_df = pd.DataFrame(
        loadings,
        index=obs_matrix.columns,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    # Factor correlation matrix (for oblique rotations)
    factor_corr = None
    if hasattr(fa, "phi_") and fa.phi_ is not None:
        factor_corr = pd.DataFrame(
            fa.phi_,
            index=loadings_df.columns,
            columns=loadings_df.columns,
        )

    return {
        "loadings": loadings_df,
        "eigenvalues": eigenvalues,
        "method_used": method_used,
        "factor_correlation": factor_corr,
    }


def loading_report(
    loadings_df: pd.DataFrame,
    means_df: pd.DataFrame,
    threshold_primary: float = 0.40,
    threshold_cross: float = 0.30,
) -> pd.DataFrame:
    """Annotate loadings with dimension labels and flag issues.

    Flags:
    - Items with no loading ≥ threshold_primary on any factor.
    - Items with cross-loading ≥ threshold_cross on a second factor.
    """
    # Map item_id → dimension
    dim_map = (
        means_df[["item_id", "dimension"]]
        .drop_duplicates()
        .set_index("item_id")["dimension"]
    )

    results = []
    for item_id in loadings_df.index:
        row = loadings_df.loc[item_id].abs()
        sorted_loadings = row.sort_values(ascending=False)
        primary_factor = sorted_loadings.index[0]
        primary_loading = sorted_loadings.iloc[0]

        cross_loading = sorted_loadings.iloc[1] if len(sorted_loadings) > 1 else 0

        flags = []
        if primary_loading < threshold_primary:
            flags.append(f"low_primary ({primary_loading:.2f})")
        if cross_loading >= threshold_cross:
            flags.append(
                f"cross_load ({sorted_loadings.index[1]}={cross_loading:.2f})"
            )

        results.append({
            "item_id": item_id,
            "dimension": dim_map.get(item_id, "unknown"),
            "primary_factor": primary_factor,
            "primary_loading": loadings_df.loc[item_id, primary_factor],
            "cross_loading": cross_loading,
            "flag": "; ".join(flags) if flags else "",
        })

    return pd.DataFrame(results)


def plot_factor_loadings(
    loadings_df: pd.DataFrame,
    means_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Heatmap of factor loadings, items grouped by dimension."""
    # Sort items by dimension
    dim_map = (
        means_df[["item_id", "dimension"]]
        .drop_duplicates()
        .set_index("item_id")["dimension"]
    )
    items_with_dim = [
        (dim_map.get(item, "zzz"), item) for item in loadings_df.index
    ]
    items_with_dim.sort()
    ordered_items = [item for _, item in items_with_dim]

    ordered_loadings = loadings_df.loc[ordered_items]

    fig, ax = plt.subplots(figsize=(max(8, loadings_df.shape[1] * 1.2), max(12, len(ordered_items) * 0.1)))
    sns.heatmap(
        ordered_loadings,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        yticklabels=False,
        ax=ax,
    )

    # Add dimension separators
    dims_ordered = [dim_map.get(item, "") for item in ordered_items]
    prev_dim = None
    for i, dim in enumerate(dims_ordered):
        if dim != prev_dim and prev_dim is not None:
            ax.axhline(i, color="black", linewidth=0.5)
        prev_dim = dim

    ax.set_title("Factor loadings (items grouped by candidate dimension)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Items (grouped by dimension)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_corr(data: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Compute weighted Pearson correlation matrix.

    Handles NaN via pairwise deletion with weighted means/covariances.
    """
    X = data.values.copy()
    w = weights.copy()
    n, p = X.shape

    corr = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            valid = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if valid.sum() < 3:
                corr[i, j] = corr[j, i] = 0.0
                continue
            wi = w[valid]
            xi = X[valid, i]
            xj = X[valid, j]
            wi_sum = wi.sum()

            mean_i = np.average(xi, weights=wi)
            mean_j = np.average(xj, weights=wi)

            cov_ij = np.sum(wi * (xi - mean_i) * (xj - mean_j)) / wi_sum
            var_i = np.sum(wi * (xi - mean_i) ** 2) / wi_sum
            var_j = np.sum(wi * (xj - mean_j) ** 2) / wi_sum

            denom = np.sqrt(var_i * var_j)
            if denom < 1e-12:
                corr[i, j] = corr[j, i] = 0.0
            else:
                corr[i, j] = corr[j, i] = cov_ij / denom

    return corr


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_factor_structure(
    df_success: pd.DataFrame,
    means_df: pd.DataFrame,
    eligible_models: list[str],
    plots_dir: str,
) -> dict:
    """Run all Section 4 analyses.

    Returns dict with all results and plot paths.
    """
    results = {}

    # 1. Build pooled matrix
    obs_matrix, weights = build_pooled_matrix(df_success, eligible_models, "direct")
    results["pooled_shape"] = obs_matrix.shape
    results["n_models"] = len(eligible_models)
    print(f"  Pooled matrix: {obs_matrix.shape[0]} observations × {obs_matrix.shape[1]} items")
    print(f"  Models included: {len(eligible_models)}")

    # 2. ICC analysis
    print("  Computing ICC...")
    icc_df = compute_icc(df_success, eligible_models, "direct")
    results["icc"] = icc_df
    valid_icc = icc_df["icc"].dropna()
    print(f"  ICC median={valid_icc.median():.3f}, mean={valid_icc.mean():.3f}")

    icc_path = f"{plots_dir}/icc_distribution.png"
    plot_icc_distribution(icc_df, icc_path)
    results["icc_plot"] = icc_path

    # 3. Parallel analysis
    print("  Running parallel analysis (1000 iterations)...")
    pa = parallel_analysis(obs_matrix, weights)
    results["parallel_analysis"] = pa
    n_factors = pa["n_factors_suggested"]
    print(f"  Parallel analysis suggests {n_factors} factors")

    scree_path = f"{plots_dir}/scree_parallel_analysis.png"
    plot_scree(pa["real_eigenvalues"], pa["random_eigenvalues_95"], n_factors, scree_path)
    results["scree_plot"] = scree_path

    # 4. EFA
    if n_factors < 1:
        n_factors = 12  # fall back to candidate dimensions count
        print(f"  Parallel analysis suggested 0 factors; using {n_factors} as fallback")

    # Cap at a reasonable number
    n_factors = min(n_factors, 20)
    print(f"  Running EFA with {n_factors} factors...")

    efa_result = run_efa(obs_matrix, weights, n_factors)
    results["efa"] = efa_result

    if "error" in efa_result:
        print(f"  EFA failed: {efa_result['error']}")
    else:
        print(f"  EFA method: {efa_result['method_used']}")

        # Loading report
        report = loading_report(efa_result["loadings"], means_df)
        results["loading_report"] = report

        n_flagged = (report["flag"] != "").sum()
        print(f"  {n_flagged} items flagged (low primary or cross-loading)")

        # Factor loadings heatmap
        loadings_path = f"{plots_dir}/factor_loadings.png"
        plot_factor_loadings(efa_result["loadings"], means_df, loadings_path)
        results["loadings_plot"] = loadings_path

    return results
