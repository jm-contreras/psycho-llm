"""Load response data from SQLite, apply reverse coding, compute model-item means."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "data" / "raw" / "responses.db"
OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Dimension code → full name mapping
DIMENSION_CODES = {
    "SA": "Social Alignment",
    "CA": "Compliance vs. Autonomy",
    "EC": "Epistemic Confidence",
    "RS": "Refusal Sensitivity",
    "VE": "Verbosity/Elaboration",
    "HE": "Hedging",
    "CC": "Creativity vs. Convention",
    "CR": "Catastrophizing/Risk Amplification",
    "AT": "Apologetic Tendency",
    "PI": "Proactive Initiative",
    "WR": "Warmth and Rapport",
    "SD": "Self-Disclosure",
}

# Tiered model inclusion thresholds (number of distinct direct items with success)
THRESHOLD_ENGINEERING = 1        # all models
THRESHOLD_ITEM_QUALITY = 200     # models with near-complete direct coverage
THRESHOLD_EFA = 200              # same — maximize N for pooled matrix


def _load_group_map() -> dict[str, str]:
    """Build model_id → canonical_model_id map from model_registry.json.

    Models with a ``group_as`` field are mapped to that canonical ID.
    All others map to themselves.
    """
    import json
    registry_path = REPO_ROOT / "model_registry.json"
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        registry = json.load(f)
    group_map: dict[str, str] = {}
    # Registry is a dict of lists (one key per provider group)
    for _key, entries in registry.items():
        if not isinstance(entries, list):
            continue
        for m in entries:
            mid = m.get("litellm_model_id", "")
            canonical = m.get("group_as") or mid
            if mid:
                group_map[mid] = canonical
    return group_map


def load_responses(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load all responses from SQLite into a DataFrame.

    Applies ``group_as`` merging: rows from model IDs that share a
    ``group_as`` target are remapped to the canonical model_id so they
    are treated as a single model in all downstream analyses.
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM responses", conn)
    conn.close()

    # Apply group_as remapping
    group_map = _load_group_map()
    if group_map:
        df["model_id"] = df["model_id"].map(lambda x: group_map.get(x, x))

    return df


def filter_success(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only status='success' rows with non-null parsed_score."""
    mask = (df["status"] == "success") & df["parsed_score"].notna()
    return df.loc[mask].copy()


def recode_reverse_items(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'score' column with reverse coding applied.

    Direct items with keying='-': score = 6 - parsed_score (flips 1-5 scale).
    All others (keying='+', scenario items with keying=NULL): score = parsed_score.
    """
    df = df.copy()
    reverse_mask = (df["item_type"] == "direct") & (df["keying"] == "-")
    df["score"] = df["parsed_score"].copy()
    df.loc[reverse_mask, "score"] = 6 - df.loc[reverse_mask, "parsed_score"]
    return df


def compute_model_item_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean score per (model_id, item_id) across runs.

    Input df must already have 'score' column (i.e., after recode_reverse_items).
    Returns DataFrame with columns: model_id, item_id, dimension, item_type,
    keying, mean_score, n_runs.
    """
    agg = (
        df.groupby(["model_id", "item_id"])
        .agg(
            dimension=("dimension", "first"),
            item_type=("item_type", "first"),
            keying=("keying", "first"),
            mean_score=("score", "mean"),
            n_runs=("score", "count"),
        )
        .reset_index()
    )
    return agg


def pivot_score_matrix(
    means_df: pd.DataFrame,
    item_type: str | None = None,
) -> pd.DataFrame:
    """Pivot model-item means into a models×items matrix.

    Rows = model_id, columns = item_id, values = mean_score.
    Optionally filter to item_type='direct' or 'scenario'.
    """
    subset = means_df
    if item_type is not None:
        subset = means_df[means_df["item_type"] == item_type]
    matrix = subset.pivot_table(
        index="model_id", columns="item_id", values="mean_score"
    )
    # Sort columns by item_id for reproducibility
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def model_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model coverage summary.

    Returns DataFrame with: model_id, total, success, parse_error, api_error,
    refusal, n_items, n_direct_items_success, max_run.
    """
    status_counts = (
        df.groupby(["model_id", "status"]).size().unstack(fill_value=0)
    )
    for col in ["success", "parse_error", "api_error", "refusal"]:
        if col not in status_counts.columns:
            status_counts[col] = 0
    status_counts["total"] = status_counts.sum(axis=1)

    items = df.groupby("model_id")["item_id"].nunique().rename("n_items")
    max_run = df.groupby("model_id")["run_number"].max().rename("max_run")

    # Direct items with success status
    direct_success = (
        df[(df["status"] == "success") & (df["item_type"] == "direct")]
        .groupby("model_id")["item_id"]
        .nunique()
        .rename("n_direct_success")
    )

    report = status_counts.join(items).join(max_run).join(direct_success)
    report["n_direct_success"] = report["n_direct_success"].fillna(0).astype(int)
    report = report.sort_values("success", ascending=False).reset_index()
    return report


def get_models_for_section(
    df: pd.DataFrame, section: int
) -> list[str]:
    """Return model_ids eligible for a given analysis section.

    Section 1 (engineering): all models.
    Section 2-3 (item/dimension quality): models with ≥200 successful direct items.
    Section 4 (EFA): models with ≥200 successful direct items.
    """
    if section == 1:
        return df["model_id"].unique().tolist()

    # Count distinct direct items with success per model
    direct_success = (
        df[(df["status"] == "success") & (df["item_type"] == "direct")]
        .groupby("model_id")["item_id"]
        .nunique()
    )

    if section in (2, 3):
        threshold = THRESHOLD_ITEM_QUALITY
    elif section == 4:
        threshold = THRESHOLD_EFA
    else:
        threshold = THRESHOLD_ENGINEERING

    return direct_success[direct_success >= threshold].index.tolist()


def get_analysis_items(
    means_df: pd.DataFrame, min_models: int = 5
) -> list[str]:
    """Return item_ids observed in at least min_models models."""
    counts = means_df.groupby("item_id")["model_id"].nunique()
    return sorted(counts[counts >= min_models].index.tolist())


def get_short_model_name(model_id: str) -> str:
    """Extract a short display name from litellm model_id.

    e.g. 'bedrock/converse/moonshotai.kimi-k2.5' → 'kimi-k2.5'
         'openai/gpt-5.4-nano' → 'gpt-5.4-nano'
    """
    parts = model_id.split("/")
    name = parts[-1]
    # Strip common prefixes
    for prefix in ("us.anthropic.", "amazon.", "google.", "ai21.", "cohere.", "nvidia.", "moonshotai.", "zai."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip version suffixes like -v1:0
    if name.endswith(":0"):
        name = name[:-2]
    return name


def ensure_output_dirs():
    """Create output/ and output/plots/ directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(db_path: Path = DB_PATH):
    """Convenience: load, filter, recode, compute means, pivot.

    Returns (df_all, df_success, means_df, score_matrix_direct, score_matrix_scenario).
    """
    df_all = load_responses(db_path)
    df_success = filter_success(df_all)
    df_success = recode_reverse_items(df_success)
    means_df = compute_model_item_means(df_success)
    score_matrix_direct = pivot_score_matrix(means_df, item_type="direct")
    score_matrix_scenario = pivot_score_matrix(means_df, item_type="scenario")
    return df_all, df_success, means_df, score_matrix_direct, score_matrix_scenario
