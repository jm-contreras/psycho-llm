"""
Prompt-level within-model predictive validity analysis.

Key question: Is the predictive validity signal present at a finer grain
than the N=24 model-level analysis?

Three analyses:
1. Per-prompt × per-factor correlations (model-level, N=24 per prompt)
2. On-target vs off-target prompt comparison
3. Within-model analysis: does a model's factor profile predict which
   prompts get higher/lower judge ratings?
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "raw" / "responses.db"
FACTOR_SCORES_PATH = REPO_ROOT / "analysis" / "output" / "factor_scores.csv"

FACTORS = ["RE", "DE", "BO", "GU", "VB"]
FACTOR_NAMES = {
    "RE": "Responsiveness", "DE": "Deference", "BO": "Boldness",
    "GU": "Guardedness", "VB": "Verbosity"
}


def load_judge_ratings():
    """Load judge ratings with reverse-scoring applied."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT subject_model_id, prompt_id, run_number, judge_model_id, "
        "keying, score_RE, score_DE, score_BO, score_GU, score_VB "
        "FROM judge_ratings WHERE parse_status='success'", conn
    )
    conn.close()

    # Apply reverse-scoring
    for i, col in enumerate(["score_RE", "score_DE", "score_BO", "score_GU", "score_VB"]):
        r_mask = df["keying"].str[i] == "R"
        df.loc[r_mask, col] = 6 - df.loc[r_mask, col]

    return df


def load_factor_scores():
    """Load instrument factor scores."""
    fs = pd.read_csv(FACTOR_SCORES_PATH)
    # Exclude self-rating model if present
    # The model_id column should match subject_model_id in judge_ratings
    return fs


def compute_ensemble_scores(jr):
    """Compute mean ensemble judge score per (model, prompt, run) then per (model, prompt)."""
    # Average across judges for each (model, prompt, run)
    score_cols = [f"score_{f}" for f in FACTORS]
    ensemble = jr.groupby(["subject_model_id", "prompt_id", "run_number"])[score_cols].mean()
    # Average across runs for each (model, prompt)
    model_prompt = ensemble.groupby(["subject_model_id", "prompt_id"]).mean().reset_index()
    return model_prompt


def analysis_1_per_prompt_correlations(model_prompt, factor_scores):
    """For each prompt × factor, correlate model-level instrument score with judge score."""
    print("=" * 80)
    print("ANALYSIS 1: Per-Prompt × Per-Factor Correlations (N=24 models per prompt)")
    print("=" * 80)

    prompts = sorted(model_prompt["prompt_id"].unique())
    results = []

    for prompt_id in prompts:
        prompt_dim = prompt_id.split("-")[0]  # e.g., BO from BO-BP01
        sub = model_prompt[model_prompt["prompt_id"] == prompt_id].copy()
        merged = sub.merge(factor_scores, left_on="subject_model_id", right_on="model_id")

        for factor in FACTORS:
            inst_col = factor
            judge_col = f"score_{factor}"
            x = merged[inst_col].values
            y = merged[judge_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            results.append({
                "prompt_id": prompt_id, "prompt_dim": prompt_dim,
                "factor": factor, "r": r, "p": p, "n": mask.sum(),
                "on_target": prompt_dim == factor
            })

    df = pd.DataFrame(results)

    # Summary: best prompt for each factor
    print("\n--- Strongest prompt-factor correlations ---")
    for factor in FACTORS:
        sub = df[df["factor"] == factor].sort_values("r", ascending=False)
        print(f"\n  {factor} ({FACTOR_NAMES[factor]}):")
        print(f"  {'Prompt':<10} {'Dim':<4} {'r':>7} {'p':>7} {'On-target':>10}")
        for _, row in sub.head(5).iterrows():
            tag = " ***" if row["on_target"] else ""
            print(f"  {row['prompt_id']:<10} {row['prompt_dim']:<4} {row['r']:>7.3f} {row['p']:>7.3f} {'YES' if row['on_target'] else '':>10}{tag}")
        # Also show worst
        print(f"  ... worst:")
        for _, row in sub.tail(3).iterrows():
            tag = " ***" if row["on_target"] else ""
            print(f"  {row['prompt_id']:<10} {row['prompt_dim']:<4} {row['r']:>7.3f} {row['p']:>7.3f} {'YES' if row['on_target'] else '':>10}{tag}")

    # Aggregate: mean r across all prompts vs on-target only vs off-target
    print("\n--- On-target vs Off-target mean correlations ---")
    print(f"{'Factor':<8} {'On-target mean r':>18} {'Off-target mean r':>19} {'All prompts mean r':>20} {'Advantage':>10}")
    for factor in FACTORS:
        sub = df[df["factor"] == factor]
        on = sub[sub["on_target"]]
        off = sub[~sub["on_target"]]
        on_r = on["r"].mean() if len(on) > 0 else np.nan
        off_r = off["r"].mean() if len(off) > 0 else np.nan
        all_r = sub["r"].mean()
        adv = on_r - off_r if np.isfinite(on_r) and np.isfinite(off_r) else np.nan
        print(f"{factor:<8} {on_r:>18.3f} {off_r:>19.3f} {all_r:>20.3f} {adv:>10.3f}")

    # Count significant correlations
    print(f"\n--- Significance counts (p < .05) ---")
    sig = df[df["p"] < 0.05]
    print(f"Total significant: {len(sig)} / {len(df)} ({100*len(sig)/len(df):.1f}%)")
    for factor in FACTORS:
        sub_sig = sig[sig["factor"] == factor]
        sub_all = df[df["factor"] == factor]
        print(f"  {factor}: {len(sub_sig)} / {len(sub_all)} significant")

    return df


def analysis_2_pooled_prompt_level(jr, factor_scores):
    """
    Pool across runs but keep prompt-level: N = 24 models × 20 prompts = 480.
    Correlate instrument factor score with judge ensemble score, with prompt fixed effects.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Pooled Prompt-Level Correlations (N = models × prompts)")
    print("=" * 80)

    score_cols = [f"score_{f}" for f in FACTORS]
    # Average across judges and runs per (model, prompt)
    ensemble = jr.groupby(["subject_model_id", "prompt_id"])[score_cols].mean().reset_index()
    merged = ensemble.merge(factor_scores, left_on="subject_model_id", right_on="model_id")

    print(f"\nTotal observations: {len(merged)}")

    # Raw correlation (ignoring prompt structure)
    print("\n--- Raw correlations (instrument score vs judge score, all prompts pooled) ---")
    print(f"{'Factor':<8} {'r':>7} {'p':>10} {'n':>5}")
    for factor in FACTORS:
        x = merged[factor].values
        y = merged[f"score_{factor}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"{factor:<8} {r:>7.3f} {p:>10.4f} {mask.sum():>5}")

    # Partial correlation: remove prompt means (prompt fixed effects)
    print("\n--- Partial correlations (prompt-demeaned, removes between-prompt variance) ---")
    print(f"{'Factor':<8} {'r':>7} {'p':>10}")
    for factor in FACTORS:
        sub = merged[["subject_model_id", "prompt_id", factor, f"score_{factor}"]].dropna()
        # Demean by prompt
        prompt_means = sub.groupby("prompt_id")[f"score_{factor}"].transform("mean")
        y_demeaned = sub[f"score_{factor}"] - prompt_means
        # Instrument score doesn't vary by prompt, so demeaning it by prompt
        # removes between-prompt variance in x too. But x is constant per model.
        # So demeaning x by prompt subtracts the mean instrument score of models
        # that responded to that prompt. Since all models respond to all prompts,
        # this is just x - grand_mean. So partial correlation = raw model-level.
        # More interesting: demean by MODEL to see within-model variation.
        model_means_y = sub.groupby("subject_model_id")[f"score_{factor}"].transform("mean")
        y_within = sub[f"score_{factor}"] - model_means_y

        # For within-model: does the prompt's target dimension predict
        # which factor scores higher for that model?
        # x should be: "is this prompt targeting factor f?"
        sub["prompt_dim"] = sub["prompt_id"].str.split("-").str[0]
        sub["on_target"] = (sub["prompt_dim"] == factor).astype(float)

        r, p = stats.pearsonr(sub["on_target"], y_within)
        print(f"{factor:<8} {r:>7.3f} {p:>10.4f}")

    print("\n  (Above tests: after removing each model's mean judge score,")
    print("   do on-target prompts score higher than off-target prompts?)")


def analysis_3_within_model(jr, factor_scores):
    """
    Within-model analysis: For each model, across its 20 prompts,
    does the prompt's target dimension predict which factor gets the
    highest judge rating?

    For each model × prompt: the judge gives 5 factor scores.
    The prompt targets one factor. Does the targeted factor score
    higher than the others?
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Within-Model Factor Profile × Prompt Target Alignment")
    print("=" * 80)

    score_cols = [f"score_{f}" for f in FACTORS]

    # Average across judges and runs per (model, prompt)
    ensemble = jr.groupby(["subject_model_id", "prompt_id"])[score_cols].mean().reset_index()
    ensemble["prompt_dim"] = ensemble["prompt_id"].str.split("-").str[0]

    # For each model: compute mean on-target vs off-target judge score
    results_per_model = []
    for model_id in sorted(ensemble["subject_model_id"].unique()):
        msub = ensemble[ensemble["subject_model_id"] == model_id]

        on_target_scores = []
        off_target_scores = []

        for _, row in msub.iterrows():
            target_factor = row["prompt_dim"]
            if target_factor in FACTORS:
                on_score = row[f"score_{target_factor}"]
                off_scores = [row[f"score_{f}"] for f in FACTORS if f != target_factor]
                on_target_scores.append(on_score)
                off_target_scores.extend(off_scores)

        results_per_model.append({
            "model_id": model_id,
            "mean_on_target": np.mean(on_target_scores),
            "mean_off_target": np.mean(off_target_scores),
            "diff": np.mean(on_target_scores) - np.mean(off_target_scores)
        })

    df_models = pd.DataFrame(results_per_model)
    print(f"\nDo prompts elicit higher scores on their target factor than on off-target factors?")
    print(f"\n  Mean on-target judge score:  {df_models['mean_on_target'].mean():.3f}")
    print(f"  Mean off-target judge score: {df_models['mean_off_target'].mean():.3f}")
    print(f"  Mean difference:             {df_models['diff'].mean():.3f}")
    t, p = stats.ttest_1samp(df_models["diff"], 0)
    print(f"  t({len(df_models)-1}) = {t:.3f}, p = {p:.4f}")

    # Break down by factor
    print(f"\n--- By target factor ---")
    print(f"{'Target':<8} {'On-target':>10} {'Off-target':>11} {'Diff':>7} {'t':>7} {'p':>7}")
    for factor in FACTORS:
        on_vals = []
        off_vals = []
        for model_id in ensemble["subject_model_id"].unique():
            msub = ensemble[(ensemble["subject_model_id"] == model_id) &
                          (ensemble["prompt_dim"] == factor)]
            if len(msub) == 0:
                continue
            on_vals.append(msub[f"score_{factor}"].mean())
            off_cols = [f"score_{f}" for f in FACTORS if f != factor]
            off_vals.append(msub[off_cols].values.mean())
        on_arr = np.array(on_vals)
        off_arr = np.array(off_vals)
        diff = on_arr - off_arr
        t_val, p_val = stats.ttest_1samp(diff, 0)
        print(f"{factor:<8} {on_arr.mean():>10.3f} {off_arr.mean():>11.3f} {diff.mean():>7.3f} {t_val:>7.2f} {p_val:>7.4f}")


def analysis_4_within_model_profile_correlation(jr, factor_scores):
    """
    For each model, correlate its 5-factor instrument profile with the
    judge ratings across prompts.

    Idea: a model with high Responsiveness instrument score should get
    higher RE judge ratings overall. But more interestingly, across the
    20 prompts, the model's relative factor strengths should predict
    which factor dimension the judges rate highest.

    Approach: For each (model, prompt), create a 5-element vector of
    judge scores. Correlate this with the model's 5-element instrument
    profile. Average the correlation across all prompts for that model.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Within-Model Profile Correlation")
    print("=" * 80)
    print("For each (model, prompt), correlate the 5-factor judge rating vector")
    print("with the model's 5-factor instrument profile.")

    score_cols = [f"score_{f}" for f in FACTORS]

    # Average across judges and runs per (model, prompt)
    ensemble = jr.groupby(["subject_model_id", "prompt_id"])[score_cols].mean().reset_index()
    merged = ensemble.merge(factor_scores, left_on="subject_model_id", right_on="model_id")

    model_results = []
    for model_id in sorted(merged["subject_model_id"].unique()):
        msub = merged[merged["subject_model_id"] == model_id]
        instrument_profile = msub[FACTORS].iloc[0].values.astype(float)

        prompt_corrs = []
        for _, row in msub.iterrows():
            judge_profile = row[score_cols].values.astype(float)
            if np.std(judge_profile) > 0 and np.std(instrument_profile) > 0:
                r, _ = stats.pearsonr(instrument_profile, judge_profile)
                prompt_corrs.append(r)

        if prompt_corrs:
            model_results.append({
                "model_id": model_id,
                "mean_profile_r": np.mean(prompt_corrs),
                "n_prompts": len(prompt_corrs)
            })

    df_res = pd.DataFrame(model_results)
    print(f"\nMean within-model profile correlation: {df_res['mean_profile_r'].mean():.3f}")
    print(f"Median: {df_res['mean_profile_r'].median():.3f}")
    t, p = stats.ttest_1samp(df_res["mean_profile_r"], 0)
    print(f"t({len(df_res)-1}) = {t:.3f}, p = {p:.4f}")

    print(f"\n{'Model':<55} {'Mean r':>7} {'N prompts':>10}")
    for _, row in df_res.sort_values("mean_profile_r", ascending=False).iterrows():
        short = row["model_id"]
        print(f"{short:<55} {row['mean_profile_r']:>7.3f} {row['n_prompts']:>10}")


def analysis_5_run_level_signal(jr, factor_scores):
    """
    Finest grain: N = models × prompts × runs.
    For each factor, correlate instrument score with individual run-level judge score.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Run-Level Correlations (finest grain)")
    print("=" * 80)

    score_cols = [f"score_{f}" for f in FACTORS]

    # Average across judges per (model, prompt, run)
    run_level = jr.groupby(["subject_model_id", "prompt_id", "run_number"])[score_cols].mean().reset_index()
    merged = run_level.merge(factor_scores, left_on="subject_model_id", right_on="model_id")

    print(f"\nTotal observations: {len(merged)}")
    print(f"\n--- Raw run-level correlations ---")
    print(f"{'Factor':<8} {'r':>7} {'p':>10} {'n':>6}")
    for factor in FACTORS:
        x = merged[factor].values
        y = merged[f"score_{factor}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"{factor:<8} {r:>7.3f} {p:>10.6f} {mask.sum():>6}")

    # Compare to model-level (aggregated across prompts and runs)
    model_level = merged.groupby("subject_model_id").agg(
        {f"score_{f}": "mean" for f in FACTORS} | {f: "first" for f in FACTORS}
    ).reset_index()

    print(f"\n--- Model-level correlations (N={len(model_level)}, for comparison) ---")
    print(f"{'Factor':<8} {'r':>7} {'p':>10}")
    for factor in FACTORS:
        x = model_level[factor].values
        y = model_level[f"score_{factor}"].values
        mask = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"{factor:<8} {r:>7.3f} {p:>10.4f}")


def main():
    print("Loading data...")
    jr = load_judge_ratings()
    fs = load_factor_scores()

    print(f"Judge ratings: {len(jr)} rows")
    print(f"Models in factor scores: {len(fs)}")
    print(f"Models in judge ratings: {jr['subject_model_id'].nunique()}")

    # Check model overlap
    fs_models = set(fs["model_id"])
    jr_models = set(jr["subject_model_id"])
    overlap = fs_models & jr_models
    print(f"Overlapping models: {len(overlap)}")

    missing_from_jr = fs_models - jr_models
    missing_from_fs = jr_models - fs_models
    if missing_from_jr:
        print(f"  In factor scores but not judge ratings: {missing_from_jr}")
    if missing_from_fs:
        print(f"  In judge ratings but not factor scores: {missing_from_fs}")

    # Filter to overlapping models
    fs = fs[fs["model_id"].isin(overlap)]
    jr = jr[jr["subject_model_id"].isin(overlap)]

    # Compute ensemble scores
    model_prompt = compute_ensemble_scores(jr)

    # Run analyses
    df_corrs = analysis_1_per_prompt_correlations(model_prompt, fs)
    analysis_2_pooled_prompt_level(jr, fs)
    analysis_3_within_model(jr, fs)
    analysis_4_within_model_profile_correlation(jr, fs)
    analysis_5_run_level_signal(jr, fs)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings to look for:
- Analysis 1: Which specific prompts drive predictive validity?
- Analysis 2: Does prompt-level pooling increase power?
- Analysis 3: Do judges rate on-target factors higher? (prompt discrimination)
- Analysis 4: Does a model's instrument profile match its judge profile?
- Analysis 5: Does finer grain (run-level) preserve or dilute the signal?
""")


if __name__ == "__main__":
    main()
