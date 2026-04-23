"""CLI entry point: python -m analysis.run_diagnostics

Run preliminary diagnostics on the psycho-llm response data.

Usage:
    python -m analysis.run_diagnostics                    # all sections
    python -m analysis.run_diagnostics --sections 1 2 3   # specific sections
    python -m analysis.run_diagnostics --output-dir /tmp   # custom output dir
"""

import argparse
import sys
import time

from . import data_loader, engineering_checks, item_quality, dimension_coherence, report


def main():
    parser = argparse.ArgumentParser(description="Run preliminary diagnostics")
    parser.add_argument(
        "--sections", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Sections to run (1-5, default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    # Setup
    if args.output_dir:
        data_loader.OUTPUT_DIR = data_loader.Path(args.output_dir)
        data_loader.PLOTS_DIR = data_loader.OUTPUT_DIR / "plots"
    data_loader.ensure_output_dirs()
    plots_dir = str(data_loader.PLOTS_DIR)
    output_dir = str(data_loader.OUTPUT_DIR)

    start = time.time()

    # Load data
    print("Loading data...")
    df_all, df_success, means_df, sm_direct, sm_scenario = data_loader.prepare_data()
    print(f"  {len(df_all)} total rows, {len(df_success)} success")
    print(f"  Direct matrix: {sm_direct.shape}, Scenario matrix: {sm_scenario.shape}")

    # Filter matrices to eligible models for sections 2-3
    eligible_23 = data_loader.get_models_for_section(df_all, 2)
    print(f"  Sections 2-3: {len(eligible_23)} eligible models")
    sm_direct_filtered = sm_direct.loc[sm_direct.index.isin(eligible_23)]
    sm_scenario_filtered = sm_scenario.loc[sm_scenario.index.isin(eligible_23)]
    means_filtered = means_df[means_df["model_id"].isin(eligible_23)]

    results = {
        "engineering": None,
        "item_quality": None,
        "dimension_coherence": None,
        "factor_structure": None,
    }

    # Section 1
    if 1 in args.sections:
        print("\n=== Section 1: Engineering Checks ===")
        results["engineering"] = engineering_checks.run_engineering_checks(df_all)
        summary = results["engineering"]["model_summary"]
        print(f"  {len(summary)} models")
        n_flagged = len(results["engineering"]["flagged_models"])
        n_oor = len(results["engineering"]["out_of_range"])
        print(f"  {n_flagged} models flagged, {n_oor} out-of-range scores")

    # Section 2
    if 2 in args.sections:
        print("\n=== Section 2: Item-Level Quality ===")
        results["item_quality"] = item_quality.run_item_quality(
            df_all, means_filtered, sm_direct_filtered, sm_scenario_filtered
        )
        v = results["item_quality"]["item_variance"]
        print(f"  {v['flagged_zero_var'].sum()} zero-variance items")
        r = results["item_quality"]["item_refusals"]
        print(f"  {r['flagged_refusal'].sum()} high-refusal items")
        itr = results["item_quality"]["item_total_r"]
        if len(itr) > 0:
            print(f"  {itr['flagged_low_r'].sum()} low item-total r items")

    # Section 3
    if 3 in args.sections:
        print("\n=== Section 3: Dimension-Level Coherence ===")
        results["dimension_coherence"] = dimension_coherence.run_dimension_coherence(
            sm_direct_filtered, sm_scenario_filtered, means_filtered, plots_dir
        )
        rel = results["dimension_coherence"]["reliability_direct"]
        print("  Direct item reliability:")
        for _, row in rel.iterrows():
            flag = " *** FLAGGED" if row["flagged_low_alpha"] else ""
            print(f"    {row['dimension']}: α={row['alpha']:.3f}, mean r={row['mean_inter_item_r']:.3f}{flag}")

    # Section 4
    if 4 in args.sections:
        print("\n=== Section 4: Preliminary Factor Structure ===")
        try:
            from . import factor_structure

            eligible_4 = data_loader.get_models_for_section(df_all, 4)
            print(f"  {len(eligible_4)} eligible models for EFA")

            results["factor_structure"] = factor_structure.run_factor_structure(
                df_success, means_df, eligible_4, plots_dir
            )
        except ImportError as e:
            print(f"  Skipping EFA: {e}")
            print("  Install with: pip install factor_analyzer")

    # Section 5: Report
    if 5 in args.sections:
        print("\n=== Section 5: Generating Report ===")

        # Fill in defaults for skipped sections
        if results["engineering"] is None:
            results["engineering"] = engineering_checks.run_engineering_checks(df_all)
        if results["item_quality"] is None:
            results["item_quality"] = item_quality.run_item_quality(
                df_all, means_filtered, sm_direct_filtered, sm_scenario_filtered
            )
        if results["dimension_coherence"] is None:
            results["dimension_coherence"] = dimension_coherence.run_dimension_coherence(
                sm_direct_filtered, sm_scenario_filtered, means_filtered, plots_dir
            )

        report_path = f"{output_dir}/diagnostic_report.md"
        report.generate_report(
            results["engineering"],
            results["item_quality"],
            results["dimension_coherence"],
            results["factor_structure"],
            report_path,
        )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
