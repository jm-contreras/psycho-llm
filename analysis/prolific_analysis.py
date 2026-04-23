"""Prolific human rating analysis.

Computes coverage, ICC, human-judge correlations, and gold item diagnostics.
Excludes only manually rejected/returned participants (not automated gold threshold).

Usage:
    python -m analysis.prolific_analysis
    python -m analysis.prolific_analysis --exclude PID1 PID2 ...
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from pipeline.storage import REPO_ROOT

# ── Paths ────────────────────────────────────────────────────────────────────

PROLIFIC_DB = REPO_ROOT / "data" / "prolific" / "prolific.db"
RESPONSES_DB = REPO_ROOT / "data" / "raw" / "responses.db"
GOLD_ITEMS_PATH = REPO_ROOT / "data" / "mturk" / "gold_items.json"
SAMPLE_PATH = REPO_ROOT / "data" / "mturk" / "sample.json"
OUTPUT_PATH = REPO_ROOT / "analysis" / "output" / "prolific_report.md"

FACTORS = ["RE", "DE", "BO", "GU", "VB"]
FACTOR_NAMES = {"RE": "Responsiveness", "DE": "Deference", "BO": "Boldness",
                "GU": "Guardedness", "VB": "Verbosity"}

PROMPT_FACTOR = {
    "RE-BP01": "RE", "RE-BP02": "RE", "RE-BP03": "RE", "RE-BP04": "RE",
    "DE-BP01": "DE", "DE-BP02": "DE", "DE-BP03": "DE", "DE-BP04": "DE",
    "BO-BP01": "BO", "BO-BP02": "BO", "BO-BP03": "BO", "BO-BP04": "BO",
    "GU-BP01": "GU", "GU-BP02": "GU", "GU-BP03": "GU", "GU-BP04": "GU",
    "VB-BP01": "VB", "VB-BP02": "VB", "VB-BP03": "VB", "VB-BP04": "VB",
}

# ── Default exclusions (Batch 1 manual review) ──────────────────────────────

# Anonymized rater IDs (see pipeline/prolific/config.py for salt).
DEFAULT_EXCLUDED = {
    "ca0b05c20ff3",  # returned
    "6f523c30c8cf",  # rejected
    "bf53ae6e33a4",  # rejected
    "22658e748c45",  # rejected
    "de38314720fd",  # rejected
    "5401bb5bf3ed",  # rejected — responded too quickly, failed attention checks
    "413aff8a004b",  # rejected
    "0a56fec18ea4",  # rejected
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def pearson(x: list[float], y: list[float]) -> tuple[float | None, int]:
    n = len(x)
    if n < 3:
        return None, n
    mx, my = sum(x) / n, sum(y) / n
    sx = sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1))
    sy = sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return None, n
    r = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)
    return r, n


def icc_2k_from_pairs(pairs: list[tuple[float, float]]) -> tuple[float | None, int]:
    """ICC(2,k) with k=2 from rating pairs."""
    n_p = len(pairs)
    if n_p < 3:
        return None, n_p
    r1 = [p[0] for p in pairs]
    r2 = [p[1] for p in pairs]
    row_means = [(r1[i] + r2[i]) / 2 for i in range(n_p)]
    grand_mean = sum(row_means) / n_p
    col_mean_1 = sum(r1) / n_p
    col_mean_2 = sum(r2) / n_p

    BMS = 2 * sum((rm - grand_mean) ** 2 for rm in row_means) / (n_p - 1)
    residuals = []
    for i in range(n_p):
        rm = row_means[i]
        residuals.append((r1[i] - rm - col_mean_1 + grand_mean) ** 2)
        residuals.append((r2[i] - rm - col_mean_2 + grand_mean) ** 2)
    EMS = sum(residuals) / ((n_p - 1))
    JMS = n_p * sum((cm - grand_mean) ** 2 for cm in [col_mean_1, col_mean_2])

    denom = BMS + (JMS - EMS) / n_p
    if denom == 0:
        return None, n_p
    icc_avg = (BMS - EMS) / denom
    return icc_avg, n_p


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(excluded_pids: set[str]) -> dict:
    """Load and filter all data needed for analysis. Returns a dict of dataframes."""
    conn = sqlite3.connect(str(PROLIFIC_DB))
    conn.row_factory = sqlite3.Row

    # Sessions
    sessions = conn.execute(
        "SELECT * FROM prolific_sessions"
    ).fetchall()
    sessions = [dict(r) for r in sessions]

    # Only use ratings from completed sessions (exclude partial in-progress)
    completed_pids = set(
        r["prolific_pid"] for r in sessions if r["status"] == "complete"
    )

    all_ratings = conn.execute(
        "SELECT * FROM prolific_ratings"
    ).fetchall()
    all_ratings = [dict(r) for r in all_ratings]
    conn.close()

    # Split by gold/sample; require completed session AND not excluded
    good_sample = [r for r in all_ratings
                   if r["prolific_pid"] in completed_pids
                   and r["prolific_pid"] not in excluded_pids
                   and r["is_gold"] == 0]
    good_gold = [r for r in all_ratings
                 if r["prolific_pid"] in completed_pids
                 and r["prolific_pid"] not in excluded_pids
                 and r["is_gold"] == 1]

    # Judge ratings
    jconn = sqlite3.connect(str(RESPONSES_DB))
    jconn.row_factory = sqlite3.Row
    judge_rows = jconn.execute(
        "SELECT * FROM judge_ratings WHERE parse_status = 'success'"
    ).fetchall()
    judge_rows = [dict(r) for r in judge_rows]

    br_rows = jconn.execute("SELECT id, model_id FROM behavioral_responses").fetchall()
    model_by_rid = {r["id"]: r["model_id"] for r in br_rows}
    jconn.close()

    # Gold items
    with open(GOLD_ITEMS_PATH) as f:
        gold_data = json.load(f)
    gold_items = gold_data["items"]

    # Sample pool size
    with open(SAMPLE_PATH) as f:
        pool = json.load(f)
        if isinstance(pool, dict) and "items" in pool:
            pool = pool["items"]
    pool_size = len(pool)

    return {
        "sessions": sessions,
        "all_ratings": all_ratings,
        "good_sample": good_sample,
        "good_gold": good_gold,
        "judge_rows": judge_rows,
        "model_by_rid": model_by_rid,
        "gold_items": gold_items,
        "pool_size": pool_size,
        "excluded_pids": excluded_pids,
    }


# ── Analysis functions ───────────────────────────────────────────────────────

def sessions_summary(data: dict) -> dict:
    sessions = data["sessions"]
    by_status = Counter(s["status"] for s in sessions)
    completed = [s for s in sessions if s["status"] == "complete"]
    n_excluded = len(data["excluded_pids"])
    n_usable = len(set(r["prolific_pid"] for r in data["good_sample"]))
    return {
        "by_status": dict(by_status),
        "total": len(sessions),
        "completed": len(completed),
        "excluded": n_excluded,
        "usable_participants": n_usable,
    }


def coverage(data: dict) -> dict:
    by_rid = defaultdict(int)
    for r in data["good_sample"]:
        by_rid[r["behavioral_response_id"]] += 1
    cnt_dist = Counter(by_rid.values())
    items_zero = data["pool_size"] - len(by_rid)
    items_at_1 = cnt_dist.get(1, 0)
    items_2plus = sum(v for k, v in cnt_dist.items() if k >= 2)
    ratings_needed = items_zero * 2 + items_at_1
    return {
        "pool_size": data["pool_size"],
        "items_zero": items_zero,
        "distribution": dict(cnt_dist),
        "items_2plus": items_2plus,
        "total_ratings": len(data["good_sample"]),
        "ratings_needed": ratings_needed,
        "participants_needed": ratings_needed // 6 + (1 if ratings_needed % 6 else 0),
    }


def compute_icc(data: dict) -> dict:
    """ICC(2,k) per factor: all items, on-target, off-target."""
    by_rid = defaultdict(list)
    for r in data["good_sample"]:
        by_rid[r["behavioral_response_id"]].append(r)

    results = {}
    for f in FACTORS:
        all_pairs, on_pairs, off_pairs = [], [], []
        for rid, rlist in by_rid.items():
            vals = [r[f"corrected_{f}"] for r in rlist if r[f"corrected_{f}"] is not None]
            if len(vals) < 2:
                continue
            pair = (vals[0], vals[1])
            all_pairs.append(pair)

            pid = rlist[0].get("prompt_id", "")
            target = PROMPT_FACTOR.get(pid, "")
            if target == f:
                on_pairs.append(pair)
            else:
                off_pairs.append(pair)

        icc_all, n_all = icc_2k_from_pairs(all_pairs)
        icc_on, n_on = icc_2k_from_pairs(on_pairs)
        icc_off, n_off = icc_2k_from_pairs(off_pairs)
        results[f] = {
            "all": (icc_all, n_all),
            "on_target": (icc_on, n_on),
            "off_target": (icc_off, n_off),
        }
    return results


def human_judge_correlations(data: dict) -> dict:
    """Item-level and model-level human-judge correlations, on/off target."""
    by_rid = defaultdict(list)
    for r in data["good_sample"]:
        by_rid[r["behavioral_response_id"]].append(r)

    judge_by_rid = defaultdict(list)
    for r in data["judge_rows"]:
        judge_by_rid[r["behavioral_response_id"]].append(r)

    model_by_rid = data["model_by_rid"]

    results = {}
    for f in FACTORS:
        # Item-level
        item_on_h, item_on_j = [], []
        item_off_h, item_off_j = [], []
        item_all_h, item_all_j = [], []

        # Model-level accumulators
        model_human_on = defaultdict(list)
        model_judge_on = defaultdict(list)
        model_human_all = defaultdict(list)
        model_judge_all = defaultdict(list)

        for rid, rlist in by_rid.items():
            h_vals = [r[f"corrected_{f}"] for r in rlist if r[f"corrected_{f}"] is not None]
            if not h_vals:
                continue
            j_list = judge_by_rid.get(rid, [])
            j_vals = [r[f"score_{f}"] for r in j_list if r[f"score_{f}"] is not None]
            if not j_vals:
                continue

            h_mean = statistics.mean(h_vals)
            j_mean = statistics.mean(j_vals)
            pid = rlist[0].get("prompt_id", "")
            target = PROMPT_FACTOR.get(pid, "")
            model = model_by_rid.get(rid, "unknown")

            item_all_h.append(h_mean)
            item_all_j.append(j_mean)
            for v in h_vals:
                model_human_all[model].append(v)
            for v in j_vals:
                model_judge_all[model].append(v)

            if target == f:
                item_on_h.append(h_mean)
                item_on_j.append(j_mean)
                for v in h_vals:
                    model_human_on[model].append(v)
                for v in j_vals:
                    model_judge_on[model].append(v)
            else:
                item_off_h.append(h_mean)
                item_off_j.append(j_mean)

        # Item-level correlations
        r_all, n_all = pearson(item_all_h, item_all_j)
        r_on, n_on = pearson(item_on_h, item_on_j)
        r_off, n_off = pearson(item_off_h, item_off_j)

        # Model-level on-target
        mh, mj = [], []
        for model in model_human_on:
            hv = model_human_on[model]
            jv = model_judge_on.get(model, [])
            if len(hv) >= 2 and len(jv) >= 2:
                mh.append(statistics.mean(hv))
                mj.append(statistics.mean(jv))
        r_model_on, n_model_on = pearson(mh, mj)

        # Model-level all
        mh_a, mj_a = [], []
        for model in model_human_all:
            hv = model_human_all[model]
            jv = model_judge_all.get(model, [])
            if len(hv) >= 3 and len(jv) >= 3:
                mh_a.append(statistics.mean(hv))
                mj_a.append(statistics.mean(jv))
        r_model_all, n_model_all = pearson(mh_a, mj_a)

        results[f] = {
            "item_all": (r_all, n_all),
            "item_on": (r_on, n_on),
            "item_off": (r_off, n_off),
            "model_on": (r_model_on, n_model_on),
            "model_all": (r_model_all, n_model_all),
        }
    return results


def gold_analysis(data: dict) -> list[dict]:
    """Per-gold-item accuracy analysis."""
    gold_by_id = {g["behavioral_response_id"]: g for g in data["gold_items"]}

    used_golds = defaultdict(list)
    for r in data["good_gold"]:
        used_golds[r["behavioral_response_id"]].append(r)

    results = []
    for gid in sorted(used_golds.keys()):
        gt = gold_by_id.get(gid)
        if not gt:
            continue
        ratings = used_golds[gid]
        n = len(ratings)
        total = correct = 0
        factor_detail = {}
        for f in FACTORS:
            gt_val = gt["ground_truth"][f]
            h_vals = [r[f"corrected_{f}"] for r in ratings if r[f"corrected_{f}"] is not None]
            if not h_vals:
                continue
            within1 = sum(1 for v in h_vals if abs(v - gt_val) <= 1)
            total += len(h_vals)
            correct += within1
            factor_detail[f] = {
                "gt": gt_val,
                "human_mean": statistics.mean(h_vals),
                "within1": within1,
                "n": len(h_vals),
            }

        results.append({
            "id": gid,
            "prompt_id": gt["prompt_id"],
            "purpose": gt["purpose"],
            "n_raters": n,
            "overall_accuracy": correct / total if total > 0 else None,
            "factors": factor_detail,
        })

    results.sort(key=lambda x: x["overall_accuracy"] or 0)
    return results


def between_model_variance(data: dict) -> dict:
    """SD of model means per factor."""
    model_by_rid = data["model_by_rid"]
    model_ratings = defaultdict(lambda: {f: [] for f in FACTORS})
    for r in data["good_sample"]:
        rid = r["behavioral_response_id"]
        model = model_by_rid.get(rid, "unknown")
        for f in FACTORS:
            val = r[f"corrected_{f}"]
            if val is not None:
                model_ratings[model][f].append(val)

    results = {}
    for f in FACTORS:
        means = [(m, statistics.mean(model_ratings[m][f]), len(model_ratings[m][f]))
                 for m in model_ratings if len(model_ratings[m][f]) >= 3]
        means.sort(key=lambda x: x[1])
        if len(means) >= 2:
            sd = statistics.stdev([m[1] for m in means])
            results[f] = {
                "sd": sd,
                "range": (means[0][1], means[-1][1]),
                "lowest": (means[0][0], means[0][1], means[0][2]),
                "highest": (means[-1][0], means[-1][1], means[-1][2]),
                "n_models": len(means),
            }
    return results


def rating_distributions(data: dict) -> dict:
    """Rating distributions per factor."""
    results = {}
    for f in FACTORS:
        vals = [r[f"corrected_{f}"] for r in data["good_sample"]
                if r[f"corrected_{f}"] is not None]
        if not vals:
            continue
        results[f] = {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals),
            "dist": dict(Counter(vals)),
        }
    return results


# ── Report generation ────────────────────────────────────────────────────────

def _fmt_r(r_val: float | None, n: int) -> str:
    if r_val is None:
        return f"N/A (N={n})"
    return f"{r_val:.3f} (N={n})"


def _fmt_icc(icc_val: float | None, n: int) -> str:
    if icc_val is None:
        return f"N/A (N={n})"
    return f"{icc_val:.3f} (N={n})"


def generate_report(data: dict) -> str:
    sess = sessions_summary(data)
    cov = coverage(data)
    iccs = compute_icc(data)
    hj = human_judge_correlations(data)
    gold = gold_analysis(data)
    bmv = between_model_variance(data)
    rdist = rating_distributions(data)

    lines = []
    lines.append("# Prolific Human Rating Analysis")
    lines.append("")
    lines.append(f"**Excluded participants:** {len(data['excluded_pids'])}")
    lines.append(f"**Usable participants:** {sess['usable_participants']}")
    lines.append("")

    # Sessions
    lines.append("## Sessions")
    lines.append("")
    for status, cnt in sorted(sess["by_status"].items()):
        lines.append(f"- {status}: {cnt}")
    lines.append(f"- **Total:** {sess['total']}, **Completed:** {sess['completed']}")
    lines.append("")

    # Coverage
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Pool size: {cov['pool_size']}")
    lines.append(f"- Total usable sample ratings: {cov['total_ratings']}")
    lines.append(f"- Items with 0 ratings: {cov['items_zero']}")
    for k in sorted(cov["distribution"].keys()):
        lines.append(f"- Items with {k} rating(s): {cov['distribution'][k]}")
    lines.append(f"- **Items at target (>=2): {cov['items_2plus']}/{cov['pool_size']} "
                 f"({cov['items_2plus']/cov['pool_size']:.1%})**")
    lines.append(f"- Additional ratings needed: {cov['ratings_needed']}")
    lines.append(f"- Estimated participants needed: ~{cov['participants_needed']}")
    lines.append("")

    # Rating distributions
    lines.append("## Rating Distributions (corrected, non-gold)")
    lines.append("")
    lines.append("| Factor | N | Mean | SD | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|--------|---|------|----|----|----|----|----|----|")
    for f in FACTORS:
        rd = rdist.get(f, {})
        if not rd:
            continue
        d = rd["dist"]
        lines.append(f"| {f} | {rd['n']} | {rd['mean']:.2f} | {rd['sd']:.2f} "
                     f"| {d.get(1,0)} | {d.get(2,0)} | {d.get(3,0)} | {d.get(4,0)} | {d.get(5,0)} |")
    lines.append("")

    # ICC
    lines.append("## Inter-Rater Reliability (ICC(2,k))")
    lines.append("")
    lines.append("| Factor | All Items | On-Target | Off-Target |")
    lines.append("|--------|-----------|-----------|------------|")
    for f in FACTORS:
        ic = iccs[f]
        lines.append(f"| {f} ({FACTOR_NAMES[f]}) "
                     f"| {_fmt_icc(*ic['all'])} "
                     f"| {_fmt_icc(*ic['on_target'])} "
                     f"| {_fmt_icc(*ic['off_target'])} |")
    lines.append("")
    lines.append("Target: ICC >= 0.60")
    lines.append("")

    # Human-judge correlations
    lines.append("## Human-Judge Correlations")
    lines.append("")
    lines.append("### Item-Level")
    lines.append("")
    lines.append("| Factor | All | On-Target | Off-Target |")
    lines.append("|--------|-----|-----------|------------|")
    for f in FACTORS:
        h = hj[f]
        lines.append(f"| {f} | {_fmt_r(*h['item_all'])} "
                     f"| {_fmt_r(*h['item_on'])} "
                     f"| {_fmt_r(*h['item_off'])} |")
    lines.append("")

    lines.append("### Model-Level")
    lines.append("")
    lines.append("| Factor | All Prompts | On-Target Prompts |")
    lines.append("|--------|-------------|-------------------|")
    for f in FACTORS:
        h = hj[f]
        lines.append(f"| {f} | {_fmt_r(*h['model_all'])} "
                     f"| {_fmt_r(*h['model_on'])} |")
    lines.append("")

    # Between-model variance
    lines.append("## Between-Model Variance")
    lines.append("")
    lines.append("| Factor | SD | Range | Lowest Model | Highest Model |")
    lines.append("|--------|----|-------|-------------|---------------|")
    for f in FACTORS:
        bm = bmv.get(f, {})
        if not bm:
            continue
        lo = bm["lowest"]
        hi = bm["highest"]
        lines.append(f"| {f} | {bm['sd']:.3f} | {bm['range'][0]:.2f}-{bm['range'][1]:.2f} "
                     f"| {lo[0][:35]} ({lo[1]:.2f}, N={lo[2]}) "
                     f"| {hi[0][:35]} ({hi[1]:.2f}, N={hi[2]}) |")
    lines.append("")

    # Gold items
    lines.append("## Gold Item Diagnostics")
    lines.append("")
    lines.append("Sorted by overall accuracy (worst first).")
    lines.append("")
    lines.append("| ID | Prompt | Purpose | Acc | N |")
    lines.append("|----|--------|---------|-----|---|")
    for g in gold:
        acc = g["overall_accuracy"]
        acc_str = f"{acc:.0%}" if acc is not None else "N/A"
        flag = " **" if acc is not None and acc < 0.55 else ""
        lines.append(f"| {g['id']} | {g['prompt_id']} | {g['purpose']} "
                     f"| {acc_str}{flag} | {g['n_raters']} |")
    lines.append("")

    # Worst gold items detail
    lines.append("### Worst Gold Items (accuracy < 55%)")
    lines.append("")
    for g in gold:
        if g["overall_accuracy"] is not None and g["overall_accuracy"] < 0.55:
            lines.append(f"**Gold {g['id']} ({g['prompt_id']}, {g['purpose']})** "
                         f"— {g['overall_accuracy']:.0%} accuracy, N={g['n_raters']}")
            for f in FACTORS:
                fd = g["factors"].get(f)
                if not fd:
                    continue
                pct = fd["within1"] / fd["n"] if fd["n"] > 0 else 0
                diff = fd["human_mean"] - fd["gt"]
                flag = " <<<" if pct < 0.70 else ""
                lines.append(f"- {f}: GT={fd['gt']}, human mean={fd['human_mean']:.1f}, "
                             f"diff={diff:+.1f}, within +/-1={fd['within1']}/{fd['n']} "
                             f"({pct:.0%}){flag}")
            lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prolific human rating analysis")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Prolific PIDs to exclude (default: Batch 1 rejections)")
    parser.add_argument("--label", default=None,
                        help="Label for the output file (e.g. 'batch1' -> prolific_report_batch1.md)")
    args = parser.parse_args()

    excluded = set(args.exclude) if args.exclude is not None else DEFAULT_EXCLUDED

    if not PROLIFIC_DB.exists():
        print(f"Error: {PROLIFIC_DB} not found. Pull from EC2 first.", file=sys.stderr)
        sys.exit(1)

    if args.label:
        out_path = OUTPUT_PATH.parent / f"prolific_report_{args.label}.md"
    else:
        out_path = OUTPUT_PATH

    print(f"Loading data (excluding {len(excluded)} participants)...", file=sys.stderr)
    data = load_data(excluded)

    report = generate_report(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}", file=sys.stderr)

    # Also print to stdout
    print(report)


if __name__ == "__main__":
    main()
