#!/usr/bin/env bash
# End-to-end reproduction: fresh clone -> venv -> data -> analyses -> headline numbers.
#
# Assumes you've already downloaded + extracted the OSF data archive into data/
# (see README.md). If data/raw/responses.db is missing, the script exits early.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f data/raw/responses.db ]]; then
  echo "ERROR: data/raw/responses.db not found." >&2
  echo "Download the OSF data archive first — see README.md." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p analysis/output

echo "==> Running primary analyses (EFA + CFA + reliability)"
python -m analysis.primary_analyses

echo "==> BFI-44 analysis"
python -m analysis.bfi_analysis

echo "==> Judge ensemble analysis"
python -m analysis.judge_analysis

echo "==> Prolific human-rating analysis"
python -m analysis.prolific_analysis

echo "==> Predictive validity + bootstrap CIs"
python -m analysis.predictive_validity
python -m analysis.bootstrap_ci
python -m analysis.within_prompt_validity
python -m analysis.mixed_model_validity

echo "==> Acquiescence audit + model-level EFA"
python -m analysis.acquiescence_audit
python -m analysis.model_level_efa

echo "==> Appendix tables"
for s in make_appendix_tables make_hero_profile make_metadata_aggregation \
         make_method_convergence make_mtmm_factor make_ocean_profile \
         make_paired_profiles make_per_model_reliability \
         make_unified_profile_table; do
  echo "    -> $s"
  python -m analysis.$s || echo "    (skipped: $s)"
done

echo
echo "=============================================================="
echo "  Headline numbers (compare to paper/main.tex)"
echo "=============================================================="
python scripts/print_headline_numbers.py
