# psycho-llm

Code and materials for **An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Modelss**, by Juan Manuel Contreras.

This repository accompanies the arXiv preprint (link forthcoming). It contains the item pool, data-collection pipeline, and analysis code used to construct and validate a 5-factor LLM-native psychometric instrument (Responsiveness, Deference, Boldness, Guardedness, Verbosity) on 25 models across 9 API providers.

## Data

Raw response data, LLM-judge ratings, and anonymized Prolific human ratings are hosted on OSF:

> **OSF DOI:** _to be filled at release_ (`psycho-llm-data-v1.tar.gz`)

Download the archive and extract into `data/` before running any analysis:

```bash
mkdir -p data
curl -L <OSF_DOWNLOAD_URL> -o data.tar.gz
tar -xzf data.tar.gz -C data/ --strip-components=1
```

Prolific participant IDs are replaced with stable 12-character hashes (salt: `psycho-llm-osf-v1`). The anonymization script is included in the OSF archive.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A `.env` is only required if you intend to re-collect data from the model APIs (`cp .env.example .env` and fill in provider credentials). Analyses only need the OSF data.

## Reproducing the paper

```bash
bash scripts/reproduce.sh
```

This runs every analysis entry point in `analysis/` and prints the headline numbers for side-by-side comparison against the manuscript (Cronbach α, Tucker φ, sample sizes, convergence correlations). Outputs land in `analysis/output/`.

Expected runtime: ~20–45 minutes on a modern laptop.

## Layout

```
paper/           LaTeX source of the manuscript (main.tex, references.bib)
items/           AI-native item pool (machine-readable)
pipeline/        Data collection pipeline (litellm-based; not needed for analyses)
analysis/        EFA, CFA, reliability, validity, appendix-table generators
scripts/         Reproducibility entry points
model_registry.json          Model routing metadata for 25 configurations
behavioral_prompts_v2.md     20 behavioral prompts used in Phase 3
llm_native_item_pool_v0.2.md Full item pool (300 items)
osf_preregistration_v3.md    Archival preregistration
```

See [`CLAUDE.md`](CLAUDE.md) for deeper architectural notes.

## Citation

```bibtex
@article{contreras2026psycholm,
  title  = {An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models},
  author = {Contreras, JM},
  year   = {2026},
  note   = {arXiv preprint (forthcoming)}
}
```

## License

Code is released under the MIT License. Data on OSF is released under CC-BY 4.0.
