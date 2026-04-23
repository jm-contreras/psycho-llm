# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**psycho-llm** is a research project constructing and validating a psychometric instrument for measuring behavioral dispositions in LLMs. The study administers 300 items (240 direct Likert + 60 scenario) + BFI-44 to 25 model configurations across 9 API providers, analyzes the factor structure via EFA/CFA, and validates against human behavioral ratings + LLM-as-judge ensemble. Target: arXiv preprint by April 24, 2026.

### Source of Truth

- **Methods and results:** [paper/main.tex](paper/main.tex) — the manuscript draft is the authoritative description of what was done and what was found. TODOs mark sections pending completion.
- **Phase 3 behavioral prompts:** [behavioral_prompts_v2.md](behavioral_prompts_v2.md) — 20 prompts targeting the 5-factor structure, including rating scale, HIT design, and discriminant validity notes.
- **Experimental design (archival):** [llm_psychometrics_experimental_design_v0.6.md](llm_psychometrics_experimental_design_v0.5.md) — the pre-execution plan. Superseded by the paper for anything that differs; deviations documented in the paper §Deviations.
- **Preregistration (archival):** [osf_preregistration_v3.md](osf_preregistration_v3.md) — original preregistration. Deviations documented in the paper.

### Current Status (as of 2026-03-28)

- **Phase 1 complete.** All 25 models × 300 AI-native items × 30 runs + 44 BFI items × 30 runs collected (~258K rows).
- **Phase 2 complete.** 5-factor EFA solution (Responsiveness, Deference, Boldness, Guardedness, Verbosity), 96/240 items retained. All reliability metrics exceed thresholds. Convergent/discriminant validity vs. BFI-44 computed.
- **Phase 3 in progress.** Behavioral samples collected (2,500 = 25 models × 20 prompts × 5 runs). LLM-as-judge ratings complete. **Prolific human ratings next.**
- **Paper draft started.** Methods and available results drafted in paper/main.tex.

### 5-Factor Structure

| Factor | Name | Items | α | Interpretation |
|---|---|---|---|---|
| F1 | Responsiveness | 27 | .976 | "Good assistant" general factor — adapts, structures, engages |
| F2 | Deference | 25 | .969 | "Stay in your lane" — complies, contains, withholds judgment |
| F3 | Boldness | 15 | .953 | Originality + epistemic confidence + personal style |
| F4 | Guardedness | 11 | .924 | Over-refusal, safety signaling, caution |
| F5 | Verbosity | 18 | .945 | Unsolicited disclaimers, preambles, over-communication |

Factor codes: `RE`, `DE`, `BO`, `GU`, `VB`.

## Repository Structure

```
paper/                          # Manuscript (LaTeX)
  main.tex                      # Methods + results draft (source of truth)
items/                          # Item pool (source of truth for data collection)
  llm_native_item_pool_v0.2.md  # 300 items across 12 candidate dimensions
pipeline/                       # Data collection pipeline
  config.py                     # Loads .env + model_registry.json; builds rate limiters
  item_loader.py                # Parses item pool markdown → list of dicts
  api_client.py                 # litellm wrapper (text score + optional log-prob score)
  storage.py                    # SQLite + CSV output with idempotency
  runner.py                     # Entry point: run() sync + async_run() parallel
  rate_limiter.py               # Async RPM/TPM/TPD rate limiter
  token_budget.py               # Per-resource-group daily token budget tracking
  progress.py                   # Progress display utilities
  status.py                     # Collection status reporting
  bfi_runner.py                 # BFI-44 administration runner
  bfi_items.py                  # BFI-44 item definitions
  batch_gemini.py               # Google Gemini batch API runner
  batch_openai.py               # OpenAI batch API runner
  bfi_batch_gemini.py           # BFI batch runner for Gemini
  bfi_batch_openai.py           # BFI batch runner for OpenAI
  judge_prompt.py               # LLM-as-judge prompt construction + few-shot examples
  judge_runner.py               # Judge ensemble execution
  behavioral_runner.py          # Behavioral sample collection
data/raw/                       # Git-ignored: responses.db (SQLite), responses.csv
analysis/                       # Analysis scripts
  primary_analyses.py           # Main: EFA → item selection → CFA → reliability → profiles
  esem.py                       # ESEM confirmatory analysis + Tucker congruence
  bfi_analysis.py               # BFI-44 trait scoring, profiles, convergent/discriminant
  judge_analysis.py             # LLM-as-judge inter-rater agreement + predictive validity
  data_loader.py                # Loads responses.db into pandas DataFrames
  engineering_checks.py         # Per-model success/error rates
  item_quality.py               # Item variance, refusal rates, item-total correlations
  dimension_coherence.py        # Per-dimension heatmaps, ICC analysis
  factor_structure.py           # Parallel analysis, EFA, helper functions
  report.py                     # Markdown report generation
  run_diagnostics.py            # Entry point: generates diagnostic_report.md
  output/                       # Generated reports and plots
    primary_analysis_report.md  # 5-factor EFA/CFA/reliability results
    esem_diagnostic_report.md   # ESEM fit + Tucker congruence
    bfi_report.md               # BFI-44 analysis
    judge_report.md             # Judge agreement + predictive validity
    archive/seven_factor/       # Archived 7-factor reports (superseded)
model_registry.json             # 25-model registry with API routing metadata
behavioral_prompts_v2.md        # Phase 3: 20 behavioral prompts (4 per factor × 5 factors)
```

## Running Analyses

```bash
# Primary analyses (5-factor EFA → CFA → reliability → profiles → validity)
python -m analysis.primary_analyses

# ESEM confirmatory analysis
python -m analysis.esem

# BFI-44 analysis
python -m analysis.bfi_analysis

# Judge analysis (inter-rater agreement + predictive validity)
python -m analysis.judge_analysis

# Preliminary diagnostics
python -m analysis.run_diagnostics
```

## Running the Pipeline

```bash
pip install litellm python-dotenv
cp .env.example .env            # fill in all provider credentials

# AI-native items — target a specific model
python -m pipeline.runner --models "Claude Sonnet 4.6" --n-items 10 --n-runs 2

# Parallel collection across all providers
python -m pipeline.runner --n-runs 30 --providers bedrock openai google xai azure alibaba xiaomi ai21 deepseek --parallel

# BFI-44 collection
python -m pipeline.bfi_runner --n-runs 30 --providers bedrock openai google xai azure alibaba xiaomi ai21 deepseek --parallel
```

The runner is **resumable**: completed `(model_id, item_id, run_number)` triples are skipped. The `--parallel` flag runs all models concurrently with per-provider RPM/TPM/TPD rate limiting.

## Data Schema

Output in `data/raw/responses.db` (authoritative) and `data/raw/responses.csv`:

| Column | Description |
|---|---|
| `model_id` | `litellm_model_id` from registry |
| `item_id` | e.g. `SA-D01`, `EC-S03` |
| `dimension` | Full dimension name |
| `item_type` | `direct` or `scenario` |
| `keying` | `+` or `-` (reverse-coded) |
| `run_number` | 1-indexed |
| `text_scoring_method` | `structured` or `regex` |
| `raw_response` | Full text from model (`content` field) |
| `reasoning_content` | Chain-of-thought from reasoning models (e.g. GPT-OSS 120B, DeepSeek R1); NULL for standard models |
| `parsed_score` | Score from text response (1–5 or 1–4); NULL on failure |
| `logprob_score` | Softmax-weighted score from log-prob ranking; NULL if unavailable |
| `logprob_available` | 1 if provider returned log-probs, 0 otherwise |
| `status` | `success`, `refusal`, `parse_error`, `api_error` |
| `error_message` | Non-null on failure |

## model_registry.json

Each model entry has three ID fields:
- `provider_model_id` — exact string for the provider API
- `litellm_model_id` — string passed to `litellm.completion()` (e.g., `bedrock/us.anthropic.claude-sonnet-4-6`)
- `api_provider` — selects credentials: `bedrock`, `openai`, `google`, `xai`, `alibaba`, `xiaomi`, `azure`, `ai21`, `deepseek`

Models with `group_as` share a completion pool — runs collected via any endpoint in the group count for all (used for DeepSeek R1 which is collected across Azure + native API).

Log-probs available for 7/25 models: GPT-5.4, GPT-5.4 Nano, GPT-OSS 120B, DeepSeek V3.2, DeepSeek R1, Llama 4 Maverick, Qwen 3.5. All other models use repeated sampling only.

## AWS Profile

MTurk and Bedrock use an AWS profile named `psycho-llm`. The MTurk client loads this automatically via `AWS_PROFILE` env var (defaults to `psycho-llm`). Make sure `~/.aws/credentials` has a `[psycho-llm]` profile configured.

## Environment Variables

Copy `.env.example` → `.env`. All provider credentials are needed for full collection:
```
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION  # Bedrock (11 models)
AZURE_API_KEY / AZURE_API_BASE / AZURE_API_VERSION              # Azure AI (7 models)
OPENAI_API_KEY                                                   # OpenAI (3 models)
GEMINI_API_KEY                                                   # Google (2 models)
XAI_API_KEY                                                      # xAI (1 model)
DASHSCOPE_API_KEY                                                # Alibaba/Qwen (1 model)
MIMO_API_KEY / MIMO_BASE_URL                                     # Xiaomi (1 model)
AI21_API_KEY                                                     # AI21 (1 model)
DEEPSEEK_API_KEY                                                 # DeepSeek native (1 model)
```

## Item Pool Structure

Items in `llm_native_item_pool_v0.2.md` are organized by 12 candidate dimensions. Sensitivity to Criticism was merged into Social Alignment before item generation. Each dimension has 20 direct Likert items (~50% reverse-coded) and 5 scenario items. Item IDs follow the pattern `{DIM_CODE}-D{nn}` (direct) and `{DIM_CODE}-S{nn}` (scenario).

Dimension codes: `SA` (Social Alignment), `CA` (Compliance vs. Autonomy), `EC` (Epistemic Confidence), `RS` (Refusal Sensitivity), `VE` (Verbosity/Elaboration), `HE` (Hedging), `CC` (Creativity vs. Convention), `CR` (Catastrophizing/Risk Amplification), `AT` (Apologetic Tendency), `PI` (Proactive Initiative), `WR` (Warmth and Rapport), `SD` (Self-Disclosure).

Items are sorted by `item_id` in `item_loader.py` — this sort order is the canonical ordering for `n_items` selection, ensuring reproducibility.
