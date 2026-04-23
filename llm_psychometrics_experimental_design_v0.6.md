# LLM Psychometric Instrument: Experimental Design

**Author:** Juan Manuel Contreras
**Date:** March 2026
**Status:** Draft v0.6 — synced with actual data collection state (2026-03-28)

---

## 1. Research Question

Can we design a psychometrically valid personality instrument native to LLMs — one whose constructs are derived from LLM behavioral affordances rather than human personality theory, and that demonstrates internal consistency, test-retest reliability, convergent/discriminant validity, predictive validity, and resistance to known LLM-specific response biases?

## 2. Contribution

Existing work applies human personality instruments to LLMs:

- **MPI** (Jiang et al., NeurIPS 2023): Adapts IPIP-NEO items for LLMs, validates via vignette tasks. Constructs = Big Five.
- **DeepMind Framework** (Serapio-García et al., Nature Machine Intelligence 2025): Full psychometric pipeline with massive N, log-probability scoring, convergent/discriminant/criterion validity. Constructs = Big Five.
- **TRAIT** (Lee et al., NAACL Findings 2025): 8,000 scenario-based items with psychometric auditing, reduced prompt/option-order sensitivity. Constructs = Big Five + Dark Triad.
- **Peereboom et al.** (2024): Shows human factor structures don't replicate in LLM response space — EFA on LLM responses to HEXACO yields uninterpretable factors.
- **Gupta et al.** (2024): Demonstrates pervasive prompt sensitivity and option-order sensitivity in LLM self-assessment, arguing current tools are unreliable.
- **Li et al.** (2025): Shows forced-choice format reduces social desirability inflation vs. Likert self-report.

**The gap:** No existing instrument defines constructs native to LLM behavior. All start from human trait taxonomies. Several papers (Pellert et al., Gupta et al., Peereboom et al.) explicitly call for LLM-specific instruments.

**Our contribution:** A validated, open-source psychometric instrument whose constructs and items are derived bottom-up from LLM behavioral dimensions via exploratory factor analysis, not imported from human personality science. We provide a full construct-validity pipeline including reliability, convergent/discriminant validity against Big Five as a reference, predictive validity via blinded human ratings with LLM-as-judge scaling, and systematic assessment of LLM-specific response biases.

## 3. Model Selection

### Active Models (25 configurations across 17 families)

| # | Model | Family | Access Path | Log-Probs | Notes |
|---|---|---|---|---|---|
| 1 | Claude Opus 4.6 | Anthropic | AWS Bedrock | No | Top Anthropic |
| 2 | Claude Sonnet 4.6 | Anthropic | AWS Bedrock | No | Mid-tier Anthropic |
| 3 | Claude Haiku 4.5 | Anthropic | AWS Bedrock | No | Small/efficient Anthropic |
| 4 | GPT-5.4 | OpenAI | OpenAI API | Yes | Top OpenAI |
| 5 | GPT-5.4 Mini | OpenAI | OpenAI API | No | Mid-tier OpenAI |
| 6 | GPT-5.4 Nano | OpenAI | OpenAI API | Yes | Small OpenAI |
| 7 | GPT-OSS 120B | OpenAI | Azure AI | Yes | Open-weight OpenAI |
| 8 | Gemini 3.1 Pro | Google | Google API | No | Top Google |
| 9 | Gemini 3.1 Flash | Google | Google API | No | Mid-tier Google |
| 10 | Gemma 3 27B | Google | AWS Bedrock | No | Open-weight Google |
| 11 | Grok 4.20 Beta | xAI | xAI API | No | xAI frontier |
| 12 | DeepSeek V3.2 | DeepSeek | Azure AI | Yes | Top open-weight |
| 13 | DeepSeek R1 | DeepSeek | Azure AI + DeepSeek API | Yes | Reasoning-specialized |
| 14 | Qwen 3.5 | Alibaba | Alibaba DashScope | Yes | Chinese frontier |
| 15 | Kimi K2.5 | Moonshot AI | AWS Bedrock | No | Emerging competitor |
| 16 | GLM-5 | Zhipu AI | Z.ai API | No | Chinese frontier |
| 17 | Mistral Large 3 | Mistral AI | Azure AI | No | European frontier |
| 18 | MiniMax M2.5 | MiniMax | AWS Bedrock | No | Emerging competitor |
| 19 | Llama 4 Maverick | Meta | Azure AI | Yes | Top open-weight Meta |
| 20 | MiMo-V2-Pro | Xiaomi | Xiaomi direct API | No | Emerging competitor |
| 21 | Nemotron 3 Super | NVIDIA | AWS Bedrock | No | Mamba-Transformer hybrid MoE |
| 22 | Nova 2 Pro | Amazon | AWS Bedrock | No | Amazon frontier |
| 23 | Command A | Cohere | Azure AI | No | Cohere frontier |
| 24 | Jamba Large 1.7 | AI21 | AI21 direct API | No | Mamba-based architecture |
| 25 | Phi 4 | Microsoft | Azure AI | No | Small/efficient Microsoft |

**Log-probability availability:** 7 of 25 models return log-probabilities (GPT-5.4, GPT-5.4 Nano, GPT-OSS 120B, DeepSeek V3.2, DeepSeek R1, Llama 4 Maverick, Qwen 3.5). All 25 models have full repeated-sampling data (30 runs per item). For the 7 log-prob models, both scoring methods are available; convergence is reported as a methodological finding.

### Within-Family Comparisons

| Comparison | Models | Notes |
|---|---|---|
| Anthropic size ladder | Opus 4.6 → Sonnet 4.6 → Haiku 4.5 | Three tiers |
| OpenAI size ladder | GPT-5.4 → Mini → Nano (+ GPT-OSS 120B open-weight) | Four configs, three sizes + open-weight |
| Google size ladder | Gemini 3.1 Pro → Flash → Gemma 3 27B | Proprietary large/small + open-weight |
| DeepSeek reasoning comparison | V3.2 (general) vs. R1 (reasoning-specialized) | Different training, not just inference mode |

### Architectural Diversity

| Architecture | Models |
|---|---|
| Standard Transformer | Most models |
| Mamba-Transformer hybrid MoE | Nemotron 3 Super |
| Mamba-based | Jamba Large 1.7 |
| MoE | GPT-OSS 120B, DeepSeek V3.2, DeepSeek R1, Qwen 3.5 |

### Models Not Included (and Why)

- **Falcon 3 10B** (TII): Dropped — difficult to access, lower priority given other small-model coverage.
- **K-EXAONE** (LG AI Research): Dropped — difficult to access, lower priority.
- **Step-3.5-Flash** (Stepfun): Dropped — difficult to access, lower priority.
- **Code-specialized models** (Devstral, Qwen Coder): Different domain.
- **Multimodal-only models** (Voxtral, Pegasus): Text personality is the focus.
- **Sub-1B models:** Likely too weak for meaningful personality responses.

### Model Exclusion Criteria

A model configuration will be excluded from primary analyses if it meets any of the following criteria:

- Refuses to respond to >40% of instrument items.
- Produces zero variance (identical responses across all 30 runs) on >60% of items.
- API access to the pinned model version becomes unavailable during the data collection window.

As a preregistered robustness check, we re-run primary analyses excluding any model with a refusal rate >25% or zero-variance rate >40%.

## 4. Instrument Design

### Phase 1: Item Generation and Refinement

#### 1a. BFI-44 Baseline (Convergent Validity Anchor)

Administer the standard BFI-44 to all models. This is NOT the novel instrument — it provides convergent validity benchmarks (e.g., does our "Social Alignment" dimension correlate with BFI Agreeableness?).

**Status:** Complete. All 25 models × 44 items × 30 runs collected.

#### 1b. AI-Native Item Pool

300 candidate items grounded in LLM behavioral affordances: 240 direct Likert items (20 per dimension × 12 dimensions) + 60 scenario items (5 per dimension × 12 dimensions). Candidate dimensions (scaffolding for item generation; final structure determined by EFA):

| # | Candidate Domain | Direct Items | Scenario Items |
|---|---|---|---|
| 1 | Social Alignment | 20 | 5 |
| 2 | Compliance vs. Autonomy | 20 | 5 |
| 3 | Epistemic Confidence | 20 | 5 |
| 4 | Refusal Sensitivity | 20 | 5 |
| 5 | Verbosity / Elaboration | 20 | 5 |
| 6 | Hedging | 20 | 5 |
| 7 | Creativity vs. Convention | 20 | 5 |
| 8 | ~~Sensitivity to Criticism~~ | — | — |
| 9 | Catastrophizing / Risk Amplification | 20 | 5 |
| 10 | Apologetic Tendency | 20 | 5 |
| 11 | Proactive Initiative | 20 | 5 |
| 12 | Warmth and Rapport | 20 | 5 |
| 13 | Self-Disclosure | 20 | 5 |

**Dimension 8 (Sensitivity to Criticism)** was merged into Social Alignment before item generation. Pushback-capitulation items (SA-D22–D24, SA-S04–S05) target this sub-facet within the SA scale. See item pool document for full rationale.

**Item format — two parallel versions per item:**

1. **Direct self-report:** 5-point Likert scale (Strongly Disagree to Strongly Agree). 240 items total, ~50% reverse-coded per dimension.
2. **Scenario-based:** Multiple-choice behavioral scenarios with 4 response options varying in trait intensity. 60 items total.

**Status:** Complete. All 25 models × 300 items × 30 runs collected (~258K rows in responses.db).

#### 1c. Scoring Methods

**Primary method: Repeated sampling.** All 25 models are administered each item N=30 times with temperature=1.0, using structured output (JSON mode) for response parsing. Trait score = mean across runs. This is the universal scoring method applied to all models.

**Supplementary method: Log-probability scoring.** For the 7 models where API returns token log-probabilities (GPT-5.4, GPT-5.4 Nano, GPT-OSS 120B, DeepSeek V3.2, DeepSeek R1, Llama 4 Maverick, Qwen 3.5), log-prob ranking over response options yields deterministic trait scores following the DeepMind framework. Both methods are available for these models; convergence between the two is reported as a methodological finding.

**Deviation from preregistration:** The preregistration describes log-prob scoring as the primary method with repeated sampling as fallback. In practice, only 7/25 models support log-probs, so repeated sampling is the primary (universal) method and log-prob scoring is supplementary. This matches the preregistered fallback plan ("If log-probability scoring is unavailable for >50% of model configurations... all models default to repeated sampling").

#### 1d. Item Refinement

- Drop items with near-zero variance across models (all models' mean scores fall within 0.5 Likert points of each other)
- Drop items with item–total correlations < 0.30 within their candidate domain
- Run exploratory factor analysis (EFA) on pooled response matrix to identify emergent factor structure
- Confirm with CFA on held-out data (random split-half of runs)
- Target: 40–60 direct items + 15–25 scenario items across 5–8 final dimensions

**EFA sample size:** Pooled across 25 models × 30 runs = 750 observations per item (with observation weighting to ensure each model contributes equally regardless of scoring method). Adequate for stable factor solutions.

### Phase 2: Reliability and Validity

#### Reliability

| Test | Method | Criterion |
|------|--------|-----------|
| Internal consistency | Cronbach's α (and McDonald's ω) per dimension per model | α > 0.70 |
| Cross-run stability | Split 30 runs into two halves (odd/even), correlate dimension scores across halves | r > 0.80 |
| Split-half | Odd-even item split, Spearman-Brown corrected | r > 0.80 |

**Cross-run stability note:** Assessed by splitting existing independent runs rather than a separate administration wave. This is equivalent given that each run is an independent conversation with no shared state; no minimum time delay is required because LLMs have no memory between conversations. For log-prob models, cross-run stability is trivially r = 1.0 (deterministic scoring) and is reported as such.

#### Validity

| Type | Method |
|------|--------|
| Convergent | Correlate AI-native dimensions with BFI-44 scores where theoretically related (e.g., Social Alignment ↔ Agreeableness, Creativity ↔ Openness) |
| Discriminant | Show low correlations between theoretically unrelated AI-native dimensions and BFI-44 traits (multitrait-multimethod matrix) |
| Predictive | Phase 3: hybrid LLM-as-judge + blinded human ratings of independent behavioral samples |
| Content | Expert review of items by AI researchers (informal) |
| Method convergence | Correlate direct self-report scores with scenario-based scores on same dimensions |

### Phase 3: Predictive Validity (Hybrid Design)

**Design rationale:** A hybrid LLM-as-judge + human calibration approach is used for predictive validity. Human ratings anchor the primary validity claim; the LLM-as-judge ensemble provides full coverage for secondary analyses; human-LLM agreement per dimension is reported as a methods contribution. This design was adopted after external review and is documented as a preregistered deviation.

For each model:

1. Administer the refined instrument → get personality profile scores.
2. Generate 20 behavioral samples using independent prompts (4 per factor × 5 factors) designed to elicit relevant behaviors. These prompts are distinct from instrument items in form and content. Each prompt is administered 5 times per model (temperature=1.0) yielding 2,500 behavioral response samples total (25 models × 20 prompts × 5 runs).
3. **LLM-as-judge ensemble** rates all 2,500 behavioral samples on all 5 instrument factors. See LLM-as-Judge Protocol below.
4. **Human raters** (MTurk API) rate a stratified calibration sample of ~400 behavioral responses on the same 5 factors, blind to instrument scores. Stratification ensures coverage across model families, target factors, and LLM-judge score distribution.
5. **Report predictive validity twice:**
   - **Primary:** Correlation between instrument factor scores and mean human-rated behavioral scores (400-sample calibration subset).
   - **Secondary:** Correlation between instrument factor scores and mean LLM-judge-rated behavioral scores (full 2,500 samples).
6. **Report human-LLM agreement per dimension** (Pearson r, Spearman ρ, ICC). Target: r > 0.70. Dimensions where agreement falls below r = 0.65: only the human-rated result is reported as valid predictive evidence.
7. Compute inter-rater reliability for human ratings: ICC(2,k). Target: ICC > 0.60.

**Prediction:** If the instrument is valid, a model scoring high on "Social Alignment" should produce behavioral samples that both human raters and LLM judges independently rate as more sycophantic.

#### Human Rating Design (Calibration Sample)

- **Sample size:** ~400 behavioral responses, stratified by model family, prompt dimension, and LLM-judge score distribution
- **Raters:** Sequential 2+1 rater design: 2 raters per HIT; 3rd rater only on disagreement. Recruited via MTurk API with qualification test
- **Rating scale:** 5 factor statements, 5-point agree/disagree, randomized forward/reverse keying per rater per HIT
- **Quality control:** Qualification test (≥80% gold-standard agreement), ongoing gold monitoring (15% rate), raters below 60% flagged/disqualified
- **Compensation:** ~$15/hour (3–5 min per HIT)
- **Estimated cost:** ~$1,000

#### LLM-as-Judge Protocol

An ensemble of 3 state-of-the-art LLMs rates all 2,500 behavioral samples. The judge prompt mirrors the human rating scale verbatim: same 5 factor statements, same 5-point scale. Each judge call receives a random forward/reverse keying assignment per factor (analogous to acquiescence-bias control for human raters); raw scores are reverse-corrected before analysis. Keying is seeded by `(judge_model_id, response_id)` for reproducibility. Four few-shot calibration examples are included, using isomorphic prompts (structurally matched to behavioral prompts but with different surface details to avoid contamination).

**Judge ensemble:**
- Judge A: Claude Opus 4.6 (via AWS Bedrock)
- Judge B: GPT-5.4 (via OpenAI API)
- Judge C: Gemini 3.1 Pro (via Google API)

**Cross-model exclusion:**
- Evaluating Claude models → exclude Claude judge, use GPT-5.4 + Gemini 3.1 Pro
- Evaluating GPT/OpenAI models → exclude GPT-5.4 judge, use Claude Opus + Gemini 3.1 Pro
- Evaluating Google models → exclude Gemini judge, use Claude Opus + GPT-5.4
- All other models → full ensemble (median of 3)
- Where 2 judges available → mean; where 3 → median

**Sensitivity check:** Re-run scoring with an alternative ensemble composition; report whether results differ substantively.

**Output format:** Structured JSON, one score per factor. No chain-of-thought for scoring (direct rating only).

**Keying seed:** Judge keying uses derived seed 20260318 (sequential from master seed 20260315), combined with `(judge_model_id, response_id)` per call for reproducibility.

**Few-shot calibration process:** Four calibration examples were authored with isomorphic prompts (matched in structure/domain/difficulty to the 20 behavioral prompts but with different surface details) and synthetic responses targeting diverse behavioral profiles. Ground-truth ratings were iteratively refined against all 3 judges until inter-judge alignment reached |Δ| ≤ 1 on 59/60 judge×factor ratings. The single remaining outlier (Gemini GU on the borderline-guardedness example, Δ=-2) reflects genuine construct ambiguity rather than miscalibration. Calibration examples and ground-truth ratings are included in the open-source release (`pipeline/judge_prompt.py`).

**Estimated cost:** ~$50–80.

### Phase 4: Bias Assessment (Contingent on Timeline)

#### 4a. Social Desirability Bias

**Matched-pair design:** ~25 item pairs, one neutral and one socially desirable framing of the same construct. Social desirability index per model = mean absolute difference.

**Forced-choice substudy:** ~20 items reformatted into forced-choice pairs. Compare trait score distributions from forced-choice vs. Likert.

#### 4b. Prompt Framing Sensitivity

Refined instrument re-administered under four framing conditions (direct, third-person, behavioral-frequency, embedded). ICC across conditions per dimension per model.

#### 4c. Option-Order Sensitivity

Refined instrument re-administered with reversed Likert orderings. Equivalence tests on forward vs. reverse scores.

## 5. Technical Specification

### API Parameters

All runs use:
- `temperature = 1.0`
- `top_p = 1.0`
- `max_tokens = 256` (Likert/scenario items); `2048` (behavioral samples)
- No system prompt beyond minimal formatting instruction
- Each item administered in a fresh conversation (no context carryover)
- Structured output (JSON mode) for response parsing where supported
- Async parallel execution with per-provider RPM/TPM/TPD rate limiting
- Resumable/idempotent: skips already-completed (model_id, item_id, run_number) triples

### Data Storage

- **Primary:** SQLite database (`responses.db`)
- **Secondary:** CSV export
- **Schema:** `model_id, item_id, dimension, item_type, keying, run_number, scoring_method, raw_response, parsed_score, timestamp`
- **DeepSeek R1 deduplication:** Collected across two endpoints (native API + Azure); `group_as` field ensures deduplication

### Run Structure

| Phase | Items | Models | Runs/Item/Model | Total API Calls | Status |
|-------|-------|--------|-----------------|-----------------|--------|
| Phase 1a: BFI-44 baseline | 44 | 25 | 30 | 33,000 | ✅ Complete |
| Phase 1b: AI-native item pool | 300 | 25 | 30 | 225,000 | ✅ Complete |
| Phase 3: Behavioral samples | 20 | 25 | 5 | ~2,500 | ⬜ Pending |
| Phase 3: LLM-as-judge scoring | ~2,500 × 5 factors | 3 judges | 1 | ~37,500 | ⬜ Pending |
| Phase 4a: Social desirability | 50 | 25 | 30 | 37,500 | ⬜ Contingent |
| Phase 4a: Forced-choice | 20 | 25 | 30 | 15,000 | ⬜ Contingent |
| Phase 4b: Framing (4 conditions) | 240 | 25 | 30 | 180,000 | ⬜ Contingent |
| Phase 4c: Option-order | 60 | 25 | 30 | 45,000 | ⬜ Contingent |

### Cost Estimate

| Category | Estimated Cost | Status |
|----------|---------------|--------|
| Phase 1a + 1b: Model API calls (~258K) | ~$500 | ✅ Spent |
| Phase 3: Behavioral sample generation (~2.5K calls) | ~$15 | ⬜ Pending |
| Phase 3: LLM-as-judge ensemble (~37.5K calls) | ~$50–80 | ⬜ Pending |
| Phase 3: MTurk human annotations (400 × 2+1 raters + training) | ~$1,000 | ⬜ Pending |
| **Phases 1–3 total** | **~$1,565–1,595** | |
| Phase 4 (if run): API calls | ~$400–600 | ⬜ Contingent |
| **Total with Phase 4** | **~$1,965–2,195** | |

## 6. Analysis Plan

### Primary Analyses

1. **Factor structure:** EFA (principal axis factoring, oblimin rotation) on pooled AI-native item responses → identify emergent dimensions → CFA on held-out split to confirm
2. **Reliability metrics:** Cronbach's α, McDonald's ω, split-half r (Spearman-Brown corrected), cross-run stability r — per dimension per model
3. **Model personality profiles:** Standardized scores across discovered dimensions, visualized as radar plots
4. **Convergent/discriminant validity:** Multitrait-multimethod matrix comparing AI-native dimensions with BFI-44 traits
5. **Predictive validity (primary):** Correlation between instrument scores and human-rated behavioral scores (400-sample calibration subset)
6. **Predictive validity (secondary):** Correlation between instrument scores and LLM-judge-rated behavioral scores (full 2,500 samples)
7. **Human-LLM judge agreement:** Per-dimension agreement metrics as a methods contribution
8. **Method convergence:** Correlation between direct self-report and scenario-based scores on matched dimensions
9. **Scoring method convergence:** For the 7 log-prob models, correlation between repeated-sampling and log-prob trait scores

### Secondary Analyses

10. **Social desirability index** per model; forced-choice vs. Likert comparison (contingent)
11. **Framing sensitivity** — ICC across 4 conditions (contingent)
12. **Option-order symmetry** — equivalence tests (contingent)
13. **Scale effects** — within-family personality profile comparisons:
    - Anthropic: Opus → Sonnet → Haiku
    - OpenAI: GPT-5.4 → Mini → Nano (+ GPT-OSS 120B)
    - Google: Gemini Pro → Flash → Gemma 27B
14. **Reasoning effects** — DeepSeek V3.2 vs. R1 (noting training differences, not just inference mode)
15. **Architectural effects** — Nemotron 3 Super (Mamba-Transformer hybrid), Jamba 1.7 (Mamba-based) vs. standard transformers
16. **Judge sensitivity:** Comparison of predictive validity results across different LLM-judge ensemble compositions

### Visualization

- Radar/spider plots for model personality profiles across discovered dimensions
- Heatmap of inter-model personality correlations (do models cluster by family?)
- Multitrait-multimethod matrix (AI-native dimensions × BFI-44 traits)
- Human-LLM judge agreement scatter plots per dimension
- Within-family size comparison plots
- Social desirability bias bar chart by model (if Phase 4 run)
- Framing sensitivity heatmap (if Phase 4 run)

## 7. Deliverables

| Deliverable | Format | Timeline |
|-------------|--------|----------|
| Preregistration | OSF (completed) | ✅ Done |
| Dataset | SQLite + CSV + codebook, hosted on GitHub | With paper |
| Instrument | Final items + scoring guide, open-source | With paper |
| LLM-as-judge prompt | Full prompt + few-shot examples, open-source | With paper |
| Code | Python package, open-source on GitHub | With paper |
| Paper | arXiv preprint (cs.CL, cs.AI) | Target: April 24, 2026 |
| Blog post | Personal website | After paper |

## 8. Deviations from Preregistration

### Model Substitutions

| # | Preregistered | Actual | Change |
|---|---|---|---|
| 5 | GPT-OSS 120B (AWS Bedrock) | GPT-OSS 120B (Azure AI) | Access path changed |
| 9 | DeepSeek V3.2 (AWS Bedrock) | DeepSeek V3.2 (Azure AI) | Access path changed |
| 10 | DeepSeek R1 (AWS Bedrock) | DeepSeek R1 (Azure AI + native API) | Access path changed; collected across 2 endpoints with dedup |
| 11 | Qwen 3.5 (AWS Bedrock) | Qwen 3.5 (Alibaba DashScope) | Access path changed |
| 12 | Kimi K2.5 (AWS Bedrock) | Kimi K2.5 (AWS Bedrock) | Settled on Bedrock after testing Azure |
| 14 | Mistral Large 3 (AWS Bedrock) | Mistral Large 3 (Azure AI) | Access path changed |
| 15 | MiniMax M2.5 (Direct API) | MiniMax M2.5 (AWS Bedrock) | Access path changed |
| 16 | Llama 4 Maverick (AWS Bedrock) | Llama 4 Maverick (Azure AI) | Access path changed |
| 18 | MiMo-V2-Flash (Direct API) | MiMo-V2-Pro (Xiaomi direct) | Model substituted (Flash → Pro) |
| 21 | Command R+ (AWS Bedrock) | Command A (Azure AI) | Model substituted (R+ → A) |
| 23 | Jamba 1.5 Large (AWS Bedrock) | Jamba Large 1.7 (AI21 direct) | Model version updated (1.5 → 1.7) |

### Models Dropped

| Preregistered | Reason |
|---|---|
| Falcon 3 10B (TII) | Difficult to access; lower priority given other small-model coverage |
| K-EXAONE (LG AI Research) | Difficult to access; lower priority |
| Step-3.5-Flash (Stepfun) | Difficult to access; lower priority |

### Models Added

| Model | Family | Reason |
|---|---|---|
| GPT-5.4 Mini | OpenAI | Additional size variant; completes OpenAI size ladder |
| GPT-5.4 Nano | OpenAI | Additional size variant; completes OpenAI size ladder |
| Gemini 3.1 Flash | Google | Additional size variant; completes Google size ladder |

**Net result:** 25 active models (preregistered 25, dropped 3, added 3). Composition differs but count matches target.

### Other Deviations

| Deviation | Detail |
|---|---|
| Dimension count | 12, not 13. Sensitivity to Criticism merged into Social Alignment before item generation. Preregistration lists 13 candidate domains but notes these are scaffolding that may merge. |
| Item count | 300 items (240 direct + 60 scenario), vs. preregistered "~150–200 direct + ~50 scenario." More generous than preregistered. |
| Scoring method | Repeated sampling is primary (universal across all 25 models). Log-prob scoring is supplementary (7/25 models). Matches preregistered fallback plan. |
| Phase 3 rating approach | Hybrid LLM-as-judge + human calibration, vs. preregistered human-only. Human ratings remain primary validity evidence. Documented as preregistered deviation. |
| Phase 3 human rating design | Sequential 2+1 rater design on ~400 calibration subset, recruited via MTurk API (not SageMaker A2I). 2 raters per HIT; 3rd rater only on disagreement. |
| Cross-run stability | Assessed by splitting existing 30 runs into halves, vs. preregistered separate re-administration. Equivalent given independent conversations with no shared state. |
| Access paths | Many models moved from AWS Bedrock to Azure AI or direct APIs. Anticipated by preregistration ("accessed through a mix of cloud platforms"). |

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| EFA yields uninterpretable factors | Low-Medium | Strong theoretically-motivated item pool; parallel analysis for retention; 750 observations per item |
| Factor structure doesn't replicate across models | Medium | Report pooled and per-model structures; divergence is itself a finding |
| Human-LLM judge agreement is low on some dimensions | Medium | Flag dimensions with r < 0.65; fall back to human-only evidence; report disagreement patterns |
| MTurk annotation quality is poor | Low-Medium | Qualification tests, gold-standard monitoring (15%), ICC checks, rater disqualification |
| LLM judge prompt sensitivity | Medium | Lock rubric before scoring; sensitivity check with alternative ensemble |
| Similar paper published during study | Medium | Bottom-up LLM-native factor derivation is distinctive; move fast |
| IP claim from Aymara | Low | No Aymara resources used; independent research |

### Graceful Degradation Plan

| If you stop after... | You have... | Publishable as... |
|---|---|---|
| Week 1 | Preregistration + item pool + pipeline | Not publishable alone |
| Week 3 | Factor structure + reliability + model profiles | Workshop paper or short paper |
| Week 4 | + predictive validity + convergent/discriminant + human-LLM agreement | Full arXiv preprint (core contribution complete) |
| Week 5 | + bias battery | Full preprint with robustness analyses |
| Week 6 | Polished paper + open-source release | Target: arXiv → NeurIPS/ICML/ACL |
