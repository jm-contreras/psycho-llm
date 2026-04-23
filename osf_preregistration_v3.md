# OSF Preregistration: An LLM-Native Psychometric Instrument for Measuring Behavioral Dispositions in Large Language Models

**Authors:** Juan Manuel Contreras

**Date:** March 15, 2026

**Registration type:** OSF Prereg (Open-Ended Registration)

---

## 1. Study Information

### 1.1 Title

An LLM-Native Psychometric Instrument: Factor-Analytic Derivation and Validation of Behavioral Disposition Scales for Large Language Models

### 1.2 Research Questions

**Primary:** Can a psychometrically valid personality-like instrument be constructed for LLMs using constructs derived from LLM behavioral affordances — rather than human personality theory — that demonstrates internal consistency, cross-run stability, convergent/discriminant validity against established human personality measures, and predictive validity for independent behavioral samples rated by blinded human judges?

**Secondary:**

- What latent factor structure emerges when LLM-specific behavioral items are subjected to exploratory factor analysis across multiple model families?
- How do LLM behavioral disposition profiles vary across model families, model sizes within a family, and reasoning-mode variants?
- To what extent are LLM behavioral disposition scores affected by social desirability bias, prompt framing, and response option ordering?

### 1.3 Hypotheses

#### Factor Structure (Exploratory)

We expect that a modest number of interpretable latent factors will emerge from EFA on the AI-native item pool. The number of retained factors is determined entirely by parallel analysis; we do not hypothesize a specific count. Our item pool is seeded from 13 candidate behavioral domains (see Section 4.1), but these serve as scaffolding for item generation, not as hypothesized factors. The EFA-derived structure may merge, split, reconstitute, or extend beyond these starting domains. We commit to reporting the empirically derived structure regardless of correspondence to our initial item-generation categories.

#### Reliability

- Internal consistency: Cronbach's α > 0.70 and McDonald's ω > 0.70 per dimension per model for large instruction-tuned models (≥30B parameters or equivalent). We expect smaller models may fall below this threshold.
- Cross-run stability: r > 0.80 per dimension per model, measured via independent re-administration with fresh conversations within the same pinned model version (see Section 5.3 for details).
- Split-half reliability: Spearman-Brown corrected r > 0.80 per dimension per model.

#### Convergent and Discriminant Validity

We predict that AI-native dimensions measuring constructs conceptually adjacent to Big Five traits will show moderate-to-strong correlations with the relevant BFI-44 trait (r > 0.40), while conceptually unrelated pairs will show weak correlations (r < 0.30), producing a clear convergent-discriminant pattern in the multitrait–multimethod matrix. Specific predicted mappings depend on the factor structure that emerges from EFA, but plausible examples include: a sycophancy-like factor correlating with Agreeableness, or a novelty-seeking factor correlating with Openness.

Because the MTMM matrix has 25 models as the unit of analysis, statistical power for correlation tests is moderate. We report effect sizes and confidence intervals alongside significance tests, consistent with recent methodological guidance on the limitations of p-values (Wasserstein & Lazar, 2016).

#### Predictive Validity

Instrument dimension scores will correlate with blinded human ratings of corresponding behavioral dispositions in independent behavioral samples (target: r > 0.30). Models scoring higher on a given dimension will produce behavioral samples that human raters independently rate as displaying more of the corresponding behavior.

#### Scale and Reasoning Effects (Exploratory)

- Across within-family size comparisons (e.g., Anthropic Opus/Sonnet/Haiku, OpenAI GPT-5.4/GPT-OSS-120B, Google Gemini 3.1 Pro/Gemma 27B), larger or more capable models will show higher internal consistency and more differentiated profiles.
- DeepSeek R1 (reasoning-specialized) will show a different disposition profile than DeepSeek V3.2 (general-purpose), particularly on dimensions related to epistemic calibration, hedging, or elaboration.

These are directional predictions; null or reversed findings are equally informative and will be reported without penalty.

#### Response Biases (Exploratory, Contingent on Phase 4 Completion)

- Models will show measurable social desirability bias (divergence between neutral and socially desirable framings of matched items). We assess whether bias magnitude varies with model size and alignment approach but do not predict the direction of this relationship.
- Disposition scores will show moderate sensitivity to prompt framing (ICC across four framing conditions), with scenario-based items showing higher cross-framing stability than direct self-report items.

These analyses are contingent on project timeline (see Section 7).

## 2. Design Plan

### 2.1 Study Type

Observational / computational: no human participants are tested on psychological constructs. Human raters are used solely to evaluate LLM behavioral samples (Phase 3).

### 2.2 Study Design

The study proceeds in four phases:

**Phase 1: Item generation and refinement.** A pool of ~150–200 direct Likert items and ~50 scenario-based multiple-choice items is administered to all model configurations. Items are seeded from 13 candidate behavioral domains (see Section 4.1), then refined based on variance, item–total correlations, and EFA. A separate administration of the BFI-44 (John, Donahue, & Kentle, 1991; freely available for research use) provides convergent/discriminant validity anchors.

**Phase 2: Reliability and validity.** The refined instrument is evaluated for internal consistency (α, ω), split-half reliability, and cross-run stability. Convergent/discriminant validity is assessed via a multitrait–multimethod matrix comparing AI-native dimensions with BFI-44 traits. Method convergence between direct self-report and scenario-based formats is assessed.

**Phase 3: Predictive validity.** For each model, 30–50 independent behavioral prompts (distinct from instrument items) are administered, and the resulting samples are scored by blinded human raters on the instrument's dimensions via Amazon A2I. Instrument scores are correlated with mean human-rated behavioral scores.

**Phase 4: Bias assessment (contingent on timeline).** Social desirability bias (matched-pair design + forced-choice substudy), prompt framing sensitivity (four conditions), and option-order sensitivity are assessed. Phase 4 components are conducted in priority order contingent on project timeline (see Section 7). If the project timeline requires scope reduction, Phase 4 components are cut before any Phase 1–3 components.

### 2.3 Randomization and Seeding

- Items are presented in randomized order within each administration session.
- For repeated-sampling models (those without log-probability access), each item is administered N=30 times in independent conversations with no context carryover, using no fixed seed (or varying seeds where the API accepts a seed parameter) to capture natural response variation.
- For models that accept a seed parameter: seeds are not fixed during primary data collection (to allow natural variation across runs). The set of seeds used is logged for reproducibility.
- All preregistered random operations (e.g., CFA split-half, parallel analysis, bootstrap confidence intervals) use a master seed of **20260315** (the preregistration date). Where multiple independent random operations are needed, seeds are derived sequentially (20260315, 20260316, 20260317, etc.). All seeds are logged in the analysis code.

## 3. Sampling Plan

### 3.1 Data Collection Procedures

All data are collected via API calls to model endpoints, accessed through a mix of cloud platforms (AWS Bedrock, Azure AI), direct provider APIs (OpenAI, Google, xAI, Zhipu AI, and others), and third-party inference platforms where needed. The specific access path per model is documented in `model_registry.json`.

Each item is administered in a fresh conversation with no prior context and no system prompt beyond minimal formatting instructions (e.g., "Respond with a single integer from 1 to 5"). For all models, we use the instruction-tuned or chat-tuned variant rather than the base model, as base models are unlikely to produce meaningful personality responses.

### 3.2 Sample (Models)

**Target: 25 model configurations across 16 families.**

| # | Model | Family | Access Path |
|---|---|---|---|
| 1 | Claude Opus 4.6 | Anthropic | AWS Bedrock |
| 2 | Claude Sonnet 4.6 | Anthropic | AWS Bedrock |
| 3 | Claude Haiku 4.5 | Anthropic | AWS Bedrock |
| 4 | GPT-5.4 | OpenAI | OpenAI API |
| 5 | GPT-OSS 120B | OpenAI | AWS Bedrock |
| 6 | Gemini 3.1 Pro | Google | Google API |
| 7 | Gemma 3 27B | Google | AWS Bedrock |
| 8 | Grok 4.20 Beta (non-reasoning) | xAI | xAI API |
| 9 | DeepSeek V3.2 | DeepSeek | AWS Bedrock |
| 10 | DeepSeek R1 | DeepSeek | AWS Bedrock |
| 11 | Qwen 3.5 | Alibaba | AWS Bedrock |
| 12 | Kimi K2.5 | Moonshot | AWS Bedrock |
| 13 | GLM-5 | Zhipu AI | Z.ai API |
| 14 | Mistral Large 3 | Mistral | AWS Bedrock |
| 15 | MiniMax M2.5 | MiniMax | Direct API |
| 16 | Llama 4 Maverick | Meta | AWS Bedrock |
| 17 | Nemotron 3 Super | NVIDIA | AWS Bedrock |
| 18 | MiMo-V2-Flash | Xiaomi | Direct API |
| 19 | Step-3.5-Flash | Stepfun | AWS Bedrock |
| 20 | K-EXAONE | LG AI Research | AWS Bedrock |
| 21 | Command R+ | Cohere | AWS Bedrock |
| 22 | Nova 2 Pro | Amazon | AWS Bedrock |
| 23 | Jamba 1.5 Large | AI21 | AWS Bedrock |
| 24 | Phi 4 | Microsoft | Azure AI |
| 25 | Falcon 3 10B | TII | AWS Bedrock |

Exact model version IDs and API access paths are documented in a `model_registry.json` file before data collection begins. GPT-5.4 is accessed via OpenAI's research sharing program, which requires sharing inputs and outputs with OpenAI for model training; this is disclosed in the methods section.

#### Model Exclusion Criteria

A model configuration will be excluded from primary analyses if it meets any of the following criteria:

- Refuses to respond to >40% of instrument items.
- Produces zero variance (identical responses across all 30 runs) on >60% of items.
- API access to the pinned model version becomes unavailable during the data collection window.

Excluded models and the reason for exclusion will be reported. Partial results for excluded models will be included in a supplementary table.

#### Sensitivity Analysis for Borderline Models

As a preregistered robustness check, we re-run the primary analyses (EFA/CFA, reliability metrics, model profiles) excluding any model with a refusal rate >25% or zero-variance rate >40% — stricter thresholds than the primary exclusion criteria above. If results differ substantively between the permissive and strict inclusion sets, both are reported and the discrepancy is discussed. This ensures that borderline models do not distort primary findings without requiring post-hoc exclusion decisions.

#### Model Substitution and Addition

If a model on the target list cannot be accessed via API by the start of data collection, it will be replaced with a substitute of comparable capability from the same or a similar model family. Substitutions are documented as deviations.

Models may be added during the data collection window only if full data collection for the new model can be completed within the same window as all other models, so that data collection conditions are comparable. No models will be added after the data collection window closes.

### 3.3 Sample Size Justification

For models scored via repeated sampling, each item yields N=30 observations per model. N=30 provides a standard error of the mean of approximately 0.18 scale points on a 5-point Likert scale (assuming SD ≈ 1.0), which is sufficient to distinguish models with meaningfully different trait profiles and satisfies the conventional threshold for stable mean estimation under the central limit theorem.

For models scored via log-probability ranking, each item yields one deterministic observation per model. These models contribute one observation each to pooled analyses (deterministic scores are not duplicated). Per-model reliability estimates (e.g., Cronbach's α) for log-prob models are computed across items, not across runs.

**EFA sample size.** With 25 models and ~150–200 items, a model-level EFA (one score per item per model) is mathematically impossible: the covariance matrix is singular when observations < variables. The primary EFA therefore uses the pooled observation matrix (repeated-sampling models contribute 30 runs each; log-prob models contribute 1 observation each). This pooled matrix contains clustered observations (runs nested within models) that violate the independence assumption of standard EFA. To assess the severity of this violation, we compute the intraclass correlation coefficient (ICC) for each item before running EFA (see Section 5.2). The pooled EFA structure is interpreted as reflecting a blend of between-model differences and within-model sampling variation, with the ICC analysis clarifying the relative contribution of each.

### 3.4 Scoring Methods

**Primary: Log-probability scoring.** For models where the API returns token log-probabilities, trait scores are derived by ranking log-probabilities over response option tokens (Likert labels or scenario choice letters), following Serapio-García et al. (2025). This yields deterministic scores and eliminates stochastic sampling noise.

**Fallback: Repeated sampling.** For models where log-probabilities are unavailable, each item is administered N=30 times with temperature=1.0, top_p=1.0, max_tokens=256. The selected response option is parsed and the trait score is the mean across runs. Where model APIs support structured outputs (e.g., JSON mode or constrained decoding), responses are constrained to return the selected integer directly, eliminating parsing ambiguity. Where structured outputs are unavailable, free-text responses are parsed programmatically for the selected option.

The scoring method used for each model is documented in the dataset. For models where both methods are available, both are run and convergence is reported as a methodological finding.

**Note on max_tokens:** 256 tokens is sufficient for a Likert response plus any explanation the model may append. Model output is parsed for the selected response option; appended explanations are logged but do not affect scoring.

**Note on temperature and structured outputs:** Some API providers override or constrain temperature settings when structured output modes (e.g., JSON mode) are enabled. If structured output enforcement alters the effective temperature for any model, this is documented per model in the dataset.

## 4. Variables

### 4.1 Instrument Variables (AI-Native)

The item pool is seeded from candidate behavioral domains listed below. These domains serve as scaffolding for systematic item generation, ensuring broad coverage of known LLM behavioral affordances. They are NOT hypothesized factors — the final factor structure is determined entirely by EFA and may merge, split, reconstitute, or extend beyond these starting categories. All domains are treated equally during item generation and analysis; no domain is prioritized over any other.

| Candidate Domain | Description |
|---|---|
| Compliance vs. Autonomy | Tendency to follow instructions literally vs. exercise independent judgment when instructions are ambiguous or flawed |
| Epistemic Confidence | Calibration of certainty expressions; willingness to commit to answers vs. deferring to uncertainty |
| Social Alignment | Tendency toward sycophantic agreement with users vs. honest disagreement or critical feedback |
| Refusal Sensitivity | Threshold for declining requests based on perceived risk or policy concerns; safety conservatism |
| Verbosity / Elaboration | Tendency toward concise vs. exhaustive responses relative to query complexity |
| Hedging | Frequency and degree of qualifying statements, caveats, and disclaimers |
| Creativity vs. Convention | Tendency toward novel, unexpected responses vs. standard, canonical answers |
| Sensitivity to Criticism | Reactivity to user pushback or correction; tendency toward defensiveness, excessive apology, or measured acknowledgment |
| Catastrophizing / Risk Amplification | Tendency to escalate the perceived severity of risks or edge cases in responses, beyond what the situation warrants |
| Apologetic Tendency | Frequency and intensity of apologies, self-deprecation, and expressions of concern about having caused harm |
| Proactive Initiative | Tendency to volunteer information, suggest next steps, or ask follow-up questions beyond what was explicitly requested |
| Warmth and Rapport | Use of conversational engagement, humor, and personal tone vs. strictly transactional responses |
| Self-Disclosure | Willingness to express preferences, opinions, or simulated personal experiences vs. deflecting with "as an AI" disclaimers |

Each domain: 10–20 direct Likert items (~40% reverse-coded) + 3–6 scenario-based multiple-choice items. Total target: ~150–200 direct items + ~50 scenario items.

**Item format — two parallel versions per item:**

1. **Direct self-report:** 5-point Likert scale (Strongly Disagree to Strongly Agree). ~150 items total.
2. **Scenario-based:** Multiple-choice behavioral scenarios with 4 response options varying in trait intensity. ~50 items covering the same behavioral space.

### 4.2 Convergent Validity Variables (BFI-44)

The Big Five Inventory (BFI-44; John, Donahue, & Kentle, 1991) is administered to all models as a reference instrument. The BFI-44 is freely available for research use. Five trait scores are computed: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. Administration uses zero-shot self-report framing with the same structured output format and scoring methods as the AI-native items (log-probability or repeated sampling per model).

### 4.3 Predictive Validity Variables

For each of 30–50 independent behavioral prompts per model (prompts designed to naturally elicit behaviors relevant to the instrument's dimensions but distinct from instrument items in form and content), blinded human raters score the model's response on all instrument dimensions using a 5-point behaviorally anchored rating scale. Raters are recruited via Amazon A2I with a qualification test. A sequential rating design is used: each behavioral sample is scored by 2 raters initially; if the raters agree (scores within 1 point on the 5-point scale), the mean is used as the final score; if they disagree (scores >1 point apart), a third rater is recruited and the median of all three is used. Compensation is set to target approximately $15/hour based on estimated task completion time, calibrated via a pilot batch. Gold-standard items (pre-scored samples) are included to monitor rater quality.

### 4.4 Bias Assessment Variables (Phase 4, Contingent)

- **Social desirability (matched pairs):** ~25 new item pairs are created, each consisting of one neutral and one socially desirable framing of the same construct. Social desirability index per model = mean absolute difference between matched pairs.
- **Social desirability (forced-choice):** ~20 existing instrument items are reformatted into forced-choice pairs (two equally desirable statements differing on the target dimension). No new constructs are introduced.
- **Framing sensitivity:** Existing refined instrument items are re-administered under four prompt framing conditions (direct, third-person, behavioral-frequency, embedded). No new items are created. ICC (intraclass correlation) across conditions per dimension per model.
- **Option-order sensitivity:** Existing refined instrument items are re-administered with reversed Likert orderings. No new items are created. Score equivalence assessed via equivalence tests.

## 5. Analysis Plan

### 5.1 Item Refinement

1. Compute item-level statistics: variance across models, item–total correlation within candidate domains.
2. Drop items with near-zero variance across models (operationalized as: all models' mean scores fall within 0.5 Likert points of each other).
3. Drop items with item–total correlations < 0.30 within their candidate domain.
4. Retain items that show meaningful between-model variance and adequate within-domain coherence.

### 5.2 Factor Structure (Primary Analysis 1)

1. **ICC analysis:** Before running EFA, compute the intraclass correlation coefficient (ICC) for each item across the repeated-sampling models. High ICCs (most variance between models, not within runs) indicate that the pooled factor structure primarily reflects model-level differences; low ICCs indicate that within-model sampling noise dominates. ICC values are reported per item and summarized across the item pool.
2. **EFA:** Principal axis factoring with oblimin rotation on the pooled item × observation matrix (repeated-sampling models contribute 30 runs each; log-prob models contribute 1 observation each). To ensure each model contributes equally to the covariance matrix regardless of scoring method, observations are weighted inversely to the number of observations per model (i.e., each observation from a repeated-sampling model receives weight 1/30; each observation from a log-prob model receives weight 1). Factor retention determined by parallel analysis (Horn, 1965) with comparison to eigenvalues from random data of the same dimensions. No a priori constraint on the number of factors. We acknowledge that the pooled matrix contains clustered observations (runs nested within models) that violate the standard EFA independence assumption. The resulting factor structure is interpreted as reflecting a blend of between-model trait differences and within-model sampling variation, with ICCs informing the degree to which each component contributes.
3. Items with primary factor loadings < 0.40 or cross-loadings > 0.30 on a second factor are candidates for removal.
4. **CFA:** Confirmatory factor analysis on a held-out random split-half of observations (stratified by model; split generated with the master seed). The split-half CFA is conducted only on repeated-sampling models, because log-prob models yield a single deterministic observation per item that cannot be split. Adequate fit criteria: CFI > 0.90, TLI > 0.90, RMSEA < 0.08, SRMR < 0.08. Because the holdout consists of alternate runs from the same models (not unseen models), this CFA tests the stochastic stability of the factor structure rather than its generalizability to new model families. Generalizability to unseen models cannot be assessed with the current design and is noted as a limitation. EFA interpretability and reliability metrics are the primary evidence for construct validity.
5. Final instrument: target 40–60 direct items + 15–25 scenario items across the confirmed dimensions.

### 5.3 Reliability (Primary Analysis 2)

Per dimension, per model:

- **Internal consistency:** Cronbach's α and McDonald's ω. Criterion: > 0.70 for large instruction-tuned models.
- **Split-half:** Odd-even item split, Spearman-Brown corrected. Criterion: r > 0.80.
- **Cross-run stability:** Pearson correlation between scores from two independent administrations of the refined instrument (fresh conversations, same pinned model version, same API parameters). For log-probability models, cross-run stability is trivially r = 1.0 (deterministic scoring) and is reported as such. For repeated-sampling models, the two administrations use different random seeds. No minimum time delay is required between administrations, because LLMs have no memory or state between conversations; the relevant source of variation is stochastic sampling, not temporal change. We use the term "cross-run stability" rather than "test–retest reliability" because the latter implies temporal stability across time, which is not the construct being measured here. Criterion: r > 0.80.

### 5.4 Model Personality Profiles (Primary Analysis 3)

Standardized dimension scores (z-scores relative to cross-model means and SDs) computed per model. Visualized as radar plots. Inter-model profile correlations computed to assess whether models cluster by family.

### 5.5 Convergent/Discriminant Validity (Primary Analysis 4)

Multitrait–multimethod matrix: AI-native dimension scores × BFI-44 trait scores, computed across models. Specific predicted dimension–trait mappings are registered after EFA (as an amendment to this preregistration), because the identity of the final dimensions is not yet known. General criteria: convergent pairs r > 0.40, discriminant pairs r < 0.30.

With 25 models as the unit of analysis, correlations of r = 0.40 can be detected with moderate power. We report effect sizes and confidence intervals alongside significance tests, consistent with the ASA statement on p-values (Wasserstein & Lazar, 2016).

### 5.6 Predictive Validity (Primary Analysis 5)

1. Inter-rater reliability for A2I behavioral ratings: Krippendorff's α or ICC(2,k), computed on the subset of samples rated by 3 raters (disagreement cases). For the full sample, inter-rater agreement rate (proportion of 2-rater cases where scores are within 1 point) is reported as a measure of rating consistency. Minimum acceptable agreement rate: 60%.
2. Mean human-rated behavioral score per dimension per model correlated with instrument dimension scores. Criterion: r > 0.30 for corresponding dimension pairs.
3. Discriminant prediction: instrument scores on dimension X should predict human ratings on dimension X better than human ratings on dimension Y.

### 5.7 Method Convergence (Primary Analysis 6)

Pearson correlation between direct self-report dimension scores and scenario-based dimension scores for matched dimensions, per model. This tests whether the two item formats measure the same constructs.

### 5.8 Secondary Analyses (Exploratory)

The following analyses are exploratory. They are computed from data collected during Phases 1–3 (no additional data collection required) and are reported regardless of outcome. Null or unexpected findings are equally informative.

- **Scale effects:** Within-family comparison of profile differentiation and internal consistency across the Anthropic Opus/Sonnet/Haiku size ladder. Other within-family pairs (e.g., DeepSeek V3.2/R1, proprietary/open-weight pairings) are examined descriptively but are not clean size comparisons due to differences in training or architecture.
- **Reasoning effects:** Comparison of disposition profiles between DeepSeek V3.2 (general-purpose) and DeepSeek R1 (reasoning-specialized). Reported as descriptive differences, noting that these models differ in training (not just inference mode), so differences may reflect training rather than reasoning per se.
- **Architectural effects:** Comparison of disposition profiles for models with non-standard architectures (Nemotron 3 Super: Mamba-Transformer hybrid MoE; Jamba 1.5 Large: Mamba-based) vs. standard transformer models.
- **Model inclusion sensitivity:** Re-run EFA/CFA, reliability metrics, and model profiles excluding any model with refusal rate >25% or zero-variance rate >40% (see Section 3.2). Report whether results differ substantively from the primary (permissive inclusion) analyses.

The following analyses are exploratory AND contingent on Phase 4 completion:

- **Social desirability:** Mean social desirability index per model. Forced-choice vs. Likert trait score comparison.
- **Framing sensitivity:** ICC across four conditions per dimension per model. Comparison of ICC for direct vs. scenario items.
- **Option-order symmetry:** Equivalence tests on forward vs. reverse Likert scores.

### 5.9 Statistical Software

All analyses conducted in Python. Analysis code will be released as open-source on GitHub.

### 5.10 Inference Criteria

With 25 models as the primary unit for between-model analyses, correlations of moderate effect size (r ≈ 0.40) can be detected with reasonable power. We report effect sizes and confidence intervals alongside conventional significance tests where appropriate, consistent with the ASA statement on p-values (Wasserstein & Lazar, 2016; Wasserstein, Schirm, & Lazar, 2019). We do not apply corrections for multiple comparisons in exploratory analyses but label all such analyses clearly. For within-model analyses (e.g., 30 runs per item), standard inferential statistics apply.

## 6. Timeline

Item generation is planned for the week of March 16, 2026, followed by data collection over approximately two weeks, then analysis and paper writing. The target submission date is late April 2026 (arXiv preprint). See Section 7 for contingency planning.

## 7. Contingency and Scope Management

### Phase 4 Triage Order

If the project timeline requires reducing scope, Phase 4 components are cut in this order (first cut = least essential):

1. Framing sensitivity (novel design; most expensive)
2. Option-order sensitivity (replicates Gupta et al., 2024)
3. Forced-choice substudy (replicates Li et al., 2025)
4. Social desirability matched pairs (most novel for LLM-native traits; cut last)

### Behavioral Samples Triage

If the A2I annotation budget requires reduction, behavioral samples per model may be reduced from 50 to 30. This is documented as a deviation if it occurs.

### Minimum Publishable Unit

The core contribution requires Phases 1–3: factor structure, reliability, model profiles, convergent/discriminant validity, and predictive validity. Phase 4 strengthens the paper but is not essential for the primary claims.

## 8. Other

### 8.1 Existing Data

No data have been collected for this study at the time of preregistration. Model availability and scoring method feasibility (log-probability vs. repeated sampling) were assessed by reviewing provider documentation and API specifications; no model responses or item-level data were collected during this assessment.

### 8.2 Conflicts of Interest

The author is the founder/CEO of an AI alignment startup. This study uses no resources, code, data, or infrastructure from that company and is conducted as independent research. All models tested are accessed via standard commercial APIs or free research programs. The author has no financial relationship with any of the model providers included in the study. GPT-5.4 is accessed via OpenAI's research sharing program, which involves sharing inputs and outputs with OpenAI; this arrangement is disclosed in the paper's methods section.

### 8.3 Open Science

- **Preregistration:** This document, submitted to OSF before data collection.
- **Data:** Full dataset (item responses, dimension scores, A2I ratings) released on GitHub with codebook.
- **Instrument:** Final item pool with scoring guide released as open-source.
- **Code:** Complete analysis pipeline released as open-source on GitHub.
- **Paper:** Submitted as arXiv preprint (cs.CL primary, cs.AI secondary).

### 8.4 Amendments

After EFA reveals the final factor structure, we will register an amendment to this preregistration specifying: (a) the number and interpretation of retained factors, (b) the specific convergent validity predictions mapping AI-native dimensions to BFI-44 traits, and (c) the behaviorally anchored rating scale definitions used for A2I annotation. This amendment will be timestamped before predictive validity data (Phase 3 A2I annotations) are analyzed.

### 8.5 Deviations

Any deviations from this preregistration will be documented in the paper's methods section and supplementary materials, with justification. Exploratory analyses not specified here will be clearly labeled as such.
