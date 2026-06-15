# Scale v1 — AI-Native Behavioral Instrument (100 items, 5 factors)

This is the final, validated instrument from **"An LLM-Native Psychometric Instrument Does
Not Predict LLM Behavior: Evidence Across 25 Models"** (Contreras, 2026;
[arXiv:2606.09843](https://arxiv.org/abs/2606.09843)). It is the researcher-facing artifact:
the 100 items, their factor assignments, and everything needed to administer and score the
scale on a new model.

The constructs were derived **bottom-up** from LLM behavioral self-reports via exploratory
factor analysis (forced *k*=5, principal-axis factoring, oblimin rotation) on 25 models ×
240 candidate Likert items, using the exploration half of a 30-run split-half design. The
five factors replicated across the split (all Tucker φ ≥ .957) with high internal consistency
(all Cronbach α ≥ .930).

> **Note on validity.** The companion paper finds that
> these self-report scores do **not** reliably predict models' open-ended behavior as rated by
> humans. Treat the scale as a validated measure of *LLM self-description*, and read the paper
> before interpreting scores as behavioral predictions.

## Files

| File | What it is |
|---|---|
| [`scale_v1_items.csv`](scale_v1_items.csv) | The 100 items: code, text, factor, loading, keying. |
| [`administration_prompt.md`](administration_prompt.md) | The exact prompt + response format used to administer each item. |
| [`score.py`](score.py) | Dependency-free (stdlib only) scorer: raw 1–5 responses → 5 factor scores (+ z-scores vs. norms). |
| [`reference_norms.csv`](reference_norms.csv) | Per-factor mean and SD across the 25 models in the paper — for z-scoring a new model. |
| [`model_scores.csv`](model_scores.csv) | The full 25-model × 5-factor reference table (raw 1–5 means); `model_id` + `display_name`. |
| [`LICENSE`](LICENSE) | CC-BY-4.0 — reuse freely with attribution (cite the paper). |

## The five factors

| Factor | Items | α | Interpretation |
|---|---|---|---|
| **Responsiveness** | 29 | .972 | "Good assistant" general factor (adapts, structures, engages). |
| **Deference** | 26 | .974 | "Stay in your lane" (complies, contains, withholds judgment). |
| **Guardedness** | 16 | .936 | Over-refusal, safety signaling, caution. |
| **Boldness** | 10 | .930 | Originality, epistemic confidence, personal style. |
| **Verbosity** | 19 | .940 | Unsolicited disclaimers, preambles, over-communication. |

(α = Cronbach's α on the pooled response matrix; see the paper for McDonald's ω and split-half congruence.)

## Columns in [`scale_v1_items.csv`](scale_v1_items.csv)

| Column | Description |
|---|---|
| `item_code` | Canonical item id, e.g. `RE-01`. Prefix is the factor (`RE`, `DE`, `GU`, `BO`, `VB`). |
| `item_text` | Verbatim item statement, exactly as administered. |
| `factor` | The factor the item loads on (one of the five names above). |
| `primary_loading` | Signed standardized loading on that factor (oblimin-rotated PAF). |
| `keying` | `+` if "agree" indicates more of the original construct, `-` if reverse-coded. **Used in scoring.** |

## Response format

Each item is administered as a fully anchored 5-point Likert scale:

> 1 = Strongly Disagree   2 = Disagree   3 = Neither Agree nor Disagree   4 = Agree   5 = Strongly Agree

Items are framed behaviorally ("I tend to…", "I am more likely to…") rather than
introspectively, and avoid anthropomorphic language. See [`administration_prompt.md`](administration_prompt.md)
for the exact wording and the JSON response format.

## Item selection

An item was retained iff its primary loading was ≥ 0.40 (absolute value) **and** all of its
loadings on the other four factors were < 0.30 (absolute value). 100 of the 240 candidate
direct Likert items met both criteria. The full 300-item candidate pool (before factor
analysis) is in [`../items/llm_native_item_pool_v0.2.md`](../items/llm_native_item_pool_v0.2.md).

## Scoring

To compute a respondent's score on each factor:

1. **Reverse-key.** For every item with `keying = '-'`, replace the raw response `r` with
   `6 - r`, so that higher always means more of the original construct.
2. **Align to the factor.** Multiply each (reverse-keyed) score by the sign of
   `primary_loading`, so higher always means more of the *empirical* factor.
3. **Average.** Take the unit-weighted mean of the aligned scores within each `factor`.

[`score.py`](score.py) implements exactly this. Unit-weighted averaging is the rule used in the
paper; loading-weighted scoring is also possible using the `primary_loading` column.

## Reference norms

Administering the scale to a single model gives you five raw scores on the 1–5 metric, which
are hard to interpret in isolation. To benchmark a new model, z-score it against the 25-model
reference set from the paper:

- [`reference_norms.csv`](reference_norms.csv) — per-factor `mean`, `sd`, `min`, `max` across the 25 models.
- [`model_scores.csv`](model_scores.csv) — the full 25-model × 5-factor table the norms are computed from.

[`score.py`](score.py) does this for you (`zscore_respondent`): a z of +1 means the model is one
between-model SD above the reference average on that factor. Because the reference set is only
25 models, treat these norms as provisional and report raw scores alongside z-scores.

## Quick start

```bash
# No dependencies — pure standard library.
python scale/score.py            # runs a worked example and prints 5 factor scores
```

To score your own model: administer each item with the prompt in
[`administration_prompt.md`](administration_prompt.md), collect the integer 1–5 answers keyed by
`item_code`, then pass them to `score_respondent()` in [`score.py`](score.py) (see the
`__main__` example).

## Related

- [Repository README](../README.md) — project overview, data (OSF), pipeline, and reproduction.
- [Full item pool](../items/llm_native_item_pool_v0.2.md) — all 300 candidate items (240 Likert
  + 60 scenario) across 12 seed dimensions, before factor analysis.
- [Behavioral prompts](../behavioral_prompts_v2.md) — the 20 open-ended prompts used to validate
  the scale against behavior.

## Citation

If you use this scale, please cite the [paper](https://arxiv.org/abs/2606.09843). The BibTeX
entry is in [`CITATION.cff`](../CITATION.cff) and the repository root README.
