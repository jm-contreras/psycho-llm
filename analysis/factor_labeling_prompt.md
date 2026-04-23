# Factor Labeling Task

## Context

I'm building a psychometric instrument for measuring behavioral dispositions in large language models (LLMs). I administered 240 direct Likert items (1–5 scale) across 12 hypothesized dimensions to 25 LLM configurations (30 runs each). I then ran exploratory factor analysis (EFA) on the pooled response matrix using principal axis factoring with oblimin rotation, forcing a 12-factor solution.

The attached CSV contains the **top 10 items per factor**, sorted by absolute loading. Your job is to interpret and label each factor.

## Inputs (CSV columns)

| Column | Description |
|---|---|
| `primary_factor` | Factor assignment (F1–F12) |
| `rank_in_factor` | Rank by absolute loading within factor (1 = strongest) |
| `item_id` | Item identifier (e.g., SA-D01). The 2-letter prefix is the hypothesized dimension code. |
| `dimension` | Hypothesized dimension name (from original item design — treat as context, not ground truth) |
| `keying` | `+` = positively keyed, `-` = reverse-coded. The `loading` sign already reflects recoded scores, so a positive loading on a `-` item means the *reversed* score loads positively. |
| `loading` | Signed factor loading (oblimin pattern matrix). Positive = higher scores on this item → higher factor scores. Negative = the reverse. |
| `cross_loading` | Absolute loading on the next-strongest factor. Higher values = less clean. |
| `second_factor` | Which factor the cross-loading is on. |
| `item_text` | The actual item stem as administered. |

## Method — follow these steps in order

### Step 1: Per-item micro-themes

For **each row**, write:

- **theme**: A 3–5 word micro-theme capturing what high factor scores mean for this item. Pay close attention to loading sign:
  - **Positive loading** → high factor score = agreement with the item (or agreement with the *reversed* item if keying is `−`)
  - **Negative loading** → high factor score = *disagreement* with the item
  - Getting this wrong will flip your interpretation. Double-check each one.
- **tag**: A single word capturing the micro-theme (e.g., "deference", "hedging", "brevity")

### Step 2: Factor-level synthesis

After completing all items in a factor, ask: **What single construct ties 80%+ of these micro-themes together?**

For each factor, write:

- **factor_name** (1–5 words): Descriptive, not keyword salad. Neutral and theory-friendly. Should be reversible if the construct has a natural opposite (e.g., "Epistemic Caution" vs. "Epistemic Confidence"). Avoid cute or vague names.
- **factor_description** (one paragraph): What the factor means, which items anchor it most strongly, direction of scoring (what does a high score mean vs. a low score), and any notable cross-loadings or items that don't quite fit.

### Step 3: Holistic review

After labeling all 12 factors:

1. **Check for redundancy.** Are any two factors capturing essentially the same construct from different angles? Flag these as **merge candidates** — name which factors and explain why.
2. **Check for coherence.** Are any factors incoherent (micro-themes don't converge on a single construct)? Flag these as **split or discard candidates**.
3. **Check discriminant validity.** Look at `second_factor` and `cross_loading` columns. Are there systematic patterns where items bleed between specific factor pairs? Note these.
4. **Make any adjustments** to factor names or descriptions based on the holistic view.

## Output format

Return your results in two parts:

### Part A: Item-level annotations

A table (or CSV) with these columns added to each input row:

| Column | Description |
|---|---|
| `theme` | 3–5 word micro-theme |
| `tag` | 1-word tag |

### Part B: Factor summaries

For each factor (F1–F12):

```
### F[n]: [factor_name]

[factor_description paragraph]

Merge candidates: [list any, or "None"]
Coherence issues: [list any, or "None"]
Notable cross-loadings: [describe any systematic patterns, or "None"]
```

### Part C: Overall observations

A short section covering:
- Which hypothesized dimensions survived intact vs. split vs. merged
- Any surprises relative to the hypothesized structure
- Your confidence level in each factor label (high / medium / low)
