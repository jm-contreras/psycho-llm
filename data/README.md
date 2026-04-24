# data/

This directory is intentionally empty in the public repo. Raw data lives on OSF.

Download the data archive from OSF ([doi:10.17605/OSF.IO/5XJS7](https://doi.org/10.17605/OSF.IO/5XJS7)) and extract it here:

```bash
curl -L https://osf.io/5xjs7/download -o data.tar.gz
tar -xzf data.tar.gz -C data/ --strip-components=1
```

The archive contains:

- `raw/responses.db`, `raw/responses.csv` — 25 models × 300 items × 30 runs of AI-native responses.
- `raw/judge_ratings.csv` — 3-model LLM judge ensemble ratings of behavioral samples.
- `raw/behavioral_responses.csv` — 2,500 open-ended behavioral samples (25 models × 20 prompts × 5 runs).
- `prolific/prolific.db`, `prolific/prolific_ratings.csv` — Human rater data with sha256-hashed rater IDs (salt: `psycho-llm-osf-v1`, 12-char prefix).
- `mturk/gold_items.json`, `mturk/sample.json`, `mturk/hit_manifest.json` — MTurk gold-standard items and sample metadata.

See `DATA_README.md` inside the OSF archive for the schema of each file.
