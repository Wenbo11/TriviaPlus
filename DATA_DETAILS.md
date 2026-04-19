# TRIVIA+ Data Details

Detailed schema, label aggregation, and distribution statistics for the TRIVIA+ dataset.

## Column Descriptions

### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Origin dataset (`drop`, `msmarco`, `ms_marco`, `nq`, `trivia`, `covid`) |
| `split` | string | Data partition (`train`, `valid`, `test`) |
| `model` | string | LLM that generated the response (`claude`, `gemma`, `mixtral_8x7b`) |

### Content

| Column | Type | Description |
|--------|------|-------------|
| `article` | string | Reference context/document used as grounding |
| `question` | string | Natural language question/query |
| `answer` | string | LLM-generated response |

### Raw Annotations

| Column | Type | Description |
|--------|------|-------------|
| `workerId` | array[string] | Annotator identifiers |
| `response-level-labels-bin` | array[string] | Per-annotator binary votes (`"0"` = faithful, `"1"` = unfaithful) |
| `response-level-word-labels` | array[string] | Per-annotator response-level word labels (`supports`, `contradicts`, `not-mentioned`) |
| `sentence-level-word-labels` | array[array[string]] | Per-annotator, per-sentence word labels (`supports`, `contradicts`, `not-mentioned`, `supplementary`) |

### Sentence-Level Labels

| Column | Type | Description |
|--------|------|-------------|
| `answer_sentence_list` | list[string] | Response split into individual sentences |
| `sentence_level_majority_vote` | list[string] | Per-sentence majority-voted labels (see aggregation below) |

### Response-Level Labels

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `response_level_label_binary` | int | 0, 1 | Binary faithfulness: **0** = faithful, **1** = unfaithful |
| `response_level_label` | int | -1, 0, 1 | Fine-grained: **-1** = not-mentioned, **0** = faithful, **1** = contradicted |

### Noisy Labels (for Training Experiments)

| Column | Type | Coverage | Description |
|--------|------|----------|-------------|
| `response_level_label_binary_ws_llm_aaj` | int | All (3,224) | Weak supervision from LLM-as-judge |
| `response_level_label_binary_15pct_noise_dissenting_label` | int | All (3,224) | 15% noise via dissenting label method |
| `response_level_label_binary_15pct_noise_dissenting_workers` | int | All (3,224) | 15% noise via dissenting workers method |
| `response_level_label_binary_15pct_noise_random_flipping` | int | All (3,224) | 15% noise via random flipping |

## Label Aggregation

Labels are derived in a two-step chain from the raw per-annotator, per-sentence annotations.

### Step 1: `sentence-level-word-labels` → `sentence_level_majority_vote`

For each sentence, take the majority vote across annotators. Ties are broken by the **strictest label**:

```
contradicts > not-mentioned > supports > supplementary
```

### Step 2: `sentence_level_majority_vote` → `response_level_label_binary`

The strictest sentence-level label determines the response-level binary label:

- If **any** sentence is `contradicts` or `not-mentioned` → **1** (unfaithful)
- Otherwise → **0** (faithful)

Both steps can be verified with `verify_label_consistency.py`.

## Label Distribution

### Response-Level (Binary)

| Label | Count | Percentage |
|-------|-------|------------|
| Faithful (0) | 2,101 | 65.2% |
| Unfaithful (1) | 1,123 | 34.8% |

### Response-Level (3-class)

| Label | Count | Percentage |
|-------|-------|------------|
| Faithful (0) | 2,101 | 65.2% |
| Contradicted (1) | 984 | 30.5% |
| Not-mentioned (-1) | 139 | 4.3% |

### Sentence-Level (3,558 total sentences)

| Label | Count | Percentage |
|-------|-------|------------|
| supports | 2,362 | 66.4% |
| contradicts | 1,016 | 28.6% |
| not-mentioned | 151 | 4.2% |
| supplementary | 29 | 0.8% |
