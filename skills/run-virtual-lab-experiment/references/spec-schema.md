# Experiment specification schema

## Contents

1. Required fields
2. Feature definitions
3. Target definitions
4. Constraints
5. Validation and search settings
6. Virtual Lab settings
7. Complete example

## Required fields

The runner accepts one JSON file with these top-level objects:

- `experiment_name`: short filesystem-safe name.
- `description`: scientific purpose and relevant domain context.
- `dataset`: input file information.
- `features`: numeric model inputs and search bounds.
- `targets`: numeric outputs, goals, expectations, and weights.
- `constraints`: optional feature limits.
- `validation`: holdout and cross-validation settings.
- `search`: candidate search and decision settings.
- `virtual_lab`: live/offline agent settings.
- `output`: artifact and optional Obsidian locations.

Supported dataset types are `.csv` and `.tsv`. All selected feature and target columns must be numeric after parsing.

## Feature definitions

Each feature is an object:

```json
{"name": "temperature", "bounds": [300.0, 700.0]}
```

`bounds` is optional. When omitted, the observed dataset minimum and maximum are used. Search never goes beyond explicit or observed bounds.

Categorical features are not supported in version 1. Convert them to scientifically meaningful numeric or one-hot columns before running the skill.

## Target definitions

Each target is an object with:

- `name`: dataset column.
- `goal`: `minimize`, `maximize`, or `target`.
- `weight`: positive relative importance; defaults to 1.
- `target_value`: required when `goal` is `target` unless an `expected_range` is supplied.
- `expected_range`: optional `[lower, upper]`; use `null` for an open end.

Examples:

```json
{"name": "yield", "goal": "maximize", "weight": 2, "expected_range": [0.85, null]}
```

```json
{"name": "defect_rate", "goal": "minimize", "weight": 1, "expected_range": [null, 0.02]}
```

```json
{"name": "pH", "goal": "target", "target_value": 7.2, "weight": 1, "expected_range": [7.0, 7.4]}
```

The final result reports whether every predicted target meets its expected range. Expectations are acceptance checks, not evidence of causal or experimental validation.

## Constraints

Constraints narrow feature values during candidate search:

```json
{"feature": "pressure", "min": 1.0, "max": 5.0}
```

At least one of `min` or `max` is required. Constraints must lie inside the corresponding feature bounds.

Row-wise algebraic constraints are not supported in version 1. Encode derived feasible variables before running or extend `pipeline_core.py` with a tested constraint function.

## Validation and search settings

`validation`:

- `test_fraction`: 0 to 1, default 0.3.
- `cv_folds`: integer at least 2, default 5.
- `group_column`: optional grouping column for leakage-resistant `GroupKFold`.
- `random_seed`: integer, default 42.

`search`:

- `candidate_count`: random continuous candidates inside bounds; default 10,000.
- `decision_method`: `auto`, `achievement_scalarization`, `weighted_sum`, or `distance_to_expectation`.
- `sensitivity_samples`: number of random weight vectors; default 300.

When `decision_method` is `auto`, the Agent discussion chooses a method. If the choice cannot be parsed or executed safely, the runner records and uses `achievement_scalarization` as the verified fallback.

## Virtual Lab settings

`virtual_lab`:

- `mode`: `auto`, `live`, or `offline`.
- `provider`: `openai`, `deepseek`, `anthropic`, `google`, or `openai_compatible`.
- `model`: exact model identifier accepted by the selected provider.
- `api_key_env`: optional environment-variable name containing the key.
- `base_url`: optional provider endpoint override; required for `openai_compatible`.
- `parallel_runs`: independent creative meetings; default 3.
- `meeting_rounds`: specialist discussion rounds; default 2.

Default credential variables are `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. `VIRTUAL_LAB_API_KEY` is a provider-independent fallback. `auto` uses live agents when a configured credential is available and otherwise uses a clearly labeled deterministic offline team.

Use `openai_compatible` for providers exposing a compatible Chat Completions endpoint:

```json
{
  "mode": "live",
  "provider": "openai_compatible",
  "model": "provider-model-id",
  "base_url": "https://provider.example/v1/chat/completions",
  "api_key_env": "PROVIDER_API_KEY"
}
```

Never add an `api_key`, `token`, `secret`, or `password` field. The runner rejects inline credentials. For interactive entry, run with `--prompt-api-key`; the input is hidden and not persisted.

## Output settings

`output`:

- `directory`: parent directory for timestamped run artifacts.
- `obsidian_directory`: optional folder for the complete Markdown handoff.

Do not put secrets in the spec, output directory, or Obsidian notes.

## Complete example

Use [the bundled template](../assets/experiment_spec.template.json) as the complete editable example. Remove placeholder features and targets rather than leaving unused entries in a real spec.
