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
- `output`: artifact and optional Markdown handoff locations.

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

- `candidate_count`: continuous candidates inside bounds before support screening; default 10,000.
- `decision_method`: `auto`, `achievement_scalarization`, `weighted_sum`, or `distance_to_expectation`.
- `sensitivity_samples`: number of random weight vectors; default 300.
- `model_families`: optional subset of `RandomForest`, `ExtraTrees`, `GradientBoosting`, and `KNN`; the live analysis plan supplies this when omitted.
- `candidate_strategy`: `latin_hypercube_plus_observed` or `random_uniform_plus_observed`; the live analysis plan supplies this when omitted.

When `decision_method` is `auto`, the Agent discussion chooses a method through structured `analysis_plan.json`. Unsupported, malformed, or contradictory plans fail explicitly; the runner never infers a choice from keyword mentions and never silently substitutes another method.

## Virtual Lab settings

`virtual_lab`:

- `mode`: `auto`, `live`, or `offline`.
- `provider`: `openai`, `deepseek`, `anthropic`, `google`, or `openai_compatible`.
- `model`: exact model identifier accepted by the selected provider.
- `api_key_env`: optional environment-variable name containing the key.
- `base_url`: optional provider endpoint override; required for `openai_compatible`.
- `independent_runs`: sequential independent creative meetings; default 3.
- `meeting_rounds`: specialist discussion rounds; default 2.

Default credential variables are `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. `VIRTUAL_LAB_API_KEY` is a provider-independent fallback. `auto` remains offline unless `--allow-live-auto` explicitly authorizes use of a discovered credential. Live runs require interactive analysis-plan approval or `--approve-plan` after review.

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

Never add an `api_key`, `token`, `secret`, or `password` field at any nesting level. The runner rejects credential-like fields recursively. For interactive entry, run with `--prompt-api-key`; the input is hidden and not persisted. Custom endpoints must use HTTPS, contain no embedded credentials, query, or fragment, and require `--allow-custom-endpoint` in live mode after the host is reviewed.

## Output settings

`output`:

- `directory`: parent directory for timestamped run artifacts.
- `handoff_directory`: optional folder for the complete Markdown handoff.

Do not put secrets in the spec, output directory, or Markdown handoff.

## Complete example

Use [the bundled template](../assets/experiment_spec.template.json) as the complete editable example. Remove placeholder features and targets rather than leaving unused entries in a real spec.
