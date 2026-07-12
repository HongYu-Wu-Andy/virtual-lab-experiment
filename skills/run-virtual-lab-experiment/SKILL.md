---
name: run-virtual-lab-experiment
description: Run an agent-led Virtual Lab for numeric tabular experiment datasets with a user-selected LLM provider and model, automatically creating domain-specific scientist agents, conducting and merging research meetings, training and validating multi-target regression models, searching parameter candidates, and recording every conversation, generated code, execution output, metric, Pareto candidate, and final result. Use when a user provides a CSV/TSV dataset and asks to optimize parameters, build an ML surrogate, create a digital/virtual laboratory, choose an AI provider/model, or save a reproducible Markdown experiment handoff.
---

# Run Virtual Lab Experiment

Use the bundled runner for numeric tabular regression experiments. Treat its selected parameters as a proposed next experiment, never as experimental validation.

## Gather the experiment contract

Obtain or infer these inputs:

1. Dataset file (`.csv` or `.tsv`).
2. Experiment name and scientific purpose.
3. Numeric feature columns to control or search.
4. Numeric target columns.
5. Goal for each target: `minimize`, `maximize`, or `target`.
6. Desired value/range for each target when the user has one.
7. Feature bounds and hard constraints.
8. Optional group column for leakage-resistant validation.
9. Output directory and optional Markdown handoff directory.
10. Live-mode provider and exact model identifier.
11. Credential method: provider environment variable, `VIRTUAL_LAB_API_KEY`, or hidden interactive prompt.

Inspect the dataset before asking questions. Ask only for missing choices that materially change the experiment. Do not guess target directions, units, or safety-critical constraints.

Read [references/spec-schema.md](references/spec-schema.md) when constructing or repairing the JSON spec. Start from [assets/experiment_spec.template.json](assets/experiment_spec.template.json), copy only applicable fields, and save the resolved spec beside the user's experiment artifacts.

## Preserve scientific integrity

- Keep feature and target names exactly aligned with dataset columns.
- Keep goals and expected ranges distinct: a goal drives optimization; an expected range screens the result.
- Use only observed or explicitly approved feature bounds. Reject undocumented extrapolation.
- Set `group_column` when samples share batches, patients, specimens, compositions, time series, sites, or other leakage-prone groups.
- Keep `decision_method` as `auto` unless the user explicitly requires a method. Let the Agent discussion choose; do not prescribe a named method in the meeting agenda.
- State that model predictions are hypotheses for the next real experiment.
- Require domain-specific safety, ethics, and experimental approval outside the ML runner when applicable.

## Check dependencies

The scripts require Python 3.11+, NumPy, pandas, and scikit-learn.

Before execution, check imports with the intended interpreter. If missing, use `scripts/requirements.txt` to create or update an isolated environment. Request approval before downloading dependencies when the environment requires it.

Never ask the user to place an API key in the spec, command line, chat transcript, or a tracked file. Accept credentials through the provider environment variable, `VIRTUAL_LAB_API_KEY`, or `--prompt-api-key`. If a user pastes a key into chat, do not repeat it; recommend revocation and replacement.

Supported native providers are `openai`, `deepseek`, `anthropic`, and `google`. Use `openai_compatible` with an HTTPS `base_url` for other providers exposing a Chat Completions-compatible endpoint. Read [references/spec-schema.md](references/spec-schema.md) for provider fields and default environment-variable names.

## Run the Virtual Lab

Execute:

```bash
python scripts/run_virtual_lab.py --spec experiment_spec.json
```

Useful variants:

```bash
python scripts/run_virtual_lab.py --spec experiment_spec.json --mode live
```

```bash
python scripts/run_virtual_lab.py --spec experiment_spec.json --mode live \
  --provider openai --model YOUR_MODEL --prompt-api-key
```

```bash
python scripts/run_virtual_lab.py --spec experiment_spec.json --mode offline --quick
```

```bash
python scripts/run_virtual_lab.py --spec experiment_spec.json --handoff-dir path/to/report/folder
```

Resolve script paths relative to this skill directory. Do not copy the skill's scripts into the user's source repository unless the user asks.

## Understand execution modes

- `auto`: use live agents when the configured provider key is available; otherwise use offline mode.
- `live`: use the selected provider/model to generate three experiment-specific scientist agents, run creative meetings, merge them with criticism, choose a decision method, and review the executed result.
- `offline`: create a clearly labeled deterministic team and use the verified generic pipeline. Use for testing and reproducibility, not as evidence of independent LLM deliberation.

The generated scientist team contains:

- a domain science specialist;
- an experimental-design/statistics specialist;
- an ML/optimization specialist;
- plus the fixed PI and Scientific Critic.

## Verify the run

Read [references/output-contract.md](references/output-contract.md) before accepting or delivering results.

Confirm:

1. Process exit code is zero.
2. `results/results.json` says `status: success`.
3. Feature bounds and target goals match the spec.
4. The selected parameter set obeys all constraints.
5. Holdout and CV metrics are finite and plausible.
6. `expectation_checks` are reported for every expected range.
7. Decision weights and sensitivity are recorded.
8. `conversations.json`, `generated_pipeline.py`, and `execution.json` exist.
9. `virtual_lab_report.md` contains conversations, code, output, results, and limitations.
10. Any requested Markdown handoff copy exists.
11. Provider and model metadata match the requested configuration, and no credential value appears in any artifact.

If validation fails, report the exact failure and retain partial artifacts. Do not summarize a failed run as successful.

## Deliver the result

Lead with:

- selected feature/parameter values;
- predicted targets;
- whether each expectation is predicted to pass;
- model/CV quality;
- decision sensitivity;
- run mode (`live` or `offline`);
- required real-world confirmation.

Link the report and the run directory. Mention that the report contains every Agent turn, the executed code, stdout/stderr, metrics, Pareto front, and machine-readable result.

## Supported scope

Version 1 supports:

- CSV/TSV input;
- numeric features;
- numeric regression targets;
- one or multiple targets;
- minimize, maximize, or target-value goals;
- box constraints on individual features;
- random or grouped cross-validation;
- continuous candidate sampling inside observed/configured bounds.

Do not silently apply it to classification, images, spectra requiring preprocessing, censored outcomes, time-series forecasting, causal inference, or row-wise algebraic constraints. Either preprocess/extend with explicit validation or explain the limitation.
