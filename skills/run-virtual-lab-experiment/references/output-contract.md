# Output and provenance contract

## Run directory

Every run creates a timestamped directory containing:

- `experiment_spec.json`: validated, resolved spec used by the run.
- `dataset_profile.json`: schema, quality, bounds, distributions, and correlations.
- `agents.json`: generated Title/Expertise/Goal/Role definitions.
- `conversations.json`: structured speaker turns for every meeting.
- `conversations.md`: readable complete transcript.
- `generated_pipeline.py`: exact ML program executed.
- `execution.json`: command, return code, stdout, stderr, and duration.
- `results/metrics.csv`: holdout/CV metrics for each target and model.
- `results/pareto_front.csv`: non-dominated candidates and predicted targets.
- `results/selected_result.csv`: selected feature values, predictions, and expectation checks.
- `results/results.json`: complete machine-readable result and limitations.
- `virtual_lab_report.md`: experiment summary, agents, conversations, code, output, and result.

If `obsidian_directory` is provided, `virtual_lab_report.md` is copied there with a unique experiment/run name.

## Result acceptance

Treat a run as computationally successful only when:

- the dataset and spec validate;
- every required artifact exists;
- model metrics are finite;
- the selected candidate obeys feature bounds and constraints;
- every target direction matches the spec;
- the decision method and weights are recorded;
- execution exits successfully;
- limitations explicitly state that predictions require experimental validation.

The `expectations_met` field is a predicted screening outcome. It does not prove the real experiment will meet the requested parameters.

## Failure handling

Save partial conversations and errors. Never present a missing or failed result as success. When live Agent code fails, retain the failed code and error, then use the verified generic pipeline only when the report labels the fallback.
