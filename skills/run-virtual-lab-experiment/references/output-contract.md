# Output and provenance contract

## Run directory

Every run creates a timestamped directory containing:

- `experiment_spec.json`: validated, resolved spec used by the run.
- `dataset_profile.json`: schema, quality, bounds, distributions, and correlations.
- `agents.json`: generated Title/Expertise/Goal/Role definitions.
- `analysis_plan.json`: structured PI plan controlling model families, candidate strategy, decision method, blockers, and rationale.
- `conversations.json`: structured speaker turns for every meeting.
- `conversations.md`: readable complete transcript.
- `executed_pipeline.py`: exact verified ML program executed; it is bundled code, not falsely represented as LLM-generated source.
- `execution.json`: command, return code, stdout, stderr, and duration.
- `results/metrics.csv`: holdout/CV metrics for each target and model.
- `results/pareto_front.csv`: supported non-dominated candidates, predicted targets, screening uncertainty, and support distance.
- `results/selected_result.csv`: selected feature values, predictions, and expectation checks.
- `results/results.json`: complete machine-readable result and limitations.
- `virtual_lab_report.md`: experiment summary, agents, conversations, code, output, and result.

`execution.json`, `results.json`, and the report record the selected mode, provider, model, endpoint, and credential source label. They never record the credential value.

If `handoff_directory` is provided, `virtual_lab_report.md` is copied there as a uniquely named `.md` handoff.

## Result acceptance

Treat a run as computationally successful only when:

- the dataset and spec validate;
- every required artifact exists;
- model metrics are finite;
- the grouped holdout has no group overlap when grouping is required;
- candidates pass the recorded multivariate support-distance threshold;
- prediction screening intervals are present;
- the selected candidate obeys feature bounds and constraints;
- every target direction matches the spec;
- the decision method and weights are recorded;
- execution exits successfully;
- limitations explicitly state that predictions require experimental validation.

`expectations_met` reports point-prediction checks. `conservative_expectations_met` requires the complete support-adjusted residual screening interval to satisfy every configured range. Neither proves that the real experiment will meet the requested parameters.

## Failure handling

Checkpoint every conversation turn. On any agent, approval, execution, or review failure, retain `conversations.json`, `conversations.md`, `failure.json`, `failure_report.md`, and any completed artifacts. Never present a missing or failed result as success, and never silently substitute another analysis plan or decision method.
