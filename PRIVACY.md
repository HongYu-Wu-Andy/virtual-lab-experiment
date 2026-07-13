# Privacy

## Local and offline operation

In offline mode, the plugin reads the selected dataset and writes experiment artifacts on the local machine. It does not send dataset content to an LLM provider.

## Live agent meetings

Live mode sends the following material to the provider selected by the user:

- the experiment description;
- feature and target names;
- goal directions, target values, expected ranges, bounds, and constraints;
- dataset shape, missing-value and duplicate counts;
- descriptive statistics and Spearman correlations;
- model results and candidate-selection summaries;
- meeting prompts and prior agent responses.

The runner does not send raw dataset rows. It removes absolute dataset, output, handoff, endpoint, and credential-variable details from the LLM context, including the final-review result summary. The selected provider processes submitted material under its own terms and privacy policy.

Do not use live mode with confidential, regulated, export-controlled, personal, or commercially sensitive data unless the user or organization has approved that provider and data flow. Use offline mode when external processing is not permitted.

## Local artifacts

Local reports intentionally preserve provenance and can contain:

- absolute dataset and output paths;
- experiment descriptions and column names;
- statistics, correlations, predictions, and selected parameters;
- full agent conversations;
- executed code and process output.

Review artifacts before sharing them. The optional Markdown handoff remains under the user's chosen storage and synchronization settings.

## Credentials

Live mode reads the configured provider environment variable, `VIRTUAL_LAB_API_KEY`, or a hidden interactive prompt. Supported defaults are `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`.

The runner recursively rejects credential-like fields at any nesting level. Credential values are not written to the experiment specification, report, conversation log, execution metadata, or result files. Artifacts record only the provider, model, endpoint, and credential source label. Provider errors are sanitized before persistence.

`auto` mode remains offline unless the user explicitly enables live auto-selection. Live runs require approval of the saved analysis plan, and custom endpoints require separate trust confirmation before receiving a credential.

## Public repository

The repository contains only a synthetic example dataset. Do not open an issue or pull request containing a private dataset, generated confidential report, personal path, or provider credential. Review all artifacts before sharing them publicly.
