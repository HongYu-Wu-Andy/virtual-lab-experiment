# Privacy

## Local and offline operation

In offline mode, the plugin reads the selected dataset and writes experiment artifacts on the local machine. It does not send dataset content to an LLM provider.

## Live agent meetings

Live mode sends the following material to the DeepSeek API:

- the experiment description;
- feature and target names;
- goal directions, target values, expected ranges, bounds, and constraints;
- dataset shape, missing-value and duplicate counts;
- descriptive statistics and Spearman correlations;
- model results and candidate-selection summaries;
- meeting prompts and prior agent responses.

The runner does not send raw dataset rows. It removes absolute dataset, output, and Obsidian paths from the LLM context. DeepSeek processes submitted material under its own terms and privacy policy.

Do not use live mode with confidential, regulated, export-controlled, personal, or commercially sensitive data unless the user or organization has approved that provider and data flow. Use offline mode when external processing is not permitted.

## Local artifacts

Local reports intentionally preserve provenance and can contain:

- absolute dataset and output paths;
- experiment descriptions and column names;
- statistics, correlations, predictions, and selected parameters;
- full agent conversations;
- executed code and process output.

Review artifacts before sharing them. Obsidian output is optional and remains under the user's chosen storage and synchronization settings.

## Credentials

Live mode reads `DEEPSEEK_API_KEY` from the process environment. The key is not written to the experiment specification, report, conversation log, or result files by the plugin.
