# Virtual Lab Experiment

Virtual Lab Experiment is a Codex plugin for reproducible, agent-led machine-learning experiments on numeric CSV or TSV datasets. Users choose the LLM provider and model that run the scientist team. The plugin conducts structured meetings, compares regression models, searches multi-objective candidates, and preserves the complete reasoning and execution trail.

The output is a proposal for the next real experiment. Model predictions are not experimental validation and should not be presented as causal or safety evidence.

## What it does

1. Validates the dataset, features, targets, directions, bounds, and constraints.
2. Creates a Principal Investigator, Scientific Critic, domain scientist, experimental-design/statistics specialist, and machine-learning/optimization specialist.
3. Uses the user's selected OpenAI, DeepSeek, Anthropic, Google Gemini, or OpenAI-compatible provider and model.
4. Runs independent meetings and a critic-assisted merge.
5. Compares Random Forest, Extra Trees, Gradient Boosting, and scaled KNN for each target.
6. Uses holdout validation plus grouped or shuffled cross-validation.
7. Lets the agent discussion choose the multi-objective decision method when `decision_method` is `auto`; TOPSIS is not prescribed.
8. Searches feasible candidates, computes a Pareto front, and performs weight-sensitivity analysis.
9. Saves every agent message, the executed code, stdout and stderr, metrics, candidates, selected result, and a complete Markdown report.

## Repository contents

- `.codex-plugin/plugin.json` — plugin identity and install metadata.
- `marketplace.json` — private GitHub marketplace entry used by Codex.
- `SKILL.md` — the Virtual Lab workflow and scientific guardrails.
- `run_virtual_lab.py` — agent generation, meetings, orchestration, execution, and reporting.
- `pipeline_core.py` — dataset validation, model comparison, Pareto search, and final selection.
- `experiment_spec.template.json` — editable experiment input template.
- `spec-schema.md` — complete input specification.
- `output-contract.md` — generated-artifact contract.
- `synthetic_experiment.csv` — synthetic, redistributable example dataset.
- `experiment_spec.json` — runnable example configuration.
- `test_smoke.py` — end-to-end offline test.
- `test_providers.py` — mocked request/response tests for every native provider interface.
- `PRIVACY.md` — data-flow and provider disclosure.
- `SECURITY.md` — security guidance and reporting policy.
- `CHANGELOG.md` — release history.

## Install from the private repository

The repository owner must first grant the user GitHub access. The user also needs Git authentication configured on their machine.

Add the repository as a Codex marketplace:

```bash
codex plugin marketplace add HongYu-Wu-Andy/virtual-lab-experiment --ref main
```

Then install it from the Codex or ChatGPT desktop plugin directory. CLI users can run:

```bash
codex plugin add virtual-lab-experiment@virtual-lab-experiment
```

Restart Codex and begin a new task if the plugin does not appear immediately.

## Python setup

Python 3.11 or newer is recommended. Install the runtime dependencies:

```bash
python -m pip install -r skills/run-virtual-lab-experiment/scripts/requirements.txt
```

No API key is needed for deterministic offline mode. Live meetings support these provider configurations:

| Provider | `provider` value | Default key variable |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google` | `GEMINI_API_KEY` |
| Other compatible provider | `openai_compatible` | `VIRTUAL_LAB_API_KEY` |

Set the appropriate environment variable:

```bash
export OPENAI_API_KEY="your-key"
```

Alternatively, enter the key through a hidden runtime prompt:

```bash
python skills/run-virtual-lab-experiment/scripts/run_virtual_lab.py \
  --spec examples/experiment_spec.json \
  --mode live \
  --provider openai \
  --model YOUR_MODEL \
  --prompt-api-key
```

`VIRTUAL_LAB_API_KEY` can be used as a generic fallback for any provider. Never place a key in `experiment_spec.json`, a command-line argument, a chat message, or Git. The plugin records only the credential source label, never the value.

## Use in Codex

Invoke the skill explicitly:

```text
Use $run-virtual-lab-experiment with my experiment.csv dataset.
The inputs are temperature, pressure, and catalyst concentration.
Maximize yield, minimize energy use, and save the full report to my chosen folder.
Use OpenAI with model YOUR_MODEL; I will supply the key securely.
```

Codex will inspect the dataset and ask only for scientifically material information that cannot be inferred safely, such as objective direction, units, acceptance ranges, leakage groups, or physical constraints.

List the built-in provider adapters with:

```bash
python skills/run-virtual-lab-experiment/scripts/run_virtual_lab.py --list-providers
```

## Run the included example

From the repository root:

```bash
python skills/run-virtual-lab-experiment/scripts/run_virtual_lab.py \
  --spec examples/experiment_spec.json \
  --mode offline \
  --quick
```

The timestamped output includes:

- `agents.json`
- `conversations.json`
- `conversations.md`
- `dataset_profile.json`
- `experiment_spec.json`
- `generated_pipeline.py`
- `execution.json`
- `results/metrics.csv`
- `results/pareto_front.csv`
- `results/selected_result.csv`
- `results/results.json`
- `virtual_lab_report.md`

To copy the complete report to an Obsidian folder, provide `--obsidian-dir` or set `obsidian_directory` in the experiment specification.

## Supported scope

Version 0.2 supports numeric tabular regression in CSV and TSV files. It does not silently claim support for classification, images, spectra, time series, causal inference, categorical variables, or row-wise algebraic constraints.

Search stays inside observed or explicitly configured feature bounds. The selected candidate must be reviewed for domain feasibility and confirmed in the physical experiment.

## Test

```bash
python -m unittest discover -s tests -v
```

## License

Released under the MIT License. See `LICENSE`.
