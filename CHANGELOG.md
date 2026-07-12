# Changelog

## 0.2.2 — 2026-07-12

- Updated the repository owner, package links, plugin marketplace source, installation command, CI badge, security URLs, and citation metadata to `WUHYA`.

## 0.2.1 — 2026-07-12

- Replaced the platform-specific handoff option with a generic Markdown handoff directory.
- Renamed the CLI option to `--handoff-dir`, the specification field to `handoff_directory`, and the execution summary field to `handoff_report`.
- Updated the public workflow diagram, documentation, schemas, examples, privacy language, and tests to use platform-neutral Markdown terminology.

## 0.2.0 — 2026-07-11

- Prepared the repository for public beta with CI across Python 3.11-3.13, packaging checks, dependency updates, and a redacted full-history secret audit.
- Added public installation guidance, a workflow diagram, an example result, project scope, roadmap, community templates, and scientific attribution.
- Added installable project metadata and the `virtual-lab-experiment` command.
- Added user-selectable OpenAI, DeepSeek, Anthropic, Google Gemini, and custom OpenAI-compatible providers.
- Added provider/model/base-URL overrides and `--list-providers`.
- Added provider-specific environment variables, generic `VIRTUAL_LAB_API_KEY`, and hidden `--prompt-api-key` entry.
- Added inline-secret rejection and ensured credential values never enter specifications or artifacts.
- Added mocked request/response tests for every native API style.

## 0.1.1 — 2026-07-11

- Updated repository-owner links and installation metadata after an earlier account rename.

## 0.1.0 — 2026-07-10

- Added the distributable Codex plugin manifest and private GitHub marketplace.
- Added dynamic experiment-specific agent generation and structured team meetings.
- Added deterministic offline mode and DeepSeek-backed live mode.
- Added multi-target regression model comparison, Pareto search, automatic decision-method selection, and sensitivity analysis.
- Added complete conversation, code, execution, metric, candidate, result, and Markdown reporting.
- Added removal of unnecessary absolute paths from live LLM context.
- Added a synthetic example, end-to-end test, privacy policy, security guidance, and MIT license.
