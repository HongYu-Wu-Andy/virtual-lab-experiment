#!/usr/bin/env python3
"""Run a generic agent-led Virtual Lab experiment from a JSON specification."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_core


@dataclass(frozen=True)
class Agent:
    title: str
    expertise: str
    goal: str
    role: str
    model: str = "deepseek-v4-pro"

    @property
    def system_prompt(self) -> str:
        return (
            f"You are the {self.title}. Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. Your role is to {self.role}. "
            "Use the supplied dataset profile and experiment specification. Separate evidence, "
            "assumptions, and hypotheses. Preserve units and target directions. Respond only as "
            "your assigned role; never simulate other participants or finish their turns."
        )

    @property
    def api_name(self) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", self.title)[:64]


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    usage: dict[str, Any]


class DeepSeekClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepseek.com/chat/completions"

    def complete(
        self,
        agent: Agent,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> Completion:
        payload = json.dumps(
            {
                "model": agent.model or self.model,
                "messages": [
                    {"role": "system", "name": agent.api_name, "content": agent.system_prompt},
                    *messages,
                ],
                "temperature": temperature,
                "max_tokens": 6000,
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(4):
            request = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices") or []
                content = choices[0].get("message", {}).get("content") if choices else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned an empty response")
                return Completion(
                    content=content.strip(),
                    model=body.get("model", agent.model),
                    usage=body.get("usage") or {},
                )
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(min(2**attempt + random.random(), 15.0))
        raise RuntimeError(f"DeepSeek call failed: {last_error}")


class OfflineClient:
    def complete(
        self,
        agent: Agent,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> Completion:
        transcript = "\n".join(item.get("content", "") for item in messages)
        latest = messages[-1].get("content", "") if messages else ""
        if "[CREATE_AGENTS_JSON]" in transcript:
            content = json.dumps(
                {
                    "agents": [
                        {
                            "title": "Domain Science Specialist",
                            "expertise": "the experiment domain, mechanisms, units, and physical constraints",
                            "goal": "keep the analysis scientifically meaningful and identify domain risks",
                            "role": "interpret variables, expected parameters, mechanisms, and validation needs",
                        },
                        {
                            "title": "Experimental Design and Statistics Specialist",
                            "expertise": "design of experiments, measurement error, leakage, uncertainty, and validation",
                            "goal": "ensure conclusions are testable and the validation design matches the experiment",
                            "role": "audit sampling, confounding, replicates, group structure, and acceptance criteria",
                        },
                        {
                            "title": "Machine Learning and Optimization Specialist",
                            "expertise": "tabular regression, model comparison, Pareto analysis, and decision sensitivity",
                            "goal": "build a reproducible surrogate and select a transparent computational candidate",
                            "role": "compare models, design validation, search candidates, and report uncertainty",
                        },
                    ]
                }
            )
        elif "[MODEL_AND_DECISION]" in transcript:
            if "Machine Learning" in agent.title:
                content = (
                    "Compare Random Forest, Extra Trees, Gradient Boosting, and scaled KNN per target. "
                    "Use holdout plus grouped CV when a group column exists. For multi-target selection, "
                    "I choose augmented achievement scalarization with explicit target weights and random "
                    "weight sensitivity. DECISION_METHOD: achievement_scalarization"
                )
            elif agent.title == "Principal Investigator" and "final summary" in latest.lower():
                content = (
                    "The team approves per-target model selection by CV normalized RMSE, Pareto filtering, "
                    "and augmented achievement scalarization with weight sensitivity. Expectations are "
                    "screening thresholds, not validation. DECISION_METHOD: achievement_scalarization"
                )
            else:
                content = (
                    f"As {agent.title}, I require leakage-aware validation, explicit assumptions, sensitivity "
                    "analysis, and real experimental confirmation of the selected parameter set."
                )
        elif "[FINAL_REVIEW]" in transcript:
            if agent.title == "Principal Investigator" and "final summary" in latest.lower():
                content = (
                    "The computational run is complete. Advance the selected parameter set only as a proposed "
                    "next experiment. Review expectation checks, model metrics, sensitivity, and limitations "
                    "before execution in the physical laboratory."
                )
            else:
                content = (
                    f"As {agent.title}, I reviewed the result and require confirmation with the real experiment; "
                    "model predictions alone do not establish mechanism or reproducibility."
                )
        elif "[PROJECT_SPECIFICATION]" in transcript:
            if agent.title == "Principal Investigator" and "final summary" in latest.lower():
                content = (
                    "The project will model the declared numeric targets from the declared features, preserve each "
                    "goal and expectation, search only inside observed/configured bounds, and record all failures."
                )
            else:
                content = (
                    f"As {agent.title}, I confirm the supplied variables and expectations while flagging that "
                    "correlation and prediction must not be presented as causal experimental proof."
                )
        elif "merge" in transcript.lower():
            content = (
                "Merged consensus: preserve target directions and bounds, use leakage-aware validation, compare "
                "multiple models, record decision sensitivity, and treat the result as a next-experiment proposal."
            )
        else:
            content = f"As {agent.title}, I provide my role-specific analysis and validation requirements."
        return Completion(content=content, model="offline-deterministic", usage={"offline_calls": 1})


PI = Agent(
    "Principal Investigator",
    "interdisciplinary research leadership and evidence-based experiment governance",
    "coordinate a reproducible Virtual Lab that produces a testable next experiment",
    "convene meetings, reconcile specialists, record decisions, and gate conclusions",
)

CRITIC = Agent(
    "Scientific Critic",
    "falsification, statistics, scientific software, reproducibility, and experimental validity",
    "prevent unsupported or non-reproducible results from being accepted",
    "audit target directions, leakage, uncertainty, code execution, and claim strength",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "experiment"


def spec_for_llm(spec: dict[str, Any]) -> dict[str, Any]:
    """Remove unnecessary local path details before sending context to an LLM."""
    shared = json.loads(json.dumps(spec))
    dataset = shared.get("dataset") or {}
    if dataset.get("path"):
        dataset["path"] = Path(str(dataset["path"])).name
    output = shared.get("output") or {}
    if output.get("directory"):
        output["directory"] = "[local output directory]"
    if output.get("obsidian_directory"):
        output["obsidian_directory"] = "[local Obsidian directory]"
    return shared


def resolve_mode(spec: dict[str, Any], override: str | None) -> str:
    mode = override or (spec.get("virtual_lab") or {}).get("mode", "auto")
    if mode == "auto":
        return "live" if os.environ.get("DEEPSEEK_API_KEY") else "offline"
    if mode not in {"live", "offline"}:
        raise ValueError("Virtual Lab mode must be auto, live, or offline")
    return mode


def parse_json_response(text: str) -> dict[str, Any]:
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + candidates
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
    raise ValueError("Could not parse JSON from agent response")


def generate_agents(client, spec: dict[str, Any], model: str, events: list[dict[str, Any]]) -> list[Agent]:
    prompt = (
        "[CREATE_AGENTS_JSON] Create exactly three scientist agents tailored to this experiment: one domain "
        "science expert, one experimental-design/statistics expert, and one machine-learning/optimization "
        "expert. Return JSON only as {\"agents\":[{\"title\":...,\"expertise\":...,\"goal\":...,"
        "\"role\":...}, ...]}. Experiment:\n" + json.dumps(spec, ensure_ascii=False)
    )
    messages = [{"role": "user", "name": "HumanResearcher", "content": prompt}]
    completion = client.complete(PI, messages, 0.2)
    events.append(
        {
            "meeting": "agent_generation",
            "speaker": "Human Researcher",
            "kind": "prompt",
            "message": prompt,
        }
    )
    events.append(
        {
            "meeting": "agent_generation",
            "speaker": PI.title,
            "kind": "response",
            "message": completion.content,
            "model": completion.model,
            "usage": completion.usage,
        }
    )
    payload = parse_json_response(completion.content)
    definitions = payload.get("agents") or []
    if len(definitions) != 3:
        raise ValueError("PI must generate exactly three scientist agents")
    agents = []
    for definition in definitions:
        required = [str(definition.get(key, "")).strip() for key in ("title", "expertise", "goal", "role")]
        if any(not value for value in required):
            raise ValueError("Generated agent lacks title, expertise, goal, or role")
        agents.append(Agent(*required, model=model))
    return agents


def ask(client, agent: Agent, history, events, meeting: str, prompt: str, temperature: float):
    history.append({"role": "user", "name": "Facilitator", "content": prompt})
    events.append({"meeting": meeting, "speaker": "Facilitator", "kind": "prompt", "message": prompt})
    completion = client.complete(agent, history, temperature)
    history.append({"role": "assistant", "name": agent.api_name, "content": completion.content})
    events.append(
        {
            "meeting": meeting,
            "speaker": agent.title,
            "kind": "response",
            "message": completion.content,
            "model": completion.model,
            "usage": completion.usage,
        }
    )
    return completion.content


def team_meeting(
    client,
    name: str,
    agenda: str,
    agents: list[Agent],
    rounds: int,
    temperature: float,
    events: list[dict[str, Any]],
) -> str:
    history: list[dict[str, str]] = []
    context = (
        f"Meeting: {name}\nAgenda:\n{agenda}\n\nEach speaker must respond only as their own role. "
        "Do not simulate later speakers. Preserve feature names, target directions, expectations, and units."
    )
    history.append({"role": "user", "name": "HumanResearcher", "content": context})
    events.append({"meeting": name, "speaker": "Human Researcher", "kind": "agenda", "message": context})
    ask(client, PI, history, events, name, "PI, give your opening analysis and focused questions only.", temperature)
    members = [*agents, CRITIC]
    for round_number in range(1, rounds + 1):
        for agent in members:
            ask(
                client,
                agent,
                history,
                events,
                name,
                f"{agent.title}, provide your own round {round_number} evidence, risks, and recommendation.",
                temperature,
            )
        if round_number < rounds:
            ask(
                client,
                PI,
                history,
                events,
                name,
                "PI, synthesize agreements, disagreements, decisions, and questions for the next round.",
                temperature,
            )
    return ask(
        client,
        PI,
        history,
        events,
        name,
        "PI, provide the final summary with decisions, rejected alternatives, risks, expectations, and next steps.",
        temperature,
    )


def parallel_and_merge(
    client,
    phase: str,
    agenda: str,
    agents: list[Agent],
    runs: int,
    rounds: int,
    events: list[dict[str, Any]],
) -> str:
    summaries = [
        team_meeting(client, f"{phase}_creative_{index}", agenda, agents, rounds, 0.8, events)
        for index in range(1, runs + 1)
    ]
    merge_prompt = (
        f"Merge the {len(summaries)} independent summaries for {phase}. Resolve contradictions, preserve "
        "minority risks, and produce one actionable answer.\n\n"
        + "\n\n".join(f"[summary {i}]\n{text}" for i, text in enumerate(summaries, 1))
    )
    history: list[dict[str, str]] = []
    initial = ask(client, PI, history, events, f"{phase}_merge", merge_prompt, 0.2)
    ask(
        client,
        CRITIC,
        history,
        events,
        f"{phase}_merge",
        "Scientific Critic, identify omissions, unsupported claims, and required corrections in the merge.",
        0.2,
    )
    return ask(
        client,
        PI,
        history,
        events,
        f"{phase}_merge",
        "PI, replace the merge with a complete corrected final answer addressing every critique.",
        0.2,
    ) or initial


def choose_decision_method(spec: dict[str, Any], discussion: str) -> str:
    configured = str((spec.get("search") or {}).get("decision_method", "auto"))
    if configured != "auto":
        return configured
    lowered = discussion.lower()
    if "weighted sum" in lowered or "weighted_sum" in lowered:
        return "weighted_sum"
    if "distance to expectation" in lowered or "distance_to_expectation" in lowered:
        return "distance_to_expectation"
    return "achievement_scalarization"


def conversation_markdown(events: list[dict[str, Any]]) -> str:
    lines = ["# Virtual Lab Conversations", ""]
    current = None
    for event in events:
        if event["meeting"] != current:
            current = event["meeting"]
            lines.extend([f"## {current}", ""])
        lines.extend([f"### {event['speaker']}", "", str(event["message"]), ""])
    return "\n".join(lines)


def report_markdown(
    run_id: str,
    mode: str,
    spec: dict[str, Any],
    agents: list[Agent],
    events: list[dict[str, Any]],
    code: str,
    execution: dict[str, Any],
    results: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "---",
            f"title: Virtual Lab Experiment - {spec['experiment_name']} - {run_id}",
            f"date: {datetime.now().date().isoformat()}",
            f"mode: {mode}",
            "tags: [virtual-lab, machine-learning, experiment]",
            "---",
            "",
            f"# Virtual Lab Experiment: {spec['experiment_name']}",
            "",
            f"**Mode:** `{mode}`",
            f"**Description:** {spec['description']}",
            f"**Decision method:** `{results['decision_method']}`",
            f"**Predicted expectations met:** `{results['selected']['expectations_met']}`",
            "",
            "## Experiment specification",
            "",
            "```json",
            json.dumps(spec, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Generated agents",
            "",
            "```json",
            json.dumps([asdict(agent) for agent in agents], indent=2, ensure_ascii=False),
            "```",
            "",
            conversation_markdown(events),
            "## Generated and executed ML code",
            "",
            "```python",
            code.rstrip(),
            "```",
            "",
            "## Execution output",
            "",
            "```text",
            execution.get("stdout", "").rstrip(),
            execution.get("stderr", "").rstrip(),
            "```",
            "",
            "## Results",
            "",
            "```json",
            json.dumps(results, indent=2, ensure_ascii=False),
            "```",
            "",
            "> Predictions are proposals for the next real experiment, not experimental validation.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "live", "offline"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--obsidian-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    validated = pipeline_core.load_spec(spec_path)
    spec = json.loads(json.dumps(validated.raw))
    mode = resolve_mode(spec, args.mode)
    settings = spec.get("virtual_lab") or {}
    model = str(settings.get("model", "deepseek-v4-pro"))
    if mode == "live":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for live mode")
        client = DeepSeekClient(key, model)
    else:
        client = OfflineClient()

    runs = 1 if args.quick else int(settings.get("parallel_runs", 3))
    rounds = 1 if args.quick else int(settings.get("meeting_rounds", 2))
    if runs < 1 or rounds < 1:
        raise ValueError("parallel_runs and meeting_rounds must be at least one")
    if args.quick:
        spec.setdefault("search", {})["candidate_count"] = min(
            int(spec.get("search", {}).get("candidate_count", 10_000)), 1_000
        )
        spec["search"]["sensitivity_samples"] = min(
            int(spec["search"].get("sensitivity_samples", 300)), 50
        )

    root_value = args.output_dir or Path((spec.get("output") or {}).get("directory", "virtual_lab_output"))
    root = root_value.expanduser()
    if not root.is_absolute():
        root = spec_path.parent / root
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root.resolve() / safe_name(spec["experiment_name"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result_dir = run_dir / "results"

    events: list[dict[str, Any]] = []
    shared_spec = spec_for_llm(spec)
    agents = generate_agents(client, shared_spec, model, events)
    (run_dir / "agents.json").write_text(
        json.dumps([asdict(agent) for agent in agents], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    data_path = pipeline_core.dataset_path(spec_path, pipeline_core.load_spec(spec_path))
    data = pipeline_core.validate_dataset(
        pipeline_core.read_dataset(data_path), pipeline_core.load_spec(spec_path)
    )
    feature_names = [item["name"] for item in spec["features"]]
    target_names = [item["name"] for item in spec["targets"]]
    profile = {
        "path": str(data_path),
        "shape": list(data.shape),
        "features": feature_names,
        "targets": target_names,
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "summary": data[feature_names + target_names].describe().to_dict(),
        "spearman": data[feature_names + target_names].corr(method="spearman").to_dict(),
    }
    (run_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    shared_profile = json.loads(json.dumps(profile))
    shared_profile["path"] = data_path.name
    context = json.dumps(
        {"spec": shared_spec, "dataset_profile": shared_profile}, ensure_ascii=False
    )
    specification_summary = parallel_and_merge(
        client,
        "project_specification",
        "[PROJECT_SPECIFICATION] Audit the dataset, features, targets, directions, expected ranges, "
        "constraints, experimental validity, and blocking ambiguities.\n" + context,
        agents,
        runs,
        rounds,
        events,
    )
    method_summary = parallel_and_merge(
        client,
        "model_and_decision",
        "[MODEL_AND_DECISION] Design the model comparison, leakage-aware validation, candidate search, "
        "multi-target decision method, weight sensitivity, and experimental confirmation. No decision "
        "method is preselected.\n" + context + "\nPrevious decision:\n" + specification_summary,
        agents,
        runs,
        rounds,
        events,
    )
    chosen_method = choose_decision_method(spec, method_summary)
    spec.setdefault("search", {})["decision_method"] = chosen_method

    resolved_spec_path = run_dir / "experiment_spec.json"
    # Make the dataset path absolute so the generated program is relocatable within this run.
    spec.setdefault("dataset", {})["path"] = str(data_path)
    resolved_spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    generated_code_path = run_dir / "generated_pipeline.py"
    code = (SCRIPT_DIR / "pipeline_core.py").read_text(encoding="utf-8")
    generated_code_path.write_text(code, encoding="utf-8")
    command = [
        sys.executable,
        str(generated_code_path),
        "--spec",
        str(resolved_spec_path),
        "--output-dir",
        str(result_dir),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
    execution = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (run_dir / "execution.json").write_text(
        json.dumps(execution, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Generated pipeline failed with code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    results = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))

    final_summary = team_meeting(
        client,
        "final_review",
        "[FINAL_REVIEW] Review the executed metrics, selected parameters, expectation checks, "
        "sensitivity, limitations, and the exact next real experiment.\n" + json.dumps(results),
        agents,
        1,
        0.2,
        events,
    )
    results["virtual_lab"] = {
        "mode": mode,
        "generated_agents": [agent.title for agent in agents],
        "decision_summary": method_summary,
        "final_review": final_summary,
    }
    (result_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "conversations.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "conversations.md").write_text(conversation_markdown(events), encoding="utf-8")
    report = report_markdown(run_id, mode, spec, agents, events, code, execution, results)
    report_path = run_dir / "virtual_lab_report.md"
    report_path.write_text(report, encoding="utf-8")

    obsidian_value = args.obsidian_dir or (
        Path(spec.get("output", {}).get("obsidian_directory"))
        if spec.get("output", {}).get("obsidian_directory")
        else None
    )
    obsidian_path = None
    if obsidian_value:
        obsidian_dir = obsidian_value.expanduser().resolve()
        obsidian_dir.mkdir(parents=True, exist_ok=True)
        obsidian_path = obsidian_dir / (
            f"Virtual Lab - {safe_name(spec['experiment_name'])} - {run_id}.md"
        )
        shutil.copy2(report_path, obsidian_path)

    print(
        json.dumps(
            {
                "status": "success",
                "mode": mode,
                "run_directory": str(run_dir),
                "report": str(report_path),
                "obsidian_report": str(obsidian_path) if obsidian_path else None,
                "selected": results["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
