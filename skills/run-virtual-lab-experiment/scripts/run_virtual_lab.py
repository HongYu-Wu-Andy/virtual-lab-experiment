#!/usr/bin/env python3
"""Run a generic agent-led Virtual Lab experiment from a JSON specification."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
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


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_style: str
    model: str
    base_url: str
    api_key_env: str


PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "openai": {
        "api_style": "openai",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": None,
    },
    "deepseek": {
        "api_style": "openai",
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-pro",
    },
    "anthropic": {
        "api_style": "anthropic",
        "base_url": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": None,
    },
    "google": {
        "api_style": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": None,
    },
    "openai_compatible": {
        "api_style": "openai",
        "base_url": None,
        "api_key_env": "VIRTUAL_LAB_API_KEY",
        "default_model": None,
    },
}

PROVIDER_ALIASES = {
    "claude": "anthropic",
    "gemini": "google",
    "custom": "openai_compatible",
    "openai-compatible": "openai_compatible",
}


def normalize_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Drop provider-specific names and merge consecutive turns with the same role."""
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = "assistant" if message.get("role") == "assistant" else "user"
        content = str(message.get("content", ""))
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n\n" + content
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def response_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item.get("text", "") for item in value if isinstance(item, dict)]
        text = "".join(part for part in parts if isinstance(part, str))
        return text or None
    return None


class HTTPClient:
    def __init__(self, api_key: str, config: ProviderConfig) -> None:
        self.api_key = api_key
        self.config = config

    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(4):
            request = urllib.request.Request(
                url,
                data=encoded,
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(min(2**attempt + random.random(), 15.0))
        raise RuntimeError(f"{self.config.provider} API call failed: {last_error}")


class OpenAICompatibleClient(HTTPClient):
    def complete(
        self,
        agent: Agent,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> Completion:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": agent.system_prompt},
                *normalize_messages(messages),
            ],
            "temperature": temperature,
        }
        token_field = "max_completion_tokens" if self.config.provider == "openai" else "max_tokens"
        payload[token_field] = 6000
        body = self.post(
            self.config.base_url,
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        choices = body.get("choices") or []
        content = response_text(choices[0].get("message", {}).get("content")) if choices else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"{self.config.provider} returned an empty response")
        return Completion(
            content=content.strip(),
            model=str(body.get("model") or self.config.model),
            usage=body.get("usage") or {},
        )


class AnthropicClient(HTTPClient):
    def complete(
        self,
        agent: Agent,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> Completion:
        body = self.post(
            self.config.base_url,
            {
                "model": self.config.model,
                "system": agent.system_prompt,
                "messages": normalize_messages(messages),
                "temperature": temperature,
                "max_tokens": 6000,
            },
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        content = response_text(body.get("content"))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("anthropic returned an empty response")
        return Completion(
            content=content.strip(),
            model=str(body.get("model") or self.config.model),
            usage=body.get("usage") or {},
        )


class GoogleClient(HTTPClient):
    def complete(
        self,
        agent: Agent,
        messages: Sequence[dict[str, str]],
        temperature: float,
    ) -> Completion:
        contents = [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in normalize_messages(messages)
        ]
        model_id = self.config.model.removeprefix("models/")
        url = (
            f"{self.config.base_url.rstrip('/')}"
            f"/{urllib.parse.quote(model_id, safe='')}:generateContent"
        )
        body = self.post(
            url,
            {
                "systemInstruction": {"parts": [{"text": agent.system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 6000,
                },
            },
            {"x-goog-api-key": self.api_key},
        )
        candidates = body.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        content = response_text(parts)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("google returned an empty response")
        return Completion(
            content=content.strip(),
            model=str(body.get("modelVersion") or self.config.model),
            usage=body.get("usageMetadata") or {},
        )


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
    if output.get("handoff_directory"):
        output["handoff_directory"] = "[local handoff directory]"
    settings = shared.get("virtual_lab") or {}
    settings.pop("api_key_env", None)
    settings.pop("base_url", None)
    for forbidden in ("api_key", "token", "secret", "password"):
        settings.pop(forbidden, None)
    return shared


def reject_inline_secrets(spec: dict[str, Any]) -> None:
    settings = spec.get("virtual_lab") or {}
    forbidden = [
        key for key in ("api_key", "token", "secret", "password") if settings.get(key)
    ]
    if forbidden:
        raise ValueError(
            "Never store credentials in experiment_spec.json. Remove "
            + ", ".join(forbidden)
            + " and use an environment variable or --prompt-api-key."
        )


def normalize_provider(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    return PROVIDER_ALIASES.get(normalized, normalized)


def validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("Provider base_url must use HTTPS, except for an HTTP loopback endpoint")
    if not parsed.netloc:
        raise ValueError("Provider base_url must be an absolute URL")
    return value


def provider_catalog() -> dict[str, dict[str, str | None]]:
    return {
        name: {
            "api_style": str(settings["api_style"]),
            "default_base_url": settings["base_url"],
            "default_api_key_env": str(settings["api_key_env"]),
            "model": "user supplied" if settings["default_model"] is None else str(settings["default_model"]),
        }
        for name, settings in PROVIDER_DEFAULTS.items()
    }


def resolve_provider_config(spec: dict[str, Any], args: argparse.Namespace) -> ProviderConfig:
    settings = spec.get("virtual_lab") or {}
    spec_provider = str(settings.get("provider", "deepseek"))
    provider = normalize_provider(args.provider or spec_provider)
    if provider not in PROVIDER_DEFAULTS:
        supported = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ValueError(f"Unsupported provider {provider!r}; choose one of: {supported}")
    defaults = PROVIDER_DEFAULTS[provider]
    provider_changed = bool(args.provider and normalize_provider(spec_provider) != provider)

    if args.model:
        model = args.model
    elif args.provider and normalize_provider(spec_provider) != provider:
        model = str(defaults.get("default_model") or "")
    else:
        model = str(settings.get("model") or defaults.get("default_model") or "")
    if not model:
        raise ValueError(f"A model identifier is required for provider {provider!r}")

    configured_base_url = None if provider_changed else settings.get("base_url")
    base_url = str(args.base_url or configured_base_url or defaults.get("base_url") or "")
    if not base_url:
        raise ValueError("A full base_url is required for openai_compatible providers")

    configured_api_key_env = None if provider_changed else settings.get("api_key_env")
    api_key_env = str(args.api_key_env or configured_api_key_env or defaults["api_key_env"])
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("api_key_env must be a valid environment-variable name")

    return ProviderConfig(
        provider=provider,
        api_style=str(defaults["api_style"]),
        model=model,
        base_url=validate_base_url(base_url),
        api_key_env=api_key_env,
    )


def resolve_api_key(config: ProviderConfig, prompt: bool) -> tuple[str | None, str | None]:
    candidates = [config.api_key_env]
    if config.api_key_env != "VIRTUAL_LAB_API_KEY":
        candidates.append("VIRTUAL_LAB_API_KEY")
    for variable in candidates:
        value = os.environ.get(variable)
        if value:
            return value, variable
    if prompt:
        value = getpass.getpass(f"Enter API key for {config.provider}: ").strip()
        if value:
            return value, "interactive prompt"
    return None, None


def build_client(config: ProviderConfig, api_key: str):
    if config.api_style == "anthropic":
        return AnthropicClient(api_key, config)
    if config.api_style == "google":
        return GoogleClient(api_key, config)
    return OpenAICompatibleClient(api_key, config)


def resolve_mode(spec: dict[str, Any], override: str | None, has_api_key: bool) -> str:
    mode = override or (spec.get("virtual_lab") or {}).get("mode", "auto")
    if mode == "auto":
        return "live" if has_api_key else "offline"
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


def generate_agents(client, spec: dict[str, Any], events: list[dict[str, Any]]) -> list[Agent]:
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
        agents.append(Agent(*required))
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
            f"**Provider:** `{results.get('virtual_lab', {}).get('provider', 'offline')}`",
            f"**Model:** `{results.get('virtual_lab', {}).get('model', 'offline-deterministic')}`",
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
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--mode", choices=("auto", "live", "offline"))
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--list-providers", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        help="copy the complete Markdown report into this folder",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.list_providers:
        print(json.dumps(provider_catalog(), indent=2))
        return
    if args.spec is None:
        parser.error("--spec is required unless --list-providers is used")

    spec_path = args.spec.expanduser().resolve()
    validated = pipeline_core.load_spec(spec_path)
    spec = json.loads(json.dumps(validated.raw))
    reject_inline_secrets(spec)
    provider_config = resolve_provider_config(spec, args)
    settings = spec.setdefault("virtual_lab", {})
    settings.update(
        {
            "provider": provider_config.provider,
            "model": provider_config.model,
            "base_url": provider_config.base_url,
            "api_key_env": provider_config.api_key_env,
        }
    )
    requested_mode = args.mode or settings.get("mode", "auto")
    api_key, credential_source = resolve_api_key(
        provider_config,
        prompt=args.prompt_api_key and requested_mode != "offline",
    )
    mode = resolve_mode(spec, args.mode, bool(api_key))
    if mode == "live":
        if not api_key:
            raise RuntimeError(
                f"Live mode requires {provider_config.api_key_env}, "
                "VIRTUAL_LAB_API_KEY, or --prompt-api-key"
            )
        client = build_client(provider_config, api_key)
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
    agents = generate_agents(client, shared_spec, events)
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
        "llm": {
            "mode": mode,
            "provider": provider_config.provider,
            "model": provider_config.model,
            "base_url": provider_config.base_url,
            "credential_source": credential_source if mode == "live" else None,
        },
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
        "provider": provider_config.provider,
        "model": provider_config.model,
        "base_url": provider_config.base_url,
        "credential_source": credential_source if mode == "live" else None,
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

    handoff_value = args.handoff_dir or (
        Path(spec.get("output", {}).get("handoff_directory"))
        if spec.get("output", {}).get("handoff_directory")
        else None
    )
    handoff_path = None
    if handoff_value:
        handoff_dir = handoff_value.expanduser().resolve()
        handoff_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = handoff_dir / (
            f"Virtual Lab Handoff - {safe_name(spec['experiment_name'])} - {run_id}.md"
        )
        shutil.copy2(report_path, handoff_path)

    print(
        json.dumps(
            {
                "status": "success",
                "mode": mode,
                "provider": provider_config.provider,
                "model": provider_config.model,
                "run_directory": str(run_dir),
                "report": str(report_path),
                "handoff_report": str(handoff_path) if handoff_path else None,
                "selected": results["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
