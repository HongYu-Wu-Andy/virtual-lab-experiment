from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "run-virtual-lab-experiment"
RUNNER = SKILL / "scripts" / "run_virtual_lab.py"
EXAMPLE_SPEC = ROOT / "examples" / "experiment_spec.json"
sys.path.insert(0, str(SKILL / "scripts"))

import run_virtual_lab


class VirtualLabSmokeTest(unittest.TestCase):
    def test_published_files_do_not_contain_private_values(self) -> None:
        forbidden = (
            "/Users/" + "andywu",
            "ALL_" + "data.csv",
            "gh" + "p_",
            "hello-" + "world01011",
            "HongYu-" + "Wu-Andy",
            "Hong-Yu-" + "Wu-Andy",
        )
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode().split("\0")
        for relative in tracked:
            if not relative:
                continue
            path = ROOT / relative
            text = path.read_text(encoding="utf-8", errors="ignore")
            for value in forbidden:
                self.assertNotIn(value, text, f"private value found in {path}")

    def test_public_files_use_platform_neutral_handoff_language(self) -> None:
        platform_name = "ob" + "sidian"
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode().split("\0")
        for relative in tracked:
            if not relative:
                continue
            path = ROOT / relative
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            self.assertNotIn(platform_name, text, f"platform-specific term found in {path}")

    def test_manifest_and_marketplace(self) -> None:
        manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
        marketplace = json.loads(
            (ROOT / ".agents" / "plugins" / "marketplace.json").read_text()
        )
        self.assertEqual(manifest["name"], "virtual-lab-experiment")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(marketplace["plugins"][0]["name"], manifest["name"])
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], project["project"]["version"])

    def test_public_release_files_and_attribution(self) -> None:
        required = [
            "CITATION.cff",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "pyproject.toml",
            "scripts/check_secrets.py",
            ".github/workflows/ci.yml",
            ".github/dependabot.yml",
            "docs/assets/social-preview.png",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("10.1038/s41586-025-09442-9", readme)
        self.assertIn("10.1016/j.mattod.2025.06.031", readme)
        self.assertNotIn("Install from the private repository", readme)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('virtual-lab-experiment = "run_virtual_lab:main"', pyproject)

    def test_offline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            handoff_dir = Path(temporary) / "markdown-handoff"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--spec",
                    str(EXAMPLE_SPEC),
                    "--mode",
                    "offline",
                    "--quick",
                    "--output-dir",
                    temporary,
                    "--handoff-dir",
                    str(handoff_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            handoff_report = Path(summary["handoff_report"])
            self.assertTrue(handoff_report.is_file())
            self.assertEqual(handoff_report.parent, handoff_dir.resolve())
            self.assertEqual(handoff_report.suffix, ".md")
            run_dir = Path(summary["run_directory"])
            required = [
                "agents.json",
                "analysis_plan.json",
                "conversations.json",
                "conversations.md",
                "dataset_profile.json",
                "experiment_spec.json",
                "executed_pipeline.py",
                "execution.json",
                "virtual_lab_report.md",
                "results/metrics.csv",
                "results/pareto_front.csv",
                "results/selected_result.csv",
                "results/results.json",
            ]
            for relative in required:
                self.assertTrue((run_dir / relative).is_file(), relative)

            results = json.loads((run_dir / "results" / "results.json").read_text())
            conversations = json.loads((run_dir / "conversations.json").read_text())
            self.assertTrue(
                any(item["meeting"].startswith("individual_") for item in conversations)
            )
            self.assertEqual(results["virtual_lab"]["mode"], "offline")
            self.assertEqual(len(results["virtual_lab"]["generated_agents"]), 3)
            self.assertIn("prediction_intervals", results["selected"])
            self.assertIn("support_distance", results["selected"])
            self.assertIn("conservative_expectations_met", results["selected"])
            self.assertEqual(
                results["candidate_strategy"], "latin_hypercube_plus_observed"
            )
            self.assertIn(results["decision_method"], {
                "achievement_scalarization",
                "weighted_sum",
                "distance_to_expectation",
            })

    def test_failed_pipeline_retains_conversations_and_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "constant.csv"
            pd.DataFrame(
                {"x": np.linspace(0.0, 1.0, 24), "constant_target": np.ones(24)}
            ).to_csv(data_path, index=False)
            spec = {
                "experiment_name": "expected_failure",
                "description": "Verify failure provenance.",
                "dataset": {"path": str(data_path)},
                "features": [{"name": "x", "bounds": [0.0, 1.0]}],
                "targets": [
                    {"name": "constant_target", "goal": "maximize", "weight": 1.0}
                ],
                "validation": {"test_fraction": 0.3, "cv_folds": 3, "random_seed": 42},
                "search": {"candidate_count": 100, "sensitivity_samples": 10},
                "virtual_lab": {"mode": "offline", "independent_runs": 1, "meeting_rounds": 1},
                "output": {"directory": str(root / "output")},
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--spec",
                    str(spec_path),
                    "--mode",
                    "offline",
                    "--quick",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            failure_files = list((root / "output").glob("**/failure.json"))
            self.assertEqual(len(failure_files), 1)
            run_dir = failure_files[0].parent
            self.assertTrue((run_dir / "failure_report.md").is_file())
            conversations = json.loads((run_dir / "conversations.json").read_text())
            self.assertGreater(len(conversations), 0)

    def test_failed_final_review_marks_pipeline_result_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
            spec["dataset"]["path"] = str(ROOT / "examples" / "synthetic_experiment.csv")
            spec["output"]["directory"] = str(root / "output")
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            original_team_meeting = run_virtual_lab.team_meeting

            def fail_final_review(client, name, *args, **kwargs):
                if name == "final_review":
                    raise RuntimeError("simulated final review failure")
                return original_team_meeting(client, name, *args, **kwargs)

            argv = [
                "virtual-lab-experiment",
                "--spec",
                str(spec_path),
                "--mode",
                "offline",
                "--quick",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_virtual_lab, "team_meeting", side_effect=fail_final_review
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated final review failure"):
                    run_virtual_lab.main()

            results_files = list((root / "output").glob("**/results/results.json"))
            self.assertEqual(len(results_files), 1)
            results = json.loads(results_files[0].read_text(encoding="utf-8"))
            self.assertEqual(results["status"], "failed")
            self.assertEqual(results["pipeline_status"], "success")
            self.assertTrue((results_files[0].parents[1] / "failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
