from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "run-virtual-lab-experiment"
RUNNER = SKILL / "scripts" / "run_virtual_lab.py"
EXAMPLE_SPEC = ROOT / "examples" / "experiment_spec.json"


class VirtualLabSmokeTest(unittest.TestCase):
    def test_published_files_do_not_contain_private_values(self) -> None:
        forbidden = (
            "/Users/" + "andywu",
            "ALL_" + "data.csv",
            "gh" + "p_",
            "hello-" + "world01011",
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for value in forbidden:
                self.assertNotIn(value, text, f"private value found in {path}")

    def test_manifest_and_marketplace(self) -> None:
        manifest = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
        marketplace = json.loads(
            (ROOT / ".agents" / "plugins" / "marketplace.json").read_text()
        )
        self.assertEqual(manifest["name"], "virtual-lab-experiment")
        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(marketplace["plugins"][0]["name"], manifest["name"])

    def test_offline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
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
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            run_dir = Path(summary["run_directory"])
            required = [
                "agents.json",
                "conversations.json",
                "conversations.md",
                "dataset_profile.json",
                "experiment_spec.json",
                "generated_pipeline.py",
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
            self.assertEqual(results["virtual_lab"]["mode"], "offline")
            self.assertEqual(len(results["virtual_lab"]["generated_agents"]), 3)
            self.assertIn(results["decision_method"], {
                "achievement_scalarization",
                "weighted_sum",
                "distance_to_expectation",
            })


if __name__ == "__main__":
    unittest.main()
