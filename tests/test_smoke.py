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
