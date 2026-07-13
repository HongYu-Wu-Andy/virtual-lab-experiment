from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "run-virtual-lab-experiment" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_core


def base_spec() -> dict:
    return {
        "experiment_name": "safeguard_test",
        "description": "Exercise scientific and validation safeguards.",
        "dataset": {"path": "data.csv"},
        "features": [{"name": "x", "bounds": [0.0, 1.0]}],
        "targets": [
            {
                "name": "y",
                "goal": "maximize",
                "weight": 1.0,
                "expected_range": [0.0, 2.0],
            }
        ],
        "validation": {"test_fraction": 0.3, "cv_folds": 3, "random_seed": 42},
        "search": {
            "candidate_count": 100,
            "sensitivity_samples": 10,
            "decision_method": "achievement_scalarization",
        },
    }


class PipelineSafeguardTest(unittest.TestCase):
    def load(self, spec: dict) -> pipeline_core.ValidatedSpec:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            return pipeline_core.load_spec(path)

    def test_nonfinite_and_reversed_numeric_contracts_are_rejected(self) -> None:
        spec = base_spec()
        spec["targets"][0]["weight"] = "nan"
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.load(spec)

        spec = base_spec()
        spec["features"][0]["bounds"] = ["nan", 1.0]
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.load(spec)

        spec = base_spec()
        spec["targets"][0]["expected_range"] = [2.0, 1.0]
        with self.assertRaisesRegex(ValueError, "reversed"):
            self.load(spec)

    def test_constant_target_fails_finite_metric_requirement(self) -> None:
        spec = self.load(base_spec())
        data = pd.DataFrame({"x": np.linspace(0.0, 1.0, 30), "y": np.ones(30)})
        with self.assertRaisesRegex(ValueError, "non-finite validation metrics"):
            pipeline_core.fit_models(data, spec)

    def test_grouped_holdout_has_no_group_overlap(self) -> None:
        raw = base_spec()
        raw["validation"]["group_column"] = "batch"
        raw["search"]["model_families"] = ["KNN"]
        spec = self.load(raw)
        x = np.linspace(0.0, 1.0, 30)
        data = pd.DataFrame(
            {
                "x": x,
                "y": 2.0 * x + np.sin(x * 3.0) * 0.01,
                "batch": np.repeat(np.arange(10), 3),
            }
        )
        _, _, validation = pipeline_core.fit_models(data, spec)
        self.assertEqual(validation["holdout"]["strategy"], "GroupShuffleSplit")
        self.assertEqual(validation["holdout"]["group_overlap"], 0)

    def test_support_filter_removes_far_feature_combinations(self) -> None:
        observed = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        candidates = np.vstack([observed, np.array([[1.0, 0.0], [0.0, 1.0]])])
        kept, distances, threshold = pipeline_core.filter_supported_candidates(
            candidates,
            observed,
            {"a": (0.0, 1.0), "b": (0.0, 1.0)},
            ["a", "b"],
        )
        self.assertEqual(len(kept), len(observed))
        self.assertTrue(np.all(distances <= threshold))


if __name__ == "__main__":
    unittest.main()
