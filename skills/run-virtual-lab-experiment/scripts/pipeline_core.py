#!/usr/bin/env python3
"""Verified generic tabular multi-target regression and candidate-selection pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ValidatedSpec:
    raw: dict[str, Any]
    features: tuple[dict[str, Any], ...]
    targets: tuple[dict[str, Any], ...]
    seed: int
    test_fraction: float
    cv_folds: int
    candidate_count: int
    sensitivity_samples: int
    decision_method: str
    model_families: tuple[str, ...]
    candidate_strategy: str


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def load_spec(path: Path) -> ValidatedSpec:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment spec not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not str(raw.get("experiment_name", "")).strip():
        raise ValueError("experiment_name is required")
    if not str(raw.get("description", "")).strip():
        raise ValueError("description is required")
    dataset = raw.get("dataset") or {}
    if not str(dataset.get("path", "")).strip():
        raise ValueError("dataset.path is required")
    features = tuple(raw.get("features") or ())
    targets = tuple(raw.get("targets") or ())
    if not features:
        raise ValueError("At least one feature is required")
    if not targets:
        raise ValueError("At least one target is required")
    feature_names = [str(item.get("name", "")).strip() for item in features]
    target_names = [str(item.get("name", "")).strip() for item in targets]
    if any(not name for name in feature_names + target_names):
        raise ValueError("Every feature and target requires a name")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique")
    if len(set(target_names)) != len(target_names):
        raise ValueError("Target names must be unique")
    overlap = set(feature_names) & set(target_names)
    if overlap:
        raise ValueError(f"Columns cannot be both features and targets: {sorted(overlap)}")

    for feature in features:
        bounds = feature.get("bounds")
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or bounds[0] is None or bounds[1] is None:
                raise ValueError(f"Feature {feature['name']} bounds must contain two finite values")
            lower = finite_float(bounds[0], f"Feature {feature['name']} lower bound")
            upper = finite_float(bounds[1], f"Feature {feature['name']} upper bound")
            if lower >= upper:
                raise ValueError(f"Feature {feature['name']} lower bound must be below upper bound")

    allowed_goals = {"minimize", "maximize", "target"}
    for target in targets:
        goal = target.get("goal")
        if goal not in allowed_goals:
            raise ValueError(f"Target {target['name']} goal must be one of {sorted(allowed_goals)}")
        weight = finite_float(target.get("weight", 1.0), f"Target {target['name']} weight")
        if weight <= 0:
            raise ValueError(f"Target {target['name']} weight must be positive")
        expected = target.get("expected_range")
        if expected is not None:
            if not isinstance(expected, (list, tuple)) or len(expected) != 2:
                raise ValueError(f"Target {target['name']} expected_range must have two entries")
            lower = None if expected[0] is None else finite_float(
                expected[0], f"Target {target['name']} expected lower bound"
            )
            upper = None if expected[1] is None else finite_float(
                expected[1], f"Target {target['name']} expected upper bound"
            )
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"Target {target['name']} expected_range is reversed")
        if target.get("target_value") is not None:
            finite_float(target["target_value"], f"Target {target['name']} target_value")
        if goal == "target" and target.get("target_value") is None:
            if not expected or expected[0] is None or expected[1] is None:
                raise ValueError(
                    f"Target {target['name']} needs target_value or a closed expected_range"
                )

    feature_bounds = {item["name"]: item.get("bounds") for item in features}
    for constraint in raw.get("constraints") or []:
        name = constraint.get("feature")
        if name not in feature_bounds:
            raise ValueError(f"Constraint references unknown feature: {name}")
        if constraint.get("min") is None and constraint.get("max") is None:
            raise ValueError(f"Constraint for {name} requires min or max")
        lower = (
            None
            if constraint.get("min") is None
            else finite_float(constraint["min"], f"Constraint {name} min")
        )
        upper = (
            None
            if constraint.get("max") is None
            else finite_float(constraint["max"], f"Constraint {name} max")
        )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"Constraint for {name} has an empty or reversed interval")
        configured = feature_bounds[name]
        if configured:
            feature_lower, feature_upper = map(float, configured)
            if lower is not None and lower < feature_lower:
                raise ValueError(f"Constraint {name} min lies outside the feature bounds")
            if upper is not None and upper > feature_upper:
                raise ValueError(f"Constraint {name} max lies outside the feature bounds")

    validation = raw.get("validation") or {}
    search = raw.get("search") or {}
    test_fraction = finite_float(validation.get("test_fraction", 0.3), "validation.test_fraction")
    if not 0 < test_fraction < 1:
        raise ValueError("validation.test_fraction must be between zero and one")
    cv_folds = int(validation.get("cv_folds", 5))
    if cv_folds < 2:
        raise ValueError("validation.cv_folds must be at least two")
    candidate_count = int(search.get("candidate_count", 10_000))
    if not 100 <= candidate_count <= 1_000_000:
        raise ValueError("search.candidate_count must be between 100 and 1,000,000")
    sensitivity_samples = int(search.get("sensitivity_samples", 300))
    if not 10 <= sensitivity_samples <= 100_000:
        raise ValueError("search.sensitivity_samples must be between 10 and 100,000")
    decision_method = str(search.get("decision_method", "achievement_scalarization"))
    allowed_methods = {
        "auto",
        "achievement_scalarization",
        "weighted_sum",
        "distance_to_expectation",
    }
    if decision_method not in allowed_methods:
        raise ValueError(f"Unsupported search.decision_method: {decision_method}")
    supported_models = tuple(model_factories(int(validation.get("random_seed", 42))))
    configured_models = tuple(search.get("model_families") or supported_models)
    if not configured_models or len(set(configured_models)) != len(configured_models):
        raise ValueError("search.model_families must contain unique supported model names")
    unknown_models = sorted(set(configured_models) - set(supported_models))
    if unknown_models:
        raise ValueError(f"Unsupported search.model_families: {unknown_models}")
    candidate_strategy = str(search.get("candidate_strategy", "latin_hypercube_plus_observed"))
    if candidate_strategy not in {"latin_hypercube_plus_observed", "random_uniform_plus_observed"}:
        raise ValueError(f"Unsupported search.candidate_strategy: {candidate_strategy}")
    return ValidatedSpec(
        raw=raw,
        features=features,
        targets=targets,
        seed=int(validation.get("random_seed", 42)),
        test_fraction=test_fraction,
        cv_folds=cv_folds,
        candidate_count=candidate_count,
        sensitivity_samples=sensitivity_samples,
        decision_method=(
            "achievement_scalarization" if decision_method == "auto" else decision_method
        ),
        model_families=configured_models,
        candidate_strategy=candidate_strategy,
    )


def dataset_path(spec_path: Path, spec: ValidatedSpec) -> Path:
    path = Path(spec.raw["dataset"]["path"]).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError("Only CSV and TSV datasets are supported")


def validate_dataset(data: pd.DataFrame, spec: ValidatedSpec) -> pd.DataFrame:
    names = [item["name"] for item in spec.features + spec.targets]
    group_column = (spec.raw.get("validation") or {}).get("group_column")
    if group_column:
        names.append(group_column)
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    selected = data.loc[:, list(dict.fromkeys(names))].copy()
    for name in [item["name"] for item in spec.features + spec.targets]:
        selected[name] = pd.to_numeric(selected[name], errors="raise")
    if selected[[item["name"] for item in spec.features + spec.targets]].isna().any().any():
        raise ValueError("Selected feature or target columns contain missing values")
    if group_column and selected[group_column].isna().any():
        raise ValueError("validation.group_column contains missing values")
    numeric = selected[[item["name"] for item in spec.features + spec.targets]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Selected feature or target columns contain non-finite values")
    if len(selected) < max(20, spec.cv_folds * 2):
        raise ValueError("Dataset is too small for the requested validation settings")
    return selected


def resolve_bounds(data: pd.DataFrame, spec: ValidatedSpec) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for feature in spec.features:
        name = feature["name"]
        configured = feature.get("bounds")
        lower = float(configured[0]) if configured else float(data[name].min())
        upper = float(configured[1]) if configured else float(data[name].max())
        observed_lower = float(data[name].min())
        observed_upper = float(data[name].max())
        if lower < observed_lower or upper > observed_upper:
            raise ValueError(
                f"Feature {name} bounds [{lower}, {upper}] extrapolate beyond observed "
                f"[{observed_lower}, {observed_upper}]"
            )
        bounds[name] = (lower, upper)
    for constraint in spec.raw.get("constraints") or []:
        name = constraint.get("feature")
        if name not in bounds:
            raise ValueError(f"Constraint references unknown feature: {name}")
        lower, upper = bounds[name]
        if constraint.get("min") is not None:
            lower = max(lower, float(constraint["min"]))
        if constraint.get("max") is not None:
            upper = min(upper, float(constraint["max"]))
        if lower >= upper:
            raise ValueError(f"Constraints create an empty interval for {name}")
        bounds[name] = (lower, upper)
    return bounds


def model_factories(seed: int):
    return {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=180, min_samples_leaf=1, n_jobs=-1, random_state=seed
        ),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=1, n_jobs=-1, random_state=seed
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=seed),
        "KNN": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray, span: float) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "nrmse": rmse / span if span > 0 else float("inf"),
    }


def fit_models(
    data: pd.DataFrame,
    spec: ValidatedSpec,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    feature_names = [item["name"] for item in spec.features]
    x = data[feature_names].to_numpy(float)
    indices = np.arange(len(data))
    group_column = (spec.raw.get("validation") or {}).get("group_column")
    if group_column:
        groups = data[group_column].to_numpy()
        group_reason = f"configured group column {group_column}"
    elif data.duplicated(subset=feature_names, keep=False).any():
        groups = pd.util.hash_pandas_object(data[feature_names], index=False).to_numpy()
        group_reason = "identical feature settings"
    else:
        groups = None
        group_reason = None

    if groups is not None:
        unique_groups = np.unique(groups)
        folds = min(spec.cv_folds, len(unique_groups))
        if folds < 2:
            raise ValueError("Grouped validation requires at least two unique groups")
        holdout = GroupShuffleSplit(
            n_splits=1, test_size=spec.test_fraction, random_state=spec.seed
        )
        train_index, test_index = next(holdout.split(x, groups=groups))
        if set(groups[train_index]) & set(groups[test_index]):
            raise RuntimeError("Grouped holdout contains overlapping groups")
        cv_description = f"GroupKFold({folds}) by {group_reason}; grouped holdout"
        holdout_description = {
            "strategy": "GroupShuffleSplit",
            "group_reason": group_reason,
            "train_groups": int(len(np.unique(groups[train_index]))),
            "test_groups": int(len(np.unique(groups[test_index]))),
            "group_overlap": 0,
        }
    else:
        train_index, test_index = train_test_split(
            indices,
            test_size=spec.test_fraction,
            random_state=spec.seed,
            shuffle=True,
        )
        cv_description = f"shuffled KFold({spec.cv_folds}); shuffled holdout"
        holdout_description = {
            "strategy": "train_test_split",
            "group_reason": None,
            "group_overlap": None,
        }

    all_factories = model_factories(spec.seed)
    factories = {name: all_factories[name] for name in spec.model_families}
    fitted: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for target in spec.targets:
        name = target["name"]
        y = data[name].to_numpy(float)
        span = float(y.max() - y.min())
        target_rows = []
        for model_name, factory in factories.items():
            holdout_model = factory()
            holdout_model.fit(x[train_index], y[train_index])
            holdout = metrics(y[test_index], holdout_model.predict(x[test_index]), span)
            # Recreate GroupKFold splits for each model because split generators are consumable.
            if groups is not None:
                cv_object = GroupKFold(n_splits=min(spec.cv_folds, len(np.unique(groups))))
                predicted = cross_val_predict(
                    factory(), x, y, groups=groups, cv=cv_object, n_jobs=1
                )
            else:
                cv_object = KFold(n_splits=spec.cv_folds, shuffle=True, random_state=spec.seed)
                predicted = cross_val_predict(factory(), x, y, cv=cv_object, n_jobs=1)
            cross_validation = metrics(y, predicted, span)
            residual_q95 = float(np.quantile(np.abs(y - predicted), 0.95))
            row = {
                "target": name,
                "model": model_name,
                **{f"holdout_{key}": value for key, value in holdout.items()},
                **{f"cv_{key}": value for key, value in cross_validation.items()},
                "cv_residual_q95": residual_q95,
            }
            rows.append(row)
            target_rows.append(row)
        best = min(target_rows, key=lambda item: (item["cv_nrmse"], -item["cv_r2"]))
        numeric_metrics = [
            value
            for key, value in best.items()
            if key.startswith(("holdout_", "cv_")) and isinstance(value, (int, float))
        ]
        if not all(math.isfinite(float(value)) for value in numeric_metrics):
            raise ValueError(
                f"Target {name} produced non-finite validation metrics; check constant targets, "
                "sample size, grouping, and data quality"
            )
        final_model = factories[best["model"]]()
        final_model.fit(x, y)
        fitted[name] = final_model
        selection[name] = {
            "model": best["model"],
            "cv_r2": best["cv_r2"],
            "cv_mae": best["cv_mae"],
            "cv_nrmse": best["cv_nrmse"],
            "cv_rmse": best["cv_rmse"],
            "prediction_error_q95": best["cv_residual_q95"],
            "holdout_r2": best["holdout_r2"],
            "holdout_mae": best["holdout_mae"],
            "holdout_nrmse": best["holdout_nrmse"],
        }
    return fitted, pd.DataFrame(rows), {
        "models": selection,
        "cv": cv_description,
        "holdout": holdout_description,
    }


def feasible_mask(values: np.ndarray, feature_names: list[str], bounds) -> np.ndarray:
    mask = np.ones(len(values), dtype=bool)
    for index, name in enumerate(feature_names):
        lower, upper = bounds[name]
        mask &= values[:, index] >= lower
        mask &= values[:, index] <= upper
    return mask


def candidates(data: pd.DataFrame, spec: ValidatedSpec, bounds) -> np.ndarray:
    feature_names = [item["name"] for item in spec.features]
    observed = data[feature_names].to_numpy(float)
    observed = observed[feasible_mask(observed, feature_names, bounds)]
    lower = np.array([bounds[name][0] for name in feature_names])
    upper = np.array([bounds[name][1] for name in feature_names])
    rng = np.random.default_rng(spec.seed)
    if spec.candidate_strategy == "latin_hypercube_plus_observed":
        dimensions = len(feature_names)
        unit = np.empty((spec.candidate_count, dimensions), dtype=float)
        for column in range(dimensions):
            unit[:, column] = (rng.permutation(spec.candidate_count) + rng.random(spec.candidate_count)) / spec.candidate_count
        sampled = lower + unit * (upper - lower)
    else:
        sampled = rng.uniform(lower, upper, size=(spec.candidate_count, len(feature_names)))
    combined = np.vstack([observed, sampled])
    rounded = np.round(combined, 10)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    return combined[np.sort(unique)]


def filter_supported_candidates(
    candidate_values: np.ndarray,
    observed_values: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reject candidate combinations far outside the observed multivariate support."""
    lower = np.array([bounds[name][0] for name in feature_names], dtype=float)
    span = np.array([bounds[name][1] - bounds[name][0] for name in feature_names], dtype=float)
    span[span == 0] = 1.0
    observed_scaled = (observed_values - lower) / span
    candidate_scaled = (candidate_values - lower) / span
    if len(observed_scaled) < 2:
        raise ValueError("At least two observations are required for support-distance screening")
    neighbor_model = NearestNeighbors(n_neighbors=2).fit(observed_scaled)
    observed_distances = neighbor_model.kneighbors(observed_scaled, return_distance=True)[0][:, 1]
    threshold = max(float(np.quantile(observed_distances, 0.95) * 1.5), np.finfo(float).eps)
    candidate_model = NearestNeighbors(n_neighbors=1).fit(observed_scaled)
    distances = candidate_model.kneighbors(candidate_scaled, return_distance=True)[0][:, 0]
    keep = distances <= threshold
    if not np.any(keep):
        raise ValueError("No candidate remains after multivariate support-distance screening")
    return candidate_values[keep], distances[keep], threshold


def target_reference(target: dict[str, Any]) -> float:
    if target.get("target_value") is not None:
        return float(target["target_value"])
    expected = target.get("expected_range")
    if expected and expected[0] is not None and expected[1] is not None:
        return 0.5 * (float(expected[0]) + float(expected[1]))
    raise ValueError(f"Target goal for {target['name']} requires a reference value")


def minimization_values(
    predictions: np.ndarray,
    spec: ValidatedSpec,
    uncertainty: np.ndarray | None = None,
) -> np.ndarray:
    columns = []
    uncertainty = np.zeros(predictions.shape[1]) if uncertainty is None else uncertainty
    for index, target in enumerate(spec.targets):
        values = predictions[:, index]
        target_uncertainty = uncertainty[:, index] if uncertainty.ndim == 2 else uncertainty[index]
        if target["goal"] == "minimize":
            columns.append(values + target_uncertainty)
        elif target["goal"] == "maximize":
            columns.append(-(values - target_uncertainty))
        else:
            columns.append(np.abs(values - target_reference(target)) + target_uncertainty)
    return np.column_stack(columns)


def pareto_indices(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values[:, 0], kind="stable")
    front: list[int] = []
    for candidate_index in order:
        candidate = values[candidate_index]
        if front:
            current = values[np.asarray(front)]
            if np.any(np.all(current <= candidate, axis=1) & np.any(current < candidate, axis=1)):
                continue
            keep = ~(
                np.all(candidate <= current, axis=1) & np.any(candidate < current, axis=1)
            )
            front = [front[index] for index in np.flatnonzero(keep)]
        front.append(int(candidate_index))
    return np.asarray(front)


def normalized(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    span[span == 0] = 1.0
    return (values - minimum) / span


def expectation_distance(predictions: np.ndarray, spec: ValidatedSpec) -> np.ndarray:
    distances = np.zeros_like(predictions, dtype=float)
    for index, target in enumerate(spec.targets):
        values = predictions[:, index]
        expected = target.get("expected_range")
        scale = max(float(values.max() - values.min()), np.finfo(float).eps)
        if expected:
            lower, upper = expected
            below = np.maximum((float(lower) - values) if lower is not None else 0.0, 0.0)
            above = np.maximum((values - float(upper)) if upper is not None else 0.0, 0.0)
            distances[:, index] = (below + above) / scale
        elif target["goal"] == "target":
            distances[:, index] = np.abs(values - target_reference(target)) / scale
        else:
            objective = values if target["goal"] == "minimize" else -values
            distances[:, index] = (objective - objective.min()) / max(
                float(objective.max() - objective.min()), np.finfo(float).eps
            )
    return distances


def score_candidates(
    pareto_objectives: np.ndarray,
    pareto_predictions: np.ndarray,
    spec: ValidatedSpec,
    weights: np.ndarray,
) -> np.ndarray:
    normalized_objectives = normalized(pareto_objectives)
    if spec.decision_method == "weighted_sum":
        return np.sum(normalized_objectives * weights, axis=1)
    if spec.decision_method == "distance_to_expectation":
        return np.sum(expectation_distance(pareto_predictions, spec) * weights, axis=1)
    weighted = normalized_objectives * weights
    return np.max(weighted, axis=1) + 0.05 * np.sum(weighted, axis=1)


def check_expectation(value: float, target: dict[str, Any]) -> bool | None:
    expected = target.get("expected_range")
    if not expected:
        return None
    lower, upper = expected
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def run_pipeline(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    data_path = dataset_path(spec_path, spec)
    raw_data = read_dataset(data_path)
    data = validate_dataset(raw_data, spec)
    bounds = resolve_bounds(data, spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    models, metric_frame, selection = fit_models(data, spec)
    metric_frame.to_csv(output_dir / "metrics.csv", index=False)
    feature_names = [item["name"] for item in spec.features]
    target_names = [item["name"] for item in spec.targets]
    candidate_values = candidates(data, spec, bounds)
    observed_values = data[feature_names].to_numpy(float)
    observed_values = observed_values[feasible_mask(observed_values, feature_names, bounds)]
    candidate_values, support_distances, support_threshold = filter_supported_candidates(
        candidate_values, observed_values, bounds, feature_names
    )
    predictions = np.column_stack(
        [models[name].predict(candidate_values) for name in target_names]
    )
    base_uncertainty = np.array(
        [selection["models"][name]["prediction_error_q95"] for name in target_names],
        dtype=float,
    )
    uncertainty = base_uncertainty[None, :] * (
        1.0 + support_distances[:, None] / support_threshold
    )
    objectives = minimization_values(predictions, spec, uncertainty)
    front_index = pareto_indices(objectives)
    front_x = candidate_values[front_index]
    front_y = predictions[front_index]
    front_f = objectives[front_index]
    front_support_distances = support_distances[front_index]
    front_uncertainty = uncertainty[front_index]

    raw_weights = np.array([float(item.get("weight", 1.0)) for item in spec.targets])
    weights = raw_weights / raw_weights.sum()
    decision_scores = score_candidates(front_f, front_y, spec, weights)
    winner = int(np.argmin(decision_scores))

    rng = np.random.default_rng(spec.seed + 101)
    sampled_weights = rng.dirichlet(np.ones(len(spec.targets)), size=spec.sensitivity_samples)
    winner_counts = np.zeros(len(front_x), dtype=int)
    for sampled_weight in sampled_weights:
        index = int(np.argmin(score_candidates(front_f, front_y, spec, sampled_weight)))
        winner_counts[index] += 1

    selected_x = front_x[winner]
    selected_y = front_y[winner]
    selected_uncertainty = front_uncertainty[winner]
    expectation_checks = {
        target["name"]: check_expectation(float(selected_y[index]), target)
        for index, target in enumerate(spec.targets)
    }
    check_values = [value for value in expectation_checks.values() if value is not None]
    expectations_met = bool(all(check_values)) if check_values else None
    prediction_intervals = {
        name: {
            "lower": float(selected_y[index] - selected_uncertainty[index]),
            "upper": float(selected_y[index] + selected_uncertainty[index]),
            "method": "cross_validated_absolute_residual_q95_scaled_by_support_distance",
        }
        for index, name in enumerate(target_names)
    }
    conservative_expectation_checks = {
        target["name"]: (
            None
            if not target.get("expected_range")
            else check_expectation(
                prediction_intervals[target["name"]]["lower"],
                {
                    **target,
                    "expected_range": [target["expected_range"][0], None],
                },
            )
            and check_expectation(
                prediction_intervals[target["name"]]["upper"],
                {
                    **target,
                    "expected_range": [None, target["expected_range"][1]],
                },
            )
        )
        for target in spec.targets
    }
    conservative_values = [
        value for value in conservative_expectation_checks.values() if value is not None
    ]
    conservative_expectations_met = (
        bool(all(conservative_values)) if conservative_values else None
    )

    pareto = pd.DataFrame(front_x, columns=feature_names)
    for index, name in enumerate(target_names):
        pareto[f"predicted_{name}"] = front_y[:, index]
        pareto[f"prediction_error_q95_{name}"] = front_uncertainty[:, index]
    pareto["support_distance"] = front_support_distances
    pareto["decision_score"] = decision_scores
    pareto["sensitivity_wins"] = winner_counts
    pareto["selected"] = False
    pareto.loc[winner, "selected"] = True
    pareto.to_csv(output_dir / "pareto_front.csv", index=False)

    selected_row = {name: float(selected_x[index]) for index, name in enumerate(feature_names)}
    selected_row.update(
        {f"predicted_{name}": float(selected_y[index]) for index, name in enumerate(target_names)}
    )
    for name in target_names:
        selected_row[f"prediction_interval_lower_{name}"] = prediction_intervals[name]["lower"]
        selected_row[f"prediction_interval_upper_{name}"] = prediction_intervals[name]["upper"]
    selected_row["support_distance"] = float(front_support_distances[winner])
    selected_row.update(
        {
            f"expectation_met_{name}": expectation_checks[name]
            for name in target_names
        }
    )
    selected_row.update(
        {
            f"conservative_expectation_met_{name}": conservative_expectation_checks[name]
            for name in target_names
        }
    )
    pd.DataFrame([selected_row]).to_csv(output_dir / "selected_result.csv", index=False)

    profile = {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_feature_settings": int(data.duplicated(subset=feature_names, keep=False).sum()),
        "feature_summary": data[feature_names].describe().to_dict(),
        "target_summary": data[target_names].describe().to_dict(),
        "spearman": data[feature_names + target_names].corr(method="spearman").to_dict(),
    }
    (output_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )

    results = {
        "status": "success",
        "experiment_name": spec.raw["experiment_name"],
        "dataset": {"path": str(data_path), "sha256": _sha256(data_path), "rows": len(data)},
        "features": feature_names,
        "targets": [
            {
                "name": item["name"],
                "goal": item["goal"],
                "weight": float(item.get("weight", 1.0)),
                "expected_range": item.get("expected_range"),
                "target_value": item.get("target_value"),
            }
            for item in spec.targets
        ],
        "feature_bounds": {name: list(value) for name, value in bounds.items()},
        "validation": selection,
        "model_families": list(spec.model_families),
        "candidate_strategy": spec.candidate_strategy,
        "candidate_count": int(len(candidate_values)),
        "support_distance_threshold": support_threshold,
        "pareto_count": int(len(front_x)),
        "decision_method": spec.decision_method,
        "decision_weights": {name: float(weights[index]) for index, name in enumerate(target_names)},
        "selected": {
            "features": {name: float(selected_x[index]) for index, name in enumerate(feature_names)},
            "predictions": {name: float(selected_y[index]) for index, name in enumerate(target_names)},
            "prediction_intervals": prediction_intervals,
            "expectation_checks": expectation_checks,
            "conservative_expectation_checks": conservative_expectation_checks,
            "expectations_met": expectations_met,
            "conservative_expectations_met": conservative_expectations_met,
            "decision_score": float(decision_scores[winner]),
            "support_distance": float(front_support_distances[winner]),
            "sensitivity_win_fraction": float(winner_counts[winner] / spec.sensitivity_samples),
        },
        "sensitivity": {
            "samples": spec.sensitivity_samples,
            "distinct_winners": int(np.count_nonzero(winner_counts)),
            "maximum_win_fraction": float(winner_counts.max() / spec.sensitivity_samples),
        },
        "limitations": [
            "Predictions are screening estimates and require real experimental validation.",
            "Candidate search is limited to observed or explicitly bounded numeric feature space.",
            "Candidates outside the observed multivariate support-distance threshold are rejected.",
            "Prediction intervals use cross-validated residual quantiles and are screening intervals, not guarantees.",
            "Random or grouped cross-validation does not establish causality.",
            "Version 0.3 supports numeric tabular regression targets only.",
        ],
        "artifacts": {
            "metrics": "metrics.csv",
            "pareto_front": "pareto_front.csv",
            "selected_result": "selected_result.csv",
            "dataset_profile": "dataset_profile.json",
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_pipeline(args.spec.resolve(), args.output_dir.resolve())
    print(json.dumps(result["selected"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
