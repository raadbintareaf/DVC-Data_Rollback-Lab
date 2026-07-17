"""Measure the carbon footprint of two Random Forest setups on a real CSV dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


TASK_DIR = Path(__file__).resolve().parent
DATA_FILE = TASK_DIR / "winequality-red.csv"
EMISSIONS_FILE = TASK_DIR / "emissions.csv"
METRICS_FILE = TASK_DIR / "metrics.json"
RANDOM_STATE = 42


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the UCI Wine Quality CSV and create a binary quality target."""
    data = pd.read_csv(DATA_FILE, sep=";")
    features = data.drop(columns=["quality"])
    target = (data["quality"] >= 6).astype(int)
    return train_test_split(
        features.to_numpy(),
        target.to_numpy(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_model(n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
    """Build a Random Forest classifier for the selected training budget."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_experiment(
    name: str,
    n_estimators: int,
    max_depth: int | None,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int | str | None]:
    """Train one model setup while CodeCarbon measures the training impact."""
    model = build_model(n_estimators=n_estimators, max_depth=max_depth)
    tracker = EmissionsTracker(
        project_name=name,
        output_dir=str(TASK_DIR),
        output_file=EMISSIONS_FILE.name,
        measure_power_secs=1,
        save_to_file=True,
        log_level="error",
    )

    start_time = time.perf_counter()
    tracker.start()
    model.fit(x_train, y_train)
    emissions_kg = tracker.stop()
    duration_seconds = time.perf_counter() - start_time

    probabilities = model.predict_proba(x_test)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    return {
        "experiment": name,
        "dataset": "UCI Wine Quality Red Wine CSV",
        "dataset_file": DATA_FILE.name,
        "dataset_source": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "target_definition": "quality >= 6 is labeled as high quality",
        "architecture": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_accuracy": round(float(accuracy), 6),
        "test_log_loss": round(float(loss), 6),
        "co2_emitted_kg": round(float(emissions_kg or 0.0), 10),
        "duration_seconds": round(float(duration_seconds), 3),
    }


def add_carbon_roi(results: list[dict[str, float | int | str | None]]) -> None:
    """Add efficiency metrics and a compact interpretation."""
    baseline = results[0]
    comparison = results[1]

    for result in results:
        emissions = float(result["co2_emitted_kg"])
        accuracy = float(result["test_accuracy"])
        result["accuracy_per_kg_co2"] = (
            round(accuracy / emissions, 2) if emissions > 0 else None
        )

    accuracy_gain = float(comparison["test_accuracy"]) - float(baseline["test_accuracy"])
    emissions_gain = float(comparison["co2_emitted_kg"]) - float(baseline["co2_emitted_kg"])

    if emissions_gain > 0 and accuracy_gain <= 0:
        interpretation = (
            "The larger Random Forest emitted more CO2 without improving accuracy, so the "
            "smaller forest has the better Carbon ROI for this dataset."
        )
    elif emissions_gain > 0:
        interpretation = (
            "The larger Random Forest improved accuracy, but the added CO2 should be "
            "justified against the size of that gain."
        )
    else:
        interpretation = (
            "The larger Random Forest did not emit more CO2 in this run, so both quality and "
            "emissions should be rechecked on a larger workload before deployment."
        )

    results.append(
        {
            "experiment": "comparison_summary",
            "accuracy_gain_larger_minus_smaller": round(accuracy_gain, 6),
            "co2_gain_larger_minus_smaller_kg": round(emissions_gain, 10),
            "interpretation": interpretation,
        }
    )


def main() -> None:
    """Run both Green AI experiments and save their metrics."""
    if EMISSIONS_FILE.exists():
        EMISSIONS_FILE.unlink()
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    x_train, x_test, y_train, y_test = load_dataset()
    experiments = [
        ("small_random_forest_50_trees", 50, 8),
        ("larger_random_forest_400_trees", 400, None),
    ]

    results = [
        run_experiment(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        for name, n_estimators, max_depth in experiments
    ]
    add_carbon_roi(results)

    METRICS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results[:2]:
        print(
            f"{result['experiment']} | Accuracy: {result['test_accuracy']} | "
            f"Loss: {result['test_log_loss']} | "
            f"CO2 Emitted: {result['co2_emitted_kg']} kg"
        )
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"CodeCarbon emissions saved to: {EMISSIONS_FILE}")


if __name__ == "__main__":
    main()
