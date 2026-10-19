"""Golden regression tests (2.1, 2.2, 2.8) and their comparison helpers."""

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _golden_cases import (
    GoldenCaseError,
    case_paths,
    compare_result,
    describe_changes,
    load_case,
)

from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

METHODS = ("pls", "svr")
METRICS = ("r2", "mse", "mae", "rmse")


@pytest.fixture(scope="module")
def detector_cache() -> dict[str, EnsembleOutlierSampleDetector]:
    """Hold fitted detectors by case name so each case is fitted once."""
    return {}


@pytest.fixture
def fitted_detector(
    golden_case: dict[str, Any],
    reference_data: tuple[pd.DataFrame, pd.Series],
    detector_cache: dict[str, EnsembleOutlierSampleDetector],
) -> EnsembleOutlierSampleDetector:
    """Return the detector fitted with the case parameters."""
    name = golden_case["case"]
    if name not in detector_cache:
        X, y = reference_data
        detector = EnsembleOutlierSampleDetector(**golden_case["params"])
        detector.fit(X.copy(), y.copy())
        detector_cache[name] = detector
    return detector_cache[name]


def detector_result(detector: EnsembleOutlierSampleDetector) -> dict[str, Any]:
    """Collect the fitted values in the layout of a golden result."""
    return {
        "outlier_support": detector.outlier_support_,
        "inlier_support": detector.inlier_support_,
        "n_iter": detector.n_iter_,
        "median_prediction": detector._median_prediction,
    }


def fail_on_mismatch(message: str | None) -> None:
    """Fail the current test with ``message`` unless it is None."""
    if message is not None:
        pytest.fail(message, pytrace=False)


# --- Golden comparison -----------------------------------------------------


def test_golden_cases_cover_all_configurations() -> None:
    expected = {f"{m}-{k}" for m, k in itertools.product(METHODS, METRICS)}
    assert {path.stem for path in case_paths()} == expected


def test_matches_legacy_golden(
    golden_case: dict[str, Any],
    fitted_detector: EnsembleOutlierSampleDetector,
) -> None:
    fail_on_mismatch(
        compare_result(
            golden_case["case"],
            "legacy",
            golden_case["legacy"]["result"],
            detector_result(fitted_detector),
        )
    )


def test_matches_modern_golden(
    golden_case: dict[str, Any],
    fitted_detector: EnsembleOutlierSampleDetector,
) -> None:
    fail_on_mismatch(
        compare_result(
            golden_case["case"],
            "modern",
            golden_case["modern"]["result"],
            detector_result(fitted_detector),
            compare_median=True,
        )
    )


# --- Helper behavior on synthetic data (no golden file is touched) ---------


def synthetic_result(
    outliers: list[int], n_iter: int = 7, n_samples: int = 100
) -> dict[str, Any]:
    """Build a fitted-style result with the given outlier indices."""
    outlier_support = np.zeros(n_samples, dtype=bool)
    outlier_support[outliers] = True
    return {
        "outlier_support": outlier_support,
        "inlier_support": ~outlier_support,
        "n_iter": n_iter,
        "median_prediction": np.linspace(4.0, 6.0, n_samples),
    }


def as_golden(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a result to the JSON layout of a case file."""
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


GOLDEN = as_golden(synthetic_result([0, 1, 2]))


def test_compare_result_accepts_identical_result() -> None:
    actual = synthetic_result([0, 1, 2])
    message = compare_result(
        "pls-r2", "modern", GOLDEN, actual, compare_median=True
    )
    assert message is None


def test_compare_result_reports_case_and_sample_indices() -> None:
    actual = synthetic_result([0, 1, 2, 3, 42])
    message = compare_result("pls-r2", "legacy", GOLDEN, actual)
    assert message == (
        "case 'pls-r2' does not match the legacy golden:\n"
        "  outlier_support: 2 of 100 samples differ, indices [3, 42]\n"
        "  inlier_support: 2 of 100 samples differ, indices [3, 42]\n"
        "  n_iter: expected 7, actual 7"
    )


def test_compare_result_shows_at_most_20_indices() -> None:
    actual = synthetic_result([0, 1, 2, *range(50, 75)])
    message = compare_result("svr-mae", "legacy", GOLDEN, actual)
    assert message is not None
    shown = list(range(50, 70))
    assert message.splitlines()[1] == (
        f"  outlier_support: 25 of 100 samples differ, indices {shown} "
        "(first 20 shown)"
    )


def test_compare_result_reports_n_iter_mismatch() -> None:
    actual = synthetic_result([0, 1, 2], n_iter=8)
    message = compare_result("pls-mse", "modern", GOLDEN, actual)
    assert message == (
        "case 'pls-mse' does not match the modern golden:\n"
        "  n_iter: expected 7, actual 8"
    )


@pytest.mark.parametrize(
    ("mask", "problem"),
    [
        (np.zeros(100, dtype=np.int64), "expected a boolean array, got int64"),
        (np.zeros(99, dtype=bool), "expected shape (100,), got (99,)"),
    ],
)
def test_compare_result_rejects_malformed_mask(
    mask: np.ndarray, problem: str
) -> None:
    actual = {**synthetic_result([0, 1, 2]), "outlier_support": mask}
    message = compare_result("pls-mae", "legacy", GOLDEN, actual)
    assert message is not None
    assert message.splitlines()[1] == f"  outlier_support: {problem}"


def test_compare_result_reports_median_outside_tolerance() -> None:
    actual = synthetic_result([0, 1, 2])
    actual["median_prediction"][5] += 1e-6
    message = compare_result(
        "svr-r2", "modern", GOLDEN, actual, compare_median=True
    )
    assert message is not None
    lines = message.splitlines()
    assert lines[0] == "case 'svr-r2' does not match the modern golden:"
    assert lines[1] == (
        "  median_prediction: 1 of 100 samples differ, indices [5]"
    )
    assert lines[-1] == "  n_iter: expected 7, actual 7"


def test_compare_result_accepts_median_within_tolerance() -> None:
    actual = synthetic_result([0, 1, 2])
    actual["median_prediction"] += 1e-12
    message = compare_result(
        "svr-r2", "modern", GOLDEN, actual, compare_median=True
    )
    assert message is None


def test_compare_result_ignores_median_unless_requested() -> None:
    actual = synthetic_result([0, 1, 2])
    actual["median_prediction"] += 1.0
    assert compare_result("svr-r2", "legacy", GOLDEN, actual) is None


DATA_SHA256 = "a" * 64


def write_case(directory: Path, **changes: Any) -> Path:
    """Write a minimal case file named ``pls-r2.json``."""
    document = {
        "schema_version": 1,
        "case": "pls-r2",
        "data": {"sha256": DATA_SHA256},
        **changes,
    }
    path = directory / "pls-r2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_case_returns_valid_case_file(tmp_path: Path) -> None:
    document = load_case(write_case(tmp_path), DATA_SHA256)
    assert document["case"] == "pls-r2"


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"schema_version": 2}, "unsupported schema_version 2"),
        ({"case": "svr-r2"}, "expected case 'pls-r2', found 'svr-r2'"),
        ({"data": {"sha256": "b" * 64}}, f"data.sha256 is {'b' * 64}"),
    ],
)
def test_load_case_rejects_unusable_case_file(
    tmp_path: Path, changes: dict[str, Any], problem: str
) -> None:
    path = write_case(tmp_path, **changes)
    with pytest.raises(GoldenCaseError, match=re.escape(problem)):
        load_case(path, DATA_SHA256)


def test_describe_changes_lists_every_kind_of_change() -> None:
    before = {"a.json": "1", "b.json": "2", "c.json": "3"}
    after = {"a.json": "1", "b.json": "9", "d.json": "4"}
    assert describe_changes(before, before) is None
    message = describe_changes(before, after)
    assert message is not None
    assert "modified: b.json" in message
    assert "added: d.json" in message
    assert "removed: c.json" in message
