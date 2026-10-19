"""Load golden case files and compare detector results with them.

Shared by ``conftest.py`` and ``test_golden.py``. Case files are only
ever opened for reading here.
"""

import hashlib
import json
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DATA_PATH = REPO_ROOT / "example" / "example_chache.csv"
CASES_DIR = TESTS_DIR / "golden" / "cases"

SCHEMA_VERSION = 1
MASK_KEYS = ("outlier_support", "inlier_support")
MAX_SHOWN_SAMPLES = 20
# (rtol, atol) for median_prediction against the modern block (2.8), by
# method. SVR medians differ by up to ~4e-4 (relative) between the macOS
# arm64 machine that generated the golden and Linux x86_64 CI runners,
# whatever the dependency versions, so SVR uses a looser tolerance that
# still catches a change in the selected hyperparameters (see
# tests/golden/README.md).
Tolerance = tuple[float, float]
MEDIAN_TOLERANCE: dict[str, Tolerance] = {
    "pls": (1e-7, 1e-10),
    "svr": (1e-3, 1e-3),
}


class GoldenCaseError(Exception):
    """A case file that the golden tests cannot compare with."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_paths(directory: Path = CASES_DIR) -> list[Path]:
    """Return the case files in ``directory``, sorted by name."""
    return sorted(directory.glob("*.json"))


def load_case(path: Path, data_sha256: str) -> dict[str, Any]:
    """Read a case file and check that it can be compared with.

    Raises ``GoldenCaseError`` when the schema version is unsupported,
    the case name differs from the file name, or the case was generated
    from reference data other than the one hashing to ``data_sha256``.
    """
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise GoldenCaseError(
            f"{path.name}: unsupported schema_version {version!r}; "
            f"these tests read schema_version {SCHEMA_VERSION} only"
        )
    if document.get("case") != path.stem:
        raise GoldenCaseError(
            f"{path.name}: expected case {path.stem!r}, "
            f"found {document.get('case')!r}"
        )
    recorded = document["data"]["sha256"]
    if recorded != data_sha256:
        raise GoldenCaseError(
            f"{path.name}: data.sha256 is {recorded}, but the reference "
            f"data {DATA_PATH.relative_to(REPO_ROOT)} hashes to "
            f"{data_sha256}; the golden was generated from other data"
        )
    return document


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    """Map the name of each file in ``paths`` to its SHA-256 digest."""
    return {path.name: file_sha256(path) for path in paths}


def describe_changes(
    before: Mapping[str, str], after: Mapping[str, str]
) -> str | None:
    """Describe how two snapshots differ, or return None if they match."""
    changes = {
        "modified": sorted(
            name
            for name in before.keys() & after.keys()
            if before[name] != after[name]
        ),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
    }
    lines = [
        f"  {kind}: {', '.join(names)}"
        for kind, names in changes.items()
        if names
    ]
    if not lines:
        return None
    return "\n".join(
        ["golden case files changed while the tests ran:", *lines]
    )


def describe_samples(indices: list[int], n_samples: int) -> str:
    """Summarize mismatching sample indices, showing at most 20."""
    text = (
        f"{len(indices)} of {n_samples} samples differ, "
        f"indices {indices[:MAX_SHOWN_SAMPLES]}"
    )
    if len(indices) > MAX_SHOWN_SAMPLES:
        text += f" (first {MAX_SHOWN_SAMPLES} shown)"
    return text


def mask_problem(key: str, expected: ArrayLike, actual: ArrayLike) -> str:
    """Return how a boolean mask differs, or "" if it matches exactly."""
    expected_mask = np.asarray(expected, dtype=bool)
    actual_mask = np.asarray(actual)
    if actual_mask.dtype != np.bool_:
        return f"{key}: expected a boolean array, got {actual_mask.dtype}"
    if actual_mask.shape != expected_mask.shape:
        return (
            f"{key}: expected shape {expected_mask.shape}, "
            f"got {actual_mask.shape}"
        )
    indices = np.flatnonzero(actual_mask != expected_mask).tolist()
    if not indices:
        return ""
    return f"{key}: {describe_samples(indices, expected_mask.size)}"


def median_problem(
    expected: ArrayLike, actual: ArrayLike, tolerance: Tolerance
) -> str:
    """Return how median predictions differ beyond the tolerance, or ""."""
    rtol, atol = tolerance
    expected_median = np.asarray(expected, dtype=np.float64)
    actual_median = np.asarray(actual, dtype=np.float64)
    try:
        np.testing.assert_allclose(
            actual_median, expected_median, rtol=rtol, atol=atol
        )
    except AssertionError as error:
        detail = str(error).strip()
    else:
        return ""
    if actual_median.shape != expected_median.shape:
        return f"median_prediction: {detail}"
    close = np.isclose(
        actual_median, expected_median, rtol=rtol, atol=atol, equal_nan=True
    )
    samples = describe_samples(
        np.flatnonzero(~close).tolist(), expected_median.size
    )
    return f"median_prediction: {samples}\n{detail}"


def compare_result(
    case_name: str,
    block: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    median_tolerance: Tolerance | None = None,
) -> str | None:
    """Compare a fitted result with a golden block.

    The masks and ``n_iter`` must match exactly. With ``median_tolerance``
    as ``(rtol, atol)``, ``median_prediction`` must also match within it.
    Returns None on a match, otherwise a failure message naming the case,
    the differing sample indices, and both ``n_iter``.
    """
    checks = [
        mask_problem(key, expected[key], actual[key]) for key in MASK_KEYS
    ]
    if median_tolerance is not None:
        checks.append(
            median_problem(
                expected["median_prediction"],
                actual["median_prediction"],
                median_tolerance,
            )
        )
    problems = [problem for problem in checks if problem]
    expected_n_iter, actual_n_iter = expected["n_iter"], actual["n_iter"]
    if not problems and actual_n_iter == expected_n_iter:
        return None
    lines = [
        *problems,
        f"n_iter: expected {expected_n_iter}, actual {actual_n_iter}",
    ]
    body = textwrap.indent("\n".join(lines), "  ")
    return f"case {case_name!r} does not match the {block} golden:\n{body}"
