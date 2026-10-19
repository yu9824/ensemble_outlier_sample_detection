"""Shared fixtures: reference data, golden cases and golden immutability."""

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from _golden_cases import (
    CASES_DIR,
    DATA_PATH,
    GoldenCaseError,
    case_paths,
    describe_changes,
    file_sha256,
    load_case,
    snapshot,
)


@pytest.fixture(scope="session")
def reference_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` of the example dataset.

    ``comment="#"`` skips the attribution lines at the top of the CSV;
    they stay in the file untouched (1.7).
    """
    frame = pd.read_csv(DATA_PATH, index_col=0, comment="#")
    return frame.iloc[:, 1:], frame.iloc[:, 0]


@pytest.fixture(scope="session")
def reference_data_sha256() -> str:
    """Return the SHA-256 digest of the reference CSV."""
    return file_sha256(DATA_PATH)


@pytest.fixture(params=[path.stem for path in case_paths()])
def golden_case(
    request: pytest.FixtureRequest, reference_data_sha256: str
) -> dict[str, Any]:
    """Return one case file per parameter, checked before use."""
    try:
        return load_case(
            CASES_DIR / f"{request.param}.json", reference_data_sha256
        )
    except GoldenCaseError as error:
        problem = str(error)
    # Failing outside the except block keeps the report to one message.
    pytest.fail(problem, pytrace=False)


@pytest.fixture(scope="session", autouse=True)
def golden_cases_unchanged() -> Iterator[None]:
    """Fail the session if a case file changed while the tests ran (2.3)."""
    before = snapshot(case_paths())
    yield
    changes = describe_changes(before, snapshot(case_paths()))
    if changes is not None:
        pytest.fail(changes, pytrace=False)
