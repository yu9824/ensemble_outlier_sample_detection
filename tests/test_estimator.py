"""Estimator contract tests (2.6, 2.7, 4.1-4.4, 4.6-4.13, 5.3, 5.5).

Every fit uses small parameters (PLS, ``n_estimators<=2``,
``max_iter <= 3``, ``n_jobs=1``) so that the file runs in seconds.
"""

import re
import warnings
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import (
    ConvergenceWarning,
    DataConversionWarning,
    NotFittedError,
)

from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

# With these parameters the reference data converges after 2 iterations,
# so max_iter=3 converges and max_iter=1 stops one iteration short.
SMALL: dict[str, Any] = {
    "method": "pls",
    "n_estimators": 2,
    "max_iter": 3,
    "random_state": 8,
    "n_jobs": 1,
    "progress_bar": False,
}
N_ITER_SMALL = 2
# With this seed instead, SMALL needs 3 iterations: the mask of
# iteration 2 differs from that of iteration 1, and iteration 3 repeats
# the mask of iteration 2.
SEED_3_ITERATIONS = 14
ARRAY_ATTRIBUTES = {
    "outlier_support_",
    "inlier_support_",
    "n_iter_",
    "n_features_in_",
}
FRAME_ATTRIBUTES = ARRAY_ATTRIBUTES | {"feature_names_in_"}
ACCEPTED = {
    "method": ("pls", "PLS", "svr", "SVR"),
    "metric": ("r2", "mse", "mae", "rmse"),
}

Data = tuple[pd.DataFrame, pd.Series]


class Fitted(NamedTuple):
    detector: EnsembleOutlierSampleDetector
    returned: Any
    params_before: dict[str, Any]


class Run(NamedTuple):
    detector: EnsembleOutlierSampleDetector
    convergence_warnings: list[warnings.WarningMessage]


@pytest.fixture(scope="module")
def fitted(reference_data: Data) -> Fitted:
    """Fit one detector on the DataFrame input and keep what fit returned."""
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**SMALL)
    params_before = detector.get_params()
    returned = detector.fit(X, y)
    return Fitted(detector, returned, params_before)


@pytest.fixture(scope="module")
def fitted_array(reference_data: Data) -> EnsembleOutlierSampleDetector:
    """Fit one detector on the same data given as ndarrays."""
    X, y = reference_data
    return EnsembleOutlierSampleDetector(**SMALL).fit(
        X.to_numpy(), y.to_numpy()
    )


@pytest.fixture(scope="module")
def runs_by_max_iter(reference_data: Data) -> dict[int, Run]:
    """Fit ``SEED_3_ITERATIONS`` with ``max_iter`` = 1, 2 and 3."""
    X, y = reference_data
    return {
        max_iter: fit_recording(
            EnsembleOutlierSampleDetector(
                **{
                    **SMALL,
                    "random_state": SEED_3_ITERATIONS,
                    "max_iter": max_iter,
                }
            ),
            X,
            y,
        )
        for max_iter in (1, 2, 3)
    }


def fit_quietly(
    detector: EnsembleOutlierSampleDetector, X: Any, y: Any
) -> EnsembleOutlierSampleDetector:
    """Fit ``detector`` while ignoring a ConvergenceWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        detector.fit(X, y)
    return detector


# Start of the detector's own ConvergenceWarning message. Matching it keeps
# the tests independent of ConvergenceWarnings that scikit-learn or libsvm
# might emit from submodels on other platforms.
NOT_CONVERGED = "EnsembleOutlierSampleDetector did not converge"


def detector_warnings(
    record: Iterable[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Keep the detector's own non-convergence warnings from ``record``."""
    return [
        w
        for w in record
        if issubclass(w.category, ConvergenceWarning)
        and str(w.message).startswith(NOT_CONVERGED)
    ]


def fit_recording(
    detector: EnsembleOutlierSampleDetector, X: Any, y: Any
) -> Run:
    """Fit ``detector`` and record its non-convergence warnings."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ConvergenceWarning)
        detector.fit(X, y)
    return Run(detector, detector_warnings(record))


def public_attributes(detector: EnsembleOutlierSampleDetector) -> set[str]:
    """Return the public instance attributes that are not parameters."""
    names = {name for name in vars(detector) if not name.startswith("_")}
    return names - set(detector.get_params())


def assert_same_result(
    actual: EnsembleOutlierSampleDetector,
    expected: EnsembleOutlierSampleDetector,
) -> None:
    np.testing.assert_array_equal(
        actual.outlier_support_, expected.outlier_support_
    )
    np.testing.assert_array_equal(
        actual.inlier_support_, expected.inlier_support_
    )
    assert actual.n_iter_ == expected.n_iter_
    np.testing.assert_array_equal(
        actual._median_prediction, expected._median_prediction
    )


# --- fit contract ----------------------------------------------------------


def test_fit_returns_self(fitted: Fitted) -> None:
    assert fitted.returned is fitted.detector


def test_get_params_is_unchanged_by_fit(fitted: Fitted) -> None:
    assert fitted.detector.get_params() == fitted.params_before


# Before 3.3, fit lower-cased the method, replaced the metric with a
# function and clipped max_components to the rank (10 here) for PLS or
# set it to inf for SVR.
@pytest.mark.parametrize(
    ("method", "metric", "n_estimators"),
    [("PLS", "rmse", 2), ("SVR", "mae", 1)],
)
def test_method_alias_and_metric_are_kept_verbatim(
    reference_data: Data, method: str, metric: str, n_estimators: int
) -> None:
    X, y = reference_data
    params = {
        **SMALL,
        "method": method,
        "metric": metric,
        "n_estimators": n_estimators,
        "max_iter": 1,
    }
    detector = fit_quietly(EnsembleOutlierSampleDetector(**params), X, y)
    defaults = EnsembleOutlierSampleDetector().get_params()
    assert detector.get_params() == {**defaults, **params}
    assert detector.get_params()["max_components"] == 30


def test_refit_overwrites_results(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    # The second dataset differs in samples, features and input type.
    X_small, y_small = X.iloc[:60, :6].to_numpy(), y.iloc[:60].to_numpy()
    detector = clone(fitted.detector).fit(X, y)
    fit_quietly(detector, X_small, y_small)

    expected = fit_quietly(
        EnsembleOutlierSampleDetector(**SMALL), X_small, y_small
    )
    assert detector.outlier_support_.shape == (60,)
    assert_same_result(detector, expected)
    assert detector.n_features_in_ == expected.n_features_in_ == 6
    assert not hasattr(detector, "feature_names_in_")


@pytest.mark.parametrize("source", ["unfitted", "fitted"])
def test_clone_gives_the_same_result(
    fitted: Fitted, reference_data: Data, source: str
) -> None:
    X, y = reference_data
    original = (
        fitted.detector
        if source == "fitted"
        else EnsembleOutlierSampleDetector(**SMALL)
    )
    twin = clone(original)
    assert twin.get_params() == original.get_params()
    assert public_attributes(twin) == set()
    assert not hasattr(twin, "_median_prediction")
    with pytest.raises(NotFittedError):
        twin.get_outlier_support()

    if source == "unfitted":
        original.fit(X, y)
    twin.fit(X, y)
    assert_same_result(twin, original)


def test_same_seed_gives_the_same_result(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    again = EnsembleOutlierSampleDetector(**SMALL).fit(X, y)
    assert_same_result(again, fitted.detector)
    # A second fit of the same instance draws from a fresh random state.
    again.fit(X, y)
    assert_same_result(again, fitted.detector)


def test_dataframe_and_ndarray_give_the_same_result(
    fitted: Fitted, fitted_array: EnsembleOutlierSampleDetector
) -> None:
    assert_same_result(fitted_array, fitted.detector)


# --- fitted attributes -----------------------------------------------------


def test_fitted_masks_are_complementary(fitted: Fitted) -> None:
    detector = fitted.detector
    assert detector.outlier_support_.dtype == np.bool_
    assert detector.outlier_support_.shape == (100,)
    np.testing.assert_array_equal(
        detector.inlier_support_, ~detector.outlier_support_
    )
    assert detector.get_outlier_support() is detector.outlier_support_
    assert detector.get_inlier_support() is detector.inlier_support_


def test_n_iter_is_the_number_of_iterations(fitted: Fitted) -> None:
    assert type(fitted.detector.n_iter_) is int
    assert fitted.detector.n_iter_ == N_ITER_SMALL
    assert not hasattr(fitted.detector, "n_iter_finished")


def test_n_iter_counts_the_iterations_run(
    runs_by_max_iter: dict[int, Run],
) -> None:
    # Two iterations are not enough and three are: n_iter_ is a count,
    # not the index of the last iteration.
    cut, converged = runs_by_max_iter[2], runs_by_max_iter[3]
    assert len(cut.convergence_warnings) == 1
    assert cut.detector.n_iter_ == 2
    assert converged.convergence_warnings == []
    assert converged.detector.n_iter_ == 3


def test_records_feature_names_of_a_dataframe(
    fitted: Fitted, reference_data: Data
) -> None:
    X, _ = reference_data
    assert fitted.detector.n_features_in_ == X.shape[1]
    assert fitted.detector.feature_names_in_.dtype == object
    np.testing.assert_array_equal(
        fitted.detector.feature_names_in_, X.columns.to_numpy()
    )


def test_records_no_feature_names_of_an_ndarray(
    fitted_array: EnsembleOutlierSampleDetector, reference_data: Data
) -> None:
    X, _ = reference_data
    assert fitted_array.n_features_in_ == X.shape[1]
    assert not hasattr(fitted_array, "feature_names_in_")


def test_records_no_non_string_column_names(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    X_numbered = X.set_axis(range(X.shape[1]), axis=1)
    detector = EnsembleOutlierSampleDetector(**SMALL).fit(X_numbered, y)
    assert detector.n_features_in_ == X.shape[1]
    assert not hasattr(detector, "feature_names_in_")
    assert_same_result(detector, fitted.detector)


def test_no_unexpected_public_attributes(
    fitted: Fitted,
    fitted_array: EnsembleOutlierSampleDetector,
    runs_by_max_iter: dict[int, Run],
) -> None:
    assert public_attributes(fitted.detector) == FRAME_ATTRIBUTES
    assert public_attributes(fitted_array) == ARRAY_ATTRIBUTES
    not_converged = runs_by_max_iter[1].detector
    assert public_attributes(not_converged) == FRAME_ATTRIBUTES


@pytest.mark.parametrize(
    "getter", ["get_outlier_support", "get_inlier_support"]
)
def test_get_support_before_fit_raises_not_fitted(getter: str) -> None:
    detector = EnsembleOutlierSampleDetector()
    with pytest.raises(NotFittedError, match="not fitted"):
        getattr(detector, getter)()


# --- convergence -----------------------------------------------------------


def test_no_convergence_warns_and_keeps_last_judgment(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, "max_iter": 1})
    with pytest.warns(ConvergenceWarning, match="max_iter=1") as record:
        detector.fit(X, y)
    caught = detector_warnings(record)
    assert len(caught) == 1
    # stacklevel=2 attributes the warning to the caller of fit.
    assert caught[0].filename == __file__
    assert detector.n_iter_ == 1
    # SMALL converges at iteration 2 by repeating the mask of iteration
    # 1, so the converged mask is also the mask of iteration 1.
    np.testing.assert_array_equal(
        detector.outlier_support_, fitted.detector.outlier_support_
    )
    np.testing.assert_array_equal(
        detector.inlier_support_, ~detector.outlier_support_
    )
    assert detector._median_prediction.shape == (100,)


def test_no_convergence_keeps_the_last_iteration(
    runs_by_max_iter: dict[int, Run],
) -> None:
    first, cut, converged = (
        runs_by_max_iter[max_iter].detector for max_iter in (1, 2, 3)
    )
    # Iteration 3 repeats the mask of iteration 2, so the converged mask
    # is the mask of iteration 2, and it differs from that of iteration 1.
    np.testing.assert_array_equal(
        cut.outlier_support_, converged.outlier_support_
    )
    assert not np.array_equal(cut.outlier_support_, first.outlier_support_)
    assert not np.array_equal(cut._median_prediction, first._median_prediction)
    np.testing.assert_array_equal(cut.inlier_support_, ~cut.outlier_support_)
    assert cut.get_outlier_support() is cut.outlier_support_
    assert cut.get_inlier_support() is cut.inlier_support_


def test_converged_fit_does_not_warn(reference_data: Data) -> None:
    X, y = reference_data
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=NOT_CONVERGED, category=ConvergenceWarning
        )
        EnsembleOutlierSampleDetector(**SMALL).fit(X, y)


def test_progress_bar_closes_without_convergence(
    reference_data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    X, y = reference_data
    params = {**SMALL, "max_iter": 1, "progress_bar": True}
    fit_quietly(EnsembleOutlierSampleDetector(**params), X, y)
    assert "Finished! [iter 1 / 1]" in capsys.readouterr().err


def test_svr_raises_no_data_conversion_warning(reference_data: Data) -> None:
    X, y = reference_data
    params = {**SMALL, "method": "svr", "n_estimators": 1, "max_iter": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        fit_quietly(EnsembleOutlierSampleDetector(**params), X, y)


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("method", "lasso"),
        ("method", None),
        ("metric", "accuracy"),
        ("metric", None),
    ],
)
def test_unsupported_method_or_metric_raises(
    reference_data: Data, name: str, value: Any
) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, name: value})
    with pytest.raises(ValueError, match=re.escape(repr(value))) as excinfo:
        detector.fit(X, y)
    message = str(excinfo.value)
    assert name in message
    missing = [v for v in ACCEPTED[name] if repr(v) not in message]
    assert missing == [], f"accepted values missing from: {message}"
    assert not hasattr(detector, "n_features_in_")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_iter", 0),
        ("max_iter", -1),
        ("max_iter", 1.5),
        ("max_iter", True),
        ("max_iter", "3"),
        ("n_estimators", 0),
        ("n_estimators", 2.0),
        ("n_estimators", False),
        ("max_components", 0),
        ("max_components", None),
        ("max_components", True),
    ],
)
def test_non_positive_integer_parameter_raises(
    reference_data: Data, name: str, value: Any
) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, name: value})
    pattern = rf"{name}.*positive integer.*{re.escape(repr(value))}"
    with pytest.raises(ValueError, match=pattern):
        detector.fit(X, y)


def test_numpy_integer_parameters_are_accepted(reference_data: Data) -> None:
    X, y = reference_data
    params = {
        **SMALL,
        "max_iter": np.int64(1),
        "n_estimators": np.int64(2),
        "max_components": np.int64(5),
    }
    detector = fit_quietly(EnsembleOutlierSampleDetector(**params), X, y)
    assert detector.n_iter_ == 1


@pytest.mark.parametrize("call", ["omitted", "none"])
def test_missing_y_raises(reference_data: Data, call: str) -> None:
    X, _ = reference_data
    detector = EnsembleOutlierSampleDetector(**SMALL)
    with pytest.raises(ValueError, match=r"y is required.*y=None"):
        if call == "omitted":
            detector.fit(X)
        else:
            detector.fit(X, None)


def test_length_mismatch_raises(reference_data: Data) -> None:
    X, y = reference_data
    with pytest.raises(
        ValueError, match=r"inconsistent numbers of samples: \[100, 99\]"
    ):
        EnsembleOutlierSampleDetector(**SMALL).fit(X, y.iloc[:-1])


@pytest.mark.parametrize("target", ["X", "y"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_raises(
    reference_data: Data, target: str, bad: float
) -> None:
    X, y = (frame.to_numpy(copy=True) for frame in reference_data)
    if target == "X":
        X[3, 2] = bad
    else:
        y[3] = bad
    problem = "NaN" if np.isnan(bad) else "infinity"
    with pytest.raises(
        ValueError, match=rf"Input {target} contains {problem}"
    ):
        EnsembleOutlierSampleDetector(**SMALL).fit(X, y)
