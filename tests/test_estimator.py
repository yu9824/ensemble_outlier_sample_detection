"""Estimator contract tests (2.6, 2.7, 4.1-4.4, 4.6-4.13, 5.3, 5.5).

Every fit uses small parameters (PLS, ``n_estimators=2``,
``max_iter <= 3``, ``n_jobs=1``) so that the file runs in seconds.
"""

import re
import warnings
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
FITTED_ATTRIBUTES = {
    "outlier_support_",
    "inlier_support_",
    "n_iter_",
    "n_features_in_",
    "feature_names_in_",
}

Data = tuple[pd.DataFrame, pd.Series]


class Fitted(NamedTuple):
    detector: EnsembleOutlierSampleDetector
    returned: Any
    params_before: dict[str, Any]


@pytest.fixture(scope="module")
def fitted(reference_data: Data) -> Fitted:
    """Fit one detector on the DataFrame input and keep what fit returned."""
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**SMALL)
    params_before = detector.get_params()
    returned = detector.fit(X, y)
    return Fitted(detector, returned, params_before)


def fit_quietly(
    detector: EnsembleOutlierSampleDetector, X: Any, y: Any
) -> EnsembleOutlierSampleDetector:
    """Fit ``detector`` while ignoring a ConvergenceWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        detector.fit(X, y)
    return detector


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


def test_method_alias_is_kept_verbatim(reference_data: Data) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, "method": "PLS"})
    fit_quietly(detector, X, y)
    assert detector.get_params()["method"] == "PLS"
    assert detector.get_params()["metric"] == "r2"
    assert detector.get_params()["max_components"] == 30


def test_refit_overwrites_results(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    X_small, y_small = X.iloc[:60].to_numpy(), y.iloc[:60].to_numpy()
    detector = clone(fitted.detector)
    detector.fit(X, y)
    fit_quietly(detector, X_small, y_small)

    expected = fit_quietly(
        EnsembleOutlierSampleDetector(**SMALL), X_small, y_small
    )
    assert detector.outlier_support_.shape == (60,)
    assert_same_result(detector, expected)
    assert detector.n_features_in_ == X.shape[1]
    assert not hasattr(detector, "feature_names_in_")


def test_clone_gives_the_same_result(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    twin = clone(fitted.detector).fit(X, y)
    assert_same_result(twin, fitted.detector)


def test_same_seed_gives_the_same_result(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    again = EnsembleOutlierSampleDetector(**SMALL).fit(X, y)
    assert_same_result(again, fitted.detector)


def test_dataframe_and_ndarray_give_the_same_result(
    fitted: Fitted, reference_data: Data
) -> None:
    X, y = reference_data
    from_array = EnsembleOutlierSampleDetector(**SMALL).fit(
        X.to_numpy(), y.to_numpy()
    )
    assert_same_result(from_array, fitted.detector)
    assert not hasattr(from_array, "feature_names_in_")


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


def test_records_feature_names_of_a_dataframe(
    fitted: Fitted, reference_data: Data
) -> None:
    X, _ = reference_data
    assert fitted.detector.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(
        fitted.detector.feature_names_in_, X.columns.to_numpy()
    )


def test_no_unexpected_public_attributes(fitted: Fitted) -> None:
    assert public_attributes(fitted.detector) == FITTED_ATTRIBUTES


def test_get_support_before_fit_raises_not_fitted() -> None:
    detector = EnsembleOutlierSampleDetector()
    with pytest.raises(NotFittedError):
        detector.get_outlier_support()
    with pytest.raises(NotFittedError):
        detector.get_inlier_support()


# --- convergence -----------------------------------------------------------


def test_no_convergence_warns_and_keeps_last_judgment(
    reference_data: Data,
) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, "max_iter": 1})
    with pytest.warns(ConvergenceWarning, match="max_iter=1"):
        detector.fit(X, y)
    assert detector.n_iter_ == 1
    assert detector.outlier_support_.shape == (100,)
    assert detector.outlier_support_.any()
    np.testing.assert_array_equal(
        detector.inlier_support_, ~detector.outlier_support_
    )
    assert detector._median_prediction.shape == (100,)


def test_converged_fit_does_not_warn(reference_data: Data) -> None:
    X, y = reference_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
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
    [("method", "lasso"), ("method", None), ("metric", "accuracy")],
)
def test_unsupported_method_or_metric_raises(
    reference_data: Data, name: str, value: Any
) -> None:
    X, y = reference_data
    detector = EnsembleOutlierSampleDetector(**{**SMALL, name: value})
    accepted = "'pls'" if name == "method" else "'rmse'"
    with pytest.raises(ValueError, match=rf"{name}.*{re.escape(accepted)}"):
        detector.fit(X, y)
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
        ("max_components", 0),
        ("max_components", None),
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
    with pytest.raises(ValueError, match="y"):
        if call == "omitted":
            detector.fit(X)
        else:
            detector.fit(X, None)


def test_length_mismatch_raises(reference_data: Data) -> None:
    X, y = reference_data
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
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
    with pytest.raises(ValueError, match="NaN|infinity"):
        EnsembleOutlierSampleDetector(**SMALL).fit(X, y)
