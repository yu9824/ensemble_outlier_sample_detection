"""Unit tests for the submodel helpers (4.5, 4.6, 6.2, 7.1, 7.3; P5)."""

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import optuna
import pytest
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ensemble_outlier_sample_detection._submodels import (
    MetricSpec,
    build_model,
    make_objective,
    n_trials,
    normalize_method,
    resolve_metric,
    suggest_params,
)

CV = 3
N_JOBS = 1
MAX_COMPONENTS = 3
# Typed as Any so mypy accepts it where MethodName is expected.
UNNORMALIZED: Any = "PLS"

METRIC_TABLE = {
    "r2": (r2_score, "maximize"),
    "mse": (mean_squared_error, "minimize"),
    "mae": (mean_absolute_error, "minimize"),
    "rmse": (root_mean_squared_error, "minimize"),
}
FIXED_PARAMS: dict[str, dict[str, float]] = {
    "pls": {"n_components": 2},
    "svr": {"C": 4.0, "epsilon": 0.05, "gamma": 0.1},
}


def fixed_trial(params: Mapping[str, float]) -> Any:
    """Return a ``FixedTrial``; it duck-types ``Trial`` (not a subclass)."""
    return optuna.trial.FixedTrial(dict(params))


class RecordingTrial:
    """Fake trial that records every suggest call with resolved options."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, float, Any, bool]] = []

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        self.calls.append(("int", name, low, high, step, log))
        return high

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self.calls.append(("float", name, low, high, step, log))
        return (low * high) ** 0.5


@pytest.fixture(scope="module")
def train_data() -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    """Return standardized ``(X, y_1d, scaler_y)`` of a small dataset."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 4))
    y = X @ np.array([1.5, -2.0, 0.5, 0.0]) + 0.3 * rng.normal(size=30)
    X_scaled = StandardScaler().fit_transform(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
    return X_scaled, y_scaled, scaler_y


def reference_model(method: str, params: Mapping[str, float]) -> Any:
    """Build the model the way the pre-refactoring ``Objective`` did."""
    if method == "pls":
        return PLSRegression(**params, scale=False)
    return SVR(**params, kernel="rbf")


def legacy_score(
    method: str,
    params: Mapping[str, float],
    metric: Callable[..., float],
    X: NDArray[np.float64],
    y_1d: NDArray[np.float64],
    scaler_y: Any,
) -> float:
    """Reproduce ``Objective.__call__`` of the pre-refactoring module.

    It passes ``y`` as a 2-D column (as ``fit`` stored it) and calls
    ``metric(y_pred, y_true)``.
    """
    y_2d = y_1d.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        y_oof = cross_val_predict(
            reference_model(method, params), X, y_2d, cv=CV, n_jobs=N_JOBS
        )
    return metric(
        scaler_y.inverse_transform(np.reshape(y_oof, (-1, 1))),
        scaler_y.inverse_transform(y_2d),
    )


def definition_order(
    score: Callable[..., float],
) -> Callable[[Any, Any], float]:
    """Adapt ``score`` to the ``metric(y_pred, y_true)`` calls of ``Objective``.

    The same adapter as the golden generator's ``r2-argument-order`` patch,
    applied to every metric: the pre-refactoring ``Objective`` plus this
    adapter is the reference after the R² fix (6.2, 6.3).
    """
    return lambda y_pred, y_true: score(y_true, y_pred)


def legacy_trial_objective(
    method: str,
    max_components: int,
    metric: Callable[..., float],
    X: NDArray[np.float64],
    y_1d: NDArray[np.float64],
    scaler_y: Any,
) -> Callable[[optuna.trial.Trial], float]:
    """Return a copy of the pre-refactoring search space and scoring."""

    def objective(trial: optuna.trial.Trial) -> float:
        params: dict[str, float]
        if method == "pls":
            params = {
                "n_components": trial.suggest_int(
                    "n_components", 1, max_components
                ),
            }
        else:
            params = {
                "C": trial.suggest_float("C", 2**-5, 2**11, log=True),
                "epsilon": trial.suggest_float(
                    "epsilon", 2**-10, 2**1, log=True
                ),
                "gamma": trial.suggest_float("gamma", 2**-20, 2**11, log=True),
            }
        return legacy_score(method, params, metric, X, y_1d, scaler_y)

    return objective


# --- normalize_method (4.5, 4.6) -------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [("pls", "pls"), ("PLS", "pls"), ("svr", "svr"), ("SVR", "svr")],
)
def test_normalize_method_accepts_aliases(given: str, expected: str) -> None:
    assert normalize_method(given) == expected


@pytest.mark.parametrize("given", ["ridge", "Pls", "", None, ["pls"]])
def test_normalize_method_rejects_unknown_values(given: Any) -> None:
    with pytest.raises(ValueError, match="method") as excinfo:
        normalize_method(given)
    message = str(excinfo.value)
    for accepted in ("'pls'", "'PLS'", "'svr'", "'SVR'"):
        assert accepted in message


# --- resolve_metric (4.5, 4.6) ---------------------------------------------


@pytest.mark.parametrize("name", list(METRIC_TABLE))
def test_resolve_metric_table(name: str) -> None:
    score, direction = METRIC_TABLE[name]
    spec = resolve_metric(name)
    assert isinstance(spec, MetricSpec)
    assert spec.score is score
    assert spec.direction == direction


def test_metric_spec_is_frozen() -> None:
    spec = resolve_metric("mae")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.direction = "maximize"  # type: ignore[misc]


@pytest.mark.parametrize("given", ["R2", "accuracy", "", None])
def test_resolve_metric_rejects_unknown_values(given: Any) -> None:
    with pytest.raises(ValueError, match="metric") as excinfo:
        resolve_metric(given)
    message = str(excinfo.value)
    for accepted in ("'r2'", "'mse'", "'mae'", "'rmse'"):
        assert accepted in message


# --- n_trials ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "max_components", "expected"),
    [
        ("pls", 1, 1),
        ("pls", 7, 7),
        ("pls", 100, 100),
        ("pls", 150, 100),
        ("svr", 1, 100),
        ("svr", 1000, 100),
    ],
)
def test_n_trials_rules(
    method: Any, max_components: int, expected: int
) -> None:
    assert n_trials(method, max_components) == expected


# --- suggest_params ---------------------------------------------------------


def test_suggest_params_pls_search_space() -> None:
    trial = RecordingTrial()
    params = suggest_params(trial, "pls", 7)  # type: ignore[arg-type]
    assert trial.calls == [("int", "n_components", 1, 7, 1, False)]
    assert params == {"n_components": 7}


def test_suggest_params_svr_order_and_ranges() -> None:
    trial = RecordingTrial()
    params = suggest_params(trial, "svr", 7)  # type: ignore[arg-type]
    assert trial.calls == [
        ("float", "C", 2**-5, 2**11, None, True),
        ("float", "epsilon", 2**-10, 2**1, None, True),
        ("float", "gamma", 2**-20, 2**11, None, True),
    ]
    assert list(params) == ["C", "epsilon", "gamma"]
    assert params == {
        "C": (2**-5 * 2**11) ** 0.5,
        "epsilon": (2**-10 * 2**1) ** 0.5,
        "gamma": (2**-20 * 2**11) ** 0.5,
    }


@pytest.mark.parametrize("method", ["pls", "svr"])
def test_suggest_params_with_fixed_trial(method: Any) -> None:
    trial = fixed_trial(FIXED_PARAMS[method])
    params = suggest_params(trial, method, MAX_COMPONENTS)
    assert params == FIXED_PARAMS[method]


# --- build_model ------------------------------------------------------------


def test_build_model_pls_fixes_scale_false() -> None:
    params = {"n_components": 3}
    model = build_model("pls", params)
    assert type(model) is PLSRegression
    assert model.get_params()["scale"] is False
    assert model.get_params()["n_components"] == 3
    assert params == {"n_components": 3}


def test_build_model_svr_fixes_rbf_kernel() -> None:
    params = {"C": 2.0, "epsilon": 0.5, "gamma": 0.25}
    model = build_model("svr", params)
    assert type(model) is SVR
    got = model.get_params()
    assert got["kernel"] == "rbf"
    assert (got["C"], got["epsilon"], got["gamma"]) == (2.0, 0.5, 0.25)
    assert params == {"C": 2.0, "epsilon": 0.5, "gamma": 0.25}


# --- unnormalized method names in the internal helpers -----------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: n_trials(UNNORMALIZED, 3),
        lambda: suggest_params(fixed_trial({}), UNNORMALIZED, 3),
        lambda: build_model(UNNORMALIZED, {}),
    ],
    ids=["n_trials", "suggest_params", "build_model"],
)
def test_helpers_reject_unnormalized_method(call: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="'pls', 'svr'"):
        call()


def test_make_objective_rejects_unnormalized_method(
    train_data: tuple[NDArray[np.float64], NDArray[np.float64], Any],
) -> None:
    X, y, scaler_y = train_data
    with pytest.raises(ValueError, match="'pls', 'svr'"):
        make_objective(
            "SVR",  # type: ignore[arg-type]
            X,
            y,
            scaler_y,
            resolve_metric("r2"),
            CV,
            N_JOBS,
            MAX_COMPONENTS,
        )


# --- make_objective (score(y_true, y_pred), 6.2) ---------------------------


@pytest.mark.parametrize("metric_name", list(METRIC_TABLE))
@pytest.mark.parametrize("method", ["pls", "svr"])
def test_make_objective_scores_oof_y_true_first(
    method: Any,
    metric_name: str,
    train_data: tuple[NDArray[np.float64], NDArray[np.float64], Any],
) -> None:
    X, y, scaler_y = train_data
    spec = resolve_metric(metric_name)
    objective = make_objective(
        method, X, y, scaler_y, spec, CV, N_JOBS, MAX_COMPONENTS
    )
    value = objective(fixed_trial(FIXED_PARAMS[method]))

    y_oof = cross_val_predict(
        reference_model(method, FIXED_PARAMS[method]),
        X,
        y,
        cv=CV,
        n_jobs=N_JOBS,
    )
    y_pred = scaler_y.inverse_transform(y_oof.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y.reshape(-1, 1))
    score, _ = METRIC_TABLE[metric_name]
    assert value == score(y_true, y_pred)

    # Regression guard: same value as the pre-refactoring Objective (which
    # passed y as a 2-D column to cross_val_predict) with the R² fix.
    assert value == legacy_score(
        method, FIXED_PARAMS[method], definition_order(score), X, y, scaler_y
    )


def test_make_objective_r2_is_y_true_first(
    train_data: tuple[NDArray[np.float64], NDArray[np.float64], Any],
) -> None:
    X, y, scaler_y = train_data
    objective = make_objective(
        "pls", X, y, scaler_y, resolve_metric("r2"), CV, N_JOBS, 1
    )
    value = objective(fixed_trial({"n_components": 1}))

    y_oof = cross_val_predict(
        PLSRegression(n_components=1, scale=False), X, y, cv=CV
    )
    y_pred = scaler_y.inverse_transform(y_oof.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
    assert value == r2_score(y_true, y_pred)
    # R² by its definition, 1 - SS_res / SS_tot around the mean of y_true.
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    np.testing.assert_allclose(value, 1.0 - ss_res / ss_tot, rtol=1e-12)
    # The old order is a different number on this data, so the test can
    # tell the two apart.
    assert value != r2_score(y_pred, y_true)


@pytest.mark.parametrize(
    ("method", "metric_name", "trials"),
    [("pls", "r2", MAX_COMPONENTS), ("svr", "rmse", 6)],
)
def test_make_objective_matches_legacy_study(
    method: Any,
    metric_name: str,
    trials: int,
    train_data: tuple[NDArray[np.float64], NDArray[np.float64], Any],
) -> None:
    """Same TPE seed → same sampled params, values and best_params (P5)."""
    X, y, scaler_y = train_data
    spec = resolve_metric(metric_name)
    count = min(trials, n_trials(method, MAX_COMPONENTS))

    def run(objective: Callable[[optuna.trial.Trial], float]) -> Any:
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=12345),
            direction=spec.direction,
        )
        study.optimize(objective, n_trials=count)
        return study

    new = run(
        make_objective(
            method, X, y, scaler_y, spec, CV, N_JOBS, MAX_COMPONENTS
        )
    )
    old = run(
        legacy_trial_objective(
            method,
            MAX_COMPONENTS,
            definition_order(spec.score),
            X,
            y,
            scaler_y,
        )
    )
    assert [t.params for t in new.trials] == [t.params for t in old.trials]
    assert [t.value for t in new.trials] == [t.value for t in old.trials]
    assert new.best_params == old.best_params
