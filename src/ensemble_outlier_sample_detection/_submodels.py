"""Submodel helpers: method and metric tables, search space, objective.

This module has no dependency on the rest of the package. The detector
uses it to normalize ``method`` and ``metric``, to tune each bootstrap
submodel with Optuna and to build the tuned model.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

import numpy as np
import optuna
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

__all__ = [
    "Direction",
    "MethodName",
    "MetricName",
    "MetricSpec",
    "Model",
    "build_model",
    "make_objective",
    "n_trials",
    "normalize_method",
    "resolve_metric",
    "suggest_params",
]

MethodName: TypeAlias = Literal["pls", "svr"]
MetricName: TypeAlias = Literal["r2", "mse", "mae", "rmse"]
Direction: TypeAlias = Literal["maximize", "minimize"]
Model: TypeAlias = PLSRegression | SVR


@dataclass(frozen=True)
class MetricSpec:
    """Score function of a metric and the direction Optuna optimizes it.

    Attributes
    ----------
    score : callable
        ``score(y_true, y_pred)``, a scikit-learn regression metric.
    direction : {"maximize", "minimize"}
        Direction passed to :func:`optuna.create_study`.
    """

    score: Callable[[NDArray[np.float64], NDArray[np.float64]], float]
    direction: Direction


_METHODS: Final[tuple[MethodName, ...]] = get_args(MethodName)
_METHOD_ALIASES: Final[Mapping[str, MethodName]] = {
    "pls": "pls",
    "PLS": "pls",
    "svr": "svr",
    "SVR": "svr",
}
_METRICS: Final[Mapping[str, MetricSpec]] = {
    "r2": MetricSpec(r2_score, "maximize"),
    "mse": MetricSpec(mean_squared_error, "minimize"),
    "mae": MetricSpec(mean_absolute_error, "minimize"),
    "rmse": MetricSpec(root_mean_squared_error, "minimize"),
}
_MAX_TRIALS: Final = 100


def _unsupported(
    kind: str, value: object, accepted: Iterable[str]
) -> ValueError:
    """Return a ``ValueError`` that lists the accepted values."""
    listed = ", ".join(repr(name) for name in accepted)
    return ValueError(
        f"Unsupported {kind} {value!r}. Accepted values: {listed}."
    )


def normalize_method(method: str) -> MethodName:
    """Return the canonical name of a submodel method.

    Parameters
    ----------
    method : {"pls", "PLS", "svr", "SVR"}
        Method name given by the user.

    Returns
    -------
    {"pls", "svr"}
        Lower-case method name.

    Raises
    ------
    ValueError
        If ``method`` is not one of the accepted values.
    """
    if isinstance(method, str) and method in _METHOD_ALIASES:
        return _METHOD_ALIASES[method]
    raise _unsupported("method", method, _METHOD_ALIASES)


def resolve_metric(metric: str) -> MetricSpec:
    """Return the score function and optimization direction of a metric.

    Parameters
    ----------
    metric : {"r2", "mse", "mae", "rmse"}
        Metric name given by the user. ``"r2"`` is maximized; the others
        are minimized.

    Returns
    -------
    MetricSpec
        ``r2_score``, ``mean_squared_error``, ``mean_absolute_error`` or
        ``root_mean_squared_error`` with its direction.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the accepted values.
    """
    if isinstance(metric, str) and metric in _METRICS:
        return _METRICS[metric]
    raise _unsupported("metric", metric, _METRICS)


def n_trials(method: MethodName, max_components: int) -> int:
    """Return the number of Optuna trials for one submodel.

    Parameters
    ----------
    method : {"pls", "svr"}
        Normalized method name.
    max_components : int
        Upper bound of ``n_components`` for PLS. Ignored for SVR.

    Returns
    -------
    int
        ``min(100, max_components)`` for PLS and ``100`` for SVR.

    Raises
    ------
    ValueError
        If ``method`` is not normalized.
    """
    if method == "pls":
        return min(_MAX_TRIALS, max_components)
    if method == "svr":
        return _MAX_TRIALS
    raise _unsupported("method", method, _METHODS)


def suggest_params(
    trial: optuna.trial.Trial, method: MethodName, max_components: int
) -> dict[str, int | float]:
    """Suggest the hyperparameters of one submodel.

    PLS searches the integer ``n_components`` in ``[1, max_components]``.
    SVR searches ``C`` in ``[2**-5, 2**11]``, ``epsilon`` in
    ``[2**-10, 2**1]`` and ``gamma`` in ``[2**-20, 2**11]`` on a log
    scale, suggested in this order so that the sampler draws its random
    numbers in the same order as before.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial that samples the values.
    method : {"pls", "svr"}
        Normalized method name.
    max_components : int
        Upper bound of ``n_components`` for PLS. Ignored for SVR.

    Returns
    -------
    dict of str to int or float
        Suggested hyperparameters keyed by the estimator argument name.

    Raises
    ------
    ValueError
        If ``method`` is not normalized.
    """
    if method == "pls":
        return {
            "n_components": trial.suggest_int(
                "n_components", 1, max_components
            ),
        }
    if method == "svr":
        return {
            "C": trial.suggest_float("C", 2**-5, 2**11, log=True),
            "epsilon": trial.suggest_float("epsilon", 2**-10, 2**1, log=True),
            "gamma": trial.suggest_float("gamma", 2**-20, 2**11, log=True),
        }
    raise _unsupported("method", method, _METHODS)


def build_model(
    method: MethodName, params: Mapping[str, int | float]
) -> Model:
    """Build an unfitted submodel.

    Parameters
    ----------
    method : {"pls", "svr"}
        Normalized method name.
    params : mapping of str to int or float
        Tuned hyperparameters, as returned by :func:`suggest_params`.

    Returns
    -------
    PLSRegression or SVR
        ``PLSRegression(**params, scale=False)`` or
        ``SVR(**params, kernel="rbf")``.

    Raises
    ------
    ValueError
        If ``method`` is not normalized.
    """
    if method == "pls":
        return PLSRegression(**params, scale=False)
    if method == "svr":
        return SVR(**params, kernel="rbf")
    raise _unsupported("method", method, _METHODS)


def make_objective(
    method: MethodName,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    scaler_y: StandardScaler,
    metric: MetricSpec,
    cv: int,
    n_jobs: int | None,
    max_components: int,
) -> Callable[[optuna.trial.Trial], float]:
    """Return the Optuna objective of one bootstrap submodel.

    The objective suggests hyperparameters, computes out-of-fold
    predictions with :func:`~sklearn.model_selection.cross_val_predict`
    and scores them against ``y_train``, both mapped back to the original
    scale of ``y`` with ``scaler_y``.

    Parameters
    ----------
    method : {"pls", "svr"}
        Normalized method name.
    X_train : ndarray of shape (n_samples, n_features)
        Standardized features of the bootstrap sample.
    y_train : ndarray of shape (n_samples,)
        Standardized target of the bootstrap sample.
    scaler_y : StandardScaler
        Scaler fitted on the bootstrap target, used to undo the
        standardization before scoring.
    metric : MetricSpec
        Metric to score the out-of-fold predictions with.
    cv : int
        Number of folds for ``cross_val_predict``.
    n_jobs : int or None
        Number of jobs for ``cross_val_predict``.
    max_components : int
        Upper bound of ``n_components`` for PLS. Ignored for SVR.

    Returns
    -------
    callable
        ``objective(trial) -> float`` to pass to ``study.optimize``.

    Raises
    ------
    ValueError
        If ``method`` is not normalized.
    """
    if method not in _METHODS:
        raise _unsupported("method", method, _METHODS)
    y_true = scaler_y.inverse_transform(np.reshape(y_train, (-1, 1)))

    def objective(trial: optuna.trial.Trial) -> float:
        """Score one set of suggested hyperparameters."""
        params = suggest_params(trial, method, max_components)
        model = build_model(method, params)
        y_oof = cross_val_predict(
            model, X_train, y_train, cv=cv, n_jobs=n_jobs
        )
        # SVR returns 1-D predictions; scalers need a column.
        y_pred = scaler_y.inverse_transform(np.reshape(y_oof, (-1, 1)))
        # Definition order of scikit-learn metrics: score(y_true, y_pred).
        # Only "r2" is not symmetric in its arguments (6.2).
        return float(metric.score(y_true, y_pred))

    return objective
