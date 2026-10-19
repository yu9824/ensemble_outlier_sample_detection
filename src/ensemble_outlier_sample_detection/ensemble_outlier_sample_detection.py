"""Ensemble-based outlier sample detection for regression data.

Copyright © 2021 yu9824
"""

import numbers
import warnings

import numpy as np
import optuna
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _runtime, _submodels

__all__ = ["EnsembleOutlierSampleDetector"]

# Upper bound (exclusive) of the seeds drawn from the random state.
_SEED_RANGE = 2**32


def _check_positive_int(name: str, value: object) -> int:
    """Return ``value`` as an int, or raise if it is not a positive integer.

    ``bool`` is rejected although it is an ``Integral``, and so are
    integral floats such as ``2.0``.
    """
    if isinstance(value, numbers.Integral) and not isinstance(
        value, bool | np.bool_
    ):
        as_int = int(value)
        if as_int >= 1:
            return as_int
    raise ValueError(
        f"{name} must be a positive integer (an int >= 1), got {value!r}."
    )


def _fit_predict_submodel(
    X: NDArray[np.float64],
    X_remained: NDArray[np.float64],
    y_remained: NDArray[np.float64],
    *,
    method: _submodels.MethodName,
    metric: _submodels.MetricSpec,
    cv: int,
    n_jobs: int | None,
    max_components: int,
    rng: np.random.RandomState,
) -> tuple[NDArray[np.float64], int]:
    """Tune one submodel on a bootstrap sample and predict every sample.

    Two seeds are drawn from ``rng``, first for the bootstrap and then
    for the TPE sampler (P1).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        All samples, which the tuned submodel predicts.
    X_remained, y_remained : ndarray
        Samples judged as inliers in the previous iteration, from which
        the bootstrap sample is drawn.
    method : {"pls", "svr"}
        Normalized method name.
    metric : MetricSpec
        Metric that Optuna optimizes.
    cv : int
        Number of folds for the out-of-fold predictions.
    n_jobs : int or None
        Number of jobs for the out-of-fold predictions.
    max_components : int
        Running upper bound of ``n_components`` for PLS.
    rng : numpy.random.RandomState
        Random state that the seeds are drawn from.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predictions for ``X`` on the original scale of ``y``.
    max_components : int
        Updated running upper bound. For PLS it never exceeds the rank
        of the standardized bootstrap features; the caller carries it
        over to later bootstraps and iterations (P2).
    """
    X_boot, y_boot = resample(
        X_remained,
        y_remained,
        random_state=rng.randint(_SEED_RANGE),
        n_samples=len(X_remained),
    )

    # Drop the features that are constant in the bootstrap sample.
    selector = VarianceThreshold(threshold=0)
    X_train = selector.fit_transform(X_boot)
    X_all = selector.transform(X)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_all = scaler_X.transform(X_all)

    # StandardScaler needs a column; the submodels get a 1-D target.
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(np.reshape(y_boot, (-1, 1))).ravel()

    if method == "pls":
        rank = int(np.linalg.matrix_rank(X_train))
        max_components = min(max_components, rank)

    objective = _submodels.make_objective(
        method, X_train, y_train, scaler_y, metric, cv, n_jobs, max_components
    )
    sampler = optuna.samplers.TPESampler(seed=rng.randint(_SEED_RANGE))
    study = optuna.create_study(sampler=sampler, direction=metric.direction)
    study.optimize(
        objective, n_trials=int(_submodels.n_trials(method, max_components))
    )

    model = _submodels.build_model(method, study.best_params)
    model.fit(X_train, y_train)
    y_pred = scaler_y.inverse_transform(
        np.reshape(model.predict(X_all), (-1, 1))
    ).ravel()
    return y_pred, max_components


def _judge_outliers(
    y: NDArray[np.float64],
    predictions: NDArray[np.float64],
    previous_outlier: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Judge outliers from the ensemble predictions of one iteration.

    A sample is an outlier when its distance from the median prediction
    exceeds ``3 * 1.4826 * MAD``, a robust 3-sigma. The MAD is taken over
    the samples that were inliers in the previous iteration (P3).

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    predictions : ndarray of shape (n_samples, n_estimators)
        Prediction of every submodel for every sample.
    previous_outlier : ndarray of shape (n_samples,)
        Outlier mask of the previous iteration.

    Returns
    -------
    outlier : ndarray of shape (n_samples,)
        Outlier mask of this iteration.
    median : ndarray of shape (n_samples,)
        Median prediction of the ensemble for each sample.
    """
    median = np.median(predictions, axis=1)
    remaining = ~previous_outlier
    mad = np.median(
        np.abs(predictions[remaining] - np.median(median[remaining]))
    )
    outlier = np.abs(y - median) > 3 * 1.4826 * mad
    return outlier, median


class EnsembleOutlierSampleDetector(BaseEstimator):
    """Detect outlier samples of regression data with a submodel ensemble.

    Each iteration fits ``n_estimators`` submodels (PLS or RBF-SVR) on
    bootstrap samples of the current inliers, tunes each one with Optuna
    on its out-of-fold score and predicts every sample. A sample whose
    target is farther than ``3 * 1.4826 * MAD`` from the median
    prediction is an outlier. The iterations stop when the outlier mask
    no longer changes or no inlier is left.

    Parameters
    ----------
    method : {"pls", "PLS", "svr", "SVR"}, default="pls"
        Submodel type.
    max_iter : int, default=30
        Maximum number of iterations. Must be a positive integer.
    n_estimators : int, default=100
        Number of submodels per iteration. Must be a positive integer.
    random_state : int, RandomState instance or None, default=None
        Seeds the bootstraps and the Optuna samplers. Pass an int for
        reproducible results.
    cv : int, default=5
        Number of folds for the out-of-fold predictions.
    metric : {"r2", "mse", "mae", "rmse"}, default="r2"
        Score that tunes the submodels. ``"r2"`` is maximized and the
        others are minimized.
    n_jobs : int or None, default=1
        Number of jobs for the out-of-fold predictions.
    max_components : int, default=30
        Maximum number of PLS components. Must be a positive integer.
        Ignored for SVR.
    progress_bar : bool, default=True
        Whether to show a progress bar during ``fit``.

    Attributes
    ----------
    outlier_support_ : ndarray of shape (n_samples,)
        Boolean mask, True for the outlier samples.
    inlier_support_ : ndarray of shape (n_samples,)
        Boolean mask, True for the inlier samples.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of the features seen during ``fit``. Defined only when
        ``X`` has feature names that are all strings.
    """

    def __init__(
        self,
        method: str = "pls",
        max_iter: int = 30,
        n_estimators: int = 100,
        random_state: int | np.random.RandomState | None = None,
        cv: int = 5,
        metric: str = "r2",
        n_jobs: int | None = 1,
        max_components: int = 30,
        progress_bar: bool = True,
    ) -> None:
        self.method = method
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cv = cv
        self.metric = metric
        self.n_jobs = n_jobs
        self.max_components = max_components
        self.progress_bar = progress_bar

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> "EnsembleOutlierSampleDetector":
        """Detect the outlier samples of ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features. NaN and infinity are not allowed.
        y : array-like of shape (n_samples,)
            Target values. Required despite the default ``None``.

        Returns
        -------
        self : EnsembleOutlierSampleDetector
            The fitted detector.

        Raises
        ------
        ValueError
            If a parameter has an unsupported value, ``y`` is missing,
            or ``X`` and ``y`` are inconsistent or not finite.

        Warns
        -----
        ConvergenceWarning
            If the outlier mask still changes after ``max_iter``
            iterations. The attributes then hold the last judgment.
        """
        method = _submodels.normalize_method(self.method)
        metric = _submodels.resolve_metric(self.metric)
        max_iter = _check_positive_int("max_iter", self.max_iter)
        n_estimators = _check_positive_int("n_estimators", self.n_estimators)
        max_components = _check_positive_int(
            "max_components", self.max_components
        )
        if y is None:
            raise ValueError(
                "y is required: EnsembleOutlierSampleDetector judges "
                "outliers by their target values, but y=None was given."
            )
        X_valid, y_valid = validate_data(self, X, y, y_numeric=True)
        rng = check_random_state(self.random_state)

        n_samples = X_valid.shape[0]
        previous_outlier = np.zeros(n_samples, dtype=bool)
        converged = False
        with (
            _runtime.suppress_optuna_logging(),
            _runtime.make_progress_bar(
                self.progress_bar, max_iter * n_estimators
            ) as pbar,
        ):
            # max_iter >= 1, so the loop binds n_iter, outlier and median.
            for n_iter in range(1, max_iter + 1):
                pbar.set_description(f"[iter {n_iter} / {max_iter}]")
                remaining = ~previous_outlier
                X_remained = X_valid[remaining]
                y_remained = y_valid[remaining]
                predictions = np.empty((n_samples, n_estimators))
                for j in range(n_estimators):
                    predictions[:, j], max_components = _fit_predict_submodel(
                        X_valid,
                        X_remained,
                        y_remained,
                        method=method,
                        metric=metric,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        max_components=max_components,
                        rng=rng,
                    )
                    pbar.update(1)
                outlier, median = _judge_outliers(
                    y_valid, predictions, previous_outlier
                )
                # P4: the mask is unchanged, or no inlier is left.
                if np.array_equal(outlier, previous_outlier) or outlier.all():
                    converged = True
                    break
                previous_outlier = outlier
            pbar.set_description(f"Finished! [iter {n_iter} / {max_iter}]")
            pbar.update((max_iter - n_iter) * n_estimators)

        self.outlier_support_ = outlier
        self.inlier_support_ = ~outlier
        self.n_iter_ = n_iter
        self._median_prediction = median
        if not converged:
            warnings.warn(
                "EnsembleOutlierSampleDetector did not converge within "
                f"max_iter={max_iter} iterations: the outlier mask still "
                "changed in the last iteration. The attributes hold that "
                "last judgment; increase max_iter to let it converge.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def get_outlier_support(self) -> NDArray[np.bool_]:
        """Return the outlier mask.

        Returns
        -------
        ndarray of shape (n_samples,)
            ``outlier_support_``, True for the outlier samples.

        Raises
        ------
        NotFittedError
            If the detector is not fitted yet.
        """
        check_is_fitted(self, ["outlier_support_", "inlier_support_"])
        return self.outlier_support_

    def get_inlier_support(self) -> NDArray[np.bool_]:
        """Return the inlier mask.

        Returns
        -------
        ndarray of shape (n_samples,)
            ``inlier_support_``, True for the inlier samples.

        Raises
        ------
        NotFittedError
            If the detector is not fitted yet.
        """
        check_is_fitted(self, ["outlier_support_", "inlier_support_"])
        return self.inlier_support_
