# Changelog

All notable changes to this project are documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

Version 1.0.0 supports current Python, NumPy, scikit-learn and Optuna
releases, and makes `EnsembleOutlierSampleDetector` follow the
scikit-learn estimator conventions. The detection algorithm is
unchanged except for the fixes listed below. Regression tests require
the outlier masks and iteration counts to match results recorded from
the 0.1.0 code, which was patched only so that it runs (and, for
`metric="r2"`, with the R² fix). See
[Migrating from 0.x](#migrating-from-0x) for code examples.

### Breaking changes

- `fit` returns the fitted detector (`self`) instead of `None`.
- `n_iter_` is the number of iterations that `fit` ran, between 1 and
  `max_iter`. In 0.1.0 it was the 0-based index of the last iteration,
  so the new value is larger by one.
- `n_iter_finished` has been removed. Its value in 0.1.0 equals
  `n_iter_ - 1` in 1.0.0.
- An unsupported `method` or `metric` raises `ValueError`, whose
  message lists the accepted values, instead of `NotImplementedError`.
  `max_iter`, `n_estimators` and `max_components` must be positive
  integers, and other values raise `ValueError` when `fit` is called.
  0.1.0 did not check them, and such values typically failed later with
  unrelated errors.
- `fit` no longer modifies the constructor parameters. In 0.1.0 it
  lower-cased `method`, replaced `metric` with a function and changed
  `max_components` (clipped to the rank of the bootstrap samples for
  PLS, set to infinity for SVR), so `get_params()` changed after `fit`,
  and calling `fit` again or fitting a `clone` of a fitted detector
  raised `NotImplementedError`. Now `get_params()` returns the same values
  before and after `fit`, a second `fit` overwrites the previous
  results, and `sklearn.base.clone` works.
- `fit` no longer stores its intermediate state in public attributes.
  The attributes `rng_`, `direction`, `RANGE`, `max_iter_optuna`,
  `X_bootstrapped`, `y_bootstrapped` and `scaler_y` are no longer set.
  The public fitted attributes are `outlier_support_`,
  `inlier_support_`, `n_iter_`, `n_features_in_` and
  `feature_names_in_`.
- Python 3.9 and earlier are no longer supported; Python 3.10 or later
  is required. The minimum versions of the dependencies are NumPy 1.23,
  scikit-learn 1.6, Optuna 3.0 and tqdm 4.40.
- pandas is no longer a runtime dependency. DataFrame and Series inputs
  still work, but pandas must be installed separately (or through the
  `example` extra).
- The package namespace exports only `EnsembleOutlierSampleDetector`.
  The `Objective` class has been removed. In 0.1.0 it was reachable as
  `ensemble_outlier_sample_detection.Objective` because the package
  star-imported its implementation module, which also exposed names
  such as `np`, `pd`, `optuna` and `tqdm`; none of these are available
  from the package any more.
- The module attributes `__author_email__`, `__user_id__`,
  `__app_name__` and `__url__`, which only `setup.py` used, have been
  removed.
- The package is built from `pyproject.toml` with a `src/` layout.
  `setup.py` and `requirements.txt` have been removed. This affects
  only builds and installs from a source checkout.

### Changed

- `metric="r2"` computes R² as `r2_score(y_true, y_pred)`. 0.1.0 passed
  the arguments in the reverse order, and R² is not symmetric in them.
  Because `"r2"` is the default metric, the tuned hyperparameters and
  the detected outliers can differ from those of 0.1.0 even with the
  same `random_state`. The other metrics are symmetric and unaffected.
- New fitted attributes: `n_features_in_`, and `feature_names_in_` when
  `X` has column names that are all strings.
- If the outlier mask still changes after `max_iter` iterations, `fit`
  emits a `sklearn.exceptions.ConvergenceWarning`, and the fitted
  attributes hold the judgment of the last iteration.
- The progress bar comes from `tqdm.auto`: a widget in Jupyter and text
  elsewhere. The iteration shown in the bar starts from 1, and the bar
  is closed even when `fit` raises.
- `__version__` is derived from the git tag at build time, and
  `__license__` is the SPDX identifier `"Apache-2.0"` (previously
  `"Apache License 2.0"`).
- The package ships type hints (`py.typed`), and the public API has
  NumPy-style docstrings.

### Fixed

- `method="svr"` works. In 0.1.0 it failed whenever `n_estimators` was
  greater than 1, including the default of 100, because the 1-D SVR
  predictions were concatenated into one long array (`ValueError: Item
  wrong length ...`). With current scikit-learn (checked with 1.9) it
  failed for any `n_estimators`, because
  `StandardScaler.inverse_transform` rejects the 1-D predictions. It
  also no longer emits `DataConversionWarning`s.
- `fit` no longer raises `AttributeError` when it does not converge
  within `max_iter` iterations, and it no longer leaves
  `outlier_support_` and `inlier_support_` unset in that case.
- `fit` restores Optuna's logging configuration. During `fit`, Optuna's
  verbosity is set to `WARNING` to hide the per-trial logs, and the
  previous verbosity is restored afterwards, also when `fit` raises.
  0.1.0 called `optuna.logging.disable_default_handler()`, which
  removed Optuna's log output for the rest of the Python session.
- Compatibility with current dependencies. 0.1.0 failed with
  NumPy 1.24 to 1.26, which removed `np.bool` (NumPy 2.0 reintroduced
  it), and with `metric="rmse"` on scikit-learn 1.6 or later, which
  removed the `squared` argument of `mean_squared_error`. It also used
  `trial.suggest_loguniform`, which Optuna 3.0 deprecated. 1.0.0 uses
  none of these APIs and emits no `DeprecationWarning` or
  `FutureWarning` of its own. It requires NumPy 1.23, scikit-learn 1.6
  and Optuna 3.0 or later, and CI runs the tests with the latest
  releases on Python 3.10 to 3.14.
- Running the implementation module as a script no longer starts a
  debugger; the debug harness that ended in `pdb.set_trace()` has been
  removed.

### Migrating from 0.x

`fit` returns the detector, so the result can be read in one chain. The
0.1.0 style still works.

```python
# 0.1.0
elo = EnsembleOutlierSampleDetector(random_state=334, n_jobs=-1)
elo.fit(X, y)
outlier = elo.outlier_support_

# 1.0.0
elo = EnsembleOutlierSampleDetector(random_state=334, n_jobs=-1).fit(X, y)
outlier = elo.outlier_support_
```

`n_iter_finished` is gone, and `n_iter_` counts the iterations.

```python
# 0.1.0
last_index = elo.n_iter_finished  # 0-based index of the last iteration
n_iterations = elo.n_iter_finished + 1

# 1.0.0
last_index = elo.n_iter_ - 1  # same value as n_iter_finished in 0.1.0
n_iterations = elo.n_iter_
```

Unsupported values raise `ValueError` instead of `NotImplementedError`.

```python
# 0.1.0
try:
    EnsembleOutlierSampleDetector(metric="accuracy").fit(X, y)
except NotImplementedError:
    ...

# 1.0.0
try:
    EnsembleOutlierSampleDetector(metric="accuracy").fit(X, y)
except ValueError:
    ...
```

Non-convergence is a warning instead of an `AttributeError`. To treat
it as an error, turn the warning into an exception.

```python
import warnings

from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    # Raises ConvergenceWarning if fit does not converge within
    # max_iter iterations.
    elo.fit(X, y)
```

A detector can be fitted again and cloned, so there is no need to
create a new instance for each dataset.

```python
from sklearn.base import clone

elo.fit(X_train, y_train)
elo.fit(X_other, y_other)  # overwrites the previous results
twin = clone(elo)  # same parameters, not fitted
```

Other points to check:

- Results with `metric="r2"` (the default) can differ from 0.1.0. There
  is no option to restore the old argument order.
- If your code relied on Optuna staying silent after `fit`, set the
  verbosity yourself with
  `optuna.logging.set_verbosity(optuna.logging.WARNING)`.
- If you pass pandas objects, install pandas explicitly.
- On Python 3.9 and earlier, keep using 0.x
  (`pip install "ensemble-outlier-sample-detection<1"`).

## [0.1.0] - 2021-07-15

Initial release.
