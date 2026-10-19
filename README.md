# Ensemble Outlier Sample Detection
![python_badge](https://img.shields.io/pypi/pyversions/ensemble-outlier-sample-detection)
![license_badge](https://img.shields.io/pypi/l/ensemble-outlier-sample-detection)
![Total_Downloads_badge](https://pepy.tech/badge/ensemble-outlier-sample-detection)

Ensemble-based outlier sample detection for regression data.

`EnsembleOutlierSampleDetector` flags the samples whose target values do
not agree with an ensemble of PLS or SVR models fitted on bootstrap
samples, following the method of Kaneko (see Reference below).

This README describes version 1.0.0. Version 1.0.0 has breaking changes
from 0.x; see Upgrading from 0.x below.

## Installation

```bash
pip install ensemble-outlier-sample-detection
```

- Python 3.10 or later is required. The package is tested on Python
  3.10 to 3.14.
- The runtime dependencies are NumPy (>= 1.23), scikit-learn (>= 1.6),
  Optuna (>= 3.0) and tqdm (>= 4.40). pip installs them automatically.
- pandas is optional. You need it only to pass pandas DataFrames or
  Series, or to run the example notebook.

## Usage

```python
import numpy as np

from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

rng = np.random.RandomState(0)
X = rng.normal(size=(100, 5))
y = X @ rng.normal(size=5) + 0.1 * rng.normal(size=100)
y[:3] += 10.0  # make the first three samples outliers

elo = EnsembleOutlierSampleDetector(random_state=334, n_jobs=-1).fit(X, y)
elo.outlier_support_  # boolean ndarray of shape (n_samples,), True = outlier
elo.inlier_support_  # boolean ndarray, the complement of outlier_support_
elo.n_iter_  # number of iterations that fit ran
print(np.flatnonzero(elo.outlier_support_))  # indices of the outliers
```

`fit` returns the detector itself, so the call can be chained. `X` can
be a NumPy array or a pandas DataFrame, and `y` is a 1-D numeric target.
To use the result, select the inliers with `X[elo.inlier_support_]` (or
`df[elo.inlier_support_]` for a DataFrame).

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `"pls"` | Submodel: `"pls"` (PLS regression) or `"svr"` (SVR with an RBF kernel). `"PLS"` and `"SVR"` are also accepted. |
| `metric` | `"r2"` | Score of the out-of-fold predictions that tunes each submodel: `"r2"` (maximized), `"mse"`, `"mae"` or `"rmse"` (minimized). |
| `n_estimators` | `100` | Number of submodels in each iteration. |
| `max_iter` | `30` | Maximum number of iterations. |
| `cv` | `5` | Number of cross-validation folds for the out-of-fold predictions. |
| `max_components` | `30` | Maximum number of PLS components. Must be a positive integer. It is validated but otherwise ignored for SVR. |
| `n_jobs` | `1` | Number of jobs for the cross-validation. `-1` uses all processors. |
| `random_state` | `None` | Seed of the bootstraps and the Optuna samplers. Pass an int for reproducible results. |
| `progress_bar` | `True` | Whether to show a progress bar during `fit` (a widget in Jupyter, text elsewhere). |

`fit` raises `ValueError` for an unsupported `method` or `metric`, for
`max_iter`, `n_estimators` or `max_components` that is not a positive
integer, and for invalid inputs (missing `y`, mismatched lengths, NaN or
infinity).

### Fitted attributes

| Attribute | Description |
| --- | --- |
| `outlier_support_` | Boolean mask of shape `(n_samples,)`, True for an outlier. Also returned by `get_outlier_support()`. |
| `inlier_support_` | Boolean mask of shape `(n_samples,)`, `~outlier_support_`. Also returned by `get_inlier_support()`. |
| `n_iter_` | Number of iterations run, between 1 and `max_iter`. |
| `n_features_in_` | Number of features seen during `fit`. |
| `feature_names_in_` | Column names of `X`. Set only when `X` has column names that are all strings, such as a DataFrame. |

If the outlier mask still changes after `max_iter` iterations, `fit`
emits a `sklearn.exceptions.ConvergenceWarning`, and the attributes hold
the judgment of the last iteration.

## How it works

Each iteration of `fit` runs these steps.

1. Take the samples that the previous iteration judged as inliers (all
   samples in the first iteration).
2. Fit `n_estimators` submodels, each on a bootstrap sample of those
   samples, after dropping constant features and standardizing `X` and
   `y`. The hyperparameters of each submodel (`n_components` for PLS;
   `C`, `epsilon` and `gamma` for SVR) are tuned with the TPE sampler of
   Optuna, which optimizes `metric` on the out-of-fold predictions of
   `cv`-fold cross-validation. The tuned submodel then predicts every
   sample.
3. Take the median of the ensemble predictions of each sample. A sample
   is an outlier when `|y - median| > 3 * 1.4826 * MAD`, a robust
   counterpart of the 3-sigma rule, where the MAD is computed from the
   ensemble predictions of the previous inliers.

The iterations stop when the outlier mask is the same as in the
previous iteration, or when every sample is an outlier. Optuna's
per-trial logs are hidden during `fit`, and Optuna's logging
configuration is restored afterwards. See
`help(EnsembleOutlierSampleDetector)` for the details.

## Example

The [example notebook](https://github.com/yu9824/ensemble_outlier_sample_detection/tree/main/example)
runs the detector on the bundled example data. It also needs pandas,
matplotlib and seaborn, which the `example` extra installs:

```bash
pip install "ensemble-outlier-sample-detection[example]"
```

## Upgrading from 0.x

Version 1.0.0 changes the API in a few incompatible ways: `fit` returns
the detector, `n_iter_` is the number of iterations (the old
`n_iter_finished` is gone), unsupported values raise `ValueError`,
Python 3.9 and earlier are no longer supported, and results with
`metric="r2"` can differ. See the
[CHANGELOG](https://github.com/yu9824/ensemble_outlier_sample_detection/blob/main/CHANGELOG.md)
for the full list and migration examples.

## Reference
### Paper
* https://www.sciencedirect.com/science/article/abs/pii/S0169743917305919?via%3Dihub

### Sites
* https://github.com/hkaneko1985/ensemble_outlier_sample_detection
* https://datachemeng.com/outlier_samples_detectionc_python/


## LICENSE
Copyright © 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
