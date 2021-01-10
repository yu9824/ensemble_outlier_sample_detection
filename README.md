# ensemble_outlier_sample_detection
A method for removing outlier samples.

## How to use.
You can see more details in the [example](https://github.com/yu-9824/ensemble_outlier_sample_detection/tree/main/example).
```python
from ensemble_outlier_sample_detection import EnsembleOutlierSampleDetector

elo = EnsembleOutlierSampleDetector(random_state = 334, n_jobs = -1)
elo.fit(X, y)
elo.outlier_support_    # boolean(np.ndarray)
```

## Reference
### Paper
* https://www.sciencedirect.com/science/article/abs/pii/S0169743917305919?via%3Dihub

### Sites
* https://github.com/hkaneko1985/ensemble_outlier_sample_detection
* https://datachemeng.com/outlier_samples_detectionc_python/