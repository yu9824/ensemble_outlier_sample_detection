# Ensemble Outlier Sample Detection
![python_badge](https://img.shields.io/pypi/pyversions/ensemble-outlier-sample-detection)
![license_badge](https://img.shields.io/pypi/l/ensemble-outlier-sample-detection)
![Total_Downloads_badge](https://pepy.tech/badge/ensemble-outlier-sample-detection)

A method for removing outlier samples.

## How to use.
You can see more details in the [example](https://github.com/yu9824/ensemble_outlier_sample_detection/tree/main/example).
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