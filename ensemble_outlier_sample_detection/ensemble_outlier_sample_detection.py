import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class ensemble_outlier_sample_detection:
    def __init__(self):
        pass

    def fit(self):
        pass

if __name__ == '__main__':
    from pdb import set_trace
    example_path = 'https://raw.githubusercontent.com/hkaneko1985/ensemble_outlier_sample_detection/0583863a8381dcde5562197e2398d906c313256f/numerical_simulation_data.csv'
    df = pd.read_csv(example_path, index_col = 0)
    set_trace()