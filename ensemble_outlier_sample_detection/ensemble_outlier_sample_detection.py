import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class ensemble_outlier_sample_detection:
    def __init__(self, method = 'pls', max_iter = 30, n_estimators = 100, random_state = None):
        '''
        method: 'pls' or 'svr'
        n_estimators: default; 100. int. The number of submodels.
        '''
        if method in ('pls', 'PLS'):
            pass
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        np.random.seed(random_state)

    def fit(self, X, y):
        n_samples = len(X)

        boolean_outlier = np.zeros(n_samples, dtype = bool)
        boolean_outlier_previous = boolean_outlier.copy()
        for i in tqdm(range(self.max_iter)):
            # 前回の外れサンプル判定の結果残るサンプル
            X_remained = self._extract(X, i = ~boolean_outlier_previous)
            y_remained = self._extract(y, i = ~boolean_outlier_previous)
            n_samples_remained = np.sum(~boolean_outlier_previous)

            for j in range(self.n_estimators):
                # いわゆるBootstrap
                # sklearn.utils.resampleを使うことも考えたが，あれは毎回random_stateを指定しないと再現性がとれないが，そのrandom_stateを決めるためにrandom_stateを生成して,,,となり逆に微妙と考えたため．
                i_bootstrapped = np.random.randint(low = 0, high = n_samples_remained, size = n_samples_remained)
                X_bootstrapped = self._extract(X_remained, i = i_bootstrapped)
                y_bootstrapped = self._extract(y_remained, i = i_bootstrapped)

                # 分散が0の（特徴量がすべてのサンプルで同じ値をとる）特徴量を削除
                selector = VarianceThreshold(threshold = 0)
                X_bootstrap_selected = selector.fit_transform(X_bootstrapped)
                X_selected = selector.transform(X)

                # 標準化したもので置き換える
                scaler_X = StandardScaler()
                X_bootstrap_selected = scaler_X.fit_transform(X_bootstrap_selected)
                X_selected = scaler_X.transform(X_bootstrapped)
                
                scaler_y = StandardScaler()
                y_bootstrapped = scaler_y.fit_transform(np.array(y_bootstrapped).reshape(-1, 1))

                
            

            
            


    
    def _extract(self, X, i):
        return X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]

        
        

if __name__ == '__main__':
    from pdb import set_trace
    example_path = 'https://raw.githubusercontent.com/hkaneko1985/ensemble_outlier_sample_detection/0583863a8381dcde5562197e2398d906c313256f/numerical_simulation_data.csv'
    df = pd.read_csv(example_path, index_col = 0)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    elo = ensemble_outlier_sample_detection(random_state=334)
    elo.fit(X, y)
    
    set_trace()