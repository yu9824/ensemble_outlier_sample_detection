import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample
from sklearn.metrics import r2_score
import optuna

class ensemble_outlier_sample_detection:
    def __init__(self, method = 'pls', max_iter = 30, n_estimators = 100, random_state = None, cv = 5, metric = 'r2', n_jobs = 1, max_components = 30):
        '''
        method: 'pls' or 'svr'
        n_estimators: default; 100. int. The number of submodels.
        n_jobs: default; 1(optunaがNoneじゃないため．)
        '''
        if method in ('pls', 'PLS'):
            self.method = 'pls'
            self.max_components = max_components
        else:
            raise NotImplementedError

        if metric == 'r2':
            self.metric = r2_score
            self.direction = 'maximize'
        else:
            self.direction = 'minimize'
            raise NotImplementedError

        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.cv = cv
        self.n_jobs = n_jobs

        self.max_iter_optuna = 100

        self.RANGE = 10 ** 9    # sklearn用の乱数を発生させる範囲
        np.random.seed(random_state)

    def fit(self, X, y):
        n_samples = len(X)

        boolean_outlier = np.zeros(n_samples, dtype = bool)
        boolean_outlier_previous = boolean_outlier.copy()
        for i in tqdm(range(self.max_iter), desc = 'iteration'):
            # 前回の外れサンプル判定の結果残るサンプル
            X_remained = self._extract(X, i = ~boolean_outlier_previous)
            y_remained = self._extract(y, i = ~boolean_outlier_previous)
            n_samples_remained = np.sum(~boolean_outlier_previous)

            # 各estimatorの出した結果を保存しておくリスト（最後にconcat）
            srs_y_pred = []

            # 各estimatorをoofで最適化したのち，予測結果を出す
            for j in tqdm(range(self.n_estimators)):
                # いわゆるBootstrap．毎回違うbootstrapにするため．np.random.randintでrandom_stateを規定
                X_bootstrapped, y_bootstrapped = resample(X_remained, y_remained, random_state = np.random.randint(self.RANGE), n_samples = n_samples_remained)

                # 分散が0の（特徴量がすべてのサンプルで同じ値をとる）特徴量を削除
                selector = VarianceThreshold(threshold = 0)
                X_bootstrapped = selector.fit_transform(X_bootstrapped)
                X = selector.transform(X)

                # 標準化したもので置き換える
                # X
                scaler_X = StandardScaler()
                X_bootstrapped = scaler_X.fit_transform(X_bootstrapped)
                X = scaler_X.transform(X)
                
                # y（二次元じゃないとStandardScaleできないので，どうせnp.ndarrayになることを見越して変換して二次元化した）
                scaler_y = StandardScaler()
                y_bootstrapped = scaler_y.fit_transform(np.array(y_bootstrapped).reshape(-1, 1))

                if self.method == 'pls':
                    model = PLSRegression
                    fixed_params = {
                        'scale':False, 
                    }

                    # 設定したself.max_componentsより有効なサンプル数が少ない場合があるので，それを求める．
                    max_components = min(self.max_components, np.linalg.matrix_rank(X_bootstrapped))

                    # n_componentsの最適化
                    def objective(trial):
                        # suggest_intは[)区間ではなく，[]区間
                        params = {
                            'n_components' : trial.suggest_int('n_components', 1, max_components),
                        }
                        estimator = model(**params, **fixed_params)
                        
                        # Out-of-Foldのyを得る（bootstrapをしている時点ですべてもうランダムに分割されていると考え，ここではshuffleしない．）
                        y_pred_oof = cross_val_predict(estimator, X_bootstrapped, y_bootstrapped, cv = self.cv, n_jobs = self.n_jobs)

                        # scaleを元に戻してscoreを算出．（y_bootstrappedをcross_val_predictの引数にいれたため，同じ形（-1, 1)にすでになっているため，このままscalerを適用できる．
                        return self.metric(scaler_y.inverse_transform(y_pred_oof), scaler_y.inverse_transform(y_bootstrapped))

                # 最適化
                optuna.logging.disable_default_handler()    # optunaのログを非表示
                sampler = optuna.samplers.TPESampler(seed = np.random.randint(self.RANGE))    # 再現性確保
                study = optuna.create_study(sampler = sampler, direction = self.direction)
                study.optimize(objective, n_trials = min(self.max_iter_optuna, max_components), n_jobs = self.n_jobs)

                # 最適なモデルを定義
                estimator = model(**study.best_params, **fixed_params)

                # fit
                estimator.fit(X_bootstrapped, y_bootstrapped)

                # predict
                y_pred = estimator.predict(X)
                
                # scaleを元に戻すしてSeriesとしたものを蓄積（inverse_transformしたものは2次元なのでflatten()）
                srs_y_pred.append(pd.Series(scaler_y.inverse_transform(y_pred).flatten(), name = 'estimator{}'.format(j)))
            
            df_y_pred = pd.concat(srs_y_pred, axis = 1)
            df_y_pred.index = y.index

            # 前回の言うところ，「普通の」（外れ値ではない）サンプル
            df_y_pred_normal = self._extract(df_y_pred, ~boolean_outlier_previous)

            median_absolute_deviation = np.median(abs(df_y_pred_normal - df_y_pred_normal.median(axis = 1).median()))

            y_error = abs(y - df_y_pred.median(axis = 1)).to_numpy()

            boolean_outlier = y_error > 3 * 1.4826 * median_absolute_deviation
            if np.all(boolean_outlier == boolean_outlier_previous):
                self.outlier_support_ = boolean_outlier
                break
            else:
                boolean_outlier_previous = boolean_outlier.copy()
            
    def _extract(self, X, i):
        return X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]

      

if __name__ == '__main__':
    from pdb import set_trace
    example_path = 'https://raw.githubusercontent.com/hkaneko1985/ensemble_outlier_sample_detection/0583863a8381dcde5562197e2398d906c313256f/numerical_simulation_data.csv'
    df = pd.read_csv(example_path, index_col = 0)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    elo = ensemble_outlier_sample_detection(random_state=334, n_jobs = -1)
    elo.fit(X, y)
    
    set_trace()