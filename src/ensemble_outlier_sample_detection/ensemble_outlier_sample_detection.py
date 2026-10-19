'''
Copyright © 2021 yu9824
'''

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.utils import check_X_y, check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
import optuna

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class EnsembleOutlierSampleDetector(BaseEstimator):
    def __init__(self, method = 'pls', max_iter = 30, n_estimators = 100, random_state = None, cv = 5, metric = 'r2', n_jobs = 1, max_components = 30, progress_bar = True):
        """You can do outlier detection.

        Parameters
        ----------
        method : str, optional
            'pls'. 'pls' or 'svr', by default 'pls'
        max_iter : int, optional
            Maximum number of iteration., by default 30
        n_estimators : int, optional
            The number of submodels., by default 100
        random_state : None, int or instance of RandomState, optional
            , by default None
        cv : int, optional
            The value of the number of divisions to be made., by default 5
        metric : str, optional
            Specify what scores should be considered when searching for outliers. 'r2' or 'rmse' or 'mae' or 'mse'., by default 'r2'
        n_jobs : int, optional
            , by default 1
        max_components : int, optional
            Maximum number of PLS's component., by default 30
        progress_bar : bool, optional
            Show progress bar or not., by default True
        """
        super().__init__()
        self.method = method
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cv = cv
        self.metric = metric
        self.n_jobs = n_jobs
        self.max_components = max_components
        self.progress_bar = progress_bar
    
    def fit(self, X, y = None):
        # ここでnp.ndarrayに変換されてしまう．
        X, y = check_X_y(X, y)

        # 引数関係の整理
        if self.method in ('pls', 'PLS'):
            self.method = 'pls'
        elif self.method in ('svr', 'SVR'):
            self.method = 'svr'
            self.max_components = float('inf')  # 使用しないが，形式上必要なため．
        else:
            raise NotImplementedError

        if self.metric == 'r2':
            self.metric = r2_score
            self.direction = 'maximize'
        elif self.metric == 'mse':
            self.metric = mean_squared_error
            self.direction = 'minimize'
        elif self.metric == 'mae':
            self.metric = mean_absolute_error
            self.direction = 'minimize'
        elif self.metric == 'rmse':
            self.metric = lambda x, y:mean_squared_error(x, y, squared = False)
            self.direction = 'minimize'
        else:
            raise NotImplementedError
        
        # 乱数関係
        self.RANGE = 2 ** 32
        self.rng_ = check_random_state(self.random_state)

        # optunaの試行回数
        self.max_iter_optuna = 100

        # optunaのログを非表示
        optuna.logging.disable_default_handler()    

        # progress bar
        if self.progress_bar:
            pbar = tqdm(total = self.max_iter * self.n_estimators)

        # 入力されたXを取っておく必要があるため．
        X_original = X.copy()

        # サンプル数
        n_samples = len(X_original)
        
        # 初期配列の定義
        boolean_outlier = np.zeros(n_samples, dtype = np.bool)
        boolean_outlier_previous = boolean_outlier.copy()
        for self.n_iter_ in range(self.max_iter):
            # 前回の外れサンプル判定の結果残るサンプル
            X_remained = self._extract(X_original, i = ~boolean_outlier_previous)
            y_remained = self._extract(y, i = ~boolean_outlier_previous)
            n_samples_remained = np.sum(~boolean_outlier_previous)

            # 各estimatorの出した結果を保存しておくリスト（最後にconcat）
            y_preds = []

            # 各estimatorをoofで最適化したのち，予測結果を出す
            for j in range(self.n_estimators):
                # いわゆるBootstrap．毎回違うbootstrapにするため．np.random.randintでrandom_stateを規定
                self.X_bootstrapped, self.y_bootstrapped = resample(X_remained, y_remained, random_state = self.rng_.randint(self.RANGE), n_samples = n_samples_remained)

                # 分散が0の（特徴量がすべてのサンプルで同じ値をとる）特徴量を削除
                selector = VarianceThreshold(threshold = 0)
                self.X_bootstrapped = selector.fit_transform(self.X_bootstrapped)
                X = selector.transform(X_original)

                # 標準化したもので置き換える
                # X
                scaler_X = StandardScaler()
                self.X_bootstrapped = scaler_X.fit_transform(self.X_bootstrapped)
                X = scaler_X.transform(X)
                
                # y（二次元じゃないとStandardScaleできないので，どうせnp.ndarrayになることを見越して変換して二次元化した）
                self.scaler_y = StandardScaler()
                self.y_bootstrapped = self.scaler_y.fit_transform(np.array(self.y_bootstrapped).reshape(-1, 1))

                # PLS用のmax_componentsよりサンプル数が少なくなってしまう場合があるので．
                if self.method == 'pls':
                    self.max_components = min(self.max_components, np.linalg.matrix_rank(self.X_bootstrapped))

                # インスタンスの生成
                objective = Objective(elo = self)

                # 最適化
                sampler = optuna.samplers.TPESampler(seed = self.rng_.randint(self.RANGE))    # 再現性確保
                study = optuna.create_study(sampler = sampler, direction = self.direction)
                study.optimize(objective, n_trials = min(self.max_iter_optuna, self.max_components))

                # 最適なモデルを定義
                estimator = objective.model(**study.best_params, **objective.fixed_params)

                # fit
                estimator.fit(self.X_bootstrapped, self.y_bootstrapped)

                # predict
                y_pred = estimator.predict(X)
                
                # scaleを元に戻すして貯めておく
                y_preds.append(self.scaler_y.inverse_transform(y_pred))

                # progressbarを一つ進める．
                if self.progress_bar:
                    pbar.set_description(desc = '[iter {0} / {1}]'.format(self.n_iter_, self.max_iter))
                    pbar.update(1)
            
            # アンサンブルした結果を一つのnp.ndarrayにまとめる．
            df_y_preds = pd.DataFrame(np.hstack(y_preds))

            # 前回の言うところ，「普通の」（外れ値ではない）サンプルだけを抽出
            df_y_preds_normal = df_y_preds[~boolean_outlier_previous]

            # median_absolute_deviation（中央絶対偏差）
            median_absolute_deviation = np.median(abs(df_y_preds_normal - df_y_preds_normal.median(axis = 1).median()))

            # 偏差を求める
            y_error = abs(y - df_y_preds.median(axis = 1)).to_numpy()

            # 所謂3σに変わる基準でそれを超えるならばoutlierであると一旦判定
            boolean_outlier = y_error > 3 * 1.4826 * median_absolute_deviation

            # 外れサンプル判定に変更がなかったら．
            if np.all(boolean_outlier == boolean_outlier_previous) or np.sum(~boolean_outlier) == 0:
                self.outlier_support_ = boolean_outlier
                self.inlier_support_ = ~self.outlier_support_
                self.n_iter_finished = self.n_iter_
                break
            else:
                boolean_outlier_previous = boolean_outlier.copy()
            
        # プログレスバーを閉じる
        if self.progress_bar:
            pbar.set_description(desc = 'Finished! [iter {0} / {1}]'.format(self.n_iter_finished + 1, self.max_iter))
            pbar.update((self.max_iter - (self.n_iter_finished + 1)) * self.n_estimators)
            pbar.close()

    def get_outlier_support(self):
        '''
        You can get outlier support_.
        '''
        check_is_fitted(self, ['outlier_support_', 'inlier_support_'])
        return self.outlier_support_

    def get_inlier_support(self):
        '''
        You can get inlier support_.
        '''
        check_is_fitted(self, ['outlier_support_', 'inlier_support_'])
        return self.inlier_support_
            
    def _extract(self, X, i):
        return X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]

class Objective:
    def __init__(self, elo):
        self.elo = elo
        if elo.method == 'pls':
            self.model = PLSRegression
            self.fixed_params = {
                'scale':False,
            }
        elif elo.method == 'svr':
            self.model = SVR
            self.fixed_params = {
                'kernel': 'rbf',
            }

    def __call__(self, trial):
        if self.elo.method == 'pls':
        # suggest_intは[)区間ではなく，[]区間
            params = {
                'n_components' : trial.suggest_int('n_components', 1, self.elo.max_components),
            }
        elif self.elo.method == 'svr':
            params = {
                'C': trial.suggest_loguniform('C', 2 ** -5, 2 ** 11),
                'epsilon': trial.suggest_loguniform('epsilon', 2 ** -10, 2 ** 1),
                'gamma': trial.suggest_loguniform('gamma', 2 ** -20, 2 ** 11),
            }

        estimator = self.model(**params, **self.fixed_params)
        
        # Out-of-Foldのyを得る（bootstrapをしている時点ですべてもうランダムに分割されていると考え，ここではshuffleしない．）
        y_pred_oof = cross_val_predict(estimator, self.elo.X_bootstrapped, self.elo.y_bootstrapped, cv = self.elo.cv, n_jobs = self.elo.n_jobs)

        # scaleを元に戻してscoreを算出．（y_bootstrappedをcross_val_predictの引数にいれたため，同じ形（-1, 1)にすでになっているため，このままscalerを適用できる．
        return self.elo.metric(self.elo.scaler_y.inverse_transform(y_pred_oof), self.elo.scaler_y.inverse_transform(self.elo.y_bootstrapped))

if __name__ == '__main__':
    from pdb import set_trace

    example_path = 'example/example_chache.csv'
    df = pd.read_csv(example_path, index_col = 0)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    elo = EnsembleOutlierSampleDetector(random_state = 334, n_jobs = -1, cv = 5)
    elo.fit(X, y)
    
    set_trace()