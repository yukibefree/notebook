"""
モデル学習と予測を行うモジュール
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

class TimeSeriesModel:
    """時系列予測モデルのクラス"""
    
    def __init__(self, model_type: str = 'lightgbm', **kwargs):
        """
        初期化
        
        Args:
            model_type (str): モデルタイプ ('lightgbm', 'random_forest', 'svr')
            **kwargs: モデル固有のパラメータ
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.selected_features = None
        self.feature_importance = None
        
        # デフォルトパラメータ
        if model_type == 'lightgbm':
            self.default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            }
        elif model_type == 'random_forest':
            self.default_params = {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif model_type == 'svr':
            self.default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1
            }
        
        # カスタムパラメータで更新
        self.default_params.update(kwargs)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                               n_trials: int = 100, cv_splits: int = 5) -> Dict:
        """
        ハイパーパラメータの最適化
        
        Args:
            X (pd.DataFrame): 特徴量
            y (pd.Series): 目的変数
            n_trials (int): 最適化試行回数
            cv_splits (int): クロスバリデーション分割数
            
        Returns:
            Dict: 最適なパラメータ
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        def objective(trial):
            if self.model_type == 'lightgbm':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'verbose': -1,
                    'random_state': 42
                }
            elif self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42
                }
            elif self.model_type == 'svr':
                params = {
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                    'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True)
                }
            
            rmses = []
            for train_idx, valid_idx in tscv.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                if self.model_type == 'lightgbm':
                    train_data = lgb.Dataset(X_train, y_train)
                    valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)
                    model = lgb.train(params, train_data, valid_sets=[valid_data], 
                                    num_boost_round=1000, 
                                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
                    y_pred = model.predict(X_valid)
                elif self.model_type == 'random_forest':
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_valid)
                elif self.model_type == 'svr':
                    model = SVR(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_valid)
                
                rmse = root_mean_squared_error(y_valid, y_pred)
                rmses.append(rmse)
            
            return np.mean(rmses)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        if self.model_type == 'lightgbm':
            self.best_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            })
        
        print(f'{self.model_type.upper()}：ベイズ最適化の結果')
        print('Best params:', self.best_params)
        print('Best CV RMSE:', study.best_value)
        
        return self.best_params
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              optimize: bool = True, n_trials: int = 100) -> Any:
        """
        モデルの学習
        
        Args:
            X (pd.DataFrame): 特徴量
            y (pd.Series): 目的変数
            optimize (bool): ハイパーパラメータ最適化を行うか
            n_trials (int): 最適化試行回数
            
        Returns:
            Any: 学習済みモデル
        """
        if optimize:
            self.optimize_hyperparameters(X, y, n_trials=n_trials)
        else:
            self.best_params = self.default_params
        
        if self.model_type == 'lightgbm':
            train_data = lgb.Dataset(X, y)
            self.model = lgb.train(self.best_params, train_data, num_boost_round=1000)
            
            # 特徴量重要度で下位20%を除外し再学習
            importances = self.model.feature_importance(importance_type='gain')
            threshold = np.percentile(importances, 20)
            self.selected_features = [f for f, imp in zip(X.columns, importances) if imp > threshold]
            self.feature_importance = dict(zip(X.columns, importances))
            
            if len(self.selected_features) < len(X.columns):
                print(f'Selected features: {len(self.selected_features)}/{len(X.columns)}')
                train_data_selected = lgb.Dataset(X[self.selected_features], y)
                self.model = lgb.train(self.best_params, train_data_selected, num_boost_round=1000)
            else:
                self.selected_features = list(X.columns)
                
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.best_params)
            self.model.fit(X, y)
            self.selected_features = list(X.columns)
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
        elif self.model_type == 'svr':
            self.model = SVR(**self.best_params)
            self.model.fit(X, y)
            self.selected_features = list(X.columns)
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測
        
        Args:
            X (pd.DataFrame): 予測対象の特徴量
            
        Returns:
            np.ndarray: 予測結果
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        if self.selected_features is not None:
            X = X[self.selected_features]
        
        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_sequential(self, X_test: pd.DataFrame, train_last_values: Dict[str, float] = None) -> np.ndarray:
        """
        逐次予測（ラグ特徴量を考慮）
        
        Args:
            X_test (pd.DataFrame): テストデータ
            train_last_values (Dict[str, float]): 訓練データの最後の値（ラグ特徴量の初期化用）
            
        Returns:
            np.ndarray: 逐次予測結果
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        X_test_copy = X_test.copy()
        predictions = []
        
        # ラグ特徴量の初期化
        if train_last_values:
            for lag_col, value in train_last_values.items():
                if lag_col in X_test_copy.columns:
                    X_test_copy.loc[X_test_copy.index[0], lag_col] = value
        
        # 逐次予測
        for i in range(len(X_test_copy)):
            # 現在の行の特徴量を取得
            current_features = X_test_copy.iloc[i:i+1]
            
            # 予測
            pred = self.predict(current_features)[0]
            predictions.append(pred)
            
            # 次の行のラグ特徴量を更新
            if i + 1 < len(X_test_copy):
                # price_actual_lag1 を更新
                lag1_col = 'price_actual_lag1'
                if lag1_col in X_test_copy.columns:
                    X_test_copy.loc[X_test_copy.index[i+1], lag1_col] = pred
                
                # price_actual_lag24 を更新（24時間後）
                lag24_col = 'price_actual_lag24'
                if lag24_col in X_test_copy.columns and i + 24 < len(X_test_copy):
                    X_test_copy.loc[X_test_copy.index[i+24], lag24_col] = pred
        
        return np.array(predictions)

def create_submission(predictions: np.ndarray, test_df: pd.DataFrame, 
                     output_path: str, filename: str = 'submission', target_col: str = 'price_actual') -> None:
    """
    提出ファイルの作成
    
    Args:
        predictions (np.ndarray): 予測結果
        test_df (pd.DataFrame): テストデータ
        output_path (str): 出力パス
        filename（str）: ファイル名
        target_col (str): 目的変数カラム名
    """
    submission = test_df[['time']].copy()
    submission[target_col] = predictions
    
    # フォーマット確認
    if 'time' in test_df.columns:
        assert submission.iloc[0, 0] == '2018-01-01 00:00:00+01:00', '1行1列目が要件を満たしません'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_filename = output_path / f'{filename}_{timestamp}.csv'
    submission.to_csv(submission_filename, index=False, header=False)
    print(f'Submission saved: {submission_filename}')

def train_and_predict(model_type: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     target_col: str = 'price_actual', optimize: bool = True,
                     sequential: bool = False, output_path: str = None, filename: str = 'submission') -> Tuple[TimeSeriesModel, np.ndarray]:
    """
    モデルの学習から予測まで一括実行
    
    Args:
        model_type (str): モデルタイプ
        train_df (pd.DataFrame): 訓練データ
        test_df (pd.DataFrame): テストデータ
        target_col (str): 目的変数カラム名
        optimize (bool): ハイパーパラメータ最適化を行うか
        sequential (bool): 逐次予測を行うか
        output_path (str): 提出ファイルの出力パス
        filename（str）: ファイル名
        
    Returns:
        Tuple[TimeSeriesModel, np.ndarray]: 学習済みモデルと予測結果
    """
    # 特徴量と目的変数の準備
    drop_cols = ['time', target_col] if target_col in train_df.columns else ['time']
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col] if target_col in train_df.columns else train_df.iloc[:, -1]
    X_test = test_df[feature_cols]
    
    # モデルの学習
    model = TimeSeriesModel(model_type)
    model.train(X_train, y_train, optimize=optimize)
    
    # 予測
    if sequential:
        # 訓練データの最後の値でラグ特徴量を初期化
        train_last_values = {}
        if 'price_actual_lag1' in X_test.columns:
            train_last_values['price_actual_lag1'] = y_train.iloc[-1]
        if 'price_actual_lag24' in X_test.columns and len(y_train) >= 24:
            train_last_values['price_actual_lag24'] = y_train.iloc[-24]
        
        predictions = model.predict_sequential(X_test, train_last_values)
    else:
        predictions = model.predict(X_test)
    
    # 提出ファイルの作成
    if output_path:
        create_submission(predictions, test_df, output_path, filename, target_col)
    
    return model, predictions
