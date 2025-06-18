"""
データ前処理のモジュール
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import japanize_matplotlib

class DataPreprocessor:
    """データ前処理を行うクラス"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初期化

        Args:
            config (Optional[Dict]): 設定パラメータ
        """
        self.config = config or {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_groups: Dict[str, List[str]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理を実行（訓練データ用）

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 前処理済みのデータフレーム
        """
        # 特徴量のグループ化
        self.feature_groups = self._get_feature_groups(df)

        # インデックスの処理
        df = self._convert_to_datetime(df, utc=True, tz='Europe/Berlin')
        
        # 欠損値の補完
        df = self._fill_missing_values(df)

        # 外れ値の処理
        df = self._handle_outliers(df)

        # 特徴量エンジニアリング
        df = self._engineer_features(df)
        
        # ラベルエンコードの実施
        df = self._label_encoder(df)

        # スケーリング
        df = self._scale_features(df)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理を実行（テストデータ用）

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 前処理済みのデータフレーム
        """
        
        # インデックスの処理
        df = self._convert_to_datetime(df, utc=True, tz='Europe/Berlin')
        
        # 欠損値の補完
        df = self._fill_missing_values(df)

        # 特徴量エンジニアリング
        df = self._engineer_features(df)
        
        # ラベルエンコードの実施
        df = self._label_encoder(df)
        
        # スケーリング
        df = self._scale_features(df, is_training=False)

        return df
      
    # 時系列データの変換
    def _convert_to_datetime(self, df, utc=False, tz='Asia/Tokyo'):
      try:
        df['time'] = pd.to_datetime(df.index, utc=utc).tz_convert(tz)
        print('データ型：',df.index.dtype)
        print('インデックスをdatetimeに変換しました')
        
        return df
      except Exception as e:
        print(f"datetime変換に失敗しました: {e}")
        
        return df
    
    
    def _get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        特徴量をグループ化する

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            Dict[str, List[str]]: グループ名と特徴量のリストの辞書
        """
        return {
            'time': ['hour', 'day_of_week', 'month', 'year'],
            'generation': [col for col in df.columns if 'generation' in col],
            'weather': [col for col in df.columns if any(x in col for x in ['temperature', 'wind_speed', 'solar_radiation'])],
            'load': [col for col in df.columns if 'load' in col],
            'price_actual': ['price_actual'] if 'price_actual' in df.columns else []
        }

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を補完する

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 欠損値を補完したデータフレーム
        """
        # 時間関連の特徴量は線形補間
        time_features = self.feature_groups['time']
        df[time_features] = df[time_features].interpolate(method='linear')

        # 発電量は0で補完
        generation_features = self.feature_groups['generation']
        df[generation_features] = df[generation_features].fillna(0)

        # 気象データは前後の平均で補完
        weather_features = self.feature_groups['weather']
        df[weather_features] = df[weather_features].interpolate(method='linear')

        # 需要データは前後の平均で補完
        load_features = self.feature_groups['load']
        df[load_features] = df[load_features].interpolate(method='linear')

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        外れ値を処理する

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 外れ値を処理したデータフレーム
        """
        # 数値型の特徴量に対して外れ値処理を実行
        numeric_features = df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature in ['time', 'hour', 'day_of_week', 'month', 'year']:
                continue

            # IQR法による外れ値の検出
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 外れ値を境界値に置き換え
            df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量エンジニアリングを実行する

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 特徴量を追加したデータフレーム
        """
        # 時間関連の特徴量
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['day_of_week'].isin([0, 6]).astype(int)  # 日曜と祝日
        df['is_peak_hour'] = df['hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).astype(int)

        # 発電量の比率
        generation_features = self.feature_groups['generation']
        if len(generation_features) > 1:
            total_generation = df[generation_features].sum(axis=1)
            for feature in generation_features:
                df[f'{feature}_ratio'] = df[feature] / total_generation

        # 需要の比率
        load_features = self.feature_groups['load']
        if len(load_features) > 1:
            total_load = df[load_features].sum(axis=1)
            for feature in load_features:
                df[f'{feature}_ratio'] = df[feature] / total_load

        return df
      
    def _label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ラベルエンコードを行う関数
        
        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: ラベルエンコード変換したデータフレーム
        """
        df_cleaned = df.copy()
        
        # 数値型に変換できる列を特定
        numeric_columns = []
        categorical_columns = []
        
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            # 数値に変換できるかテスト
            try:
                pd.to_numeric(df_cleaned[col], errors='raise')
                numeric_columns.append(col)
            except:
                categorical_columns.append(col)
        
        print(f'数値型に変換可能な列: {numeric_columns}')
        print(f'カテゴリカル列: {categorical_columns}')
        
        # 数値型に変換
        for col in numeric_columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # カテゴリカル列をラベルエンコード
        for col in categorical_columns:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df[col].values)
        
        return df_cleaned

    def _scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        特徴量のスケーリングを実行する

        Args:
            df (pd.DataFrame): 入力データフレーム
            is_training (bool, optional): 訓練データかどうか. Defaults to True.

        Returns:
            pd.DataFrame: スケーリングしたデータフレーム
        """
        # スケーリング対象の特徴量
        scale_features = (
            self.feature_groups['generation'] +
            self.feature_groups['weather'] +
            self.feature_groups['load']
        )

        if is_training:
            # 訓練データの場合、スケーラーを学習
            for feature in scale_features:
                self.scalers[feature] = StandardScaler()
                df[feature] = self.scalers[feature].fit_transform(df[[feature]])
        else:
            # テストデータの場合、学習済みのスケーラーを使用
            for feature in scale_features:
                if feature in self.scalers:
                    df[feature] = self.scalers[feature].transform(df[[feature]])

        return df 