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
        self.holiday_checker = HolidayChecker()

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
        
        # 祝日情報を追加
        df = self.holiday_checker.fit(df)
        
        # ラベルエンコードの実施
        df = self._label_encoder(df)

        # スケーリング
        df = self._scale_features(df)
        
        print('-'*20, ' 前処理終了 ', '-'*20)

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
        
        # 祝日情報を追加
        df = self.holiday_checker.fit(df)
        
        # ラベルエンコードの実施
        df = self._label_encoder(df)
        
        # スケーリング
        df = self._scale_features(df, is_training=False)
        
        print('-'*20, ' 前処理終了 ', '-'*20)

        return df
      
    # 時系列データの変換
    def _convert_to_datetime(self, df, utc=False, tz='Asia/Tokyo'):
      try:
        df.index = pd.to_datetime(df.index, utc=utc).tz_convert(tz)
        df['time'] = df.index
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
            'price_actual': ['price_actual'] if 'price_actual' in df.columns else [],
            'holiday' : [
              'is_holiday_or_weekend_flag',
              'is_next_day_holiday_or_weekend_flag',
              'is_previous_day_holiday_or_weekend_flag',
              'consecutive_holiday_or_weekend_flag'
              ]
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
        # 新しい特徴量を格納する辞書
        new_features = {}
        
        # 時間関連の特徴量
        new_features['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        new_features['is_holiday'] = df['day_of_week'].isin([0, 6]).astype(int)  # 日曜と祝日
        new_features['is_peak_hour'] = df['hour'].isin([9, 10, 11, 12, 13, 18, 19, 20, 21, 22]).astype(int)
        new_features['is_off_hour'] = df['hour'].isin([1, 2, 3, 4, 5, 6]).astype(int)

        # 発電量の比率
        generation_features = self.feature_groups['generation']
        if len(generation_features) > 1:
            total_generation = df[generation_features].sum(axis=1)
            for feature in generation_features:
                new_features[f'{feature}_ratio'] = df[feature] / total_generation

        # 需要の比率
        load_features = self.feature_groups['load']
        if len(load_features) > 1:
            total_load = df[load_features].sum(axis=1)
            for feature in load_features:
                new_features[f'{feature}_ratio'] = df[feature] / total_load
                
        # 特定の地域の気温差
        new_features['valencia_temp_diff'] = df['valencia_temp_max'] - df['valencia_temp_min']
        
        # 需要量と化石燃料以外による発電量の差
        # fossilを含まない発電量カラムを選択
        non_fossil_cols = [col for col in df[generation_features].columns if 'fossil' not in col]

        # fossil以外の発電量を合計
        new_features['non_fossil_generation'] = df[non_fossil_cols].sum(axis=1)
        new_features['demand_non_fossil_diff'] = df['total_load_actual'] - new_features['non_fossil_generation']

        # 新しい特徴量を一度にDataFrameに追加
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)

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

    # 相関分析と特徴量選択
    def _analyze_correlations(self, df, target_col, threshold=0.1):
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # 閾値以上の特徴量を選択
        selected_features = correlations[correlations >= threshold].index.tolist()
        selected_features.remove(target_col)  # 目的変数を除外
        
        print(f"相関閾値 {threshold} 以上の特徴量:")
        for feature in selected_features:
            corr_value = correlations[feature]
            print(f"{feature}: {corr_value:.3f}")
        
        return df[selected_features]


import holidays
from datetime import date, timedelta

class HolidayChecker:
    def __init__(self, country='ES', city_list=None):
        self.country = country
        self.city_list = city_list if city_list else []
        self.country_holidays = holidays.country_holidays(country)

    def fit(self, df: pd.DataFrame):
        """
        データフレームに対して全ての処理を実行
        """
        # 新しい特徴量を格納する辞書
        new_features = {}
        
        # 当日が祝日または週末かどうか
        new_features['is_holiday_or_weekend_flag'] = df.index.map(
            lambda x: self.is_holiday_or_weekend(x.date())
        )

        # 翌日が祝日または週末かどうか
        new_features['is_next_day_holiday_or_weekend_flag'] = df.index.map(
            lambda x: self.check_next_day(x.date())
        )

        # 前日が祝日または週末かどうか
        new_features['is_previous_day_holiday_or_weekend_flag'] = df.index.map(
            lambda x: self.check_previous_day(x.date())
        )

        # 連続で何日の祝日または週末かどうか
        new_features['consecutive_holiday_or_weekend_flag'] = df.index.map(
            lambda x: self.consecutive_holiday_or_weekend(x.date())
        )
        
        # 新しい特徴量を一度にDataFrameに追加
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        
        return df
      
    def is_holiday_or_weekend(self, dt: date) -> int:
        """
        指定された日付が祝日または土日かを判定する。
        """
        return int(dt in self.country_holidays or dt.weekday() >= 5)

    def is_holiday_or_weekend_by_city(self, dt: date, city: str) -> int:
        """
        指定された日付が都市における祝日または土日かを判定する。
        """
        return int(dt in holidays.country_holidays(self.country, subdiv=city) or dt.weekday() >= 5)

    def check_next_day(self, dt: date) -> int:
        """
        指定された日付の翌日が祝日または土日かを判定する。
        """
        next_day = dt + timedelta(days=1)
        return int(self.is_holiday_or_weekend(next_day))

    def check_current_day(self, dt: date) -> int:
        """
        指定された日付が祝日または土日かを判定する。
        (is_holiday_or_weekend と同じ機能だが、要求に応じて分割)
        """
        return int(self.is_holiday_or_weekend(dt))

    def check_previous_day(self, dt: date) -> int:
        """
        指定された日付の前日が祝日または土日かを判定する。
        """
        previous_day = dt - timedelta(days=1)
        return int(self.is_holiday_or_weekend(previous_day))

    def consecutive_holiday_or_weekend(self, dt: date) -> int:
        """
        指定された日付を含む連続した祝日または土日の日数を計算する。
        指定された日付が祝日または土日でない場合は0を返す。
        """
        if not self.is_holiday_or_weekend(dt):
            return 0

        count = 0
        current_date = dt
        # 前方向の連続日数をカウント
        while self.is_holiday_or_weekend(current_date):
            count += 1
            current_date -= timedelta(days=1)

        return count
      
class Visualizer:
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化

        Args:
            config (Optional[Dict]): 設定パラメータ
        """
        self.df = pd.DataFrame()
        self.config = config or {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_groups: Dict[str, List[str]] = {}

