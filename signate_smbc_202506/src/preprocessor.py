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
        self.regions = ['barcelona', 'bilbao', 'madrid', 'seville', 'valencia']
        self.weather_main_weights = {
            'clear': 10,
            'clouds': 5,
            'rain': 4,
            'mist': 3,
            'fog': 2,
            'drizzle': 1,
            'thunderstorm': 7,
            'snow': 6,
            'haze': 8,
            'other': 0
        }

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
        
        # カラムの削除
        df = self.drop_features(df)
        
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
        # 目的変数がなければダミーで追加
        if 'price_actual' not in df.columns:
            df['price_actual'] = np.nan
            
        # インデックスの処理
        df = self._convert_to_datetime(df, utc=True, tz='Europe/Berlin')
        
        # 欠損値の補完
        df = self._fill_missing_values(df)
        
        # 特徴量エンジニアリング
        df = self._engineer_features(df)
        
        # 祝日情報を追加
        df = self.holiday_checker.fit(df)
        
        # カラムの削除
        df = self.drop_features(df, is_test=True)
        
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

    def _create_lag_features(self, df: pd.DataFrame, target_cols: list, lags: list = [1, 24]) -> pd.DataFrame:
        """
        指定したカラムに対してラグ特徴量を作成する

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_cols (list): ラグを作成するカラム名リスト
            lags (list): ラグ数のリスト（例: [1, 24]）

        Returns:
            pd.DataFrame: ラグ特徴量を追加したデータフレーム
        """
        lag_features = {}
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    lag_col = f"{col}_lag{lag}"
                    lag_features[lag_col] = df[col].shift(lag)
        return pd.DataFrame(lag_features, index=df.index)

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
        
        # 地域ごとの気温をまとめる
        df_temp = self._aggregate_temperature_features(df, self.regions)
        
        # 地域ごとの天気をまとめる
        df_weather = self._aggregate_weather_main_features(df, self.regions)
        
        # 需要量と化石燃料以外による発電量の差
        # fossilを含まない発電量カラムを選択
        non_fossil_cols = [col for col in df[generation_features].columns if 'fossil' not in col]

        # fossil以外の発電量を合計
        new_features['non_fossil_generation'] = df[non_fossil_cols].sum(axis=1)
        new_features['demand_non_fossil_diff'] = df['total_load_actual'] - new_features['non_fossil_generation']

        # --- ラグ特徴量の追加 ---
        # 目的変数: price_actual のみ
        lag_target_cols = []
        if 'price_actual' in df.columns:
            lag_target_cols.append('price_actual')
        lag_features_df = self._create_lag_features(df, lag_target_cols, lags=[1, 24])
        # 新しい特徴量を一度にDataFrameに追加
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df, df_temp, df_weather, lag_features_df], axis=1)

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
      
    def _aggregate_temperature_features(self, df: pd.DataFrame, regions: list) -> pd.DataFrame:
      """
      複数の地域における気温関連の特徴量 (temp, temp_max, temp_min) を集約し、
      以下の4つの新しい特徴量を作成します。
      - avg_temp: 全地域の平均気温の平均
      - var_temp: 全地域の平均気温の分散
      - avg_diff_temp: 全地域の最高気温と最低気温の差の平均
      - var_diff_temp: 全地域の最高気温と最低気温の差の分散

      Parameters
      ----------
      df : pd.DataFrame
          元のデータフレーム。各地域の '地域名_temp', '地域名_temp_max', '地域名_temp_min' カラムを含む必要があります。
      regions : list
          処理する地域の名前のリスト (例: ['barcelona', 'bilbao', 'madrid', 'seville', 'valencia'])。

      Returns
      -------
      pd.DataFrame
          新しい4つの集約された特徴量カラムを含むデータフレーム。
      """

      # 計算用の一時的なリスト
      all_temps = []
      all_temp_diffs = []

      for region in regions:
          temp_col = f"{region}_temp"
          temp_max_col = f"{region}_temp_max"
          temp_min_col = f"{region}_temp_min"

          # 必要なカラムがDataFrameに存在するか確認
          if not all(col in df.columns for col in [temp_col, temp_max_col, temp_min_col]):
              print(f"警告: 地域 '{region}' の必要な気温カラムの一部または全てがDataFrameに見つかりませんでした。スキップします。")
              continue

          # 各地域の平均気温 (tempカラム) を追加
          all_temps.append(df[temp_col])

          # 各地域の最高気温と最低気温の差を計算して追加
          temp_diff = df[temp_max_col] - df[temp_min_col]
          all_temp_diffs.append(temp_diff)
          

      if not all_temps:
          raise ValueError("指定された地域で有効な気温カラムが見つかりませんでした。")

      # 全地域の平均気温を横方向に結合し、行ごとの平均と分散を計算
      df_all_temps = pd.concat(all_temps, axis=1)
      
      # NaNがある場合の処理を考慮
      # skipna=True でNaNを無視して計算
      avg_temp_series = df_all_temps.mean(axis=1, skipna=True)
      var_temp_series = df_all_temps.var(axis=1, skipna=True)

      # 全地域の最高気温と最低気温の差を横方向に結合し、行ごとの平均と分散を計算
      df_all_temp_diffs = pd.concat(all_temp_diffs, axis=1)
      
      avg_diff_temp_series = df_all_temp_diffs.mean(axis=1, skipna=True)
      var_diff_temp_series = df_all_temp_diffs.var(axis=1, skipna=True)

      # 新しい特徴量カラムを持つDataFrameを作成
      new_features_df = pd.DataFrame({
          'avg_temp': avg_temp_series,
          'var_temp': var_temp_series,
          'avg_diff_temp': avg_diff_temp_series,
          'var_diff_temp': var_diff_temp_series
      }, index=df.index) # 元のDataFrameのインデックスを保持

      return new_features_df
    
    def _aggregate_weather_main_features(self, df: pd.DataFrame, regions: list) -> pd.DataFrame:
      """
      複数の地域における天気に関する 'weather_main' カラムを集約し、
      指定された重み付けに基づいて数値化し、その平均と分散を計算します。

      Parameters
      ----------
      df : pd.DataFrame
          元のデータフレーム。各地域の '地域名_weather_main' カラムを含む必要があります。
      regions : list
          処理する地域の名前のリスト (例: ['barcelona', 'bilbao', 'madrid', 'seville', 'valencia'])。

      Returns
      -------
      pd.DataFrame
          新しい2つの集約された特徴量カラム ('avg_weighted_weather_main', 'var_weighted_weather_main')
          を含むデータフレーム。
      """
      all_weighted_weather_mains = []

      for region in regions:
          main_col = f"{region}_weather_main"

          if main_col not in df.columns:
              print(f"警告: カラム '{main_col}' がDataFrameに見つかりませんでした。スキップします。")
              continue

          # weather_main のカテゴリを重み付けに基づいて数値にマッピング
          weighted_series = df[main_col].apply(lambda x: self.weather_main_weights.get(str(x).lower(), self.weather_main_weights['other']))
          
          all_weighted_weather_mains.append(weighted_series)
          

      if not all_weighted_weather_mains:
          raise ValueError("指定された地域で有効な天気(main)カラムが見つかりませんでした。")

      # 全地域の重み付けされたweather_main値を横方向に結合
      df_all_weighted = pd.concat(all_weighted_weather_mains, axis=1)

      # 行ごとの平均と分散を計算
      # NaNが存在する可能性を考慮し、skipna=True を使用
      avg_weighted_weather_main_series = df_all_weighted.mean(axis=1, skipna=True)
      var_weighted_weather_main_series = df_all_weighted.var(axis=1, skipna=True)

      # 新しい特徴量カラムを持つDataFrameを作成
      new_weather_features_df = pd.DataFrame({
          'avg_weighted_weather_main': avg_weighted_weather_main_series,
          'var_weighted_weather_main': var_weighted_weather_main_series
      }, index=df.index) # 元のDataFrameのインデックスを保持

      return new_weather_features_df
    
    def drop_features(self, df, is_test=False):
      """
      特徴量を削除

      Parameters
      ----------
      df : pd.DataFrame: 特徴量削除の対象のDataFrame
      is_test（bool）: テストデータかどうかのフラグ
      Returns
      -------
      pd.DataFrame: 特徴量削除後のDataFrame
      """
      drop_columns = []
      for region in self.regions:
        # 頭に地域がつくカラムの削除
        drop_columns.extend(
          [f"{region}_temp",
          f"{region}_temp_max",
          f"{region}_temp_min",
          f"{region}_weather_id",
          f"{region}_weather_main",
          f"{region}_weather_description",
          f"{region}_weather_icon"]
        )
        
      if is_test:
        drop_columns.append('price_actual')

      return df.drop(columns=drop_columns, errors='raise')


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

