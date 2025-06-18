"""
データ読み込み・前処理用のモジュール
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_data(
    data_dir: Path,
    train_path: str = 'train.csv',
    test_path: str = 'test.csv',
    feature_desc_path: str = 'feature_description.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    データを読み込む関数

    Parameters
    ----------
    data_dir : Path
        データディレクトリのパス
    train_path : str, optional
        訓練データのファイル名, by default 'train.csv'
    test_path : str, optional
        テストデータのファイル名, by default 'test.csv'
    feature_desc_path : str, optional
        特徴量説明のファイル名, by default 'feature_description.csv'

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        訓練データ、テストデータ、特徴量説明のDataFrame
    """
    train_df = pd.read_csv(data_dir / train_path, index_col=0, header=0)
    test_df = pd.read_csv(data_dir / test_path, index_col=0, header=0)
    feature_desc_df = pd.read_csv(data_dir / feature_desc_path, index_col=0, header=0)

    return train_df, test_df, feature_desc_df


def preprocess_time(
    df: pd.DataFrame,
    tz: str = 'Asia/Tokyo'
) -> pd.DataFrame:
    """
    時刻列の前処理を行う関数

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム
    tz : str, optional
        タイムゾーン

    Returns
    -------
    pd.DataFrame
        前処理後のデータフレーム
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    
    # 時刻関連の特徴量を追加
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    
    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    特徴量をグループ化する関数

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム

    Returns
    -------
    dict
        特徴量グループの辞書
    """
    feature_groups = {
        'time': ['hour', 'day_of_week', 'month', 'day', 'year'],
        'generation': [col for col in df.columns if 'generation' in col],
        'weather': [col for col in df.columns if any(x in col for x in [
            'temp', 'pressure', 'humidity', 'wind', 'rain', 'snow', 'clouds'
        ])],
        'load': [col for col in df.columns if 'load' in col],
        'price_actual': [col for col in df.columns if 'price_actual' in col]
    }
    
    return feature_groups


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    欠損値の確認を行う関数

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム
    threshold : float, optional
        欠損値の閾値, by default 0.5

    Returns
    -------
    pd.DataFrame
        欠損値の情報を含むDataFrame
    """
    missing = df.isnull().sum()
    missing_ratio = missing / len(df)
    
    missing_info = pd.DataFrame({
        'missing_column': missing.index,
        'missing_count': missing,
        'missing_ratio': missing_ratio
    })
    
    print("欠損値があるカラム、欠損値の数、全レコードに対する割合:")
    display(missing_info)
    
    return missing_info[missing_info['missing_ratio'] > threshold]


def check_outliers(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> dict:
    """
    外れ値の確認を行う関数

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム
    columns : Optional[list], optional
        確認対象のカラム, by default None
    method : str, optional
        外れ値検出の方法, by default 'iqr'
    threshold : float, optional
        外れ値の閾値, by default 1.5

    Returns
    -------
    dict
        外れ値の情報を含む辞書
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': ((df[col] < lower_bound) | (df[col] > upper_bound)).sum(),
                'outlier_ratio': ((df[col] < lower_bound) | (df[col] > upper_bound)).mean()
            }
    
    return outliers 