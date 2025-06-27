import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.linear_model import LinearRegression
import numpy as np

def ensemble_submissions(
    submission_files: List[str],
    output_path: str,
    target_column: Optional[str] = None,
    method: str = 'mean',
    weights: Optional[List[float]] = None
):
    """
    複数のsubmissionファイルを組み合わせて新しいsubmissionを作成する（平均または加重平均）。
    1列目がインデックス、ヘッダーなしの形式で出力します。

    Parameters
    ----------
    submission_files : List[str]
        組み合わせるsubmissionファイルのパスリスト。
    output_path : str
        出力先ファイルパス。
    target_column : Optional[str]
        アンサンブル対象のカラム名（Noneなら最初の数値カラムを自動判定）。
    method : str
        'mean'（単純平均）または 'weighted'（加重平均）。
    weights : Optional[List[float]]
        加重平均の場合の重みリスト（method='weighted'時のみ有効）。
    """
    dfs = [pd.read_csv(f, index_col=0, header=None) for f in submission_files]
    if target_column is None:
        # 最初の数値カラムを自動判定
        num_cols = dfs[0].select_dtypes(include='number').columns
        if len(num_cols) == 0:
            raise ValueError('数値カラムが見つかりません')
        target_column = num_cols[0]
    # すべてのファイルでindexとtarget_columnが一致しているか確認
    for df in dfs:
        assert (df.index == dfs[0].index).all(), 'indexが一致しません'
        assert target_column in df.columns, f'{target_column}が存在しません'
    # アンサンブル
    preds = [df[target_column].values for df in dfs]
    if method == 'mean':
        ensemble_pred = sum(preds) / len(preds)
    elif method == 'weighted':
        if weights is None or len(weights) != len(preds):
            raise ValueError('weightsの長さがファイル数と一致しません')
        ensemble_pred = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
    else:
        raise ValueError('methodは"mean"または"weighted"のみ対応')
    # 新しいDataFrameを作成
    out_df = dfs[0].copy()
    out_df[target_column] = ensemble_pred
    out_df[[target_column]].to_csv(output_path, index=True, header=False)
    print(f'Ensembled submission saved to: {output_path}')

def stacking_ensemble(
    submission_files: List[str],
    output_path: str,
    target_column: Optional[str] = None,
    meta_model=None
):
    """
    スタッキングによるアンサンブルを行う関数。
    submissionファイル群の予測値を特徴量としてメタモデルで最終予測を行う。
    
    Parameters
    ----------
    submission_files : List[str]
        組み合わせるsubmissionファイルのパスリスト。
    output_path : str
        出力先ファイルパス。
    target_column : Optional[str]
        アンサンブル対象のカラム名（Noneなら最初の数値カラムを自動判定）。
    meta_model : sklearnの回帰モデル（デフォルトはLinearRegression）
    """
    dfs = [pd.read_csv(f, index_col=0, header=None) for f in submission_files]
    if target_column is None:
        num_cols = dfs[0].select_dtypes(include='number').columns
        if len(num_cols) == 0:
            raise ValueError('数値カラムが見つかりません')
        target_column = num_cols[0]
    for df in dfs:
        assert (df.index == dfs[0].index).all(), 'indexが一致しません'
        assert target_column in df.columns, f'{target_column}が存在しません'
    # 予測値を特徴量として結合
    X = np.column_stack([df[target_column].values for df in dfs])
    # メタモデルの用意
    if meta_model is None:
        meta_model = LinearRegression()
    # メタモデルの学習（ここでは平均値をターゲットとする簡易例。実運用ではバリデーションデータの正解値を使う）
    y_meta = X.mean(axis=1)  # 本来は正解値を使うべき
    meta_model.fit(X, y_meta)
    # スタッキング予測
    ensemble_pred = meta_model.predict(X)
    out_df = dfs[0].copy()
    out_df[target_column] = ensemble_pred
    out_df[[target_column]].to_csv(output_path, index=True, header=False)
    print(f'Stacking ensemble saved to: {output_path}')

def blending_ensemble(
    val_pred_files: List[str],
    val_true_file: str,
    test_pred_files: List[str],
    output_path: str,
    target_column: Optional[str] = None,
    meta_model=None
):
    """
    ブレンディングによるアンサンブルを行う関数。
    バリデーションデータの正解値と各モデルの予測値を使ってメタモデルを学習し、
    テストデータのアンサンブル予測を行う。

    Parameters
    ----------
    val_pred_files : List[str]
        バリデーションデータの各モデル予測ファイル（index_col=0, header=None）
    val_true_file : str
        バリデーションデータの正解値ファイル（index_col=0, header=None, targetカラムのみ）
    test_pred_files : List[str]
        テストデータの各モデル予測ファイル（index_col=0, header=None）
    output_path : str
        出力先ファイルパス
    target_column : Optional[str]
        アンサンブル対象のカラム名（Noneなら最初の数値カラムを自動判定）
    meta_model : sklearnの回帰モデル（デフォルトはLinearRegression）
    """
    # バリデーション予測値
    val_dfs = [pd.read_csv(f, index_col=0, header=None) for f in val_pred_files]
    # 正解値
    val_true = pd.read_csv(val_true_file, index_col=0, header=None)
    if target_column is None:
        num_cols = val_dfs[0].select_dtypes(include='number').columns
        if len(num_cols) == 0:
            raise ValueError('数値カラムが見つかりません')
        target_column = num_cols[0]
    for df in val_dfs:
        assert (df.index == val_dfs[0].index).all(), 'indexが一致しません'
        assert target_column in df.columns, f'{target_column}が存在しません'
    # バリデーション特徴量・正解
    X_val = np.column_stack([df[target_column].values for df in val_dfs])
    y_val = val_true[target_column].values
    # メタモデル
    if meta_model is None:
        meta_model = LinearRegression()
    meta_model.fit(X_val, y_val)
    # テストデータ予測値
    test_dfs = [pd.read_csv(f, index_col=0, header=None) for f in test_pred_files]
    for df in test_dfs:
        assert (df.index == test_dfs[0].index).all(), 'indexが一致しません'
        assert target_column in df.columns, f'{target_column}が存在しません'
    X_test = np.column_stack([df[target_column].values for df in test_dfs])
    ensemble_pred = meta_model.predict(X_test)
    out_df = test_dfs[0].copy()
    out_df[target_column] = ensemble_pred
    out_df[[target_column]].to_csv(output_path, index=True, header=False)
    print(f'Blending ensemble saved to: {output_path}')
