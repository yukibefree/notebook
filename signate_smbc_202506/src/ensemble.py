import pandas as pd
from pathlib import Path
from typing import List, Optional

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
