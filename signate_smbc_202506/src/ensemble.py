import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import product

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

def optimize_weights_from_history(
    lb_scores: List[float],
    weights_history: List[Tuple[float, float]],
    method: str = 'grid_search',
    grid_points: int = 20
) -> Dict:
    """
    LBスコア履歴から最適な重みを推定する関数
    
    Parameters
    ----------
    lb_scores : List[float]
        LBスコアの履歴
    weights_history : List[Tuple[float, float]]
        重みの履歴 (model1_weight, model2_weight)
    method : str
        'best_from_history' または 'grid_search'
    grid_points : int
        グリッドサーチの分割数
    
    Returns
    -------
    Dict
        最適化結果
    """
    if len(lb_scores) != len(weights_history):
        raise ValueError("LBスコアと重み履歴の長さが一致しません")
    
    # 履歴から最良の組み合わせを特定
    best_idx = np.argmin(lb_scores)  # RMSEなので最小値が最良
    best_score = lb_scores[best_idx]
    best_weight = weights_history[best_idx]
    
    result = {
        'best_from_history': {
            'score': best_score,
            'weight': best_weight,
            'weight_ratio': f"{best_weight[0]:.2f}:{best_weight[1]:.2f}"
        }
    }
    
    if method == 'grid_search':
        # グリッドサーチで最適化
        weight1_range = np.linspace(0.0, 0.5, grid_points)
        weight2_range = np.linspace(0.5, 1.0, grid_points)
        
        best_grid_score = float('inf')
        best_grid_weight = None
        
        for w1, w2 in product(weight1_range, weight2_range):
            if abs(w1 + w2 - 1.0) < 1e-6:  # 重みの合計が1になる組み合わせのみ
                # 履歴データから重みの傾向を学習（簡易的な線形補間）
                predicted_score = _predict_score_from_history(w1, w2, lb_scores, weights_history)
                
                if predicted_score < best_grid_score:
                    best_grid_score = predicted_score
                    best_grid_weight = (w1, w2)
        
        result['grid_search'] = {
            'predicted_score': best_grid_score,
            'weight': best_grid_weight,
            'weight_ratio': f"{best_grid_weight[0]:.3f}:{best_grid_weight[1]:.3f}"
        }
    
    return result

def _predict_score_from_history(
    w1: float, 
    w2: float, 
    lb_scores: List[float], 
    weights_history: List[Tuple[float, float]]
) -> float:
    """
    履歴データから重みに対するスコアを予測（簡易的な線形補間）
    """
    if len(lb_scores) < 2:
        return lb_scores[0]
    
    # 重みの距離に基づく重み付き平均
    distances = []
    for hist_w1, hist_w2 in weights_history:
        dist = np.sqrt((w1 - hist_w1)**2 + (w2 - hist_w2)**2)
        distances.append(dist)
    
    # 距離の逆数を重みとして使用
    weights = 1.0 / (np.array(distances) + 1e-8)
    weights = weights / weights.sum()
    
    predicted_score = np.sum(weights * np.array(lb_scores))
    return predicted_score

def suggest_optimal_weights(
    current_results: Dict[str, float],
    n_models: int = 2
) -> List[Tuple[float, ...]]:
    """
    現在の結果から最適な重みの組み合わせを提案
    
    Parameters
    ----------
    current_results : Dict[str, float]
        {'weight_ratio': lb_score} の形式
    n_models : int
        モデル数
    
    Returns
    -------
    List[Tuple[float, ...]]
        試すべき重みの組み合わせ
    """
    if n_models == 2:
        # 2モデルの場合
        lb_scores = []
        weights_history = []
        
        for weight_ratio, score in current_results.items():
            w1, w2 = map(float, weight_ratio.split(':'))
            lb_scores.append(score)
            weights_history.append((w1, w2))
        
        # 最適化実行
        optimization_result = optimize_weights_from_history(lb_scores, weights_history)
        
        # 推奨する重みの組み合わせ
        best_from_history = optimization_result['best_from_history']['weight']
        grid_search = optimization_result.get('grid_search', {}).get('weight', best_from_history)
        
        # 周辺での微調整
        w1_best, w2_best = best_from_history
        suggestions = [
            best_from_history,
            grid_search,
            (max(0, w1_best - 0.05), min(1, w2_best + 0.05)),
            (max(0, w1_best + 0.05), min(1, w2_best - 0.05)),
            (0.15, 0.85),  # model2をさらに重視
            (0.10, 0.90),  # model2をさらに重視
            (0.05, 0.95),  # model2をさらに重視
        ]
        
        return suggestions
    
    elif n_models == 3:
        # 3モデルの場合（model2を主体とした組み合わせ）
        return [
            (0.10, 0.70, 0.20),
            (0.05, 0.80, 0.15),
            (0.15, 0.65, 0.20),
            (0.20, 0.60, 0.20),
            (0.10, 0.75, 0.15),
        ]
    
    else:
        # その他の場合（均等重みベース）
        return [(1.0/n_models,) * n_models]

def analyze_weight_performance(
    current_results: Dict[str, float]
) -> Dict:
    """
    重みの性能を分析
    
    Parameters
    ----------
    current_results : Dict[str, float]
        {'weight_ratio': lb_score} の形式
    
    Returns
    -------
    Dict
        分析結果
    """
    # データ整理
    data = []
    for weight_ratio, score in current_results.items():
        w1, w2 = map(float, weight_ratio.split(':'))
        data.append({
            'weight_ratio': weight_ratio,
            'w1': w1,
            'w2': w2,
            'score': score
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('score')
    
    analysis = {
        'best_score': df.iloc[0]['score'],
        'best_weight': df.iloc[0]['weight_ratio'],
        'worst_score': df.iloc[-1]['score'],
        'worst_weight': df.iloc[-1]['weight_ratio'],
        'score_range': df.iloc[-1]['score'] - df.iloc[0]['score'],
        'trends': {
            'model1_high_performance': df[df['w1'] > 0.3]['score'].mean(),
            'model2_high_performance': df[df['w2'] > 0.7]['score'].mean(),
            'balanced_performance': df[(df['w1'] >= 0.2) & (df['w1'] <= 0.3)]['score'].mean()
        },
        'recommendations': []
    }
    
    # 推奨事項の生成
    if df.iloc[0]['w2'] > 0.7:
        analysis['recommendations'].append("model2の重みを高く保つことを推奨")
    
    if df.iloc[0]['w1'] < 0.3:
        analysis['recommendations'].append("model1の重みは0.3以下が効果的")
    
    if analysis['score_range'] < 0.1:
        analysis['recommendations'].append("重みの影響が小さい - 他の要因を検討")
    
    return analysis
