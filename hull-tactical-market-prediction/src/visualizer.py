import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_matrix_excluding_target(df: pd.DataFrame, target_column: str, figsize: tuple = (15, 12)):
    """
    目的変数をデータフレームから除外した上で、残りの特徴量間の相関行列を計算し、ヒートマップで表示します。

    Parameters
    ----------
    df : pd.DataFrame
        分析対象のデータフレーム。
    target_column : str
        目的変数として除外するカラムの名前。
    figsize : tuple, optional
        プロットのフィギュアサイズ。デフォルトは (15, 12)。
    """
    if target_column not in df.columns:
        print(f"エラー: 指定された目的変数 '{target_column}' がデータフレームに存在しません。")
        return

    # 目的変数を除外したデータフレームを作成
    df_features = df.drop(columns=[target_column])

    # 数値型の特徴量のみを選択（相関計算のため）
    df_numeric_features = df_features.select_dtypes(include=[np.number])

    if df_numeric_features.empty:
        print("エラー: 目的変数を除外した後、数値型の特徴量が見つかりませんでした。")
        return

    # 相関行列を計算
    correlation_matrix = df_numeric_features.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'相関行列 (目的変数 "{target_column}" を除く)', fontsize=16)
    plt.show()

    print("\n--- 相関行列の概要 ---")
    # 相対的に相関の高いペアを確認することもできます
    # 上三角行列を取得し、対角要素と重複を除く
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # NaNを除外し、スタックしてシリーズにする
    corr_pairs = upper_tri.unstack().sort_values(ascending=False).dropna()
    
    print("相関の高い上位20ペア:")
    print(corr_pairs.head(20))
    print("\n相関の低い（負の相関が強い）上位20ペア:")
    print(corr_pairs.tail(20))
    
def plot_categorical_histograms(df: pd.DataFrame, top_n: int = 20):
    """
    DataFrame内のオブジェクト型とカテゴリ型のカラムについて、ヒストグラム（棒グラフ）をプロットします。
    ユニークな値が多い場合は、上位N個のカテゴリのみを表示し、残りは「_Other_」としてまとめます。

    Parameters
    ----------
    df : pd.DataFrame
        分析対象のDataFrame。
    top_n : int, optional
        ヒストグラムに表示する上位のユニークな値の数。デフォルトは20。
    """
    print("--- オブジェクト型/カテゴリ型カラムのヒストグラムプロット ---")

    # オブジェクト型とカテゴリ型のカラムを抽出
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) == 0:
        print("オブジェクト型またはカテゴリ型のカラムは見つかりませんでした。")
        return

    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        
        # NaNを特定の値に置き換える（プロットのため）
        # 元のデータには影響しないようにコピーを使う
        col_data = df[col].fillna('_NaN_') 
        
        unique_values = col_data.nunique()
        
        if unique_values > top_n:
            # 上位 top_n 個のカテゴリを取得
            top_categories = col_data.value_counts().nlargest(top_n).index.tolist()
            
            # top_n以外のカテゴリを「_Other_」としてまとめる
            plot_data = col_data.apply(lambda x: x if x in top_categories else '_Other_')
            
            # 「_Other_」を含めたvalue_countsを再計算
            counts = plot_data.value_counts()
            
            # プロット順序を頻度が高い順にソート（_Other_は通常最後に来るように調整）
            if '_Other_' in counts.index:
                sorted_categories = counts.drop('_Other_').index.tolist() + ['_Other_']
            else:
                sorted_categories = counts.index.tolist()
            
            sns.countplot(y=plot_data, order=sorted_categories, palette='viridis', hue=plot_data, legend=False)
            plt.title(f'Distribution of {col} (Top {top_n} Categories + Other)')
        else:
            # ユニーク数がtop_n以下の場合は全て表示
            sns.countplot(y=col_data, order=col_data.value_counts().index, palette='viridis', hue=col_data, legend=False)
            plt.title(f'Distribution of {col}')
        
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

        # 詳細情報も表示
        print(f"\n--- カラム: {col} (データ型: {df[col].dtype}) ---")
        print(f"  ユニークな値の総数: {unique_values}")
        print(f"  上位 {min(unique_values, top_n)} 個の値と頻度:")
        print(col_data.value_counts(dropna=False).head(top_n).to_string())
        
        if '_NaN_' in col_data.value_counts(dropna=False).index:
            nan_count = col_data.value_counts(dropna=False)['_NaN_']
            print(f"  NaNの数: {nan_count} ({nan_count / len(df) * 100:.2f}%)")

def plot_autocorrelation(df: pd.DataFrame, column: str, lags: int = 48, figsize: tuple = (12, 6)):
    """
    指定したカラムの自己相関（autocorrelation）をプロットします。

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム。
    column : str
        自己相関を調べるカラム名。
    lags : int, optional
        ラグ数（デフォルト: 48）。
    figsize : tuple, optional
        プロットのサイズ。
    """
    from pandas.plotting import autocorrelation_plot
    import statsmodels.api as sm
    if column not in df.columns:
        print(f"カラム '{column}' がデータフレームに存在しません。")
        return
    plt.figure(figsize=figsize)
    sm.graphics.tsa.plot_acf(df[column].dropna(), lags=lags, ax=plt.gca())
    plt.title(f"Autocorrelation for '{column}' (up to {lags} lags)")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.show()

def plot_cross_correlation(df: pd.DataFrame, col_x: str, col_y: str, lags: int = 48, figsize: tuple = (12, 6)):
    """
    2つのカラム間の相互相関（cross-correlation）をプロットします。

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム。
    col_x : str
        1つ目のカラム名。
    col_y : str
        2つ目のカラム名。
    lags : int, optional
        ラグ数（デフォルト: 48）。
    figsize : tuple, optional
        プロットのサイズ。
    """
    import statsmodels.api as sm
    if col_x not in df.columns or col_y not in df.columns:
        print(f"カラム '{col_x}' または '{col_y}' がデータフレームに存在しません。"); return
    x = df[col_x].dropna()
    y = df[col_y].dropna()
    # 長さを揃える
    min_len = min(len(x), len(y))
    x = x[-min_len:]
    y = y[-min_len:]
    ccf = sm.tsa.stattools.ccf(x, y, adjusted=False)[:lags+1]
    plt.figure(figsize=figsize)
    plt.stem(range(lags+1), ccf)
    plt.title(f"Cross-correlation between '{col_x}' and '{col_y}' (up to {lags} lags)")
    plt.xlabel("Lag (y vs x)")
    plt.ylabel("Cross-correlation")
    plt.tight_layout()
    plt.show()
