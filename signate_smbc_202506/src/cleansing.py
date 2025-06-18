"""データクレンジングを行うモジュール"""
import pandas as pd
import numpy as np
from copy import deepcopy

def _convert_object_to_numeric(series: pd.Series) -> pd.Series:
  """
  指定されたPandas Seriesの'object'型を数値型に変換を試みます。
  変換できない値がある場合はValueErrorを発生させます。

  Args:
    series (pd.Series): 変換対象のSeries。

  Returns:
    pd.Series: 数値型に変換されたSeries。

  Raises:
    ValueError: 数値に変換できない値が含まれる場合。
  """
  if series.dtype == 'object':
    # 'errors='raise'' により、変換できない値があるとValueErrorを発生させる
    converted_series = pd.to_numeric(series, errors='raise')
    print(f"  列 '{series.name}' のobject型を数値型に変換しました。")
    return converted_series
  return series # object型でなければそのまま返す

def _downcast_integer_type(series: pd.Series) -> pd.Series:
  """
  指定されたPandas Seriesの整数型を、より小さいメモリフットプリントの型にダウンキャストします。
  NaNが含まれている場合はダウンキャストせず、元のSeriesを返します。

  Args:
    series (pd.Series): ダウンキャスト対象のSeries。

  Returns:
    pd.Series: ダウンキャストされたSeries、またはダウンキャストされなかった元のSeries。
  """
  # NaNが含まれていないことを確認してからダウンキャストを試みる
  if series.dtype in ('int64', 'Int64') or (series.dtype == 'float64' and not series.hasnans):
    if series.hasnans:
      # NaNがある場合は整数にダウンキャストしない
      return series 

    col_min = series.min()
    col_max = series.max()

    if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
      print(f"  列 '{series.name}' をint8に変換しました。")
      return series.astype(np.int8)
    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
      print(f"  列 '{series.name}' をint16に変換しました。")
      return series.astype(np.int16)
    elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
      print(f"  列 '{series.name}' をint32に変換しました。")
      return series.astype(np.int32)
    else:
      print(f"  列 '{series.name}' はint64のままです。")
  return series # ダウンキャスト対象外のデータ型、または最適化できない場合

def _downcast_float_type(series: pd.Series) -> pd.Series:
  """
  指定されたPandas Seriesの浮動小数点数型を、より小さいメモリフットプリントの型にダウンキャストします。

  Args:
    series (pd.Series): ダウンキャスト対象のSeries。

  Returns:
    pd.Series: ダウンキャストされたSeries、またはダウンキャストされなかった元のSeries。
  """
  if series.dtype in ('float64', 'float32'):
    # float32にダウンキャスト
    print(f"  列 '{series.name}' をfloat32に変換しました。")
    return series.astype(np.float32)
  return series # ダウンキャスト対象外のデータ型


def optimize_dataframe_memory(df: pd.DataFrame, inplace: bool=False) -> pd.DataFrame:
  """
  DataFrameのメモリ使用量を最適化します。
  各列に対して、以下の順序で処理を試みます。
  1. object型（文字列など）を数値型に変換。変換できない場合はエラーとしスキップ。
  2. 整数型（int64, Int64）をより小さい整数型にダウンキャスト。
  3. 浮動小数点数型（float64）をfloat32にダウンキャスト。

  Args:
    df (pd.DataFrame): 最適化対象のDataFrame。
    inplace (bool): Trueの場合、元のDataFrameを変更します。Falseの場合、新しいDataFrameを返します。

  Returns:
    pd.DataFrame: メモリ最適化されたDataFrame。
  """
  df_opt = pd.DataFrame()

  for col in df.columns:
    current_series = df[col].copy() # 処理する列のコピー

    print(f"列 '{col}' の最適化を開始します (現在の型: {current_series.dtype})...")
    # 0. 欠損があれば -1 を入れる （文字列型の場合はNaNのまま）
    df_opt = df[col].fillna(-1)
    
    # 1. object型を数値に変換する処理
    if current_series.dtype == 'object':
      try:
        current_series = _convert_object_to_numeric(current_series)
      except ValueError as e:
        print(f"  **エラー:** 列 '{col}' の数値変換に失敗しました: {e}")
        print(f"  → 列 '{col}' は数値型に変換されず、カテゴリ型になります。")
        df_opt[col] = df[col].astype('category')
        continue
      except Exception as e:
        print(f"  **予期せぬエラー:** 列 '{col}' の数値変換中に問題が発生しました: {e}")
        df_opt[col] = df[col]
        continue

    # 2. 数値型のメモリ最適化
    # 整数型（int64, Int64）のダウンキャスト
    # NaNを含まないfloat64もint型にダウンキャスト可能であれば試みる
    if current_series.dtype in ('int64', 'Int64'):
      current_series = _downcast_integer_type(current_series)
    # float64型またはfloat32型のダウンキャスト
    elif current_series.dtype in ('float64', 'float32'):
      current_series = _downcast_float_type(current_series)
    else:
      print(f"  列 '{col}' は最適化対象外のデータ型 ({current_series.dtype}) です。")

    # 処理が完了した列を最適化されたDataFrameに追加
    df_opt[col] = current_series

  print("\nメモリ最適化処理が完了しました。")

  if inplace:
    # inplace=Trueの場合、元のDataFrameを更新する
    for col in df_opt.columns:
      df[col] = df_opt[col]
    return df # inplace=Trueの場合は元のdfを返す
  
  return df_opt
