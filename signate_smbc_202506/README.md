# SIGNATE SMBC 2025/06 コンペティション

## プロジェクト概要
スペインの電力価格を予測する高精度なモデルを構築するコンペティション。

## 環境構築
### 前提条件
- Python 3.8以上
- uv（Pythonパッケージマネージャー）

### セットアップ手順
1. uvのインストール（未インストールの場合）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 仮想環境の作成と有効化
```bash
uv venv
source .venv/bin/activate  # Linuxの場合
# または
.venv\Scripts\activate  # Windowsの場合
```

3. 依存パッケージのインストール
```bash
uv pip install -e .
```

## プロジェクト構造
```
signate_smbc_202506/
├── 20250615/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   └── utils.py
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_model_development.ipynb
│       └── 04_model_evaluation.ipynb
└── data/
    ├── feature_description.csv
    ├── sample_submit.csv
    ├── test.csv
    └── train.csv
```

## 要件定義
### 基本情報
- 要望ID：REQ-1234
- タイトル：SIGNATE SMBC 2025/06 コンペティション対応
- 概要：スペインの電力価格予測モデルの構築

### 判断結果
- 実施判断：Yes
- 優先度：高
- 影響範囲：
  - 新規プロジェクトとして開始
  - データ分析環境の構築が必要
  - 予測モデルの開発が必要
  - 評価指標：RMSE（Root Mean Squared Error）

## 詳細設計
### 1. 変更対象ファイル・関数
#### 新規作成ファイル
- `requirements.txt`: Pythonパッケージの依存関係管理
- `src/`: ソースコード
  - `data_loader.py`: データ読み込み・前処理
  - `feature_engineering.py`: 特徴量エンジニアリング
  - `model.py`: モデル定義・学習
  - `predict.py`: 予測実行
  - `evaluate.py`: モデル評価
  - `utils.py`: ユーティリティ関数
- `notebooks/`: Jupyter Notebooks
  - `01_data_exploration.ipynb`: データ探索
  - `02_feature_engineering.ipynb`: 特徴量設計
  - `03_model_development.ipynb`: モデル開発
  - `04_model_evaluation.ipynb`: モデル評価

### 2. データ設計方針
#### 目的変数
1. 実際の電力価格 (EUR/MWh) (目標値)
   - price_actual

#### 特徴量エンジニアリング
1. 時間関連特徴量
   - 時刻（hour）
   - 曜日（day of week）
   - 月（month）
   - 季節（season）
   - 祝日フラグ（スペインの祝日）

2. 気象関連特徴量
   - 5都市の気象データ
     - 気温（現在・最低・最高）
     - 気圧
     - 湿度
     - 風速
     - 風向
     - 降雨量（1時間・3時間）
     - 降雪量（3時間）
     - 雲量
     - 天候

3. 電力関連特徴量
   - 発電実績データ（24時間前）
     - バイオマス
     - 化石燃料（褐炭、天然ガス、石炭、石油）
     - 原子力
     - 水力（揚水消費、自流式、貯水池式）
     - 太陽光
     - 風力（陸上・洋上）
   - 総電力需要実績（24時間前）

4. 派生特徴量
   - 発電量の合計
   - 再生可能エネルギー比率
   - 化石燃料比率
   - 需要供給バランス
   - 気象データの統計量（平均、標準偏差など）

### 3. モデル開発方針
#### 候補アルゴリズム
1. 時系列モデル
   - LightGBM
   - XGBoost
   - CatBoost
   - ニューラルネットワーク（LSTM/GRU）

#### 評価方法
- 評価指標：RMSE
- 交差検証：時系列分割交差検証（TimeSeriesSplit）
- ハイパーパラメータチューニング：Optuna

#### アンサンブル手法
- 複数モデルの予測値の加重平均
- スタッキング

## 開発スケジュール
1. データ探索・前処理（6/2-6/6）
   - データの可視化
   - 欠損値・外れ値の確認
   - 基本的な前処理

2. 特徴量エンジニアリング（6/7-6/13）
   - 特徴量の設計
   - 特徴量の重要度分析
   - 特徴量の選択

3. モデル開発（6/14-6/20）
   - ベースモデルの構築
   - ハイパーパラメータチューニング
   - モデルの評価

4. モデル改善（6/21-6/27）
   - アンサンブル手法の検討
   - モデルの微調整
   - 最終評価

5. 提出準備（6/28-6/30）
   - コードの整理
   - ドキュメント作成
   - 最終提出

## 開発工数
### タスク別時間
1. 環境構築・データ準備: 6時間
2. データ探索・分析: 12時間
3. 特徴量エンジニアリング: 24時間
4. モデル開発: 28時間
5. 評価・改善: 24時間
6. 提出準備: 10時間

### 合計工数
- 合計時間: 104時間
- 人日換算: 約13人日（8時間/日として）

## 未確定事項
1. データ関連
   - [ ] 欠損値の処理方針
   - [ ] 外れ値の定義と処理方針
   - [ ] 特徴量の重要度に基づく選択基準

2. モデル関連
   - [ ] 使用する具体的なアンサンブル手法
   - [ ] ハイパーパラメータの探索範囲
   - [ ] モデルの評価基準（RMSEの目標値）

3. 開発環境関連
   - [ ] 具体的なPythonバージョン
   - [ ] 使用するライブラリのバージョン
   - [ ] 開発環境の具体的な設定 