import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder # カテゴリ変数の処理用

class AdversarialValidator:
    """
    Adversarial Validation を実行するためのクラス。
    トレーニングデータとテストデータの分布の違いを検出します。
    """

    def __init__(self, model=None, n_splits=5, random_state=42):
        """
        コンストラクタ

        Parameters
        ----------
        model : object, optional
            敵対的分類に使用する分類モデル。デフォルトはLightGBMのLGBMClassifier。
            fit, predict_proba メソッドを持つ必要があります。
        n_splits : int, optional
            KFold交差検定の分割数。デフォルトは5。
        random_state : int, optional
            乱数シード。デフォルトは42。
        """
        if model is None:
            # デフォルトでLightGBMを使用
            self.model = lgb.LGBMClassifier(random_state=random_state, n_estimators=1000, learning_rate=0.05, num_leaves=31, verbose=-1)
        else:
            self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.oof_preds = None
        self.feature_importances = None
        self.auc_score = None

    def validate(self, train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_features=None):
        """
        Adversarial Validation を実行します。

        Parameters
        ----------
        train_df : pd.DataFrame
            トレーニングデータフレーム。
        test_df : pd.DataFrame
            テストデータフレーム。
        categorical_features : list, optional
            カテゴリカル特徴量のリスト。指定するとLabelEncoderで処理されます。
            LightGBMはカテゴリ特徴量を直接扱えますが、念のため指定可能にしておきます。

        Returns
        -------
        dict
            AUCスコア、重要特徴量、各レコードの予測確率を含む辞書。
        """
        print("Adversarial Validationを開始します...")

        # 1. データセットの結合とラベル付け
        # 元のDataFrameに影響を与えないようにコピーを使用
        train_copy = train_df.copy()
        test_copy = test_df.copy()

        train_copy['is_test'] = 0
        test_copy['is_test'] = 1
        
        # ここでは、両方のデータフレームに存在する列のみを対象とする
        common_cols = list(set(train_copy.columns) & set(test_copy.columns))
        
        # 'is_test'は共通なので除外
        if 'is_test' in common_cols:
            common_cols.remove('is_test')

        # 全体のデータフレームを作成
        combined_df = pd.concat([train_copy, test_copy], ignore_index=True)

        # ターゲット変数と特徴量を定義
        X = combined_df.drop('is_test', axis=1)
        y = combined_df['is_test']
        
        self.original_is_test = y # プロットのために元のis_testラベルを保存する

        # カテゴリカル特徴量の処理 (Label Encoding)
        if categorical_features:
            for col in categorical_features:
                if col in X.columns:
                    le = LabelEncoder()
                    # 訓練データとテストデータ全体でfit_transformすることで一貫性を保つ
                    X[col] = le.fit_transform(X[col].astype(str).fillna('NaN_Category')) # NaNをカテゴリとして扱う
                else:
                    print(f"Warning: Categorical feature '{col}' not found in combined dataframe.")

        # KFold交差検定の準備
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(X))
        feature_importances = pd.DataFrame(index=X.columns)

        print(f"特徴量数: {X.shape[1]}")
        print(f"レコード数: {X.shape[0]}")

        # 2. 分類モデルの学習と予測
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"--- Fold {fold+1}/{self.n_splits} ---")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # カテゴリカル特徴量がある場合、LightGBMに適切な型を指示
            lgb_params = {}
            if categorical_features:
                lgb_params['categorical_feature'] = [col for col in categorical_features if col in X.columns]
            
            self.model.fit(X_train, y_train, **lgb_params,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

            oof_preds[val_idx] = self.model.predict_proba(X_val)[:, 1]
            
            # 特徴量の重要度を蓄積
            fold_importance = pd.DataFrame(self.model.feature_importances_, 
                                          index=X.columns, 
                                          columns=[f'Fold_{fold+1}'])
            feature_importances = feature_importances.merge(fold_importance, left_index=True, right_index=True, how='left')
            
        self.oof_preds = oof_preds
        
        # 3. モデルの評価
        self.auc_score = roc_auc_score(y, oof_preds)
        print(f"\n--- Adversarial Validation AUC Score: {self.auc_score:.4f} ---")

        # 4. 重要特徴量の特定
        self.feature_importances = feature_importances.mean(axis=1).sort_values(ascending=False)
        print("\n--- Top 10 Feature Importances (indicating distribution difference) ---")
        print(self.feature_importances.head(10))

        return {
            'auc_score': self.auc_score,
            'feature_importances': self.feature_importances,
            'oof_predictions': self.oof_preds
        }

    def plot_predictions_distribution(self):
        """
        Adversarialモデルの予測確率の分布をプロットします。
        """
        if self.oof_preds is None:
            print("validate() メソッドを先に実行してください。")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 正しく分離された予測確率
        preds_train_origin = self.oof_preds[self.original_is_test == 0]
        preds_test_origin = self.oof_preds[self.original_is_test == 1]
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(preds_train_origin, color='blue', label='Train Data (Predicted as 0)', kde=True, stat='density', alpha=0.6, bins=50)
        sns.histplot(preds_test_origin, color='red', label='Test Data (Predicted as 1)', kde=True, stat='density', alpha=0.6, bins=50)

        plt.title('Distribution of Adversarial Validation Predictions')
        plt.xlabel('Predicted Probability of being Test Data (is_test=1)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("\n予測確率の分布を確認することで、訓練データとテストデータがどの程度分離可能か視覚的に判断できます。")
        print("0.5付近にピークがあるほど、分離が難しい（分布が似ている）ことを示します。")
        print("0と1に近い側にピークがあるほど、分離が容易（分布が異なる）ことを示します。")

    def get_feature_importances(self):
        """
        計算された特徴量の重要度を取得します。
        """
        if self.feature_importances is None:
            print("validate() メソッドを先に実行してください。")
            return None
        return self.feature_importances

    def get_auc_score(self):
        """
        計算されたAUCスコアを取得します。
        """
        if self.auc_score is None:
            print("validate() メソッドを先に実行してください。")
            return None
        return self.auc_score
