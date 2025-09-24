## 1. uv パッケージでの環境構築
```bash
# カーネル作成
uv add --dev ipykernel
uv run ipython kernel install --user --name=my-kernel
```

## 2. Notebookの起動
```bash
# Jupyter Notebook 起動
uv run --with jupyter jupyter notebook
```

## 3. カーネルの削除
```bash
# カーネル削除
uv run --with jupyter jupyter kernelspec uninstall my-kernel
```