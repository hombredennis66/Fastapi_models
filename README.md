# FastAPI ML & LLM 統合プロジェクト

このプロジェクトは、伝統的な機械学習（ML）モデルと現代的な大規模言語モデル（LLM）を一つの FastAPI アプリケーションに統合したオープンソースプロジェクトです。

## プロジェクト概要

このアプリケーションは、以下の2つの主要な機能を提供します：

1.  **Iris 分類 (ML)**: Scikit-learn を使用して学習されたロジスティック回帰モデルを用い、花のガクと花弁のサイズから種類を予測します。
2.  **感情分析 (LLM)**: Hugging Face Transformers を使用し、入力されたテキストの感情（ポジティブ/ネガティブ）を分析します。

## 技術スタック

*   **Framework**: FastAPI
*   **Machine Learning**: Scikit-learn, Joblib, NumPy
*   **NLP/LLM**: Transformers (DistilBERT), PyTorch
*   **Testing**: Pytest, HTTPX

## セットアップと実行

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

### モデルの学習（オプション）

既製の `model.joblib` が含まれていますが、以下のコマンドで再学習が可能です：

```bash
python train_ml_model.py
```

### サーバーの起動

```bash
uvicorn main:app --reload
```

サーバーは `http://localhost:8000` で起動します。

## API エンドポイント

### 1. ML 予測
*   **URL**: `/predict`
*   **Method**: `POST`
*   **Payload**: `{"features": [5.1, 3.5, 1.4, 0.2]}`

### 2. 感情分析
*   **URL**: `/sentiment`
*   **Method**: `POST`
*   **Payload**: `{"text": "FastAPI is amazing!"}`

## テストの実行

```bash
pytest
```

## オープンソースについて

このプロジェクトはオープンソースです。コントリビューション、Issueの報告、フォークを歓迎します。

---
*Created by Jules & hombredennis66

