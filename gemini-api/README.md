# Gemini API Learn

Google GenAI SDK (`google-genai`) を使用した Gemini API の学習用サンプルコード集です。

## 📁 収録サンプルコード一覧

| ファイル名 | 概要・用途 |
| :--- | :--- |
| **`count_tokens.py`** | **トークン数（Token Usage）の計測サンプル**<br>プロンプトに対する回答を取得すると同時に、入力（Prompt）、回答（Candidates）、思考（Thoughts）、合計（Total）の各種トークン数を出力します。 |
| **`streaming_response.py`** | **ストリーミング応答の取得サンプル**<br>リアルタイムで回答を受信・出力する2つの方法（`interactions.create` と `models.generate_content_stream`）の違いと使い方を実証しています。 |

---

## 🛠️ 環境構築・実行手順

### 1. 依存ライブラリのインストール
必要なライブラリをインストールします。
```bash
pip install google-genai python-dotenv
```

### 2. APIキーの設定 (`.env`)
リポジトリ直下に `.env` ファイルを作成し、Gemini APIキーを設定します（`.env.example` を参考にしてください）。

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. サンプルの実行
各スクリプトを実行します。

```bash
# トークン数計測サンプルの実行
python count_tokens.py

# ストリーミング応答サンプルの実行
python streaming_response.py
```
