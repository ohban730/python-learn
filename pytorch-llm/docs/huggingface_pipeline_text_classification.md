# Hugging Face Transformers: pipeline("text-classification") の使い方

`pipeline` は、Hugging Face の `transformers` ライブラリが提供する、非常に使いやすい高レベルAPIです。
たった1行で、テキストの前処理（トークナイズ）、モデルによる推論、後処理（人間が読めるラベルへの変換）をまとめて実行することができます。

---

## 1. 基本的な仕組み

`pipeline("text-classification")` を呼び出すと、内部では自動的に以下の処理フローが実行されます。

```
[ 入力テキスト ] 
      │
      ▼ (前処理: Tokenizer)
[ トークンID (数値のリスト) ]
      │
      ▼ (推論: Model)
[ 予測スコア (未正規化のLogits) ]
      │
      ▼ (後処理: Softmax & ラベル変換)
[ 確率スコアと最終ラベル ] ──> [{'label': 'POSITIVE', 'score': 0.90}]
```

1. **トークナイザー (前処理)**: テキストを単語やサブワードに分割し、モデルに入力できる数値ベクトル（トークンID）に変換します。
2. **モデル (推論)**: 数値ベクトルを入力し、分類用の予測スコア（Logits）を出力します。
3. **後処理**: スコアにSoftmax関数を適用して確率にし、最も確率の高いラベル（または全ラベル）の名前とスコアを返します。

---

## 2. 基本的なコード例

基本的な使い方および応用的な指定方法は、[huggingface_pipeline_demo.py](../basics/huggingface_pipeline_demo.py) で実際に動作させることができます。

```python
from transformers import pipeline

# 1. デフォルトモデルで実行（英語の感情分析モデルが使われます）
classifier = pipeline("text-classification")

# 単一の文章を入力する
result = classifier("I love coding with Python!")
print(result)
# 出力例: [{'label': 'POSITIVE', 'score': 0.9998}]

# 2. 複数の文章をまとめて入力する（リスト形式）
results = classifier([
    "This was a terrible experience.",
    "The weather is just okay."
])
print(results)
# 出力例: [{'label': 'NEGATIVE', 'score': 0.9995}, {'label': 'NEGATIVE', 'score': 0.8524}]
```

---

## 3. 主要なオプションと引数

`pipeline` を定義する際、または実行する際に様々なカスタマイズが可能です。

### ① `model` の明示的な指定（英語・日本語など）
デフォルトでは英語の感情分析モデルがロードされますが、`model` 引数に Hugging Face Hub にあるモデル名（あるいはローカルのモデルパス）を指定することで、任意の言語や分類タスクのモデルを利用できます。

```python
# 日本語の感情分析（ポジティブ / ネガティブ / ニュートラルの3クラス分類）
classifier_ja = pipeline(
    "text-classification", 
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)

result = classifier_ja("この本は非常に分かりやすくて最高です！")
print(result)
# 出力例: [{'label': 'ポジティブ', 'score': 0.9852}]
```

### ② `top_k`（すべてのクラスのスコアを取得）
デフォルトでは最も確率の高い1つのラベルのみが返されますが、`top_k=None` を設定すると、モデルが分類可能なすべてのラベルのスコアを高い順にソートして返します。

```python
# すべてのラベルの確率を出力
classifier_all = pipeline("text-classification", top_k=None)
result = classifier_all("The movie was not bad, but a bit long.")
print(result)
# 出力例: [{'label': 'POSITIVE', 'score': 0.85}, {'label': 'NEGATIVE', 'score': 0.15}]
```

### ③ `device`（GPUの使用）
計算をGPUで行いたい場合は、`device` 引数で指定します。
* `device=0` (最初のGPUを使用)
* `device="cuda"` もしくは `device="cpu"`
* ※デフォルトは `-1` (CPU使用)

```python
# GPUを使用して高速推論を行う
classifier_gpu = pipeline("text-classification", device=0)
```

---

## 4. よく使われる分類モデルの例

Hugging Face Hub (https://huggingface.co/models) にアップロードされている以下のモデル名を `model` 引数に渡すことで、すぐに異なる分類タスクを試すことができます。

* **`distilbert-base-uncased-finetuned-sst-2-english`** (デフォルト)
  * 英語の2クラス（ポジティブ/ネガティブ）感情分析。
* **`lxyuan/distilbert-base-multilingual-cased-sentiments-student`**
  * 日本語を含む多言語対応の3クラス（positive / negative / neutral）感情分析。追加ライブラリ（fugashiなど）なしで動作します。
* **`cardiffnlp/twitter-roberta-base-sentiment-latest`**
  * TwitterなどのSNSテキスト向けに調整された英語の感情分析（3クラス）。
* **`Hate-speech-CNERG/dehatebert-mono-english`**
  * ヘイトスピーチ（攻撃的・差別的な発言）の検出。
