# Hugging Face Hub へファインチューニングモデルをプッシュ (Push) する手順

ファインチューニングした自作モデルを **Hugging Face Hub** にアップロード（Push）して公開・共有するための実践手順を解説します。

---

## 1. 全体の流れ (3つのステップ)

1. **Hugging Face アカウントで「Write権限アクセストークン」を作成する**
2. **PC / 開発環境でログイン認証を行う**
3. **Pythonコードから `push_to_hub()` を実行する**

---

## Step 1: Write権限アクセストークンの取得

Hubにモデルを書き込む（アップロードする）には、書き込み権限（Write）を持ったトークンが必要です。

1. [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens) にアクセスします。
2. **「Create new token」** ボタンをクリックします。
3. トークンタイプで **「Write」**（書き込み権限）を選択し、名前（例: `my-finetuned-model-token`）を入力して生成します。
4. 発行された `hf_...` で始まる文字列（トークン）をコピーします。

---

## Step 2: 開発環境でのログイン認証

ターミナル（コマンドプロンプトやPowerShell）でログインコマンドを実行します。

```bash
huggingface-cli login
```

プロンプトで `Enter your token (input will not be visible):` と聞かれたら、Step 1 でコピーしたトークン（`hf_...`）を貼り付けて Enter を押します。

※ Pythonコード内で直接ログインしたい場合は以下のように記述することも可能です：
```python
from huggingface_hub import login
login(token="hf_あなたのトークン文字列")
```

---

## Step 3: Python コードからのプッシュ実行

モデルをプッシュする方法には、**「Trainer API を使う方法」** と **「モデル単体を直接プッシュする方法」** の2種類があります。

### 方法A: `Trainer` API を使ってプッシュする (おすすめ)

`TrainingArguments` で `push_to_hub=True` を設定し、学習完了後に `trainer.push_to_hub()` を呼び出します。

```python
from transformers import TrainingArguments, Trainer

# 1. push_to_hub=True を指定
training_args = TrainingArguments(
    output_dir="distilbert-base-uncased-finetuned-emotion", # Hub上のリポジトリ名にもなります
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    num_train_epochs=2,
    push_to_hub=True,  # ★ Hubへの自動準備を有効化
    hub_model_id="あなたのユーザー名/distilbert-finetuned-emotion" # オプション: 任意のモデルID
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    processing_class=tokenizer
)

# 2. 学習の実行
trainer.train()

# 3. Hub へのアップロード実行！
trainer.push_to_hub(commit_message="Training completed successfully")
```

---

### 方法B: 学習済みモデルとトークナイザーを直接プッシュする

すでに手元で学習済みの `model` オブジェクトと `tokenizer` オブジェクトがある場合は、各自の `.push_to_hub()` メソッドを個別に呼び出すだけで完了します。

```python
# モデルとトークナイザーを個別アップロード
hub_repo_name = "あなたのユーザー名/distilbert-finetuned-emotion"

model.push_to_hub(hub_repo_name)
tokenizer.push_to_hub(hub_repo_name)
```

---

## 4. プッシュ後の使い方

Hub へのアップロードが成功すると、自分や他の人が全世界どこからでも以下の一行であなたの自作モデルを呼び出して推論できるようになります。

```python
from transformers import pipeline

# Hub上の自作モデルを指定して一発ロード！
classifier = pipeline("text-classification", model="あなたのユーザー名/distilbert-finetuned-emotion")

results = classifier("I am so excited to build my own LLM!")
print(results)
```
