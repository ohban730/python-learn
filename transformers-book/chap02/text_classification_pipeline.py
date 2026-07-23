"""
オライリー本『機械学習エンジニアのためのTransformers』第2章 テキスト分類
2章の全体成果物：DistilBERT を用いた感情分類モデルのファインチューニングと評価パイプライン

※本スクリプトは学習・検証目的で独自に構成・解説コメントを付与した実装です。
出典: 『機械学習エンジニアのためのTransformers』 (O'Reilly Media / オライリー・ジャパン) 第2章
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def compute_metrics(pred):
    """
    モデルの評価指標 (Accuracy および F1-score) を計算するコールバック関数
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    print("=== 第2章 テキスト分類（感情分析）パイプライン ===\n")

    # 1. デバイスの設定 (GPUが使える場合はCUDA、なければCPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 2. データセットのロード (emotion データセット)
    print("データセットをロード中...")
    emotions = load_dataset("dair-ai/emotion")
    
    # ラベル情報の取得
    labels = emotions["train"].features["label"].names
    num_labels = len(labels)
    print(f"感情ラベル一覧 ({num_labels}種類): {labels}")

    # 3. トークナイザーの準備と前処理
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    print("データセットをトークン化中...")
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    # 4. ファインチューニング用モデルの初期化
    print(f"事前学習済みモデル ({model_ckpt}) をロード中...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=num_labels
    ).to(device)

    # 5. ハイパーパラメータと Trainer の設定
    batch_size = 64
    logging_steps = len(emotions_encoded["train"]) // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"

    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        log_level="error",
        push_to_hub=True,
        hub_model_id="Sapolas0730/distilbert-finetuned-emotion"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["validation"],
        processing_class=tokenizer
    )

    # 6. ファインチューニング（学習）の実行
    print("\n--- ファインチューニング開始 ---")
    trainer.train()

    # Hubへモデルをアップロード
    print("\nHugging Face Hubへモデルをアップロード中...")
    trainer.push_to_hub(commit_message="Training completed successfully")

    # 7. 検証データでの評価
    print("\n--- 検証データでの評価結果 ---")
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1-score: {eval_results['eval_f1']:.4f}")

    # 8. 混同行列 (Confusion Matrix) の描画
    print("\n混同行列を計算して表示中...")
    preds_output = trainer.predict(emotions_encoded["validation"])
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_valid = np.array(emotions_encoded["validation"]["label"])

    cm = confusion_matrix(y_valid, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_final.png")
    print("混同行列のグラフを 'confusion_matrix_final.png' に保存しました。")

    # 9. パイプラインを使ったテキスト感情予測のデモ
    print("\n--- 予測パイプラインのテスト ---")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    custom_texts = [
        "I saw a movie today and it was absolutely fantastic!",
        "I feel so lonely and hopeless about the future."
    ]

    for text in custom_texts:
        preds = classifier(text)
        print(f"入力テキスト: '{text}'")
        print(f"予測結果: Label = {preds[0]['label']} (Score = {preds[0]['score']:.4f})\n")

if __name__ == "__main__":
    main()
