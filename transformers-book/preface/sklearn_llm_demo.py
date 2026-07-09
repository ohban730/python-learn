import numpy as np
import os
from transformers import pipeline
from sklearn.linear_model import LogisticRegression

def main():
    print("=== LLM (Embeddings) + scikit-learn 高速分類デモ ===\n")

    # 1. Hugging Faceの軽量モデルを使って、テキストをベクトル（特徴量）に変換するパイプラインを起動
    # モデル: sentence-transformers/all-MiniLM-L6-v2 (約90MBの非常に軽量なモデル)
    print("1. Hugging Face からテキスト埋め込み用モデルをロード中...")
    extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    print("モデルのロードが完了しました。\n")

    # 2. 訓練用のテキストデータと正解ラベル (0: IT/テクノロジー, 1: 料理/レシピ)
    train_texts = [
        "How to install PyTorch on Windows using pip",
        "Best way to configure git remote repository",
        "Docker container is not running on my local machine",
        "Easy recipe for chocolate cake with cocoa powder",
        "How to bake sourdough bread at home step by step",
        "Delicious pasta carbonara recipe for quick dinner"
    ]
    train_labels = [0, 0, 0, 1, 1, 1]

    print("2. 訓練用テキストをベクトル（Embedding）に変換中...")
    # 各テキストを 384 次元のベクトルに変換します
    train_embeddings = []
    for text in train_texts:
        # extractor(text)[0] の形状は (Token_Length, Hidden_Dim=384)
        # 単語（トークン）方向の平均をとることで、文章全体の 1本のベクトル(384次元) にします
        feats = np.mean(extractor(text)[0], axis=0)
        train_embeddings.append(feats)
    
    train_embeddings = np.array(train_embeddings)
    print(f"訓練データのShape: {train_embeddings.shape} (データ数: {train_embeddings.shape[0]}, 次元数: {train_embeddings.shape[1]})\n")

    # 3. scikit-learn の分類器（ロジスティック回帰）の定義と学習
    # CPUだけで一瞬で学習が終わります
    print("3. scikit-learn (ロジスティック回帰) の学習を実行中...")
    clf = LogisticRegression()
    clf.fit(train_embeddings, train_labels)
    print("学習が完了しました。(学習時間: 1ミリ秒未満)\n")

    # 4. 未知のテストテキストに対する予測の実行
    test_texts = [
        "git push command failed with permission error",
        "sweet apple pie recipe with cinnamon"
    ]
    
    print("4. テストデータをベクトルに変換して予測中...")
    test_embeddings = [np.mean(extractor(text)[0], axis=0) for text in test_texts]
    test_embeddings = np.array(test_embeddings)

    # 予測クラスと、それぞれの予測確率を算出
    predictions = clf.predict(test_embeddings)
    probabilities = clf.predict_proba(test_embeddings)

    print("\n--- 予測結果 ---")
    class_names = {0: "IT/テクノロジー", 1: "料理/レシピ"}
    for i, text in enumerate(test_texts):
        pred_label = predictions[i]
        prob = probabilities[i][pred_label] * 100
        print(f"入力テキスト: \"{text}\"")
        print(f" => 予測クラス: {class_names[pred_label]} (確信度: {prob:.2f}%)\n")

if __name__ == "__main__":
    main()
