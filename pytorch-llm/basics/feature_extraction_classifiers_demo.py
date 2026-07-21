import numpy as np
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=== 特徴抽出ベースの分類器（ロジスティック回帰 vs ランダムフォレスト）デモ ===\n")

    # 1. テキストをベクトル（埋め込み）に変換するモデルのロード
    # 非常に軽量な sentence-transformers/all-MiniLM-L6-v2 (約90MB) を使用
    print("1. Hugging Face から特徴抽出用モデルをロード中...")
    extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    print("モデルのロードが完了しました。\n")

    # 2. 訓練用データ（ツイート風のテキストと感情ラベル: 0=悲しい/怒り、1=嬉しい/楽しい）
    train_texts = [
        "I hate this cold and rainy weather, it is so depressing.",
        "My computer crashed and I lost all my hard work. I am so mad!",
        "This service is absolute garbage, what a complete waste of money.",
        "I am so excited! I just got accepted to my dream university!",
        "Had a wonderful dinner with my family. The food was absolutely delicious.",
        "Such a beautiful sunny day! Spending it at the beach with friends."
    ]
    train_labels = [0, 0, 0, 1, 1, 1]

    # 3. テキストをベクトル特徴量に変換（この段階で Transformer の重みは固定/凍結）
    print("2. テキストを 384次元の隠れ状態ベクトル（特徴量）に変換中...")
    train_features = []
    for text in train_texts:
        # extractor(text)[0] の形状は (トークン数, 384)
        # トークン方向の平均（Mean Pooling）を計算し、文章全体の1本のベクトルにします
        emb = np.mean(extractor(text)[0], axis=0)
        train_features.append(emb)
    
    train_features = np.array(train_features)
    print(f"特徴量テンソルの形状: {train_features.shape} (データ数: 6, 特徴量の次元数: 384)\n")

    # --- 分類器 1: ニューラル分類層（ロジスティック回帰）の学習 ---
    # ※ ロジスティック回帰は、もっとも単純な1層のニューラル分類層 (Linear + Sigmoid/Softmax) と数学的に同等です。
    print("3. 分類器1: ロジスティック回帰（ニューラル分類層）の学習を実行中...")
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    print("ロジスティック回帰の学習完了。\n")

    # --- 分類器 2: ランダムフォレストの学習 ---
    # ※ 決定木を多数決させる、勾配計算を使わない（勾配に依存しない）定番アルゴリズムです。
    print("4. 分類器2: ランダムフォレストの学習を実行中...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(train_features, train_labels)
    print("ランダムフォレストの学習完了。\n")

    # 5. テスト用テキストの予測
    test_texts = [
        "Everything is going wrong today, I want to go home.",
        "I won the lottery! This is the best day of my life!"
    ]
    test_labels = [0, 1] # 正解ラベル

    print("5. 未知のテストデータをベクトルに変換して予測を実行中...")
    test_features = np.array([np.mean(extractor(text)[0], axis=0) for text in test_texts])

    # 予測の実行
    lr_preds = lr_clf.predict(test_features)
    lr_probs = lr_clf.predict_proba(test_features)

    rf_preds = rf_clf.predict(test_features)
    rf_probs = rf_clf.predict_proba(test_features)

    # 結果の表示
    print("\n================== 予測結果の比較 ==================")
    class_names = {0: "悲しい/怒り (0)", 1: "嬉しい/楽しい (1)"}
    
    for i, text in enumerate(test_texts):
        print(f"\n入力文: \"{text}\" (正解: {class_names[test_labels[i]]})")
        
        # ロジスティック回帰の結果
        lr_pred = lr_preds[i]
        lr_prob = lr_probs[i][lr_pred] * 100
        print(f" ├─ [ロジスティック回帰]: {class_names[lr_pred]} (確信度: {lr_prob:.2f}%)")
        
        # ランダムフォレストの結果
        rf_pred = rf_preds[i]
        rf_prob = rf_probs[i][rf_pred] * 100
        print(f" └─ [ランダムフォレスト]: {class_names[rf_pred]} (確信度: {rf_prob:.2f}%)")
    print("====================================================")

if __name__ == "__main__":
    main()
