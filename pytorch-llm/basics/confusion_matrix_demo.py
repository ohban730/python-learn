import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    print("=== 混同行列 (Confusion Matrix) 可視化デモ ===\n")

    # 1. クラス名（ラベル）のリスト (6種類の感情)
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # 2. テストデータの「正解ラベル (y_true)」と「モデルの予測結果 (y_preds)」の例
    # 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
    y_true  = [0, 0, 0, 1, 1, 2, 3, 3, 4, 5]  # 実際の正解
    y_preds = [0, 0, 3, 1, 1, 1, 3, 0, 4, 1]  # モデルが予測した値

    print(f"正解データ (y_true) : {y_true}")
    print(f"予測データ (y_preds): {y_preds}\n")

    # --- 3. 混同行列の計算 ---
    # normalize="true" を指定すると、正解クラス（行）ごとの割合 (0.0〜1.0) に正規化されます
    cm = confusion_matrix(y_true, y_preds, normalize="true")

    print("計算された混同行列 (行列データ):")
    print(cm)
    print()

    # --- 4. グラフ描画の設定 ---
    # 描画用のキャンバスを作成 (6x6 インチ)
    fig, ax = plt.subplots(figsize=(7, 7))

    # ConfusionMatrixDisplay オブジェクトを作成
    # confusion_matrix: 計算した行列データ, display_labels: 軸に表示するクラス名
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # 描画を実行
    # cmap="Blues": 青色の濃淡で数値を表現
    # values_format=".2f": 小数点以下2桁まで表示 (例: 0.71)
    # colorbar=False: 右側のカラーバーを非表示にする
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)

    plt.title("Normalized Confusion Matrix (感情分類)")
    plt.tight_layout()
    
    # 画像ファイルとして保存（GUIがない環境でも確認できるようにする）
    output_path = "confusion_matrix_demo.png"
    plt.savefig(output_path)
    print(f"混同行列のグラフを '{output_path}' に保存しました。")

if __name__ == "__main__":
    main()
