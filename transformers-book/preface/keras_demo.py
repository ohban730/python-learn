import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def main():
    print("=== Keras (High-Level API) によるモデル学習デモ ===\n")

    # 1. 訓練データおよびテストデータの定義 (NumPy 配列を使用)
    X_train = np.array([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ], dtype=np.float32)
    y_train = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    X_test = np.array([
        [-0.8, 2.8],
        [2.6, -1.6]
    ], dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int32)

    # 2. モデルの定義 (Keras Sequential スタイル)
    # 積み木のようにレイヤーを重ねるだけで定義できます
    print("1. モデルを定義中...")
    model = keras.Sequential([
        layers.Dense(30, activation='relu', input_shape=(2,)),  # 中間層1 (30ニューロン)
        layers.Dense(20, activation='relu'),                   # 中間層2 (20ニューロン)
        layers.Dense(2)                                        # 出力層 (2クラス分のスコア)
    ])

    # 3. モデルのコンパイル (学習方法の設定)
    # 最適化アルゴリズム、損失関数、および評価用のメトリクスを指定します
    print("2. 学習ルールを設定 (コンパイル) 中...")
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']  # 自動で正解率(Accuracy)も計算するように設定
    )

    # 4. 学習の実行 (Keras の本領発揮：model.fit)
    # 手動のバッチループや自動微分の処理はすべてこの 1行の内部で自動実行されます
    print("\n3. 学習 (fit) を開始します...")
    # batch_size=2, epochs=3 で実行 (訓練データが5件なので、1エポックあたり2バッチ処理されます)
    model.fit(X_train, y_train, epochs=3, batch_size=2, verbose=1)

    # 5. モデルの評価 (model.evaluate)
    # テストデータに対する損失と正解率を一瞬で測定します
    print("\n4. テストデータでモデルを評価中...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # 6. 未知のデータに対する予測の実行 (model.predict)
    print("\n5. 予測を実行中...")
    logits = model.predict(X_train, verbose=0)
    # 確率に変換するために Softmax に通す
    probas = tf.nn.softmax(logits, axis=1).numpy()
    
    print("\n--- 訓練データに対する予測確率 ---")
    print(probas)

if __name__ == "__main__":
    main()
