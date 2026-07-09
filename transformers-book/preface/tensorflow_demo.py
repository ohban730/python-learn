import tensorflow as tf
from tensorflow.keras import layers
import os

# 1. モデルの定義 (PyTorch の nn.Module に対応)
class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_outputs):
        super().__init__()
        
        # PyTorchの nn.Sequential に対応
        self.layers_seq = tf.keras.Sequential([
            layers.Dense(30, activation='relu'),  # nn.Linear(inputs, 30) + nn.ReLU()
            layers.Dense(20, activation='relu'),  # nn.Linear(30, 20) + nn.ReLU()
            layers.Dense(num_outputs)             # nn.Linear(20, outputs)
        ])

    # PyTorch の forward(self, x) に対応
    def call(self, x):
        logits = self.layers_seq(x)
        return logits

def main():
    print("=== TensorFlow による PyTorch 再現デモ (appendix-A 相当) ===\n")

    # 2. データセットの準備 (PyTorch の Dataset & DataLoader に対応)
    # X_train, y_train
    X_train = tf.constant([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ], dtype=tf.float32)
    y_train = tf.constant([0, 0, 0, 1, 1], dtype=tf.int32)

    # X_test, y_test
    X_test = tf.constant([
        [-0.8, 2.8],
        [2.6, -1.6]
    ], dtype=tf.float32)
    y_test = tf.constant([0, 1], dtype=tf.int32)

    # tf.data を使って DataLoader を作成 (PyTorch の DataLoader 相当)
    # drop_remainder=True は PyTorch の drop_last=True に対応
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_loader = train_dataset.shuffle(buffer_size=5, seed=123).batch(2, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_loader = test_dataset.batch(2)

    # 3. モデル、オプティマイザ、損失関数の初期化
    model = NeuralNetwork(num_outputs=2)
    
    # PyTorch: torch.optim.SGD 相当
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    
    # PyTorch: F.cross_entropy 相当 (from_logits=True で Softmax 前の出力を処理)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 乱数シードの設定 (モデル初期化のため)
    tf.random.set_seed(123)

    # 4. 訓練ループの実行 (PyTorch のトレーニングループに対応)
    num_epochs = 3
    for epoch in range(num_epochs):
        
        # バッチ単位でループ
        for batch_idx, (features, labels) in enumerate(train_loader):
            
            # GradientTape が順伝播の演算（勾配計算に必要な値）を記録します
            with tf.GradientTape() as tape:
                logits = model(features)          # 順伝播
                loss = loss_fn(labels, logits)    # 損失計算

            # 勾配の計算 (PyTorchの loss.backward() 相当)
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # 重みの更新 (PyTorchの optimizer.step() 相当)
            # ※TensorFlowでは zero_grad() は不要で、apply_gradients時に新しく適用されます
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch: {batch_idx+1:03d}/002"
                  f" | Train Loss: {loss:.2f}")

    # 5. 推論と精度の計算
    outputs = model(X_train)
    probas = tf.nn.softmax(outputs, axis=1)
    print("\n--- 訓練データに対する予測確率 (Softmax) ---")
    print(probas.numpy())

    # 精度計算の関数定義 (PyTorchの compute_accuracy 相当)
    def compute_accuracy(model, dataloader):
        correct = 0
        total_examples = 0
        
        for features, labels in dataloader:
            logits = model(features)
            # 最も確率の高いインデックスを取得 (PyTorchの argmax 相当)
            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            
            compare = (predictions == labels)
            correct += tf.reduce_sum(tf.cast(compare, tf.int32)).numpy()
            total_examples += len(compare)
            
        return correct / total_examples

    print("\n--- 評価 (Accuracy) ---")
    print(f"Train Acc: {compute_accuracy(model, train_loader):.4f}")
    print(f"Test Acc:  {compute_accuracy(model, test_loader):.4f}")

    # 6. 重みの保存 (PyTorch の model.state_dict() 保存相当)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "tf_model_weights.weights.h5")
    model.save_weights(save_path)
    print(f"\n=> モデルの重みを保存しました: {save_path}")

if __name__ == "__main__":
    main()
