"""
PyTorchの超初歩と基本ワークフローを学ぶデモプログラム

このコードは、PyTorchを用いたディープラーニングモデル開発の最小限の一連のフローを再現しています。
データ準備 -> モデル定義 -> 訓練設定 -> 訓練ループ -> モデル評価 -> 保存・ロード を一気に学べます。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# STEP 1: データの準備 (Dataset & DataLoader)
# ==========================================
# PyTorchでは、データを扱うために Dataset (データの塊) と DataLoader (ミニバッチを切り出す機械) を使います。

class ToyDataset(Dataset):
    """
    自作のデータセットクラス。
    Datasetクラスを継承し、__len__ と __getitem__ を定義することが義務付けられています。
    """
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        # 指定されたインデックスの「特徴量(入力データ)」と「教師ラベル」を1セット返します。
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        # データの総数を返します。
        return self.labels.shape[0]

# おもちゃデータ (X: 2次元の特徴ベクトルが5つ, y: 0か1のラベル)
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
y_test = torch.tensor([0, 1])

# Datasetのインスタンス化
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# DataLoaderの定義 (データをシャッフルし、指定したバッチサイズごとに取り出してくれます)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,       # 訓練時はデータをシャッフルする
    drop_last=True      # 端数のデータを捨てる (バッチサイズに満たない場合)
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False       # テスト・評価時はシャッフル不要
)


# ==========================================
# STEP 2: モデルの定義 (nn.Module & nn.Sequential)
# ==========================================
# ニューラルネットワークのモデルを作る際は、必ず nn.Module を継承します。

class SimpleNeuralNetwork(nn.Module):
    """
    3層構造のシンプルなニューラルネットワーク。
    入力層 (num_inputs) -> 中間層1 (30次元) -> 中間層2 (20次元) -> 出力層 (num_outputs)
    """
    def __init__(self, num_inputs, num_outputs):
        # 親クラス(nn.Module)の初期化処理を必ず呼び出す必要があります。
        super().__init__()

        # nn.Sequential は、複数のレイヤーを「直列」に繋ぐための便利なラッパーです。
        # 内部で定義されたLinearレイヤーの重み（W）は、自動的に nn.Parameter として追跡されます。
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30), # 1つ目の全結合層 (Wのサイズ: [30, num_inputs])
            nn.ReLU(),                 # 活性化関数 (非線形変換)
            
            nn.Linear(30, 20),         # 2つ目の全結合層 (Wのサイズ: [20, 30])
            nn.ReLU(),
            
            nn.Linear(20, num_outputs) # 3つ目の全結合層 (Wのサイズ: [num_outputs, 20])
        )

    def forward(self, x):
        """
        順伝播 (Forward propagation) の処理を記述します。
        ※ 使用する際は model(x) と呼び出し、直接 model.forward(x) を呼んではいけません。
        """
        logits = self.layers(x)
        return logits


# ==========================================
# STEP 3: デバイス設定、モデルと最適化アルゴリズムの設定
# ==========================================
# GPU(CUDA)が使える場合はGPUを使い、使えない場合はCPUを使います。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルをインスタンス化し、to(device)で一括してデバイスに転送します。
model = SimpleNeuralNetwork(num_inputs=2, num_outputs=2)
model = model.to(device)

# オプティマイザ（パラメータ更新エンジン）の定義。
# model.parameters() を渡すだけで、モデル内のすべての重みパラメータが自動追跡されて学習対象になります。
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 学習可能なパラメータ（重みの要素数）の総数を算出する
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"学習可能な総パラメータ数: {num_params} 個")


# ==========================================
# STEP 4: 訓練ループの実行
# ==========================================
print("\n--- 訓練ループ開始 ---")
num_epochs = 3
for epoch in range(num_epochs):
    # モデルを「学習モード」に設定します (ドロップアウトなどを有効化)
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        # データをモデルと同じデバイス(GPU or CPU)に送る
        features, labels = features.to(device), labels.to(device)

        # 1. 予測 (順伝播) -> 自動的に forward が呼ばれます
        logits = model(features)

        # 2. 損失（誤差）の計算 (Cross Entropy Loss)
        loss = F.cross_entropy(logits, labels)

        # 3. パラメータの更新（★PyTorchお決まりの3ステップ）
        optimizer.zero_grad()  # 前のバッチで計算された勾配（傾き）をリセットしてゼロにする
        loss.backward()        # 逆伝播 (Backpropagation) を実行し、全パラメータの勾配を自動計算する
        optimizer.step()       # 計算された勾配に基づいて、重みを少しだけ更新する

        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
              f"| Batch: {batch_idx+1:02d} "
              f"| Loss (誤差): {loss.item():.4f}")


# ==========================================
# STEP 5: モデル of 評価 (推論モード)
# ==========================================
print("\n--- モデルの評価 ---")

# モデルを「評価・推論モード」に設定します (ドロップアウトなどを無効化)
model.eval()

def compute_accuracy(model, dataloader):
    correct = 0.0
    total_examples = 0

    # 評価時は勾配計算が不要なため、with torch.no_grad() で囲んで
    # メモリ消費と計算グラフの構築をシャットアウトし、高速化させます。
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            
            # 各データの出力値（確率のようなスコア）のうち、最も大きなインデックスを取り出す (予測結果)
            predictions = torch.argmax(logits, dim=1)
            
            # 予測と教師ラベルを比較
            compare = (predictions == labels)
            correct += torch.sum(compare).item()
            total_examples += len(compare)
            
    return correct / total_examples

train_acc = compute_accuracy(model, train_loader)
test_acc = compute_accuracy(model, test_loader)
print(f"訓練データの正解率: {train_acc * 100:.1f}%")
print(f"テストデータの正解率: {test_acc * 100:.1f}%")


# ==========================================
# STEP 6: モデルの保存と再ロード
# ==========================================
print("\n--- モデルの保存とロード ---")
model_path = "model.pth"

# state_dict はモデル内の「パラメータ（重み）の名前と値の辞書データ」です。
# torch.save を使ってこれをファイルに書き込みます。
torch.save(model.state_dict(), model_path)
print(f"モデルのパラメータを '{model_path}' に保存しました。")

# 新しいモデルのインスタンスをゼロから作ります
new_model = SimpleNeuralNetwork(num_inputs=2, num_outputs=2).to(device)

# 保存されたパラメータファイルを読み込み、新しいインスタンスへ流し込みます
new_model.load_state_dict(torch.load(model_path, map_location=device))
print("保存された重みを新しいモデルインスタンスにロードしました。")

# ロードしたモデルで推論テスト
new_model.eval()
with torch.no_grad():
    sample_input = torch.tensor([[-0.8, 2.8]]).to(device)
    output = new_model(sample_input)
    # ソフトマックスで確率化
    prob = torch.softmax(output, dim=1)
    print(f"サンプル入力 [-0.8, 2.8] に対する予測確率: {prob.cpu().numpy()}")

# テスト用に作成したモデルファイルを削除してクリーンアップします
if os.path.exists(model_path):
    os.remove(model_path)
