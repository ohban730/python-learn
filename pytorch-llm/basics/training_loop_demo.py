import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  PyTorch 訓練ループ 内部状態追跡デモ (ゼロから重み更新まで)")
print("=" * 60)

# ------------------------------------------------------------
# 1. 訓練データセットの準備 (y = 2x + 1 を学習させる)
# ------------------------------------------------------------
# 全データ 4 件
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# バッチサイズ 2 に分割 (全2バッチ ──> 1エポック=2ステップ)
batches = [
    (x_data[0:2], y_data[0:2]),  # バッチ0 (x=[1,2], y=[3,5])
    (x_data[2:4], y_data[2:4])   # バッチ1 (x=[3,4], y=[7,9])
]

# ------------------------------------------------------------
# 2. モデルとオプティマイザの定義
# ------------------------------------------------------------
# y = w * x + b の線形モデル
model = nn.Linear(1, 1)

# パラメータ変化を追跡しやすくするため、初期値を手動で設定
# (正解は w=2.0, b=1.0)
with torch.no_grad():
    model.weight.fill_(1.0)
    model.bias.fill_(0.0)

# 最適化手法として最も単純なSGDを採用 (学習率 lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("\n--- [初期状態] パラメータ ---")
print(f"重み (weight): {model.weight.item():.4f}")
print(f"バイアス (bias) : {model.bias.item():.4f}")

# ------------------------------------------------------------
# 3. 訓練ループのシミュレーション (2エポック分)
# ------------------------------------------------------------
num_epochs = 2

for epoch in range(num_epochs):
    print(f"\n============================================================")
    print(f"  EPOCH {epoch + 1} / {num_epochs}")
    print(f"============================================================")
    
    for batch_idx, (inputs, targets) in enumerate(batches):
        print(f"\n--- [EPOCH {epoch + 1} / Batch {batch_idx}] ---")
        print(f"入力バッチ x: {inputs.flatten().tolist()}")
        print(f"正解バッチ y: {targets.flatten().tolist()}")
        
        # ------------------------------------------------------
        # STEP 1: optimizer.zero_grad()
        # ------------------------------------------------------
        optimizer.zero_grad()
        print("\n[STEP 1] optimizer.zero_grad() 実行後")
        w_grad = model.weight.grad
        b_grad = model.bias.grad
        # 最初のループやリセット直後は grad は None または 0 になります
        print(f"  重みの勾配 (weight.grad): {w_grad if w_grad is None else f'{w_grad.item():.4f}'}")
        print(f"  バイアスの勾配 (bias.grad) : {b_grad if b_grad is None else f'{b_grad.item():.4f}'}")
        print("  --> 以前のバッチの勾配が完全にクリアされました。")
        
        # ------------------------------------------------------
        # STEP 2: Forwardパス & Loss計算
        # ------------------------------------------------------
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        print(f"\n[STEP 2] Forward & Loss計算")
        print(f"  予測値 (outputs): {outputs.flatten().detach().tolist()}")
        print(f"  損失値 (loss)   : {loss.item():.6f}")
        
        # ------------------------------------------------------
        # STEP 3: loss.backward()
        # ------------------------------------------------------
        loss.backward()
        print("\n[STEP 3] loss.backward() 実行後")
        print(f"  重みの勾配 (weight.grad): {model.weight.grad.item():.4f}")
        print(f"  バイアスの勾配 (bias.grad) : {model.bias.grad.item():.4f}")
        # backwardの段階では、勾配は計算されましたが、重み自体はまだ変わっていません
        print(f"  更新前の重み (weight): {model.weight.item():.4f} (まだ変化していません)")
        print(f"  更新前のバイアス (bias) : {model.bias.item():.4f} (まだ変化していません)")
        
        # ------------------------------------------------------
        # STEP 4: optimizer.step()
        # ------------------------------------------------------
        optimizer.step()
        print("\n[STEP 4] optimizer.step() 実行後")
        # 重み = 重み - lr * 勾配  の式で値が更新されます
        print(f"  更新後の重み (weight): {model.weight.item():.4f}")
        print(f"  更新後のバイアス (bias) : {model.bias.item():.4f}")
        print("  --> 勾配方向にパラメータが微調整されました。")

print("\n" + "=" * 60)
print("  まとめ")
print("  * 各エポックの中で、すべてのバッチが順番に学習処理される。")
print("  * zero_grad() で勾配をクリアしないと、backward()のたびに古い勾配が蓄積してしまう。")
print("  * backward() は勾配の計算のみ行い、step() で初めてパラメータの値が書き換わる。")
print("=" * 60)
