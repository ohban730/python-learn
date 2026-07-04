import os
import torch
import numpy as np
import matplotlib.pyplot as plt

print("==============================================================")
print("  勾配降下法（SGD & Momentum）による等高線上の探索デモ")
print("==============================================================\n")

# 1. 損失関数の定義 (楕円形のすり鉢状の関数)
# L(w1, w2) = 0.5 * w1^2 + 2.0 * w2^2
# 最適値 (最小損失) は (0, 0)
def loss_func(w1, w2):
    return 0.5 * w1**2 + 2.0 * w2**2

# 2. 最適化プロセスをシミュレートする関数
def run_optimization(optimizer_type, lr=0.1, steps=40):
    # 初期パラメータ位置 (-4.0, 3.0)
    w = torch.tensor([-4.0, 3.0], requires_grad=True)
    
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD([w], lr=lr)
    elif optimizer_type == "Momentum":
        # 慣性係数 momentum=0.8
        optimizer = torch.optim.SGD([w], lr=lr, momentum=0.8)
    else:
        raise ValueError("Unknown optimizer type")
        
    trajectory = []
    
    for step in range(steps):
        # 現在のパラメータを記録 (グラフプロット用にテンソルからデタッチしてコピー)
        current_w = w.detach().numpy().copy()
        trajectory.append(current_w)
        
        # 損失の計算
        loss = loss_func(w[0], w[1])
        
        # 勾配の計算 (自動微分)
        optimizer.zero_grad()
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
    return np.array(trajectory)

# 3. 各オプティマイザで軌跡を生成
print("オプティマイザの探索をシミュレート中...")
steps = 30
lr = 0.4

sgd_traj = run_optimization("SGD", lr=lr, steps=steps)
momentum_traj = run_optimization("Momentum", lr=lr, steps=steps)

print(f"  SGDの初期位置     : {sgd_traj[0]}")
print(f"  SGDの最終位置     : {sgd_traj[-1]} (Loss: {loss_func(sgd_traj[-1][0], sgd_traj[-1][1]):.6f})")
print(f"  Momentumの最終位置: {momentum_traj[-1]} (Loss: {loss_func(momentum_traj[-1][0], momentum_traj[-1][1]):.6f})\n")

# 4. 等高線図の描画と可視化
print("等高線図のプロットを作成中...")
fig, ax = plt.subplots(figsize=(8, 7))

# グリッドの作成
w1_range = np.linspace(-5.0, 5.0, 100)
w2_range = np.linspace(-4.0, 4.0, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = loss_func(W1, W2)

# 等高線のプロット (20本の等高線を描画)
contours = ax.contour(W1, W2, Z, levels=20, cmap="viridis", alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

# 谷底 (最小値 0, 0) に赤い星印をプロット
ax.plot(0, 0, "r*", markersize=15, label="Global Minimum (0, 0)")

# SGDの軌跡を描画 (ジグザグに進む様子)
ax.plot(sgd_traj[:, 0], sgd_traj[:, 1], color="tab:blue", linestyle="-", linewidth=2.0, label="Standard SGD")
ax.scatter(sgd_traj[:, 0], sgd_traj[:, 1], color="tab:blue", s=25, alpha=0.8)

# Momentumの軌跡を描画 (慣性で曲がりながら進む様子)
ax.plot(momentum_traj[:, 0], momentum_traj[:, 1], color="tab:orange", linestyle="-", linewidth=2.0, label="Momentum SGD")
ax.scatter(momentum_traj[:, 0], momentum_traj[:, 1], color="tab:orange", s=25, alpha=0.8)

# スタート地点 (-4.0, 3.0) をプロット
ax.plot(-4.0, 3.0, "go", markersize=8, label="Start Point (-4, 3)")

# グラフの設定
ax.set_title("Gradient Descent Optimization Trajectory on Loss Contour", fontsize=12, fontweight="bold")
ax.set_xlabel("Parameter 1 (w1)", fontsize=10)
ax.set_ylabel("Parameter 2 (w2)", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlim([-5.0, 5.0])
ax.set_ylim([-4.0, 4.0])

# 画像として保存
output_filename = "gradient_descent_contour.png"
output_path = os.path.join(os.path.dirname(__file__), output_filename)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"プロット画像の保存完了: {output_path}")
print("==============================================================")
