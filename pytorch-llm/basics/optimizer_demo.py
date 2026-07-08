import torch
import torch.optim as optim
import math

print("=" * 60)
print("  Adam vs AdamW 数理・更新挙動の違いの実証シミュレーター")
print("=" * 60)

# 初期設定
init_val = 10.0      # 初期パラメータ値 (重み)
lr = 0.1             # 学習率 (eta)
wd = 0.1             # 重み減衰率 (lambda / weight_decay)
steps = 3            # 検証ステップ数

# 擬似的な勾配データ (各ステップで一定の勾配が発生したとする)
gradients = [0.5, 0.5, 0.5]

# Adamのデフォルト超パラメータ
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# ============================================================
# 1. 従来の Adam + L2正則化 の手動シミュレーション
# (ペナルティを勾配に加算するアプローチ)
# ============================================================
print("\n--- [CASE 1] 従来の Adam + L2正則化 (手動計算) ---")
theta_adam = init_val
m_t = 0.0
v_t = 0.0

for t_idx in range(steps):
    t = t_idx + 1
    g_raw = gradients[t_idx]
    
    # L2正則化: 勾配にペナルティを加算する (g = g + wd * theta)
    g_t = g_raw + wd * theta_adam
    
    # モーメントの更新
    m_t = beta1 * m_t + (1 - beta1) * g_t
    v_t = beta2 * v_t + (1 - beta2) * (g_t ** 2)
    
    # 偏り補正
    m_hat = m_t / (1 - (beta1 ** t))
    v_hat = v_t / (1 - (beta2 ** t))
    
    # パラメータ更新
    theta_adam_old = theta_adam
    theta_adam = theta_adam_old - (lr / (math.sqrt(v_hat) + eps)) * m_hat
    print(f"  Step {t}: {theta_adam_old:.6f} ──> {theta_adam:.6f} (勾配 g_t={g_t:.4f}, v_hat={v_hat:.6f})")

# ============================================================
# 2. 改良版 AdamW の手動シミュレーション
# (更新後に直接、重みを減衰させるアプローチ)
# ============================================================
print("\n--- [CASE 2] 改良版 AdamW (手動計算) ---")
theta_adamw = init_val
m_t = 0.0
v_t = 0.0

for t_idx in range(steps):
    t = t_idx + 1
    g_t = gradients[t_idx] # 勾配にはペナルティを加えない
    
    # モーメントの更新
    m_t = beta1 * m_t + (1 - beta1) * g_t
    v_t = beta2 * v_t + (1 - beta2) * (g_t ** 2)
    
    # 偏り補正
    m_hat = m_t / (1 - (beta1 ** t))
    v_hat = v_t / (1 - (beta2 ** t))
    
    # パラメータ更新 (直接、重み減衰を適用: theta = (1 - lr * wd) * theta)
    theta_adamw_old = theta_adamw
    theta_adamw = (1.0 - lr * wd) * theta_adamw_old - (lr / (math.sqrt(v_hat) + eps)) * m_hat
    print(f"  Step {t}: {theta_adamw_old:.6f} ──> {theta_adamw:.6f} (勾配 g_t={g_t:.4f}, v_hat={v_hat:.6f})")


# ============================================================
# 3. PyTorch公式オプティマイザーとの一致検証
# ============================================================
print("\n" + "=" * 60)
print("  PyTorch公式オプティマイザーとの一致検証")
print("=" * 60)

# PyTorchで検証用のダミーパラメータを作成
param_adam = torch.nn.Parameter(torch.tensor([init_val]))
param_adamw = torch.nn.Parameter(torch.tensor([init_val]))

# PyTorchのオプティマイザをインスタンス化
# 注: PyTorchの optim.Adam で weight_decay を指定すると、従来のL2正則化と同じ挙動になります。
opt_adam = optim.Adam([param_adam], lr=lr, weight_decay=wd)
opt_adamw = optim.AdamW([param_adamw], lr=lr, weight_decay=wd)

# 各ステップ実行
for t_idx in range(steps):
    # 勾配のセット
    opt_adam.zero_grad()
    param_adam.grad = torch.tensor([gradients[t_idx]])
    opt_adam.step()
    
    opt_adamw.zero_grad()
    param_adamw.grad = torch.tensor([gradients[t_idx]])
    opt_adamw.step()

print(f"手動 Adam  最終値: {theta_adam:.6f}")
print(f"公式 Adam  最終値: {param_adam.item():.6f}")
diff_adam = abs(theta_adam - param_adam.item())
print(f"  --> 誤差: {diff_adam:.12f}")

print(f"\n手動 AdamW 最終値: {theta_adamw:.6f}")
print(f"公式 AdamW 最終値: {param_adamw.item():.6f}")
diff_adamw = abs(theta_adamw - param_adamw.item())
print(f"  --> 誤差: {diff_adamw:.12f}")

# ------------------------------------------------------------
# 4. 数値的な違いの考察
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("  数値的差異の考察 (なぜ AdamW が優れているか)")
print("=" * 60)
print(f"Adam  の更新後パラメータ: {theta_adam:.6f}")
print(f"AdamW の更新後パラメータ: {theta_adamw:.6f}")
print(f"  * 差額: {abs(theta_adam - theta_adamw):.6f}")
print("  * 考察:")
print("    Adam (L2正則化) では、自適応学習率 (1/sqrt(v_hat)) の割り算の影響で、")
print("    本来の重み減衰項までもが意図せず拡大・縮小されてしまい、正則化が歪んでいます。")
print("    一方、AdamW では更新後に独立して直接重みを減衰させる (theta * (1 - lr * wd)) ため、")
print("    すべてのパラメータが数学的意図通りに安定して減衰でき、高い汎化性能を示します。")
print("=" * 60)
