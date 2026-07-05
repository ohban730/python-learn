import torch
import torch.nn as nn

print("==============================================================")
print("     勾配消失のシミュレーション (標準層 vs 残差接続層 20層比較)")
print("==============================================================\n")

# 特徴量次元
DIM = 32
# レイヤー数 (深さ)
NUM_LAYERS = 20

# ------------------------------------------------------------
# 1. スキップ接続なしの標準ブロック
# ------------------------------------------------------------
class StandardBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        
        # 勾配消失をシミュレートしやすくするため、少し小さな重みで初期化します
        # (層が深くなると掛け算の連鎖で急速に勾配が縮小します)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.25)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.act(self.linear(x))


# ------------------------------------------------------------
# 2. スキップ接続（残差接続）ありのブロック
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        
        # 比較のため、初期値は全く同じスケールにします
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.25)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # 出力 = 入力 x + 計算結果 F(x) (スキップ接続)
        return x + self.act(self.linear(x))


# ------------------------------------------------------------
# 3. 20層のネットワーク構築と実行
# ------------------------------------------------------------
# 比較しやすくするため、シード値を固定します
torch.manual_seed(42)

# スキップ接続なしモデルの構築
standard_layers = nn.ModuleList([StandardBlock(DIM) for _ in range(NUM_LAYERS)])
# スキップ接続ありモデルの構築
residual_layers = nn.ModuleList([ResidualBlock(DIM) for _ in range(NUM_LAYERS)])

# 同一の入力データ
x_in = torch.randn(5, DIM)

# 順伝播 (Forward Pass)
x_std = x_in.clone()
for layer in standard_layers:
    x_std = layer(x_std)

x_res = x_in.clone()
for layer in residual_layers:
    x_res = layer(x_res)

# ロスの計算 (出力ベクトルの平均)
loss_std = x_std.mean()
loss_res = x_res.mean()

# 逆伝播 (Backward Pass)
loss_std.backward()
loss_res.backward()


# ------------------------------------------------------------
# 4. 勾配の大きさ (絶対値の平均) を比較
# ------------------------------------------------------------
print("--- [結果] 各レイヤーの重みが受け取った勾配の大きさ (絶対値平均) ---")
print("※ Layer 0 が入力に最も近く、Layer 19 が出力に最も近い層です。\n")

print(f"{'Layer番号':<10} | {'標準ネット (スキップなし)':<25} | {'残差ネット (スキップあり)':<25}")
print("-" * 70)

for i in range(NUM_LAYERS):
    # スキップなしの勾配
    grad_std = standard_layers[i].linear.weight.grad
    grad_std_mean = grad_std.abs().mean().item() if grad_std is not None else 0.0
    
    # スキップありの勾配
    grad_res = residual_layers[i].linear.weight.grad
    grad_res_mean = grad_res.abs().mean().item() if grad_res is not None else 0.0
    
    # 指数表記と小数表記で出力
    print(f"Layer {i:>2d}      | {grad_std_mean:<25.6e} | {grad_res_mean:<25.6e}")

print("-" * 70)
print("\n[解説]")
print("* 標準ネット (スキップなし):")
print("  出力側の Layer 19 から入力側の Layer 0 へ逆伝播する過程で、")
print("  勾配が次々と掛け算され、急速に縮小しています。")
print(f"  入力に近い Layer 0 の勾配は {standard_layers[0].linear.weight.grad.abs().mean().item():.2e} と極限まで消失しており、")
print("  この層の重みはほとんど学習が進まない状態に陥っています。")
print("")
print("* 残差ネット (スキップあり):")
print("  どれだけ層が深く（NUM_LAYERS = 20）なろうとも、微分式の「+1」効果によるバイパスのおかげで、")
print("  勾配が掛け算で削られることなく、最下層の Layer 0 まで健康なサイズで維持されています。")
print(f"  Layer 0 でも {residual_layers[0].linear.weight.grad.abs().mean().item():.6f} というしっかりした勾配があり、")
print("  超深層になってもすべての層が完全に学習可能です。")
print("==============================================================")
