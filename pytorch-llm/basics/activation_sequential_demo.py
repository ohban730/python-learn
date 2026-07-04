import torch
import torch.nn as nn

print("==============================================================")
print("  活性化関数(ReLU)・keepdim・nn.Sequential の動作デモ")
print("==============================================================\n")

# ------------------------------------------------------------
# 1. 活性化関数 ReLU の挙動検証
# ------------------------------------------------------------
print("=== 1. 活性化関数 ReLU の挙動 ===")

# マイナス値とプラス値が混在する1次元テンソル
x = torch.tensor([-2.5, -0.5, 0.0, 1.2, 3.8])
relu = nn.ReLU()
relu_out = relu(x)

print(f"入力テンソル x : {x}")
print(f"ReLU(x) の出力 : {relu_out}")
print("=> 入力が 0 以下の部分はすべて 0.0 になり、0 より大きい部分はそのまま出力されています。\n")


# ------------------------------------------------------------
# 2. keepdim の有無による Shape の違いと引き算の検証
# ------------------------------------------------------------
print("=== 2. keepdim の有無による Shape の違い ===")

# 2行3列の行列
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print(f"元の行列 matrix (Shape: {list(matrix.shape)}):\n{matrix}\n")

# A. keepdim=False (デフォルト) で最後の次元 (dim=-1) の平均を計算
mean_no_keep = matrix.mean(dim=-1, keepdim=False)
print(f"A. keepdim=False の平均 : {mean_no_keep}")
print(f"   Shape: {list(mean_no_keep.shape)} (次元数が 1 に減少！)")

# B. keepdim=True で最後の次元 (dim=-1) の平均を計算
mean_with_keep = matrix.mean(dim=-1, keepdim=True)
print(f"B. keepdim=True  の平均 :\n{mean_with_keep}")
print(f"   Shape: {list(mean_with_keep.shape)} (元の2次元構造をキープ！)\n")

# C. 元の行列から平均を引き算する (標準化の再現)
print("--- 元の行列から平均を引き算するテスト ---")

# keepdim=True の場合
sub_with_keep = matrix - mean_with_keep
print(f"* matrix - mean_with_keep (keepdim=True) --> 成功！:\n{sub_with_keep}")
print("  (各行から、それぞれの行の平均値 [2.0 と 5.0] が正しく引かれています)\n")

# keepdim=False の場合
try:
    print("* matrix - mean_no_keep (keepdim=False) に挑戦します...")
    sub_no_keep = matrix - mean_no_keep
    print(f"  計算結果: {sub_no_keep}")
except RuntimeError as e:
    print("  [ERROR] 計算失敗！エラーが発生しました。")
    print(f"  エラー内容: {e}")
    print("  => 次元数が [2, 3] と [2] で一致しないため、PyTorchが計算を拒否しました。")
    print("     これが keepdim=True にしなければならないプログラム上の理由です。\n")


# ------------------------------------------------------------
# 3. nn.Sequential による層の直列連結
# ------------------------------------------------------------
print("=== 3. nn.Sequential による層の直列化 ===")

# Linear層 (5次元 -> 6次元) と ReLU層 を Sequential で連結
sequential_layer = nn.Sequential(
    nn.Linear(5, 6),
    nn.ReLU()
)

# 2つのサンプル、各5次元の特徴量を持つダミー入力
torch.manual_seed(123) # 書籍と再現性を合わせるためのシード設定
batch_example = torch.randn(2, 5)

# 順伝播 (forward) の実行
out = sequential_layer(batch_example)

print(f"入力テンソル (Shape: {list(batch_example.shape)}):\n{batch_example}")
print(f"出力テンソル (Shape: {list(out.shape)}):\n{out}")
print("=> 5次元から6次元に変換され、同時にすべての負の値が 0.0 にカットされています。")
print("   nn.Sequential を使うことで、この一連の流れを 1 行で処理できます。\n")
print("==============================================================")
