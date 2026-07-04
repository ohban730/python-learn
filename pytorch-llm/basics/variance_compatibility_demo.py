import torch

print("==============================================================")
print("   有偏・不偏分散 (ベッセル補正) とロード互換性の検証デモ")
print("==============================================================\n")

# ------------------------------------------------------------
# 1. データ数 n が小さい場合の比較 (n=5)
# ------------------------------------------------------------
print("--- [STEP 1] データ数 n が小さい場合 (n = 5) ---")
x_small = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

# PyTorchデフォルト: unbiased=True (分母は n-1 = 4)
var_unbiased_s = x_small.var(unbiased=True)

# TensorFlowデフォルト: unbiased=False (分母は n = 5)
var_biased_s = x_small.var(unbiased=False)

print(f"  入力データ : {x_small.tolist()}")
print(f"  不偏分散 (分母 n-1 = 4) : {var_unbiased_s.item():.4f}")
print(f"  有偏分散 (分母 n   = 5) : {var_biased_s.item():.4f}")
diff_small = var_unbiased_s - var_biased_s
print(f"  両者の差 : {diff_small.item():.4f} (約 {diff_small.item() / var_biased_s.item() * 100:.1f}% のひらきがあります)\n")


# ------------------------------------------------------------
# 2. データ数 n が大きい場合の比較 (n=768: GPT-2の埋め込み次元)
# ------------------------------------------------------------
print("--- [STEP 2] データ数 n が大きい場合 (n = 768) ---")
torch.manual_seed(42)
x_large = torch.randn(768)

var_unbiased_l = x_large.var(unbiased=True)
var_biased_l = x_large.var(unbiased=False)

print(f"  不偏分散 (分母 n-1 = 767) : {var_unbiased_l.item():.6f}")
print(f"  有偏分散 (分母 n   = 768) : {var_biased_l.item():.6f}")
diff_large = var_unbiased_l - var_biased_l
print(f"  両者の差 : {diff_large.item():.6f}")
print(f"  --> 次元数が大きくなると、統計的な数値の差はほぼゼロ (約 {diff_large.item() / var_biased_l.item() * 100:.3f}%) になり、完全に無視できます。\n")


# ------------------------------------------------------------
# 3. 誤差の蓄積（バタフライ効果）の再現シミュレーション
# ------------------------------------------------------------
print("--- [STEP 3] レイヤー通過に伴う誤差の蓄積シミュレーション ---")
print("  統計的には無視できる『0.0017』の微小な計算のズレが、")
print("  モデルの層（LayerNormやアテンション）を何十層も繰り返すことで、")
print("  どれほど巨大なズレに化けるかを再現します。\n")

# 初期値
val_tf = 1.0  # TensorFlow (有偏) 側の状態
val_pt = 1.0  # PyTorch (デフォルト / 不偏) 側の状態

# 各層を通過するごとに、わずかな差 (0.17%) が掛け算で蓄積されると仮定
error_rate = 1.0017
layers = 48  # GPT-2 Medium などのレイヤー数

print(f"  [初期状態] TensorFlow: {val_tf:.6f} | PyTorch: {val_pt:.6f} | 差: 0.00%")

for layer_idx in range(1, layers + 1):
    val_pt = val_pt * error_rate  # 毎回わずかにズレが乗算される
    # 12層、24層、36層、48層（最終層）の時点で状態を出力
    if layer_idx in [12, 24, 36, 48]:
        diff_percent = (val_pt - val_tf) / val_tf * 100
        print(f"  [{layer_idx:2d} 層通過] TensorFlow: {val_tf:.6f} | PyTorch: {val_pt:.6f} | ズレ: {diff_percent:.2f}%")

print("\n  --> [CONCLUSION]")
print("      1層あたりの差はわずか 0.17% であっても、48層の計算を通過した時点で、")
print("      元のデータ分布から 約 8.5% も数値がズレてしまいました。")
print("      これが、事前学習済みのパラメータが『デタラメな出力』に化けてしまう理由です。")
print("      だからこそ、計算仕様を完全に TensorFlow（unbiased=False）に合わせる必要があります。")
print("==============================================================")
