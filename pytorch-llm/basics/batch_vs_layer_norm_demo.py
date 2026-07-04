import torch
import torch.nn as nn

print("==============================================================")
print("     バッチ正規化 (BatchNorm) と層正規化 (LayerNorm) の比較")
print("==============================================================\n")

# ダミーデータの設定 (埋め込み次元 = 5)
sample1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
sample2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
sample3 = torch.tensor([-5.0, -10.0, -15.0, -20.0, -25.0])

# バッチA: サンプル1, 2, 3 の3サンプル (バッチサイズ = 3)
batch_A = torch.stack([sample1, sample2, sample3])

# バッチB: サンプル1, 2 の2サンプル (バッチサイズ = 2)
# (サンプル3を除外し、バッチ内の構成メンバーを変更)
batch_B = torch.stack([sample1, sample2])

print(f"サンプル1 の値 : {sample1.tolist()}\n")


# ------------------------------------------------------------
# 1. バッチ正規化 (BatchNorm1d) の検証
# ------------------------------------------------------------
print("--- [STEP 1] バッチ正規化 (BatchNorm) の挙動検証 ---")
print("  ※ 周りのメンバー構成（バッチ構成）が変わると、")
print("     同じサンプル1の正規化出力がどう変化するかを検証します。\n")

# 訓練モード（そのバッチから統計量を都度計算）の挙動を再現するため
# track_running_stats=False に設定します
bn = nn.BatchNorm1d(num_features=5, momentum=None, track_running_stats=False)

# バッチA (サンプル1, 2, 3) を通過
out_bn_A = bn(batch_A)
# バッチB (サンプル1, 2のみ) を通過
out_bn_B = bn(batch_B)

# それぞれのバッチでの「サンプル1」の正規化結果を取り出す
s1_bn_in_A = out_bn_A[0]
s1_bn_in_B = out_bn_B[0]

print(f"  バッチA (サイズ3) でのサンプル1出力: {s1_bn_in_A.tolist()}")
print(f"  バッチB (サイズ2) でのサンプル1出力: {s1_bn_in_B.tolist()}")

diff_bn = (s1_bn_in_A - s1_bn_in_B).abs().sum().item()
print(f"  出力値の差の合計 : {diff_bn:.6f}")
print("  --> [RESULT] 値が変化しました！")
print("               BatchNormはバッチ全体の平均・分散を使うため、")
print("               他のデータの有無によって自分の出力値が引っ張られて変化します。\n")


# ------------------------------------------------------------
# 2. 層正規化 (LayerNorm) の検証
# ------------------------------------------------------------
print("--- [STEP 2] 層正規化 (LayerNorm) の挙動検証 ---")
print("  ※ 同様にバッチ構成を変えた際に、")
print("     同じサンプル1の正規化出力がどうなるかを検証します。\n")

ln = nn.LayerNorm(normalized_shape=5)

# バッチA (サンプル1, 2, 3) を通過
out_ln_A = ln(batch_A)
# バッチB (サンプル1, 2のみ) を通過
out_ln_B = ln(batch_B)

# それぞれのバッチでの「サンプル1」の正規化結果を取り出す
s1_ln_in_A = out_ln_A[0]
s1_ln_in_B = out_ln_B[0]

print(f"  バッチA (サイズ3) でのサンプル1出力: {s1_ln_in_A.tolist()}")
print(f"  バッチB (サイズ2) でのサンプル1出力: {s1_ln_in_B.tolist()}")

diff_ln = (s1_ln_in_A - s1_ln_in_B).abs().sum().item()
print(f"  出力値の差の合計 : {diff_ln:.6f}")
print("  --> [RESULT] 値は完全に一致（差が 0.0）しました！")
print("               LayerNormは他のデータの影響を一切受けず、")
print("               サンプル自身の横軸（特徴量次元）だけで自己完結して計算するためです。")
print("==============================================================")
print("  要約: この『バッチサイズや他のサンプルへの完全な独立性』こそが、")
print("        訓練時と推論時でバッチサイズが変動したり、可変長のテキストを扱う")
print("        LLMにおいて、LayerNorm が必須とされる最大の理由です。")
print("==============================================================")
