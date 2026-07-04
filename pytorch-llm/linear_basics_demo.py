"""
PyTorchの torch.nn.Linear (全結合層 / 線形層) とバイアスの仕組みを理解するデモプログラム

このコードは、nn.Linearが内部にどのような重みパラメータを保持し、
どのような計算式で出力を算出しているかを、手動計算との比較によって実証します。
"""

import torch
import torch.nn as nn

print("=== 1. nn.Linear (バイアスあり) の定義 ===")
# 入力サイズ d_in = 3, 出力サイズ d_out = 2 のリニア層を作成
# (書籍の W_query などと同じサイズ設定)
torch.manual_seed(123)
linear_with_bias = nn.Linear(in_features=3, out_features=2, bias=True)

# nn.Linearが内部に自動的に作成した「重み」と「バイアス」を取り出します。
# これらは自動的に nn.Parameter として登録されています。
w_param = linear_with_bias.weight
b_param = linear_with_bias.bias

# パラメータの形状 (Shape) を確認
# ※注意: PyTorchの nn.Linear 内の weight は、数式上の転置を考慮して [d_out, d_in] (2行3列) の形で保持されています。
print(f"重み (weight) の形状: {w_param.shape}") # [2, 3]
print(f"重みの値:\n{w_param.data}\n")

print(f"バイアス (bias) の形状: {b_param.shape}") # [2]
print(f"バイアスの値: {b_param.data}\n")


print("=== 2. nn.Linear を用いた出力の計算 ===")
# テスト用の入力データ (1サンプル、3次元ベクトル)
# Shape: [1, 3]
x = torch.tensor([[0.5, 0.8, -0.2]])
print(f"入力 x: {x}")

# nn.Linearに入力を渡して計算します (正しい作法に従いインスタンスを直接呼び出します)
output_linear = linear_with_bias(x)
print(f"nn.Linearの出力: {output_linear.data}\n")


print("=== 3. 数式通りの手動計算による検証 ===")
# nn.Linearの内部計算式: y = x @ W.T + b  (入力 x と 重み W の転置の行列積に、バイアス b を足す)
# ※ w_param が [2, 3] なので、w_param.T (転置) は [3, 2] になります。
# [1, 3] @ [3, 2] -> [1, 2] の行列積になり、そこに [2] のバイアスがブロードキャストで加算されます。
output_manual = x @ w_param.T + b_param

print(f"手動計算 (x @ W.T + b) の出力: {output_manual.data}")

# 両者の計算結果が完全に一致するか検証
is_equal = torch.allclose(output_linear, output_manual)
print(f"両者の結果は完全に一致するか?: {is_equal}\n")


print("=== 4. nn.Linear (バイアスなし: bias=False) の場合 ===")
# bias=False に設定すると、バイアスパラメータは作成されません (None になります)。
# 現代のLLM（GPTなど）のアテンション射影では、この bias=False 設定が一般的です。
linear_no_bias = nn.Linear(in_features=3, out_features=2, bias=False)

print(f"bias=False 時のバイアスの状態: {linear_no_bias.bias}") # None

# 内部計算式は単純な行列積のみになります: y = x @ W.T
output_no_bias_linear = linear_no_bias(x)
output_no_bias_manual = x @ linear_no_bias.weight.T

is_equal_no_bias = torch.allclose(output_no_bias_linear, output_no_bias_manual)
print(f"バイアスなし時の出力: {output_no_bias_linear.data}")
print(f"手動計算 (x @ W.T) との一致確認: {is_equal_no_bias}")
