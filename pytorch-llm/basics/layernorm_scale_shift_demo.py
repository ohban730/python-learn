import torch
import torch.nn as nn

print("==============================================================")
print("     LayerNorm における scale と shift の分布制御デモ")
print("==============================================================\n")

# 書籍リスト4-2の LayerNorm クラス
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # nn.Parameter として定義 (学習可能なパラメータ)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # scale を掛けて、shift を足すことで分布を再調整する
        return self.scale * norm_x + self.shift


# 5次元の埋め込みを持つ、2つのサンプルのダミーデータ
# 各サンプル（行）ごとに異なるスケールや平均を持っています
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [10.0, 20.0, 30.0, 40.0, 50.0]
])

print(f"入力データ x (Shape: {list(x.shape)}):\n{x}\n")

# LayerNorm のインスタンス化 (埋め込み次元 = 5)
layer = LayerNorm(emb_dim=5)


# ------------------------------------------------------------
# STEP 1: 初期状態 (scale=1.0, shift=0.0) での標準化確認
# ------------------------------------------------------------
print("--- [STEP 1] 初期パラメータでの LayerNorm 適用 ---")
print(f"  現在の scale (初期値): {layer.scale.data}")
print(f"  現在の shift (初期値): {layer.shift.data}")

# レイヤーを通過させる
out_initial = layer(x)

# 各行の平均と分散を計算 (keepdim=True を指定して次元を保持)
mean_initial = out_initial.mean(dim=-1, keepdim=True)
var_initial = out_initial.var(dim=-1, keepdim=True, unbiased=False)

print(f"\n  出力結果 (Initial):\n{out_initial}")
print(f"  出力の平均値 (各行): {mean_initial.squeeze().tolist()} (ほぼ 0.0)")
print(f"  出力の分散値 (各行): {var_initial.squeeze().tolist()} (ほぼ 1.0)")
print("  --> [SUCCESS] 初期状態では、データは平均0・分散1に綺麗に標準化されています。\n")


# ------------------------------------------------------------
# STEP 2: パラメータを手動変更して、分布の再調整を実証
# ------------------------------------------------------------
print("--- [STEP 2] scale と shift パラメータの手動変更 ---")
print("  モデルが学習によって、平均を 10.0 にシフトし、")
print("  標準偏差を 2.5 (分散を 2.5^2 = 6.25) にスケールしたと仮定します。\n")

# パラメータの書き換え
# (実運用では nn.Parameter の中身は逆伝播の勾配で自動更新されます)
layer.scale.data = torch.tensor([2.5, 2.5, 2.5, 2.5, 2.5])
layer.shift.data = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])

print(f"  変更後の scale: {layer.scale.data}")
print(f"  変更後の shift: {layer.shift.data}")

# 同じ入力データを通す
out_modified = layer(x)

# 再度、各行の平均と分散を計算
mean_modified = out_modified.mean(dim=-1, keepdim=True)
var_modified = out_modified.var(dim=-1, keepdim=True, unbiased=False)

print(f"\n  出力結果 (Modified):\n{out_modified}")
print(f"  出力の平均値 (各行): {mean_modified.squeeze().tolist()} (平均が 10.0 へ移動！)")
print(f"  出力の分散値 (各行): {var_modified.squeeze().tolist()} (分散が 6.25 [2.5^2] へ拡大！)")
print("  --> [SUCCESS] scale と shift パラメータの力によって、")
print("                最終出力データの平均と分散が完全に制御されました！")
print("==============================================================")
