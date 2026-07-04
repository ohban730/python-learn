import torch

# --------------------------------------------------
# PyTorch の torch.arange の挙動を理解するデモ
# --------------------------------------------------

print("=== 1. 引数を1つだけ指定した場合 (0 から end-1 までの連番) ===")
# torch.arange(end) は 0 から end までの整数（ただし end は含まない）を生成します
tensor_1 = torch.arange(5)
print("torch.arange(5):", tensor_1)
print("データ型:", tensor_1.dtype)  # デフォルトは torch.int64 (整数型)
print("-" * 50)


print("\n=== 2. 開始と終了を指定した場合 (start から end-1 までの連番) ===")
# torch.arange(start, end)
tensor_2 = torch.arange(2, 7)
print("torch.arange(2, 7):", tensor_2)  # 2, 3, 4, 5, 6
print("-" * 50)


print("\n=== 3. ずらし幅 (step) を指定した場合 ===")
# torch.arange(start, end, step)
# 2から始まり、2ずつ増やしながら10未満まで生成
tensor_3 = torch.arange(2, 10, 2)
print("torch.arange(2, 10, 2):", tensor_3)  # 2, 4, 6, 8
print("-" * 50)


print("\n=== 4. なぜ位置埋め込みで使われるのか？ ===")
# max_length = 4 (コンテキスト長が4) のとき、
# 各トークンの「位置インデックス (0番目, 1番目, 2番目, 3番目)」を瞬時に作成するために使われます。
max_length = 4
pos_indices = torch.arange(max_length)
print("位置インデックスのテンソル:", pos_indices)
