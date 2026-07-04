import torch

# ==================================================
# PART 1: 基礎的な Python / PyTorch 関数のデモ
# ==================================================
print("=== PART 1: torch.empty と enumerate ===")

# 1-A. torch.empty(size)
# 初期化処理（ゼロ埋めなど）を行わずに、メモリの領域だけを確保します。
# そのため、中身は「メモリのゴミデータ（未定義の値）」が入りますが、高速にテンソルを確保できます。
empty_tensor = torch.empty(6)
print("torch.empty(6) の中身 (実行ごとに異なるゴミ値が入ります):")
print(empty_tensor)

# 1-B. enumerate(sequence)
# リストやテンソルから「インデックス (i)」と「中身のデータ (x_i)」を同時に取り出すための関数です。
sample_list = ["Your", "journey", "starts"]
print("\nenumerate(sample_list) の動作:")
for i, word in enumerate(sample_list):
    print(f"インデックス: {i}, 要素: {word}")
print("-" * 50)


# ==================================================
# PART 2: ループを用いたドット積によるアテンションスコアの計算
# ==================================================
print("\n=== PART 2: ループによるドット積 (書籍のやり方) ===")

# 6単語、各単語が3次元の埋め込みベクトル (Shape: [6, 3])
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2つ目の単語 "journey" (x^2) をクエリとする
print("Query (x^2):", query)

# アテンションスコアを格納する空のテンソルを用意 (6単語分)
attn_scores_loop = torch.empty(inputs.shape[0])

# 各入力トークン (x_i) と クエリ (query) のドット積をループで計算
for i, x_i in enumerate(inputs):
    # torch.dot は1次元ベクトル同士のドット積（内積＝要素ごとの掛け算の合計）を計算します
    # 計算例 (i=0): 0.43*0.55 + 0.15*0.87 + 0.89*0.66 = 0.9544
    attn_scores_loop[i] = torch.dot(x_i, query)

print("ループで求めたスコア:\n", attn_scores_loop)
print("-" * 50)


# ==================================================
# PART 3: 【本質】行列演算によるアテンションスコアの一括計算
# ==================================================
print("\n=== PART 3: 行列演算による一括計算 (ループなし) ===")

# 行列 inputs [6, 3] と ベクトル query [3] の積を計算します。
# `@` 演算子と `torch.matmul` は機能的に完全に同じ（前者は後者のショートカット）です。
# 形状の対応: [6, 3] x [3] -> [6]

# 方法A: @ 演算子を使用
attn_scores_matrix_at = inputs @ query

# 方法B: torch.matmul() を使用
attn_scores_matrix_matmul = torch.matmul(inputs, query)

print("方法A (@) のスコア:\n", attn_scores_matrix_at)
print("方法B (matmul) のスコア:\n", attn_scores_matrix_matmul)
print("-" * 50)


# ==================================================
# PART 4: 答え合わせ
# ==================================================
print("\n=== PART 4: 答え合わせ ===")
# すべての結果が一致しているかを検証します
is_equal_loop_at = torch.allclose(attn_scores_loop, attn_scores_matrix_at)
is_equal_at_matmul = torch.allclose(attn_scores_matrix_at, attn_scores_matrix_matmul)

print("1. ループ版 と @ 演算子版 の結果は一致するか？:", is_equal_loop_at)
print("2. @ 演算子版 と matmul() 版 の結果は一致するか？:", is_equal_at_matmul)
