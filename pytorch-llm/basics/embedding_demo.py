import torch
import torch.nn as nn

# --------------------------------------------------
# nn.Embedding を直感的に理解するためのデモプログラム
# --------------------------------------------------

# 1. 極小サイズの Embedding レイヤを作成します
# - 語彙数 (num_embeddings) = 5 (扱えるIDは 0, 1, 2, 3, 4)
# - 埋め込み次元 (embedding_dim) = 3 (1つのIDを3つの数値で表す)
vocab_size = 5
embed_dim = 3
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

print("=== 1. 内部のルックアップテーブル（重み行列） ===")
# nn.Embedding の実体はただの [5行 x 3列] のテンソル（パラメータ）です
# ※ 初期値はランダムな値で設定されます（学習によって更新されていきます）
print(embedding_layer.weight)
print("テーブルの形状 (vocab_size, embed_dim):", embedding_layer.weight.shape)
print("-" * 50)

# 2. 入力するトークンIDを用意します
# 例として、ID: 1 と ID: 3 を指定します
input_ids = torch.tensor([1, 3])
print("\n=== 2. 入力トークンID ===")
print("入力:", input_ids)
print("-" * 50)

# 3. Embeddingレイヤに入力してベクトルに変換します
embedded = embedding_layer(input_ids)
print("\n=== 3. 変換後のベクトル（出力） ===")
print("出力:\n", embedded)
print("出力の形状 (入力数, embed_dim):", embedded.shape)
print("-" * 50)

# 4. 「ただのテーブル引き（辞書引き）」であることを確認します
print("\n=== 4. 答え合わせ ===")
print("重み行列の 1行目 (Index 1):", embedding_layer.weight[1].detach())
print("出力された 1つ目のベクトル:", embedded[0].detach())
print("=> 両者が完全に一致していることが分かります。")

print("\n重み行列の 3行目 (Index 3):", embedding_layer.weight[3].detach())
print("出力された 2つ目のベクトル:", embedded[1].detach())
print("=> こちらも完全に一致しています。")
