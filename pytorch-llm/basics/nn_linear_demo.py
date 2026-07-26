import torch
import torch.nn as nn

def main():
    print("=== PyTorch nn.Linear (全結合層/線形層) の動きと次元変換デモ ===\n")

    # 1. 2次元テンソルでの例 (一般的なテーブルデータや画像特徴量)
    # Shape: (batch_size=4, in_features=10)
    in_features = 10
    out_features = 5
    linear_layer = nn.Linear(in_features, out_features)

    torch.manual_seed(42)
    x_2d = torch.randn(4, in_features)
    y_2d = linear_layer(x_2d)

    print("--- 1. 2次元テンソル [batch, in_features] の次元変換 ---")
    print(f"入力 Shape: {x_2d.shape}  (batch_size=4, in_features=10)")
    print(f"出力 Shape: {y_2d.shape}   (batch_size=4, out_features=5)")
    print("⇒ 最後の次元が 10 から 5 へ自動変換されました！\n")

    # 2. 3次元テンソルでの例 (Transformer/LLM のテキストデータ)
    # Shape: (batch_size=2, seq_len=5, hidden_dim=768)
    hidden_dim = 768
    intermediate_dim = 3072  # 一般的にFFN層では4倍(3072)に拡大する
    
    linear_expand = nn.Linear(hidden_dim, intermediate_dim)
    linear_shrink = nn.Linear(intermediate_dim, hidden_dim)

    x_3d = torch.randn(2, 5, hidden_dim)
    
    # 拡大変換
    expanded = linear_expand(x_3d)
    # 縮小変換
    shrunk = linear_shrink(expanded)

    print("--- 2. 3次元テンソル [batch, seq_len, hidden_dim] (LLM) の次元変換 ---")
    print(f"入力 Shape     : {x_3d.shape}     (batch=2, seq_len=5, 768次元)")
    print(f"拡大後の Shape : {expanded.shape} (batch=2, seq_len=5, 3072次元に拡大)")
    print(f"縮小後の Shape : {shrunk.shape}   (batch=2, seq_len=5, 768次元に還元)")
    print("⇒ batch や seq_len (2, 5) の次元はそのまま維持され、最後の次元だけが独立して変換されています！\n")

    # 3. 内部のパラメータ (重み W と バイアス b) の Shape 確認
    print("--- 3. nn.Linear 内部で保持されているパラメータ ---")
    print(f"重み W (weight) の Shape : {linear_expand.weight.shape}  (out_features, in_features)")
    print(f"バイアス b (bias) の Shape: {linear_expand.bias.shape}        (out_features)")

if __name__ == "__main__":
    main()
