import torch
import torch.nn.functional as F

def main():
    print("=== PyTorch torch.nn.functional (F) デモ ===\n")

    # 1. 活性化関数 (Activation Functions)
    print("--- 1. 活性化関数 ---")
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])
    print(f"入力 x: {x}")
    
    # ReLU: マイナスを 0 にする
    print(f"ReLU(x): {F.relu(x)}")
    
    # GELU: LLM（GPT等）で非常によく使われる活性化関数
    print(f"GELU(x): {F.gelu(x)}")
    
    # Softmax: 確率分布に変換する（合計が 1 になる）
    # dim=-1 で最後の次元に対して確率化
    probs = F.softmax(x, dim=-1)
    print(f"Softmax(x): {probs}")
    print(f"Softmaxの合計値: {probs.sum().item()}\n")

    # 2. 損失関数 (Loss Functions)
    print("--- 2. 損失関数 (クロスエントロピー) ---")
    # モデルの予測スコア（Logits）: 3つのクラスに対する予測（Batchサイズ=1）
    logits = torch.tensor([[2.0, 0.5, -1.0]])  # クラス0のスコアが最も高い
    # 正解ラベル（クラス0）
    targets = torch.tensor([0])
    
    # F.cross_entropy は Softmax適用 -> 損失計算 をまとめて行ってくれます
    loss = F.cross_entropy(logits, targets)
    print(f"Logits: {logits}")
    print(f"Target: {targets}")
    print(f"Cross Entropy Loss: {loss.item():.4f}\n")

    # 3. テンソルのパディング (Padding)
    print("--- 3. テンソルのパディング (F.pad) ---")
    # 2x2 のテンソル
    t = torch.tensor([[1, 2], 
                      [3, 4]])
    print(f"元のテンソル:\n{t}")
    
    # F.pad(テンソル, (左, 右, 上, 下), value=埋める値)
    # 左右に1セルずつ、上下に1セルずつ 0 で埋める
    padded_t = F.pad(t, (1, 1, 1, 1), mode='constant', value=0)
    print(f"パディング後 (F.pad):\n{padded_t}\n")

    # 4. 画像などのリサイズ（F.interpolate）
    print("--- 4. テンソルのリサイズ (F.interpolate) ---")
    # 形状 (Batch: 1, Channel: 1, Height: 2, Width: 2) のテンソル
    img = torch.tensor([[[[1.0, 2.0],
                          [3.0, 4.0]]]])
    print(f"元のサイズ: {img.shape}")
    
    # サイズを 2x2 から 4x4 にバイリニア補間で拡大
    resized_img = F.interpolate(img, size=(4, 4), mode='bilinear', align_corners=False)
    print(f"リサイズ後のサイズ: {resized_img.shape}")
    print(f"リサイズ後のデータ:\n{resized_img[0, 0]}\n")

    # 5. ドロップアウト (Dropout)
    print("--- 5. ドロップアウト ---")
    h = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    # training=Trueのとき、確率 p（ここでは0.5）で要素をランダムに 0 にし、
    # 残りの要素をスケールアップ（この場合は 1/0.5 = 2倍）します。
    dropped = F.dropout(h, p=0.5, training=True)
    print(f"元の値: {h}")
    print(f"Dropout後: {dropped}")

if __name__ == "__main__":
    main()
