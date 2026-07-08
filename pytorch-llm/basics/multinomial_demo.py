import torch

print("=" * 60)
print("  torch.multinomial 確率比例サンプリング 1,000回検証デモ")
print("=" * 60)

# 再現性のためのシード固定
torch.manual_seed(123)

# 1. 書籍161ページの単語リストと確率分布を定義
vocab = {
    0: "closer",
    1: "every",
    2: "effort",
    3: "forward",
    4: "inches",
    5: "moves",
    6: "pizza",
    7: "toward"
}

# 各単語の予測確率 (合計が 1.0 になっています)
probas = torch.tensor([0.073, 0.000, 0.000, 0.582, 0.002, 0.000, 0.000, 0.343])

print("\n--- [前提設定] 単語ごとの予測確率 (Softmax出力) ---")
for idx, prob in enumerate(probas.tolist()):
    print(f"  ID {idx} ({vocab[idx]:<8}): 確率 {prob * 100:>5.1f}% (元の値: {prob:.3f})")


# ============================================================
# 検証 1: 従来の argmax (決定論的選択)
# ============================================================
print("\n" + "=" * 50)
print("  検証 1: torch.argmax() による決定論的選択")
print("=" * 50)
print("  ※ argmax は何度実行しても「一番高い確率」の単語しか選びません。")

for trial in range(5):
    best_id = torch.argmax(probas).item()
    print(f"    試行 {trial + 1}: 選ばれた単語 = '{vocab[best_id]}' (確率: {probas[best_id]*100:.1f}%)")


# ============================================================
# 検証 2: multinomial (確率比例サンプリング)
# ============================================================
print("\n" + "=" * 50)
print("  検証 2: torch.multinomial() による 1,000回 サンプリング")
print("=" * 50)
print("  ※ 確率の大きさに比例した頻度で、ランダムに単語が選ばれます。")

# 復元抽出 (replacement=True) で 1,000 回の抽選を一括実行
# 形状: [1000] のテンソルが得られます
sample_indices = torch.multinomial(probas, num_samples=1000, replacement=True)

# 各IDが選ばれた回数をカウント (bincount)
# 長さを合わせるため minlength を指定
counts = torch.bincount(sample_indices, minlength=len(probas))

print("\n--- 1,000回抽選した集計結果 ---")
for idx, count in enumerate(counts.tolist()):
    expected = probas[idx].item() * 1000
    actual_pct = (count / 1000.0) * 100
    print(f"  {vocab[idx]:<8}: 実計測 {count:>3} 回 ({actual_pct:>5.1f}%) | 理論上の期待値: {expected:>5.1f} 回")

print("\n考察:")
print("  * 一番確率の高い 'forward' (58.2%) が、最も多い回数 (約580回前後) 選ばれています。")
print("  * 二番手の 'toward' (34.3%) や、約7%しかない 'closer' も、確率に比例した適度な回数選ばれています。")
print("  * 確率 0.2% の 'inches' も、1000回中ごく稀に (1〜2回程度) 引っかかります。")
print("  * 確率が完全に 0% の 'every' や 'pizza' は、1000回回しても『一度も選ばれません』。")
print("=" * 60)
