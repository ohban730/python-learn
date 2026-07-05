import torch

print("=" * 60)
print("  torch.argmax() の動作解説デモ")
print("=" * 60)

# ============================================================
# STEP 1: 最も基本的な使い方
# ============================================================
print("\n--- [STEP 1] 基本 ---")

scores = torch.tensor([0.1, 0.8, 0.3, 0.05, 0.6])
print(f"スコアのリスト: {scores}")

result = torch.argmax(scores)
print(f"最大値: {scores[result].item():.2f}")
print(f"最大値のインデックス: {result.item()}")
print(" --> インデックス 1 の位置 (0.8) が最大値")


# ============================================================
# STEP 2: LLMでの実際の使い方 (語彙から次の単語IDを選ぶ)
# ============================================================
print("\n--- [STEP 2] LLMでの実際の使い方 ---")

vocab = {0: "Every", 1: "effort", 2: "moves", 3: "you", 4: "forward"}
logits = torch.tensor([-0.5, 0.3, 1.8, 0.1, 0.6])  # 各単語の「次に来る確率スコア」

print(f"語彙: {vocab}")
print(f"ロジット（スコア）: {logits}")

next_token_id = torch.argmax(logits, dim=-1)
print(f"\n最大スコアのインデックス (= 次の単語ID): {next_token_id.item()}")
print(f"選ばれた次の単語: '{vocab[next_token_id.item()]}'")


# ============================================================
# STEP 3: dim=-1 を指定した2次元テンソルへの適用
# (バッチ × 語彙 の形状で複数文章を一括処理する)
# ============================================================
print("\n--- [STEP 3] バッチ処理での適用 (dim=-1) ---")
print("logits の形状: [バッチサイズ=2, 語彙サイズ=5]")

# バッチサイズ=2、語彙サイズ=5 の2次元テンソル
batch_logits = torch.tensor([
    [-0.5,  0.3,  1.8,  0.1,  0.6],  # バッチ1 (文章1)
    [ 0.9, -0.2,  0.4,  2.1, -0.1],  # バッチ2 (文章2)
])
print(f"\nbatch_logits:\n{batch_logits}")

# dim=-1 は「最後の次元（語彙の次元）に沿って最大値を探す」という指定
next_ids = torch.argmax(batch_logits, dim=-1)
print(f"\ntorch.argmax(dim=-1) の結果: {next_ids}")
print(f"  → バッチ1の次のトークンID: {next_ids[0].item()} ('{vocab[next_ids[0].item()]}')")
print(f"  → バッチ2の次のトークンID: {next_ids[1].item()} ('{vocab[next_ids[1].item()]}')")


# ============================================================
# STEP 4: dim を指定しないとどうなるか（落とし穴）
# ============================================================
print("\n--- [STEP 4] dim を指定しないと？（落とし穴） ---")

no_dim = torch.argmax(batch_logits)            # dim 未指定
with_dim = torch.argmax(batch_logits, dim=-1)  # dim=-1 指定

print(f"dim 未指定: {no_dim.item()} (テンソル全体を1次元に平坦化してから最大値を探す！)")
print(f"dim=-1 指定: {with_dim}   (各バッチ行ごとに独立して最大値インデックスを返す)")
print("\n★ LLMではバッチごとに処理したいので、必ず dim=-1 を指定すること！")

print("\n" + "=" * 60)
print("  まとめ")
print("  argmax = 最大値の『値』ではなく、最大値の『インデックス』を返す")
print("  LLMでは argmax(logits, dim=-1) で各バッチの次の単語IDを一括取得")
print("=" * 60)
