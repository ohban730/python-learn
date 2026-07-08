import torch
import torch.nn.functional as F

print("=" * 60)
print("  損失計算の6つのステップ (図5-7) の実証シミュレーター")
print("=" * 60)

# 再現性のための乱数シード固定
torch.manual_seed(123)

# 極小の語彙数とデータサイズを定義
batch_size = 2
seq_len = 3
vocab_size = 5

# ------------------------------------------------------------
# STEP 1: ロジット (Logits)
# ------------------------------------------------------------
print("\n--- [STEP 1] ロジット (生スコア) ---")
# 形状: [2, 3, 5]
logits = torch.randn(batch_size, seq_len, vocab_size)
print(f"logits (Shape: {logits.shape}):")
print(logits)


# ------------------------------------------------------------
# STEP 2: 確率 (Probas)
# ------------------------------------------------------------
print("\n--- [STEP 2] 確率 (Softmax適用後) ---")
# 各位置の全単語の合計確率が 1.0 になるよう変換
probas = torch.softmax(logits, dim=-1)
print(f"probas (Shape: {probas.shape}):")
print(probas)
print(f"※ 各位置の確率の合計値 (1.0になるべき): {probas.sum(dim=-1)}")


# ------------------------------------------------------------
# STEP 3: ターゲット確率 (Target Probas)
# ------------------------------------------------------------
print("\n--- [STEP 3] ターゲット確率 (正解単語の確率抽出) ---")
# 各位置の「正解単語ID」を定義 (Shape: [2, 3])
targets = torch.tensor([
    [2, 4, 1],  # バッチ0の正解単語ID
    [0, 3, 2]   # バッチ1の正解単語ID
])
print(f"正解ターゲット単語ID (targets):\n{targets}")

# アドバンスト・インデキシングによる正解確率のピンポイント抽出
target_probas_list = []
for i in range(batch_size):
    # バッチ i に対する抽出
    extracted = probas[i, [0, 1, 2], targets[i]]
    target_probas_list.append(extracted)
    print(f"  バッチ {i} のターゲット確率: {extracted}")

# 2つのバッチ分を結合して1つのフラットなテンソルにする
target_probas = torch.cat(target_probas_list)
print(f"\n結合したターゲット確率 (Shape: {target_probas.shape}):")
print(target_probas)


# ------------------------------------------------------------
# STEP 4: 対数確率 (Log Probas)
# ------------------------------------------------------------
print("\n--- [STEP 4] 対数確率 (Log Probas) ---")
# 確率の対数を取る
log_probas = torch.log(target_probas)
print(f"log_probas:")
print(log_probas)


# ------------------------------------------------------------
# STEP 5 & 6: 平均対数確率 & 負の平均対数確率 (Loss)
# ------------------------------------------------------------
print("\n--- [STEP 5 & 6] 平均対数確率 ──> 負の平均対数確率 (Loss) ---")
mean_log_proba = log_probas.mean()
loss_manual = -mean_log_proba

print(f"平均対数確率: {mean_log_proba.item():.6f}")
print(f"手動計算による損失 (loss_manual): {loss_manual.item():.6f}")


# ============================================================
# PyTorch 組み込みの cross_entropy 関数による一括計算と比較
# ============================================================
print("\n" + "=" * 60)
print("  PyTorch組み込み F.cross_entropy との比較")
print("=" * 60)

# PyTorchの cross_entropy 関数は、入力を以下のようにフラット化して受け取る必要があります
# logits:  [Batch * Seq_Len, Vocab_Size]  (2*3, 5) => (6, 5)
# targets: [Batch * Seq_Len]              (2*3)   => (6)
flat_logits = logits.view(-1, vocab_size)
flat_targets = targets.view(-1)

print(f"フラット化した logits  形状: {flat_logits.shape}")
print(f"フラット化した targets 形状: {flat_targets.shape}")

# 一括計算
loss_pytorch = F.cross_entropy(flat_logits, flat_targets)
print(f"\nPyTorch公式損失 (loss_pytorch): {loss_pytorch.item():.6f}")

# 誤差の検証
diff = torch.abs(loss_manual - loss_pytorch).item()
print(f"手動計算と公式関数の誤差: {diff:.12f}")

if diff < 1e-7:
    print("\n[SUCCESS] 手動の6ステップ計算と、PyTorch組み込み関数の結果が完全に一致しました！")
else:
    print("\n[ERROR] 誤差が発生しています。計算を見直してください。")
print("=" * 60)
