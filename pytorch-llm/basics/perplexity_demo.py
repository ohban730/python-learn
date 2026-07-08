import torch
import torch.nn.functional as F

print("=" * 60)
print("  パープレキシティ (Perplexity / PPL) 数理検証シミュレーター")
print("=" * 60)

# ============================================================
# シミュレーション 1: 完全なランダム予測 (語彙数 50,257)
# ============================================================
print("\n--- [CASE 1] 完全なランダム予測 (語彙数 50,257) ---")
vocab_size = 50257

# すべての単語に均等に確率が配分されている状態 (正解確率は 1/50257)
target_proba_random = torch.tensor(1.0 / vocab_size)
print(f"正解単語の予測確率: {target_proba_random.item():.8f} (1 / {vocab_size})")

# クロスエントロピー損失 (自然対数) を計算
loss_random = -torch.log(target_proba_random)
print(f"算出された Loss: {loss_random.item():.6f}")

# 指数 (exp) を取ってパープレキシティを計算
ppl_random = torch.exp(loss_random)
print(f"算出された PPL  : {ppl_random.item():.2f}")
print(f" --> 語彙サイズ {vocab_size} と完全に一致！(モデルは50,257語すべてで迷っている)")


# ============================================================
# シミュレーション 2: N者択一で均等に迷っている状態 (例: 4択)
# ============================================================
print("\n--- [CASE 2] 4択クイズで均等に迷っている状態 ---")
n_choices = 4

# 4つの候補に確率が25%ずつ配分されている状態 (正解確率は 1/4 = 0.25)
target_proba_4way = torch.tensor(1.0 / n_choices)
print(f"正解単語の予測確率: {target_proba_4way.item():.2f} (1 / {n_choices})")

loss_4way = -torch.log(target_proba_4way)
print(f"算出された Loss: {loss_4way.item():.6f}")

ppl_4way = torch.exp(loss_4way)
print(f"算出された PPL  : {ppl_4way.item():.2f}")
print(f" --> 選択肢の数 {n_choices} と完全に一致！(モデルは実質的に4つの単語で迷っている)")


# ============================================================
# シミュレーション 3: 実際のモデル出力からの Loss と PPL の一括計算
# ============================================================
print("\n--- [CASE 3] 実際のモデル出力からの計算 ---")
torch.manual_seed(123)

batch_size = 2
seq_len = 3
vocab_size_small = 10  # 語彙サイズ 10

# ロジットとターゲットを作成
logits = torch.randn(batch_size, seq_len, vocab_size_small)
targets = torch.tensor([[2, 9, 1], [0, 4, 2]])

# フラット化して Loss を計算
flat_logits = logits.view(-1, vocab_size_small)
flat_targets = targets.view(-1)
loss = F.cross_entropy(flat_logits, flat_targets)

# PPL の計算
ppl = torch.exp(loss)

print(f"モデルのLoss (クロスエントロピー損失): {loss.item():.6f}")
print(f"モデルのPPL (パープレキシティ)        : {ppl.item():.4f}")
print(f"  → 解釈: このモデルは、次の単語を予測する際に、")
print(f"          語彙 {vocab_size_small} 個のうち平均して約 {ppl.item():.1f} 個の単語の間で迷っている")

print("\n" + "=" * 60)
print("  まとめ")
print("  * PPL = e^Loss")
print("  * あてずっぽう (完全ランダム) の時、PPL ＝ 語彙数 となる。")
print("  * PPL は「モデルが平均して何個の単語の間で迷っているか」を物理的に表す。")
print("=" * 60)
