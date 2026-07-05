import torch
import torch.nn as nn

print("==============================================================")
print("     フィードフォワード (FFN) とアテンション (MHA) の独立性デモ")
print("==============================================================\n")

# 書籍リスト4-4の FeedForward クラス
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            # 中間層の次元を 4倍 に拡大
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            # 再び元の次元に縮小
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


# 比較検証用の簡易アテンション層 (単語間で情報を混ぜる処理)
class SimpleAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key   = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        # x shape: [Batch, Sequence, Dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # アテンションウェイト (単語間の関係性スコア)
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 重み付き和 (ここで単語同士の情報がブレンドされる)
        return torch.bmm(attn_weights, v)


# ------------------------------------------------------------
# テストデータの設定 (バッチサイズ1, トークン数3, 埋め込み次元4)
# ------------------------------------------------------------
# 3単語（単語1, 単語2, 単語3）のオリジナルの文章データ
x_original = torch.tensor([[
    [1.0, 1.0, 1.0, 1.0],  # 単語1
    [2.0, 2.0, 2.0, 2.0],  # 単語2
    [3.0, 3.0, 3.0, 3.0]   # 単語3
]])

# 単語2の値だけを大きく書き換えた文章データ
x_modified = torch.tensor([[
    [1.0, 1.0, 1.0, 1.0],  # 単語1 (同じ)
    [9.0, 9.0, 9.0, 9.0],  # 単語2 (書き換え！)
    [3.0, 3.0, 3.0, 3.0]   # 単語3 (同じ)
]])

print(f"オリジナル入力 x_original:\n{x_original}\n")
print(f"書き換え後入力 x_modified (単語2のみ変更):\n{x_modified}\n")


# ------------------------------------------------------------
# 1. フィードフォワード層 (FFN) の検証
# ------------------------------------------------------------
print("--- [STEP 1] フィードフォワード層 (FFN) の実行 ---")
ffn = FeedForward(emb_dim=4)

# 重みを固定して同一比較するため、評価モードにします
ffn.eval()

with torch.no_grad():
    out_ffn_orig = ffn(x_original)
    out_ffn_mod = ffn(x_modified)

print("\n  [結果] FFNを通過した後の出力:")
print(f"  オリジナル時の単語1・3出力: {out_ffn_orig[0, 0].tolist()} / {out_ffn_orig[0, 2].tolist()}")
print(f"  書き換え時の単語1・3出力  : {out_ffn_mod[0, 0].tolist()} / {out_ffn_mod[0, 2].tolist()}")

diff_ffn_word1 = (out_ffn_orig[0, 0] - out_ffn_mod[0, 0]).abs().sum().item()
diff_ffn_word3 = (out_ffn_orig[0, 2] - out_ffn_mod[0, 2]).abs().sum().item()

print(f"  単語1 の出力差 : {diff_ffn_word1:.6f}")
print(f"  単語3 の出力差 : {diff_ffn_word3:.6f}")
print("  --> [SUCCESS] 出力値は『完全に一致（差が 0.0）』しました！")
print("                FFNは各単語を一方向へ独立処理するため、")
print("                他の単語の値がどれだけ変わっても影響を一切受けません。\n")


# ------------------------------------------------------------
# 2. アテンション層の検証 (対比用)
# ------------------------------------------------------------
print("--- [STEP 2] アテンション層の実行 ---")
attn = SimpleAttention(emb_dim=4)
attn.eval()

with torch.no_grad():
    out_attn_orig = attn(x_original)
    out_attn_mod = attn(x_modified)

print("\n  [結果] アテンションを通過した後の出力:")
print(f"  オリジナル時の単語1・3出力: {out_attn_orig[0, 0].tolist()} / {out_attn_orig[0, 2].tolist()}")
print(f"  書き換え時の単語1・3出力  : {out_attn_mod[0, 0].tolist()} / {out_attn_mod[0, 2].tolist()}")

diff_attn_word1 = (out_attn_orig[0, 0] - out_attn_mod[0, 0]).abs().sum().item()
diff_attn_word3 = (out_attn_orig[0, 2] - out_attn_mod[0, 2]).abs().sum().item()

print(f"  単語1 の出力差 : {diff_attn_word1:.6f}")
print(f"  単語3 の出力差 : {diff_attn_word3:.6f}")
print("  --> [RESULT] 他の単語（単語2）に連動して、単語1と3の出力値も変化しました！")
print("               アテンションは文脈理解のために、単語同士の情報を混ぜるからです。")
print("==============================================================")
print("  要約: Transformerブロックは、")
print("        『アテンションで情報を混ぜ合わせ（横のブレンド）』、")
print("        『フィードフォワード(FFN)で各単語が個別に深く知識処理する（縦の抽出）』")
print("        という見事なコンビプレイで成り立っています。")
print("==============================================================")
