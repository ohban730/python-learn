import torch

def print_bar_chart(title, values, labels):
    """値を簡易的な棒グラフ（アスタリスク）としてコンソールに出力します。"""
    print(f"\n--- {title} ---")
    for label, val in zip(labels, values):
        # 割合(0.0〜1.0)を40マスの幅にマッピングしてアスタリスクを表示
        bar = "*" * int(val * 40)
        print(f"{label:10} ({val:.4f}): {bar}")

# 書籍の inputs 行列から journey (x^2) をクエリとして計算したドット積のスコア
scores = torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
labels = ["Your", "journey", "starts", "with", "one", "step"]

print("=== 元のアテンションスコア（ドット積の結果） ===")
for label, score in zip(labels, scores):
    print(f"{label:10} : {score:.4f}")
print("-" * 50)


# ==================================================
# PART 1: 単純な比率（割り算）による正規化
# ==================================================
# 各値を全体の合計で割るだけの正規化。
# 値の大小関係はそのまま縮小されます。
scores_simple = scores / scores.sum()
print_bar_chart("単純な比率（割り算）での正規化", scores_simple, labels)


# ==================================================
# PART 2: ソフトマックス関数による正規化
# ==================================================
# 指数関数 e^x を用いた正規化。
# 値が大きいトークンがより強く強調（引き上げ）されます。
scores_softmax = torch.softmax(scores, dim=0)
print_bar_chart("ソフトマックス関数による正規化", scores_softmax, labels)
print("※ 'journey' (1.4950) と 'starts' (1.4754) の僅かな差が、ソフトマックス後はより強調されています。")
print("-" * 50)


# ==================================================
# PART 3: 入力の差によるグラフ（出力形状）の変化
# ==================================================
print("\n=== PART 3: 入力値の差による出力形状の変化 ===")

# 3-A. 1つだけ値が突出している場合 (最大値を極端に際立たせる効果)
# 指数関数的に大きくなるため、少しの差が非常に大きな確率差になります。
extreme_scores = torch.tensor([1.0, 2.0, 6.0])  # 6.0 が大きい
ext_labels = ["Item A", "Item B", "Item C"]

ext_simple = extreme_scores / extreme_scores.sum()
ext_softmax = torch.softmax(extreme_scores, dim=0)

print_bar_chart("突出した入力：単純な比率 [1.0, 2.0, 6.0]", ext_simple, ext_labels)
print_bar_chart("突出した入力：ソフトマックス [1.0, 2.0, 6.0] (Cが独占)", ext_softmax, ext_labels)


# 3-B. 入力にマイナス値（負のスコア）が含まれる場合
# 類似度の計算（ドット積など）で値がマイナスになっても、指数関数 e^x は常に正の数 (>0) を返すため、
# 確率（0以上1以下）として正しくマッピングされます。
negative_scores = torch.tensor([-2.0, 0.0, 3.0])
neg_labels = ["Negative", "Zero", "Positive"]

# 単純な比率だと -2.0 / 1.0 = -2.0 などになり、確率として破綻します
neg_softmax = torch.softmax(negative_scores, dim=0)
print_bar_chart("マイナスを含む入力：ソフトマックス [-2.0, 0.0, 3.0]", neg_softmax, neg_labels)
