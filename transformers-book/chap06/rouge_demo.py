# ------------------------------------------------------------------
# Attribution Notice:
# Based on: "Natural Language Processing with Transformers" (Chapter 6.4.2)
# Authors: Lewis Tunstall, Leandro von Werra, Thomas Wolf (O'Reilly Media)
# Repository: https://github.com/nlp-with-transformers/notebooks
# ------------------------------------------------------------------

"""
ROUGE スコア (ROUGE-1, ROUGE-2, ROUGE-L) 数理アルゴリズム Pure Python 実装デモ

外部ライブラリに依存せず、ROUGE の分子・分母・最長共通部分文字列(LCS) の
動的計画法 (Dynamic Programming) 計算プロセスを追体験するプログラムです。
"""

from collections import Counter

def get_ngrams(tokens, n):
    """単語リストから n-gram のリストを生成"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_rouge_n(ref_tokens, pred_tokens, n=1):
    """ROUGE-N (Recall, Precision, F1) の計算"""
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    
    if not ref_ngrams or not pred_ngrams:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    
    # 共通して存在する n-gram のカウント (最小値の和)
    overlap_count = 0
    for ngram, count in pred_ngrams.items():
        overlap_count += min(count, ref_ngrams.get(ngram, 0))
    
    total_ref = sum(ref_ngrams.values())
    total_pred = sum(pred_ngrams.values())
    
    recall = overlap_count / total_ref if total_ref > 0 else 0.0
    precision = overlap_count / total_pred if total_pred > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    
    return {"recall": recall, "precision": precision, "f1": f1}

def lcs_length(ref_tokens, pred_tokens):
    """最長共通部分文字列 (LCS) の長さを動的計画法 (DP) で算出"""
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    return dp[m][n]

def calculate_rouge_l(ref_tokens, pred_tokens):
    """ROUGE-L (LCSベースの Recall, Precision, F1) の計算"""
    lcs_len = lcs_length(ref_tokens, pred_tokens)
    
    m, n = len(ref_tokens), len(pred_tokens)
    recall = lcs_len / m if m > 0 else 0.0
    precision = lcs_len / n if n > 0 else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    
    return {"lcs": lcs_len, "recall": recall, "precision": precision, "f1": f1}

def main():
    print("==========================================")
    print("  ROUGE スコア数理計算デモ (Pure Python)")
    print("==========================================\n")
    
    reference = "the cat is on the mat"
    prediction = "the cat is sitting on mat"
    
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    
    print(f"正解要約 (Reference) : {reference}")
    print(f"モデル出力 (Prediction): {prediction}")
    print("-" * 50)
    
    # 1. ROUGE-1
    r1 = calculate_rouge_n(ref_tokens, pred_tokens, n=1)
    print(f"[ROUGE-1 (Unigram)]")
    print(f"  - Recall    (再現率): {r1['recall']:.4f}  (一致: 5 / 正解語数: 6)")
    print(f"  - Precision (適合率): {r1['precision']:.4f}  (一致: 5 / 生成語数: 6)")
    print(f"  - F1 Score  (調和平均): {r1['f1']:.4f}")
    print("-" * 50)
    
    # 2. ROUGE-2
    r2 = calculate_rouge_n(ref_tokens, pred_tokens, n=2)
    print(f"[ROUGE-2 (Bigram)]")
    print(f"  - Recall    (再現率): {r2['recall']:.4f}")
    print(f"  - Precision (適合率): {r2['precision']:.4f}")
    print(f"  - F1 Score  (調和平均): {r2['f1']:.4f}")
    print("-" * 50)
    
    # 3. ROUGE-L
    rl = calculate_rouge_l(ref_tokens, pred_tokens)
    print(f"[ROUGE-L (最長共通部分文字列 LCS)]")
    print(f"  - LCS 長 (共通系列長): {rl['lcs']}")
    print(f"  - Recall    (再現率): {rl['recall']:.4f}")
    print(f"  - Precision (適合率): {rl['precision']:.4f}")
    print(f"  - F1 Score  (調和平均): {rl['f1']:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()
