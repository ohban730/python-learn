# ROUGE スコア（テキスト要約評価指標）の徹底解説

> **📄 出典・参考情報**
> * **参考書籍**: 『機械学習エンジニアのためのTransformers』 (オライリー・ジャパン発行, ISBN: 978-4-87311-995-3)  
> * **該当箇所**: 6.4.2 節「ROUGE」 (pp.160-161)  
> * **著者**: Lewis Tunstall, Leandro von Werra, Thomas Wolf / 訳: 中山 光樹  
> * **原論文**: Chin-Yew Lin, *"ROUGE: A Package for Automatic Evaluation of Summaries"* (2004)

---

## 💡 1. 一言でいうと？ (BLEU と ROUGE の本質的違い)

テキスト生成モデルの評価指標である **BLEU** と **ROUGE** は、どちらも「人間が書いた正解文（参照文/Reference）」と「モデルが生成した文（生成文/Prediction）」の一致度を測定しますが、**着目している視点（分子と分母関係）が逆**です。

| 指標 | 主な対象タスク | 重視する観点 | 質問のイメージ |
|---|---|---|---|
| **BLEU** | 機械翻訳 | **適合率 (Precision)** | 「モデルが生成した単語のうち、正解文に含まれる正しい単語はどれくらいあるか？」 (余計な単語や嘘がないか) |
| **ROUGE** | **テキスト要約** | **再現率 (Recall)** | 「正解文に含まれる重要な単語のうち、モデルが漏らさず生成できた単語はどれくらいあるか？」 (重要情報を網羅できたか) |

---

## 🎯 2. なぜ要約タスクでは ROUGE (再現率) が重視されるのか？

要約の目的は、**「元の長い文章から大事なポイントを漏れなくカバーすること」** です。

* もし「適合率（Precision）」だけを重視すると、正解文からたった1単語（例: `"Yes"`）だけを正しく出力した文が「精度100%」として高評価されてしまいます。
* しかし要約としては、**正解に含まれている重要単語をどれだけ網羅（カバー）できたか（Recall）** が最も大切です。
* そのため、要約評価では Recall をベースにした **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** が標準指標として用いられます。

> ⚠️ **注意（現在の実装）**:  
> Recallだけを追求すると「正解の単語を網羅するためにダラダラと長文を出力するモデル」が高得点になってしまいます。そのため、現代のHugging Face等のライブラリ（`evaluate` ライブラリの `rouge`）では、Recall と Precision の両方を考慮した **F1 スコア（調和平均）** を最終値として出力するのが一般的です。

---

## 🧩 3. ROUGE の種類とそれぞれの意味

ROUGE には注目する単位や計算方法に応じていくつかのアプローチがあります。

### ① ROUGE-N (ROUGE-1, ROUGE-2 など)
$n$-gram（$n$ 個の連続した単語のカタマリ）の一致度を計算します。

* **ROUGE-1**: 1単語（Unigram）単位での一致率。単語のカバー率を測ります。
* **ROUGE-2**: 2単語の連なり（Bigram）単位での一致率。単語の並び順・フレーズの自然さを測ります。

$$\text{ROUGE-N (Recall)} = \frac{\text{正解文と生成文で一致した } n\text{-gram の数}}{\text{正解文に含まれるすべての } n\text{-gram の数}}$$

---

### ② ROUGE-L (最長共通部分文字列 / Longest Common Subsequence)
単語の**連続性に左右されない「順序を保った共通の最長部分系列（LCS）」** を用いて計算します。

* **例**:
  * 正解文 $X$: `["The", "quick", "brown", "fox"]` (長さ4)
  * 生成文 $Y$: `["The", "fast", "brown", "fox"]` (長さ4)
  * 共通する最長系列 (LCS): `["The", "brown", "fox"]` (長さ3)
* 完全な一致でなくても、「単語の登場順序」が合っていれば柔軟にスコアを与えられるため、文構造の柔軟な要約に対応できます。
* ROUGE-L では、**文（sentence）単位**でLCSを計算し、平均をとります。

---

### ③ ROUGE-Lsum (Summary レベルの ROUGE-L)
* ROUGE-Lが「1文ごと」にLCSを計算するのに対し、**ROUGE-Lsum** は改行（`\n`）を含む**要約全体（サマリー全体）**を1つの大きなテキストとして扱い、全体のLCSを直接計算します。
* 複数文からなる長文要約の評価に適しています。

---

## 🧮 4. 具体例で理解する ROUGE-1 の計算

### 例題
* **参照文 (Reference / 正解要約)**: `"the cat is on the mat"` （全6単語）
* **生成文 (Prediction / モデル出力)**: `"the cat is sitting on mat"` （全6単語）

#### 【Step 1: 単語の抽出】
* 正解文の単語: `["the", "cat", "is", "on", "the", "mat"]` (出現頻度: the:2, cat:1, is:1, on:1, mat:1)
* 生成文の単語: `["the", "cat", "is", "sitting", "on", "mat"]` (出現頻度: the:1, cat:1, is:1, sitting:1, on:1, mat:1)

#### 【Step 2: 重なり (Overlap) のカウント】
* 一致した単語: `the`(1回), `cat`(1回), `is`(1回), `on`(1回), `mat`(1回) $\to$ 合計 5 単語一致

#### 【Step 3: スコアの算出】
* **Recall (再現率)** = $\frac{5 \text{ (一致数)}}{6 \text{ (正解文の単語数)}} = 0.8333$ (83.3%)
* **Precision (適合率)** = $\frac{5 \text{ (一致数)}}{6 \text{ (生成文の単語数)}} = 0.8333$ (83.3%)
* **F1 スコア** = $2 \times \frac{0.8333 \times 0.8333}{0.8333 + 0.8333} = 0.8333$ (83.3%)

---

## 💻 5. サンプルコードデモ

`transformers-book/chap06/rouge_demo.py` に、`evaluate` ライブラリを使用したROUGE計算の動作検証用スクリプトを配置しています。

* **実行コード**: [transformers-book/chap06/rouge_demo.py](file:///C:/Users/owner/Documents/lab/Antigravity/python-learn/transformers-book/chap06/rouge_demo.py)
