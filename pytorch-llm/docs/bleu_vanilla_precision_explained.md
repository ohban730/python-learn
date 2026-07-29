# BLEUスコアにおける「単純な精度（Vanilla Precision）」の分子の意味

評価指標 **BLEU** の解説において、単純な精度（Vanilla Precision）の計算例で登場する「分子の 6」についての詳細な解説ノートです。

---

## 1. 疑問の整理

以下の例において、なぜ単純な精度が $\frac{6}{6} = 100\%$ になるのか？分子の「6」は何を意味しているのか？

* **参照文（手本の正解文）**: `the cat is on the mat` (合計 6 単語／`the` は 2 回登場)
* **生成文（AIが作成した文）**: `the the the the the the` (合計 6 単語／`the` を 6 連呼)

---

## 2. 「通常の精度 (Precision)」の計算ルール

機械学習における精度（Precision）の定義は以下の通りです。

$$\text{精度 (Precision)} = \frac{\text{AIが生成した単語のうち、参照文の中にも存在する単語の数}}{\text{AIが生成した全単語の数}}$$

### 分母の「6」
* AIが生成した文章の**全体の単語数**です。
* `the` `the` `the` `the` `the` `the` $\rightarrow$ 計 **6 単語**。

### 分子の「6」
AIが生成した 6 つの単語を、左から1つずつ「手本文の中に存在する単語かどうか？」チェックした結果です。

1. 1つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)
2. 2つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)
3. 3つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)
4. 4つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)
5. 5つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)
6. 6つ目の `the` $\rightarrow$ 参照文 `the cat is on the mat` に入っているか？ $\rightarrow$ **入っている！** (+1)

合計して、**「参照文に存在する単語である」と判定された回数が 6 回**あるため、**分子が「6」** になります。

---

## 3. 何が問題なのか？（Vanilla Precision の欠陥）

単純な精度計算では、**「すでにカウントした単語の回数」を記憶していません**。
そのため、参照文に `the` という単語が1回でも含まれていれば、AIが `the` を何十回連呼しようと、**すべての `the` が毎回「正解単語！」と判定されて加算されてしまう**のです。

その結果：
$$\text{単純な精度} = \frac{6 \text{ (ヒットした回数)}}{6 \text{ (AIの全単語数)}} = 100\%$$
という、**あきらかに狂った高評価**になってしまいます。

---

## 4. 解決策：修正精度（Modified Precision）での「クリッピング」

この欠陥を直すために、BLEUでは**「参照文に含まれる最大出現数でカウントを打ち切る（クリッピング）」**というルール（修正精度）を導入しています。

* 参照文 `the cat is on the mat` の中で `the` は **最高 2 回** しか出てこない。
* AIが `the` を 6 回出しても、カウントの上限は **2 回** までとする。

結果、分子が 「6」 から 「2」 に補正されます。

$$\text{修正精度 (Modified Precision)} = \frac{2 \text{ (上限にクリップされたヒット数)}}{6 \text{ (AIの全単語数)}} = \frac{2}{6} \approx 33.3\%$$

これにより、同じ単語の連呼によるインチキ高得点を防ぐことができます。
