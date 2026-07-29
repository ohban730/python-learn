# Trainer 用評価関数 `compute_metrics` の完全解説

オライリー本第4章（112ページ）で登場する **`compute_metrics` 関数（採点係）** の内部処理と、なぜ複雑な変換が必要なのかを解きほぐして解説します。

---

## 1. 結論：何をする関数か？

`Trainer` から渡された **「モデルの予測データ」** と **「本当の正解データ」** を比較し、**採点結果（F1スコア）を計算して辞書形式 `{"f1": 0.8536}` で返す関数** です。

```text
 ［ Trainer から届くデータ (eval_pred) ］
   ・predictions: 各クラスの確率スコアの配列
   ・label_ids  : 正解ラベルの数値ID (パディング部には -100)

                │
                ▼  【 compute_metrics 関数の中で加工 】
                │  1. np.argmax() で一番確率が高い ID (0〜6) を選択
                │  2. 無効データ (-100) を除外して、数値IDを 'B-PER' などの文字列に戻す
                ▼

 ［ seqeval.metrics.f1_score() で採点 ］ ──> {'f1': 0.8536} を返却！
```

---

## 2. 関数のコードと 3 つのステップ解説

```python
def compute_metrics(eval_pred):
    # 【準備】 Trainer から予測スコアと正解ラベルを取り出す
    predictions, labels = eval_pred
    
    # 【Step 1】 7つのクラス確率の中で一番高いID (0〜6) を選ぶ
    predictions = np.argmax(predictions, axis=2)

    # 【Step 2】 パディング部分 (-100) を除外して、数値IDを文字列 ('B-PER'等) に戻す
    true_predictions = [
        [index2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [index2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 【Step 3】 評価ライブラリ seqeval に渡して F1 スコアを計算する
    return {"f1": f1_score(true_labels, true_predictions)}
```

---

### Step 1: `np.argmax(predictions, axis=2)`
* モデルの出力 `predictions` は `[バッチ数, 単語数, 7クラスの確率]` という 3 次元配列です。
* `np.argmax(..., axis=2)` を使うことで、7つの確率の中で最も数値が大きいインデックス番号（例: `5` ＝ `B-LOC`）を一つ選び出し、2 次元配列 `[バッチ数, 単語数]` に変換します。

### Step 2: `-100` の除去と `index2tag` によるラベル復元
* 文章の長さを揃えるパディング余白部分のラベルには **`-100`** が入っています。
* `if l != -100` を使って余白部分を採点対象から除外（フィルタリング）します。
* 残った本物の単語の数値ID（例: `1` や `5`）を、`index2tag` 辞書を使って `"B-PER"` や `"B-LOC"` というテキスト文字列に変換します。

### Step 3: `f1_score(true_labels, true_predictions)`
* 綺麗に整えられた正解文字列リスト `true_labels` と予測文字列リスト `true_predictions` を、NER評価専用のライブラリ `seqeval` の `f1_score` 関数に入力します。
* 計算された F1 スコアを `{"f1": 0.8536}` のように Trainer が読み取れる辞書形式で返します。

---

## 3. なぜ `if l != -100`（除去）が必要なのか？

データコレーターのパディング処理によって、文章の末尾の空きスペースには無意味な余白ラベル `-100` が埋め込まれています。

もし `-100` を除去せずに採点関数へ渡してしまうと、**「無意味な余白部分まで正解・不正解の判定対象に含まれてしまい、正解率や F1 スコアが歪んでしまう」** ため、`if l != -100` で綺麗に取り除いています。
