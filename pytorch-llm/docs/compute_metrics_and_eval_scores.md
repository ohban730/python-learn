# `compute_metrics` 関数の仕組みと評価指標 (Accuracy, F1-score) の解説

Hugging Face の `Trainer` API で使用する評価指標計算関数 `compute_metrics` のコード1行ごとの役割と、そこで使われている評価指標（Accuracy と F1-score）の意味を解説します。

---

## 1. `compute_metrics` 関数全体のコード

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

この関数は、`Trainer` が検証用データ（Validation data）で評価を行う際、自動的に呼び出される**コールバック関数**です。引数 `pred` にはモデルの予測結果が入った `EvalPrediction` オブジェクトが渡されます。

---

## 2. 処理コードの1行ごとの解説

### ① `labels = pred.label_ids`
* **意味**: 検証データの **「実際の正解ラベル (1次元配列)」** を取り出します。
* **例**: `[0, 1, 3, 2, 0, 5, ...]`（データごとの正解クラスID）

### ② `preds = pred.predictions.argmax(-1)`
* **意味**: モデルが出力した **「生スコア (Logits)」** から、**「最もスコアが高いクラスID」** を取り出します。
* **仕組み**:
  * `pred.predictions` の中身は、各データに対する 6クラス分の数値リスト `[[2.1, -0.5, 0.3, ...], ...]`（2次元配列）です。
  * `argmax(-1)` を使うことで、一番最後の軸（各データ内の6クラス）の中で**最大値を持つインデックス（例: 0〜5）だけを抽出**し、1次元配列の予測結果に変換します。

### ③ `f1 = f1_score(labels, preds, average="weighted")`
* **意味**: 正解 `labels` と予測 `preds` から **F1スコア** を計算します。
* **`average="weighted"` の意味**: 6つの感情ごとにクラスのデータ件数が異なる（不均衡な）場合、**データ数に応じた重み付け平均** を計算して全体のF1スコアを出します。

### ④ `acc = accuracy_score(labels, preds)`
* **意味**: 単純な **正解率 (Accuracy)** を計算します。

### ⑤ `return {"accuracy": acc, "f1": f1}`
* **意味**: 評価指標の名称と計算結果を**辞書型**で返します。
* 戻り値のキー（`"accuracy"`, `"f1"`）は、`Trainer` のログ出力時に `eval_accuracy` や `eval_f1` として表示・記録されます。

---

## 3. 評価指標の意味：Accuracy と F1-score

### ① 正解率 (Accuracy)
* **計算式**: `正しく予測できた件数 ÷ 全体件数`
* **意味**: 全体のうち何%正解できたかという最もシンプルな指標です。
* **注意点**: データセットのクラス比率が偏っている（例: 「悲しみ」が 90% を占める）場合、全データに対して「悲しみ」とだけ答える適当なモデルでも正解率 90% が出てしまうため、これ単体ではモデルの実力を見誤るリスクがあります。

### ② F1-score (F1スコア)
* **意味**: **適合率 (Precision)** と **再現率 (Recall)** のバランス（調和平均）を表す指標です。
  * **適合率 (Precision)**: モデルが「喜び」と予測したデータのうち、**本当に喜びだった割合**（空振りの少なさ）。
  * **再現率 (Recall)**: 実際に「喜び」だったデータ全体のうち、**モデルが見逃さずに正解できた割合**（見逃しの少なさ）。
* **メリット**: クラスのデータ件数に偏りがあるデータセットにおいても、モデルの真の分類精度を正確に評価できます。一般的にnlp分類タスクでは F1-score を重視します。

---

## 4. まとめ

`compute_metrics` 関数は、**「モデルが出力した生スコア（Logits）を最高確率のクラスIDに変換し、正解ラベルと比較して Accuracy（単純正解率）と F1-score（データ偏りを考慮した総合精度）を計算して返す関数」** です。
