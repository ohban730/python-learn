# 混同行列 (Confusion Matrix) の仕組みと可視化プログラムの解説

書籍の48ページに登場する、モデルの誤予測傾向をグラフ化する **「混同行列（Confusion Matrix）」** のプログラムコードの意味と、行列の見方を詳しく解説します。

---

## 1. 混同行列 (Confusion Matrix) とは？

混同行列とは、分類モデルの予測結果について、**「実際の正解 (True)」** と **「モデルの予測 (Predicted)」** をグリッド状の表（行列）にして、**「モデルがどのクラスとどのクラスを勘違いして間違えやすいか」** を視覚的に把握するための集計表です。

```text
               【 モデルの予測 (Predicted Label) 】
                 sadness   joy   anger ...
 ［ 正  sadness [  0.71    0.11   0.10  ]  <-- 本当は sadness なのに
 　 解  joy     [  0.09    0.80   0.03  ]      10% を anger と勘違い！
 　 ラ  anger   [  0.29    0.14   0.44  ]
 　 ベル ］
```

### 行列の見方の法則
* **縦軸 (Y軸 / 各行)**: **実際の正解 (True label)**
* **横軸 (X軸 / 各列)**: **モデルが予測したクラス (Predicted label)**
* **左上から右下への対角線 (対角成分)**: **正解した割合（正解率）**。青色が濃く数字が大きいほど、正しく分類できています。
* **対角線以外の場所**: **モデルが間違えて勘違いした箇所**。
  * 例: 書籍のグラフで `anger`(行) と `sadness`(列) が交差する箇所の数値が `0.29` ──> 「本当は『怒り』なのに、モデルが『悲しみ』だと勘違いしたデータが 29% もある」という意味になります。

---

## 2. プログラムコードの1行ずつ解説

書籍48ページのコードを分解して解説します。

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    # ① 混同行列の数値を集計・計算する
    cm = confusion_matrix(y_true, y_preds, normalize="true")

    # ② グラフを表示する描画キャンバス（枠組み）を作る
    fig, ax = plt.subplots(figsize=(6, 6))

    # ③ 集計データ cm と文字ラベル labels を描画用オブジェクトにセットする
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # ④ グラフの見た目を装飾して描画する
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)

    plt.title("Normalized confusion matrix")
    plt.show()
```

### 各行の詳しい役割

1. **`cm = confusion_matrix(y_true, y_preds, normalize="true")`**
   * 正解 `y_true` とモデルの予測 `y_preds` を比較して、行列データを計算します。
   * **`normalize="true"` の意味**: 単なる件数（「500件」など）ではなく、正解クラス（行）ごとの**「割合（0.0 〜 1.0）」** に正規化します。データ件数が不均衡なデータセットでも、各感情をどれくらいの確率で正解・誤認したかが一目でわかるようになります。
2. **`fig, ax = plt.subplots(figsize=(6, 6))`**
   * matplotlib を使って、縦6インチ × 横6インチの正方形のグラフ表示領域（`ax`）を準備します。
3. **`disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)`**
   * scikit-learn が用意している可視化専用のクラスです。
   * 計算した行列データ `cm` に、文字ラベル（`["sadness", "joy", ...]`）を対応付けます。
4. **`disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)`**
   * **`cmap="Blues"`**: 数値が大きい場所ほど「濃い青色」で塗りつぶすカラーマップを指定。
   * **`values_format=".2f"`**: マスの中に表示する数値を「小数点以下2桁の浮動小数点数（例: `0.71`）」にフォーマット指定。
   * **`colorbar=False`**: グラフ右側のカラーバー凡例を非表示にしてスッキリさせます。

---

## 3. なぜ `normalize="true"`（正規化）を行うのか？

データセット内で「悲しみ」が5000件、「驚き」が500件といった不均衡なデータの場合、件数のまま表示すると、全体数の多い「悲しみ」のマスの数字ばかりが大きくなってしまい、モデルが本当に優秀なのか判断しづらくなります。

`normalize="true"` を設定すると、各行の横合計が **1.0 (100%)** に変換されるため、「『驚き』というレアな感情のうち、何%を正解できたか」が公平に評価できるようになります。
