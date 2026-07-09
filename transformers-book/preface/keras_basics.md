# Keras の基礎知識と使いどころ

**Keras（ケラス）** は、Pythonで書かれたディープラーニング用の高レベルAPI（ライブラリ）です。

その最大の設計思想は **「人間第一（User-friendliness）」** であり、開発者が数学やコードの細かい実装（ループ処理や勾配の計算など）に惑わされず、**「モデルの構造設計」だけに集中できる**ように作られています。

---

## 1. Keras はどんなときに使われるのか？

### ① プロトタイプ（試作品）を爆速で作る
アイデアを思いついてから、実際にモデルを作成して学習を走らせるまでの時間を極限まで短縮できます。
PyTorchでは数十行にわたる「訓練ループ」を手書きする必要がありますが、Kerasなら **「モデルを積み木のように重ねて、`.fit()` を呼ぶだけ」** で学習が始まります。

### ② ディープラーニングの学習コストを下げる
「ニューラルネットワークの基礎」を学ぶ際、オプティマイザの初期化や逆伝播の仕組みを手書きするのは難易度が高いです。Kerasはそうした定型的な処理をすべて隠蔽してくれるため、初心者でも扱いやすいのが特徴です。

### ⚙️ 計算エンジン（バックエンド）との関係
Kerasは、自身で低レベルの数学計算（行列の掛け算など）を行うわけではありません。裏側で動くエンジン（**バックエンド**）として、**TensorFlow, PyTorch, JAX** のいずれかを呼び出して動作します。
（※最新の Keras 3 では、PyTorchをバックエンドに指定してKerasのコードを動かすことも可能になりました。）

---

## 2. Keras の代名詞：`model.fit()` スタイル

Kerasを使ったモデル構築と学習は、常に以下の **3ステップ** で完結します。

### Step 1: モデルの定義（積み木を重ねる）
```python
model = keras.Sequential([
    layers.Dense(30, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(2)
])
```

### Step 2: コンパイル（学習ルールの設定）
「どの最適化手法（Optimizer）を使い、どの損失関数（Loss）で学習するか」を登録します。
```python
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'] # ついでに正解率も自動で計算させる設定
)
```

### Step 3: フィット（学習の実行）
これだけで、バッチ処理やエポックのループ、勾配の計算、パラメータ更新がすべて裏で自動実行されます。
```python
model.fit(X_train, y_train, epochs=3, batch_size=2)
```

---

## 3. PyTorchの訓練ループとの比較

同じ「3エポック学習させる」処理におけるコード量の違いです。

*   **PyTorch**: `zero_grad()`, `backward()`, `step()`, エポックのforループ, バッチのforループをすべて手書き（約15〜20行）。
*   **Keras**: `model.fit(..., epochs=3, batch_size=2)` の**1行のみ**。
