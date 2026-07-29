# 誤差逆伝播（Backpropagation）はどこで定義・実行されているのか？

「モデルクラスの中で定義されているのか？それとも Trainer なのか？」という非常に鋭く重要な疑問に対する解説です。

---

## 1. 結論：役割分担されています！

* **自作モデルクラス (`XLMRobertaForTokenClassification`)**:
  **【順伝播 ＋ 誤差（Loss）の計算】** までを担当（定義）しています。
* **`trainer.train()`**:
  **【誤差逆伝播 (`loss.backward()`) ＋ 重みの更新 (`optimizer.step()`)】** を裏側で実行しています。

```text
 ［ 自作モデルクラス (forward) ］
   入力テキスト ──> 順伝播計算 ──> 予測結果 (logits) ──> 誤差 (Loss) の数値を算出！
                                                          │
                                                          ▼ (Loss を Trainer へ手渡す)
 ［ Trainer (trainer.train()) ］
   Loss を受領 ──> ★ loss.backward() 実行！ (誤差逆伝播で勾配計算)
                 ──> ★ optimizer.step() 実行！ (重みを更新)
```

---

## 2. 自作モデルクラスの中での定義（順伝播と Loss）

自作した `XLMRobertaForTokenClassification` の `forward` メソッドの中には、以下のコードが書かれていました。

```python
# 1. 順伝播で予測結果を出したあと、誤差を計算する定義
if labels is not None:
    loss_fct = nn.CrossEntropyLoss()
    # 予測(logits) と 正解(labels) を比較して誤差 Loss を計算！
    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

# 2. 計算した Loss を Trainer に渡すために返却する
return TokenClassifierOutput(loss=loss, logits=logits, ...)
```

ここでは、**「どうやって誤差（Loss）の大きさを計算するか」のルールを定義しているだけ**であり、まだ逆伝播（勾配の計算）は行われていません。

---

## 3. `trainer.train()` の内部で自動実行されている処理

`trainer.train()` を呼び出すと、Trainer の内部では PyTorch の標準的な学習ループ（以下と同等のコード）が自動実行されています。

```python
# 【 Trainer の内部で自動実行されているコードのイメージ 】
for batch in train_dataloader:
    # ① 自作モデルの forward() を呼び出して Loss を受け取る
    outputs = model(**batch)
    loss = outputs.loss

    # ② 誤差逆伝播の実行！ (勾配を計算)
    loss.backward()

    # ③ 重み（パラメータ）の更新！
    optimizer.step()
    optimizer.zero_grad()  # 勾配のリセット
```

---

## 4. まとめ

1. **自作モデルクラス**: 「入力から予測を出し、正解との**誤差（Loss）の数値を計算する関数**」
2. **`trainer.train()`**: 「モデルが計算した Loss を使って **`loss.backward()`（誤差逆伝播）を呼び出し、重みを書き換えていく実行役**」

このように、モデルクラスは「誤差の定義」に集中し、面倒な「逆伝播ループの実行」は Trainer がすべて裏側で引き受けてくれています。
