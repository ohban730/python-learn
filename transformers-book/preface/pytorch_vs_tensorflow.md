# PyTorch vs TensorFlow 対応ガイド

## 💡 TensorFlow とは何か？

**TensorFlow（テンソルフロー）** は、Googleが開発したオープンソースのディープラーニング（深層学習）フレームワークです。PyTorchと並び、世界中で最も広く使われています。

### 1. 歴史と進化（使いやすさの劇的向上）
*   **TensorFlow 1.x (昔)**:
    *   計算グラフ（数式）をあらかじめ定義した後にデータを流す「静的グラフ（Define-and-Run）」という設計でした。これは書いたコードがすぐ動かないためデバッグが非常に難しく、初心者にはかなり難解でした。
*   **TensorFlow 2.x (現在 / 本デモで採用)**:
    *   初心者にも扱いやすい直感的な高レベルAPIである **`Keras (ケラス)`** を標準統合しました。
    *   PyTorchと同じく、実行しながらその場で計算を定義する「動的グラフ（Define-by-Run / Eager Execution）」へと生まれ変わり、PyTorchとほぼ同じ感覚で簡単にコードを書いてデバッグできるようになりました。

### 2. PyTorch との主な違い・使い分け
*   **PyTorch (研究・論文に強い)**:
    *   コードが純粋なPythonらしく直感的で、カスタマイズが容易であるため、現在AIの**研究開発や公開される論文コードの大部分はPyTorch**で書かれています。
*   **TensorFlow (実用・本番導入に強い)**:
    *   スマートフォンやIoT機器向けの動作軽量化（TensorFlow Lite）や、ブラウザでの実行（TensorFlow.js）、大規模分散処理など、**「実サービスへの組み込みや本番運用」**に関する周辺ツール（エコシステム）が非常に強力です。

---

## 2. 概念を1対1で対応させて理解する

現在の TensorFlow 2.x（Keras）はPyTorchと設計思想が非常に似ているため、**概念を1対1で対応させて理解することが十分に可能です。**

ここでは、`appendix-A/main.py` のPyTorchプログラムと全く同じ動作をするTensorFlowプログラムの書き方を対比させて解説します。

---

## 1. 概念・コードの対応表

| 機能 | PyTorch (`torch`) | TensorFlow / Keras (`tf`) |
|---|---|---|
| **テンソルの作成** | `torch.tensor([[1.0, 2.0]])` | `tf.constant([[1.0, 2.0]])` |
| **モデルの定義** | `class NeuralNetwork(torch.nn.Module):`<br>直下で `forward(self, x)` を定義 | `class NeuralNetwork(tf.keras.Model):`<br>直下で `call(self, x)` を定義 |
| **全結合層** | `nn.Linear(in_features=2, out_features=30)` | `layers.Dense(units=30)`<br>(※入力次元は自動推論されるため指定不要) |
| **モデルの直列定義** | `nn.Sequential(...)` | `tf.keras.Sequential(...)` |
| **データセット管理** | 自作 `Dataset` + `DataLoader` | `tf.data.Dataset.from_tensor_slices` |
| **最適化（Optimizer）**| `torch.optim.SGD(model.parameters(), lr)`| `tf.keras.optimizers.SGD(learning_rate)` |
| **損失関数** | `F.cross_entropy(logits, labels)` | `tf.keras.losses.SparseCategoricalCrossentropy(`<br>`from_logits=True)` |
| **訓練中の自動微分** | `loss.backward()` + `optimizer.step()` | `with tf.GradientTape() as tape:`<br>内で順伝播を行い、`tape.gradient` で更新 |
| **重みの保存・読込** | `torch.save(model.state_dict(), path)` | `model.save_weights(path)` |

---

## 2. 実装スタイル（2つの方法）

TensorFlowには、大きく分けて2つの書き方があります。

1.  **高レベルAPI (Keras / `model.fit()` スタイル)**
    *   `model.compile()` で最適化手法や損失関数を設定し、`model.fit()` の1行で訓練を回す。極めてシンプル。
2.  **低レベルAPI (カスタム訓練ループ / `GradientTape` スタイル)**
    *   PyTorchの訓練ループ（`loss.backward()` など）と全く同じように、手動でバッチを取り出して勾配を計算し、パラメータを更新する。

`appendix-A/main.py` に対応させるため、以下の `tensorflow_demo.py` では **2の低レベルAPI（カスタム訓練ループ）** スタイルを採用し、PyTorchの訓練ループを完全に1対1で再現しています。
