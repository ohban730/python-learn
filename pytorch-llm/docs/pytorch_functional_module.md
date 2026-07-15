# PyTorch: torch.nn.functional の役割と機能一覧

PyTorch にはニューラルネットワークを構築するために `torch.nn` と `torch.nn.functional`（通常 `F` としてインポートされる）の2つが用意されています。

これらの違い、および `torch.nn.functional` で利用できる代表的な機能について解説します。

---

## 1. `torch.nn` と `torch.nn.functional` の違い

最も大きな違いは、**「状態（学習可能なパラメータ）を持つかどうか」** です。

* **`torch.nn` (クラス形式 / オブジェクト指向)**
  * **特徴**: レイヤーのなかに「重み（Weights）」や「バイアス（Bias）」などの学習用パラメータを**状態として保持**します。
  * **使いどころ**: `nn.Linear` (全結合層) や `nn.Conv2d` (畳み込み層) など、ネットワークを構成する主要な学習レイヤー。通常、モデルの `__init__` 内でインスタンス化して登録します。
* **`torch.nn.functional` (関数形式 / Pure Functions)**
  * **特徴**: 状態を保持しません。入力を受け取って、計算結果を返すだけの単純な「関数」です。
  * **使いどころ**: 活性化関数（`relu`, `gelu`, `softmax`）や、パディング（`pad`）、プール（`max_pool2d`）など、**重みの更新が必要ない単なる計算処理**。通常、モデルの `forward` メソッド内で直接呼び出します。

---

## 2. `torch.nn.functional` の主要な機能分類

`F` には、ディープラーニングに必要なあらゆる計算関数が網羅されています。
（具体的な動作コードは [pytorch_functional_demo.py](../basics/pytorch_functional_demo.py) を参照してください）

### ① 活性化関数 (Activation Functions)
テンソルに対して非線形な変換を適用します。
* **`F.relu(input)`**: 最も広く使われる活性化関数。マイナスをすべて `0` に変換します。
* **`F.gelu(input)`**: GPTやBERTなどのTransformer（LLM）で標準的に使われる活性化関数。
* **`F.sigmoid(input)` / `F.tanh(input)`**: 古典的・標準的なS字カーブ関数。
* **`F.softmax(input, dim)`**: 数値のリストを「合計が1（100%）」の確率分布に変換します。

### ② 損失関数 (Loss Functions)
モデルの出力と正解データの誤差を計算します。
* **`F.cross_entropy(input, target)`**: 分類タスク用の標準損失関数。内部でSoftmaxの計算もまとめて処理してくれます。
* **`F.mse_loss(input, target)`**: 回帰タスク用（平均二乗誤差）。
* **`F.binary_cross_entropy(input, target)`**: 二値分類用（ロジスティック損失）。

### ③ テンソル変形・パディング・ユーティリティ
テンソル（行列・多次元配列）の形状を操作したり、特殊な変換を行います。
* **`F.one_hot(tensor, num_classes)`**: 整数IDのテンソルを、one-hotベクトルに変換します。
* **`F.pad(input, pad, mode, value)`**: テンソルの周囲を特定の数値（例: `0`）で埋める（パディング）処理。テキストの長さ調整や画像処理の境界線処理に必須です。
* **`F.interpolate(input, size, mode)`**: 画像テンソルなどの縦横サイズを、線形補間などを用いて拡大・縮小します。
* **`F.dropout(input, p, training)`**: 過学習を防ぐため、指定確率 `p` で要素をランダムに `0` にリセットします。

---

## 3. 使い分けの設計ルール

PyTorch でニューラルネットワーク（`nn.Module`）を自作する場合、通常は以下のように使い分けるのがベストプラクティスです。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 重みパラメータを保持するレイヤーは nn.XXX を使う
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # パラメータを持たない活性化やドロップアウトは F.xxx を使う
        x = self.fc1(x)
        x = F.gelu(x)       # Fを使う（状態がないため）
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc2(x)
        return x
```
