# Python Learn

Pythonの基本構文や便利なライブラリの利用方法を学習・検証するためのサンプルコード集です。

## サンプルコード一覧

### 1. 正規表現によるテキストの分割・トークン化
- **ファイル名**: [regular-expression.py](./regular-expression.py)
- **概要**: `re.split` や `re.sub` を使用して、テキストの分割（トークン化）や記号周りのスペース整形を行うサンプルコードです。区切り文字となった記号も保持するように丸かっこ `()` でキャプチャグループ化されています。

### 2. 語彙（Vocab）辞書の作成とイテレーション
- **ファイル名**: [make-vocab.py](./make-vocab.py) （参照ファイル: [sample.txt](./sample.txt)）
- **概要**: 抽出した単語リストから各単語にIDを割り当てる語彙辞書を作成するサンプルコードです。`enumerate()` 関数や辞書の `.items()` メソッドの挙動を振り返るための解説コメントが含まれています。

### 3. Byte-Pair Encoding (BPE) と tiktoken によるトークン化
- **ファイル名**: [Byte-Pair_Encoding.py](./Byte-Pair_Encoding.py)
- **概要**: OpenAI の `tiktoken` ライブラリを使用し、Byte-Pair Encoding (BPE) アルゴリズムによるテキストのエンコード・デコードを行うサンプルコードです。辞書外の未知語（OOV）がサブワードへ分解される挙動や、特殊トークン（`<|endoftext|>` など）の扱いを確認できます。


