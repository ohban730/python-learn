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

### 4. GPT-2 入力埋め込みパイプライン
- **ファイル名**: [make-embedding.py](./make-embedding.py)
- **概要**: `tiktoken` トークナイザーと PyTorch の `Dataset`/`DataLoader` を組み合わせてミニバッチデータを作成し、トークン埋め込み（Token Embedding）と位置埋め込み（Positional Embedding）を合成する LLM の入力埋め込みパイプラインのサンプルコードです。

## ライセンスと出典 (License & Attribution)

本フォルダ内には、オープンソースコードをベースに学習・検証目的で改変したコードが含まれています。対象ファイルと出典・ライセンス情報は以下の通りです。

### Sebastian Raschka 氏 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 由来のコード
- **対象ファイル**: `regular-expression.py`, `make-vocab.py`, `Byte-Pair_Encoding.py`, `make-embedding.py`
- **ライセンス**: [Apache License 2.0](./LICENSE) （オリジナルコードの著作権は Sebastian Raschka 氏に帰属します）





