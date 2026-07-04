# Python & LLM 開発学習ロードマップ (Python Learn)

Pythonの基本仕様から、PyTorchを用いたLLM（大規模言語モデル）の基礎構築、さらに外部LLM API（Gemini API）の活用までを段階的に学習・検証するためのリポジトリです。

初めて学ぶ場合、または後から見返す場合は、以下の**学習ロードマップ順**に進めることを推奨します。

---

## 🗺️ 推奨学習ロードマップ (Roadmap)

### 📌 LLM学習の全体像
まず最初に、LLM開発のロードマップ全体と、自作学習スペースの構成を把握することをおすすめします。
*   **LLM自作学習スペース案内図**: [pytorch-llm/README.md](./pytorch-llm/README.md) 🌟**必読**
    *   **概要**: `pytorch-llm` ディレクトリ配下にある、LLMをスクラッチで実装するためのすべてのファイルと学習ステップの対応表です。
*   **LLM開発の3つのステージ**: [llm_development_stages.md](./pytorch-llm/llm_development_stages.md) 🌟**最初に見るのがおすすめ**
    *   **概要**: スクラッチからLLMを構築し、特定のアプリケーション（分類器やアシスタント）へファインチューニングするまでの「3つのステージ」と「アテンションの進化プロセス」を独自のMermaid図で整理したロードマップです。
*   **LLM重要用語の整理**: [llm_terminology.md](./pytorch-llm/llm_terminology.md)
    *   **概要**: アテンションとトランスフォーマーの違い、2つの「コンテキスト」の意味、エンコーダー/デコーダーの役割など、LLM学習に頻出する最重要ワードを整理した用語集です。

---

### STEP 1: Python の基本仕様とテキスト処理の基礎
LLM構築の事前準備として、Pythonの文法仕様や、テキストを前処理するための基礎知識を学びます。

*   **1-A. 例外処理とデバッグ**
    *   **コード**: [assert-and-raise.py](./assert-and-raise.py)
    *   **概要**: 前提条件チェックのための `assert` 文と、プロダクションコード用エラーハンドリングのための `raise` 文の使い分けについて学びます。
*   **1-B. 正規表現を用いたトークン化の基本**
    *   **コード**: [regular-expression.py](./regular-expression.py)
    *   **概要**: `re.split` などを活用し、テキストを単語や記号単位に分割する基本的なテキスト前処理手法を学びます。
*   **1-C. イテレータの挙動と手動操作**
    *   **コード**: [iter_and_next_demo.py](./iter_and_next_demo.py)
    *   **概要**: Pythonの `iter()` と `next()` 関数の役割や挙動、そして `for` ループの内部的な仕組みについて解説したデモプログラムです。

---

### STEP 2: LLM用トークナイザーとデータセット作成
モデルにデータを入力するための前処理パイプラインの構築について学びます。

*   **2-A. 語彙（Vocab）辞書の作成**
    *   **コード**: [make-vocab.py](./pytorch-llm/make-vocab.py) （参照: [sample.txt](./pytorch-llm/sample.txt)）
    *   **概要**: 抽出した単語リストから各単語にIDを割り当てる語彙辞書の作成と、Pythonでのイテレーション処理を学びます。
*   **2-B. サブワードトークン化 (BPE)**
    *   **コード**: [Byte-Pair_Encoding.py](./pytorch-llm/Byte-Pair_Encoding.py)
    *   **概要**: OpenAIの `tiktoken` ライブラリを使用し、未知語（OOV）に強いByte-Pair Encoding (BPE) によるエンコード・デコードの仕組みを学びます。
*   **2-C. データセットとデータローダーの作成**
    *   **解説ドキュメント**: [dataset_and_dataloader.md](./pytorch-llm/dataset_and_dataloader.md) 🌟**必読**
    *   **概要**: LLMへ入力するためのミニバッチを作成する「スライディングウィンドウ方式」について学びます。重要なパラメータである `max_length` (最大長) と `stride` (歩幅/移動幅) の違いを図解で解説しています。

---

### 💡 PyTorch基礎ワークフローの振り返り
本格的なLLMの埋め込みやアテンションに入る前に、PyTorchによる基本的な「データ準備 $\to$ モデル構築 $\to$ 訓練 $\to$ 評価 $\to$ 保存」の全体プロセスを復習します。

*   **PyTorch基礎の一気通貫デモ**
    *   **コード**: [pytorch_basics_demo.py](./pytorch-llm/pytorch_basics_demo.py) 🌟**必読**
    *   **概要**: `nn.Module` を用いた3層ニューラルネットの定義、自作DatasetとDataLoader、訓練ループの3ステップ（ゼログラ、逆伝播、パラメータ更新）、評価時の `no_grad` の意義、モデルの保存と再ロードまで、PyTorchの基礎を一括で学べるコメント付きデモコードです。
*   **Linear（全結合）層とバイアスの動作デモ**
    *   **コード**: [linear_basics_demo.py](./pytorch-llm/linear_basics_demo.py)
    *   **概要**: ニューラルネットワークの基本要素である `torch.nn.Linear` が内部に持つ重みとバイアス、およびその数式（y = x @ W.T + b）が手動の行列演算と完全に一致することを確認するデモプログラムです。

---

### STEP 3: 入力埋め込み (Input Embedding) パイプライン
モデルにトークンを渡す直前の「ベクトル表現」について学びます。

*   **3-A. nn.Embeddingの動作デモ**
    *   **コード**: [embedding_demo.py](./pytorch-llm/embedding_demo.py) 🌟**最初に見るのがおすすめ**
    *   **概要**: `torch.nn.Embedding` が内部でどのような数値テーブル（行列）を持ち、IDからベクトルへの変換（ルックアップ）がどのように行われているかを目で見て確認できる、極小サイズのサンプルコードです。
*   **3-B. PyTorch torch.arange の動作デモ**
    *   **コード**: [arange_demo.py](./pytorch-llm/arange_demo.py)
    *   **概要**: 位置インデックスを作成する際に多用される、PyTorchの連番生成関数 `torch.arange` の基本的な使い方を確認するデモプログラムです。
*   **3-C. 入力埋め込みの処理パイプライン**
    *   **コード**: [make-embedding.py](./pytorch-llm/make-embedding.py)
    *   **概要**: トークンIDを意味ベクトルに変換するトークン埋め込み（Token Embedding）と、位置情報を表す位置埋め込み（Positional Embedding）を PyTorch で合成する実装です。
*   **3-D. 埋め込みのメカニズム解説**
    *   **解説ドキュメント**: [embedding_mechanism.md](./pytorch-llm/embedding_mechanism.md) 🌟**必読**
    *   **概要**: 入力テキストがベクトルへ変換され、Transformerモデルへ入力されるまでのプロセスを独自のMermaid図を用いてステップバイステップで解説しています。

---

### STEP 4: Attentionメカニズムの基礎
LLMの心臓部であるAttention（注意機構）の仕組みと、その計算に必要な数学的処理を学びます。

*   **4-A. Attentionスコア計算の基礎とデモ**
    *   **コード**: [attention_basics_demo.py](./pytorch-llm/attention_basics_demo.py)
    *   **概要**: `torch.empty` と `enumerate` の動作確認から、ループを用いたドット積によるスコア計算、さらに効率的な「行列演算（ループなし）」への移行までを1つにまとめたデモコードです。
*   **4-B. ドット積と行列演算のビジュアル解説**
    *   **解説ドキュメント**: [attention_basics.md](./pytorch-llm/attention_basics.md) 🌟**必読**
    *   **概要**: なぜドット積が「単語の類似度」として機能するのか、そしてループ処理がどのように並列行列演算へ変換されるのかを図解した解説書です。
*   **4-C. ソフトマックス関数の挙動確認デモ**
    *   **コード**: [softmax_demo.py](./pytorch-llm/softmax_demo.py)
    *   **概要**: アテンションスコアが確率分布（重み）に変換されるプロセスを、コンソール上のテキストグラフで可視化して確認するデモプログラムです。
*   **4-D. ソフトマックス関数のグラフ特性と役割解説**
    *   **解説ドキュメント**: [softmax_basics.md](./pytorch-llm/softmax_basics.md) 🌟**必読**
    *   **概要**: 指数関数 $e^x$ が描くグラフの特性がどのように「注意の集中」に役立つのか、なぜ単純な割り算ではダメなのかを数学的・直感的に解説したドキュメントです。

---

### STEP 5: 外部LLM APIの活用 (Gemini API)
モデルを自作するだけでなく、最先端のクラウドLLMをAPIから活用する応用スキルを学びます。

*   **学習ディレクトリ**: [gemini-api/](./gemini-api/)
*   **概要**: Google GenAI SDK (`google-genai`) を使った Gemini API の実装例です。
    *   [count_tokens.py](./gemini-api/count_tokens.py): プロンプトの入力・出力・思考（Thoughts）プロセスのトークン数を計測・出力します。
    *   [streaming_response.py](./gemini-api/streaming_response.py): リアルタイムにAIの応答を表示するストリーミング処理について学びます。

---

## 📄 ライセンスと出典 (License & Attribution)

本リポジトリ内には、学習・検証目的で以下のオープンソースを参考に作成または改変したコードが含まれています。

### Sebastian Raschka 氏 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 由来のコード
*   **対象ファイル**: `regular-expression.py`, `pytorch-llm/make-vocab.py`, `pytorch-llm/Byte-Pair_Encoding.py`, `pytorch-llm/make-embedding.py`, `pytorch-llm/attention_basics_demo.py`, `pytorch-llm/softmax_demo.py`
*   **ライセンス**: [Apache License 2.0](./LICENSE) （オリジナルコードの著作権は Sebastian Raschka 氏に帰属します）





