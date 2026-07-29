# Python & LLM 開発学習ロードマップ (python-learn)

Pythonの基本仕様から、PyTorchを用いたLLM（大規模言語モデル）のスクラッチ構築、さらにHugging Face / Transformersライブラリを活用した自然言語処理の実践応用までを段階的に学習・検証するための統合ワークスペースです。

---

## ⚙️ 開発・実行環境およびプロジェクト運用ルール

本リポジトリでのコード実行および資材の管理は以下の統一ルールに従って行います。

* **仮想環境 (Python Virtual Environment)**:
  * 実行環境: `llm-sandbox`
  * パス: `C:\Users\owner\miniconda3\envs\llm-sandbox`
* **質問回答・解説ノート保管先 (Q&A & Docs)**:
  * フォルダ: [pytorch-llm/docs/](./pytorch-llm/docs/) 🌟 **110件以上の解説ノート・数理解説を集約**
  * インデックス: [docs/README.md](./pytorch-llm/docs/README.md)
* **サンプルプログラムの配置規則 (Code Directory Structure)**:
  * **PyTorch基礎・単機能検証**: [pytorch-llm/basics/](./pytorch-llm/basics/) （テンソル操作、Linear層、LayerNorm、活性化関数、勾配降下法など）
  * **LLMスクラッチ自作実装**: [pytorch-llm/src/](./pytorch-llm/src/) （語彙辞書作成、BPEトークナイザー、埋め込み、Attention等）
  * **Transformers実践・応用**: [transformers-book/](./transformers-book/) （『機械学習エンジニアのためのTransformers』の各章別分類）
* **📄 著作権・ライセンスおよび出典表記方針 (Copyright & Compliance)**:
  * 本リポジトリ内のコード・ドキュメントは、個人学習・研究・検証を目的として作成されています。
  * 書籍・公式サンプルコード（オープンソース等）を参考にしたプログラムには、ファイルヘッダーおよび該当READMEに出典（書籍名・著者・出版社・公式GitHubリポジトリURL）および元のライセンス（MIT, Apache 2.0等）を明記し、著作権に配慮した管理を行います。

---

## 🗺️ 2大学習証跡ロードマップ (Learning Roadmaps)

本リポジトリには、2冊の主要書籍に基づく実践学習の証跡が整理されています。

```text
python-learn/
├── pytorch-llm/                # 🛠️ 『つくりながら学ぶ！LLM自作入門』学習スペース
│   ├── basics/                 # 💡 PyTorch基本・前提数理デモコード
│   ├── src/                    # 🚀 LLM構築フロー順の本番ソースコード
│   └── docs/                   # 📚 質問回答・ビジュアル解説ノート & 用語集 (全110件超)
│       └── README.md           # 🌟 docs全件カテゴリ別インデックス
│
├── transformers-book/          # 🤗 『機械学習エンジニアのためのTransformers』学習スペース
│   ├── preface/                # まえがき (Conv2d, PyTorch vs TF, Scikit-learn, Keras)
│   ├── chap01/                 # 1章: Transformersの紹介 & RNN/LSTM/GRU
│   ├── chap02/                 # 2章: テキスト分類
│   ├── chap03/                 # 3章: Transformerの解剖学
│   └── chap04/                 # 4章: 多言語固有表現抽出 (NER)
│
└── import-basics / gemini-api  # Python基礎文法 & API連携デモ
```

---

## 📖 主な案内図と重要ドキュメント

### 1. 『LLM自作入門』学習案内
* 🌟 **LLMスクラッチ自作メイン案内マップ**: [pytorch-llm/README.md](./pytorch-llm/README.md)
* 🌟 **LLM開発の3つのステージとAttention進化**: [llm_development_stages.md](./pytorch-llm/docs/llm_development_stages.md)
* 🌟 **解説ノート全件テーマ別目次**: [docs/README.md](./pytorch-llm/docs/README.md)
* 📖 **基礎知識三種の神器**:
  1. [LLM一般概念・AI定義編](./pytorch-llm/docs/llm_terminology.md)
  2. [PyTorch基礎・Linear数理編](./pytorch-llm/docs/pytorch_basics.md)
  3. [テンソル変形・メモリ最適化編](./pytorch-llm/docs/pytorch_tensor_operations.md)

### 2. 『機械学習エンジニアのためのTransformers』学習案内
* 🌟 **Transformers本学習ノート・章別インデックス**: [transformers-book/README.md](./transformers-book/README.md)
* 📘 **章別進捗と学習トピック**:
  * **まえがき**: Conv2d基礎, PyTorch vs TensorFlow, Hugging Face関係性, sklearn/Keras基礎
  * **1章**: RNN/tanh/LSTM/GRU基礎, 活性化関数比較, Transformerとの違い
  * **2章〜4章**: Hugging Face datasets/trainer/evaluate を用いたテキスト分類、多言語NER (PAN-X) など

---

## 📌 STEP別学習フロー

### STEP 1: Python 基本仕様 & テキスト前処理
* **例外処理**: [assert-and-raise.py](./assert-and-raise.py)
* **正規表現トークン化**: [regular-expression.py](./regular-expression.py)
* **イテレータ動作**: [iter_and_next_demo.py](./iter_and_next_demo.py)
* **動的インポート & sys.path 解決**: [import-basics/import_demo.py](./import-basics/import_demo.py) ([解説](./import-basics/README.md))

### STEP 2: PyTorch 基礎 & 数理シミュレーション
* **PyTorch一気通貫デモ**: [pytorch_basics_demo.py](./pytorch-llm/basics/pytorch_basics_demo.py)
* **Linear層とバイアス数理**: [linear_basics_demo.py](./pytorch-llm/basics/linear_basics_demo.py)
* **勾配降下法・等高線可視化**: [gradient_descent_demo.py](./pytorch-llm/basics/gradient_descent_demo.py) ([解説](./pytorch-llm/docs/gradient_descent.md))
* **LayerNorm / BatchNorm 比較**: [batch_vs_layer_norm_demo.py](./pytorch-llm/basics/batch_vs_layer_norm_demo.py) ([解説](./pytorch-llm/docs/batch_vs_layer_normalization.md))
* **GELU / SwiGLU 比較**: [gelu_swiglu_demo.py](./pytorch-llm/basics/gelu_swiglu_demo.py) ([解説](./pytorch-llm/docs/gelu_swiglu_normal_distribution.md))

### STEP 3: トークナイザー & LLMスクラッチ構築
* **語彙辞書作成**: [make-vocab.py](./pytorch-llm/src/make-vocab.py)
* **BPEトークン化**: [Byte-Pair_Encoding.py](./pytorch-llm/src/Byte-Pair_Encoding.py)
* **入力埋め込み (Token + Positional)**: [make-embedding.py](./pytorch-llm/src/make-embedding.py)
* **Attention基礎デモ**: [attention_basics_demo.py](./pytorch-llm/src/attention_basics_demo.py) ([解説](./pytorch-llm/docs/attention_basics.md))

### STEP 4: Hugging Face Ecosystem & トランスフォーマー応用
* **テキスト分類 Pipeline デモ**: [huggingface_pipeline_demo.py](./pytorch-llm/basics/huggingface_pipeline_demo.py)
* **BLEUスコア & Downstream Task 解説**: [bleu_metric_and_downstream_task.md](./pytorch-llm/docs/bleu_metric_and_downstream_task.md)
* **Transformers各章読書ノート**: [transformers-book/README.md](./transformers-book/README.md)
