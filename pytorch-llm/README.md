# PyTorchによるLLMスクラッチ自作 (pytorch-llm)

本ディレクトリは、書籍『つくりながら学ぶ！LLM自作入門』の設計思想に基づき、PyTorchを用いて大規模言語モデル（LLM）を一から手書きで実装・学習するためのメインワークスペースです。

---

## 📁 フォルダ構成マップ (Directory Structure)

GitHub上でどこに何があるかを把握するための視覚的ツリー図です。

```text
pytorch-llm/              # 🛠️ LLMスクラッチ自作メインスペース
│
├── basics/               # 💡 PyTorch基本・前提知識デモコード (単機能で学ぶ)
│   ├── pytorch_basics_demo.py      # -> データから訓練・ロードまでの一連の基礎
│   ├── linear_basics_demo.py       # -> Linear層とバイアスの数理実証
│   ├── layernorm_scale_shift_demo.py # -> LayerNormのscale/shiftの動作検証
│   ├── gelu_swiglu_demo.py         # -> GELU近似検証とSwiGLU実装および可視化
│   ├── feed_forward_demo.py        # -> FFNのトークン独立性とMHAとの挙動対比
│   ├── ffn_key_value_demo.py       # -> FFNのKey-Valueメモリ動作シミュレータ
│   ├── skip_connection_demo.py     # -> 20層モデルでの勾配流・勾配消失検証
│   ├── batch_vs_layer_norm_demo.py # -> BatchNormとLayerNormの動作挙動対比
│   ├── variance_compatibility_demo.py # -> 有偏・不偏分散の差と誤差蓄積の検証
│   ├── activation_sequential_demo.py # -> ReLU、keepdim、Sequentialの検証
│   ├── keepdim_visual_demo.py      # -> keepdim(False/True)形状の視覚的デモ
│   ├── gradient_descent_demo.py    # -> 勾配降下法の等高線探索ビジュアルデモ
│   ├── odds_ratio_demo.py          # -> オッズ、オッズ比、ロジット逆変換の実証
│   ├── embedding_demo.py           # -> Embeddingのルックアップの仕組み
│   ├── arange_demo.py              # -> 位置エンコーディング用の連番生成
│   ├── softmax_demo.py             # -> ソフトマックスの数値的安定性と挙動
│   ├── view_basics_demo.py         # -> viewとメモリ上のデータ解釈
│   └── contiguous_basics_demo.py   # -> メモリ連続性とおまじないの正体
│
├── src/                  # 🚀 LLM構築フロー順の本番ソースコード (Step 1〜4)
│   ├── make-vocab.py               # -> [Step 1] テキストからの語彙（辞書）作成
│   ├── Byte-Pair_Encoding.py       # -> [Step 2] BPEトークン化とデータローダー
│   ├── make-embedding.py           # -> [Step 3] トークンと位置の埋め込み合成
│   ├── attention_basics_demo.py    # -> [Step 4] 重みパラメータ付きアテンション
│   └── sample.txt                  # -> トークン化テスト用の入力テキスト
│
└── docs/                 # 📚 ビジュアル解説ドキュメント & 用語集
    ├── llm_development_stages.md    # -> LLMの3ステージとAttentionロードマップ
    ├── gpt_architecture.md         # -> GPTモデル全体構成とデータフロー図解
    ├── layernorm_scale_shift.md    # -> LayerNormのscaleとshiftの役割解説
    ├── gelu_swiglu_normal_distribution.md # -> GELU/SwiGLUの数理とガウス分布の役割
    ├── feed_forward_network.md     # -> FFNの役割とアテンションとの違い
    ├── ffn_knowledge_processing.md # -> FFNのKey-Valueメモリとしての知識処理メカニズム
    ├── skip_connection_history.md  # -> スキップ接続の歴史と勾配流の数理
    ├── batch_vs_layer_normalization.md # -> バッチ正規化と層正規化の仕組み・違い
    ├── variance_and_compatibility.md # -> 有偏分散と公式モデルとのロード互換性
    ├── keepdim_details.md          # -> keepdim形状と自動拡張のテキスト図解
    ├── activation_and_sequential.md# -> 活性化関数、keepdim、Sequentialの解説
    ├── gradient_descent.md         # -> 勾配降下法・等高線探索の数理と可視化
    ├── odds_ratio.md               # -> オッズ, ロジット、シグモイドの接続ストーリー
    ├── llm_terminology.md          # -> 一般概念編 (アテンションの違い、ファインチューニング等)
    ├── pytorch_basics.md           # -> PyTorch基礎編 (Parameter、Module、Linear数理等)
    ├── pytorch_tensor_operations.md# -> テンソル編 (Rank、tril、stack/cat、Storage、contiguous)
    ├── attention_basics.md         # -> Attention数理（加重平均、スケーリング、マスク等）
    ├── dataset_and_dataloader.md   # -> データローダー（max_length, stride等）
    ├── embedding_mechanism.md      # -> 埋め込み（Token + Positional等）
    ├── softmax_basics.md           # -> Softmax（次元指定、オーバーフロー対策等）
    ├── final_normalization_and_output_head.md # -> 最後の層正規化と線形出力層の役割解説
    ├── weight_tying_basics.md      # -> 重み共有 (Weight Tying) のメカニズム解説
    └── generation_loop_stopping.md # -> テキスト生成ループの停止メカニズム解説
```

---

## 🗺️ LLM構築の学習ロードマップ

学習と実装は、以下のライフサイクルに従って進めていきます。
開発ステージ全体の解説と詳細なダイアグラムは、[llm_development_stages.md](./docs/llm_development_stages.md) を参照してください。

```mermaid
graph TD
    Data["1) データの準備と前処理"] --> Token["2-B) BPEトークン化"]
    Token --> Embed["3) 入力埋め込み (Token + Positional)"]
    Embed --> Attn["4) Attentionメカニズム"]
    Attn --> Arch["5) LLMアーキテクチャの構築"]
```

---

## 📂 各ステップと学習資材の対応表

| 学習フェーズ / ステップ | 実行コード (.py) | 解説ドキュメント (.md) | 学べる内容 |
| :--- | :--- | :--- | :--- |
| **PyTorchの前提知識** | [pytorch_basics_demo.py](./basics/pytorch_basics_demo.py)<br>[linear_basics_demo.py](./basics/linear_basics_demo.py)<br>[activation_sequential_demo.py](./basics/activation_sequential_demo.py)<br>[batch_vs_layer_norm_demo.py](./basics/batch_vs_layer_norm_demo.py) | [pytorch_basics.md](./docs/pytorch_basics.md)<br>[activation_and_sequential.md](./docs/activation_and_sequential.md)<br>[batch_vs_layer_normalization.md](./docs/batch_vs_layer_normalization.md) | `nn.Module`モデルの定義、Linear層の挙動、活性化関数（ReLU）、統計量計算時の `keepdim` の役割、および「バッチ正規化（BatchNorm）」と「層正規化（LayerNorm）」の計算方向の違いとLLMでの選択理由 |
| **数理の前提知識** | [odds_ratio_demo.py](./basics/odds_ratio_demo.py)<br>[gradient_descent_demo.py](./basics/gradient_descent_demo.py)<br>[variance_compatibility_demo.py](./basics/variance_compatibility_demo.py)<br>[gelu_swiglu_demo.py](./basics/gelu_swiglu_demo.py)<br>[feed_forward_demo.py](./basics/feed_forward_demo.py)<br>[ffn_key_value_demo.py](./basics/ffn_key_value_demo.py)<br>[skip_connection_demo.py](./basics/skip_connection_demo.py) | [odds_ratio.md](./docs/odds_ratio.md)<br>[gradient_descent.md](./docs/gradient_descent.md)<br>[variance_and_compatibility.md](./docs/variance_and_compatibility.md)<br>[gelu_swiglu_normal_distribution.md](./docs/gelu_swiglu_normal_distribution.md)<br>[feed_forward_network.md](./docs/feed_forward_network.md)<br>[ffn_knowledge_processing.md](./docs/ffn_knowledge_processing.md)<br>[skip_connection_history.md](./docs/skip_connection_history.md) | 「オッズ比」、損失地形のパラメータ探索軌跡、有偏・不偏分散とパラメータロード互換性、GELU/SwiGLU活性化曲線の滑らかさ、アテンション（横のブレンド）とフィードフォワード（縦の個別処理）のトークン独立性、FFNのKey-Valueメモリ動作、およびResNetの歴史的劣化問題と微分式「+1」が敷く勾配流高速道路の数理 |
| **テンソル・メモリの前提知識** | [view_basics_demo.py](./basics/view_basics_demo.py)<br>[contiguous_basics_demo.py](./basics/contiguous_basics_demo.py)<br>[keepdim_visual_demo.py](./basics/keepdim_visual_demo.py) | [pytorch_tensor_operations.md](./docs/pytorch_tensor_operations.md)<br>[keepdim_details.md](./docs/keepdim_details.md) | `view`によるメモリ空間データ解釈（Storage）、転置後のメモリ非連続性と `contiguous()` の本質、および `keepdim=True/False` の形状の違いとブロードキャスト（自動拡張）の仕組み |
| **Step 1: データの前処理** | [make-vocab.py](./src/make-vocab.py) | [llm_terminology.md](./docs/llm_terminology.md) | テキストから語彙（辞書）を作成する基本的な前処理と、事前学習・ファインチューニングなどのLLM一般用語 |
| **Step 2: トークン化(BPE)** | [Byte-Pair_Encoding.py](./src/Byte-Pair_Encoding.py) | [dataset_and_dataloader.md](./docs/dataset_and_dataloader.md) | サブワード分割（BPE）と、スライディングウィンドウ方式によるデータローダー作成（`max_length`, `stride` の役割） |
| **Step 3: 入力埋め込み** | [make-embedding.py](./src/make-embedding.py) <br>[embedding_demo.py](./basics/embedding_demo.py)<br>[arange_demo.py](./basics/arange_demo.py) | [embedding_mechanism.md](./docs/embedding_mechanism.md) | トークン埋め込み（`nn.Embedding`）と位置埋め込み（`torch.arange`）を足し合わせて入力テンソルを作るプロセス |
| **Step 4: Attentionの基礎** | [attention_basics_demo.py](./src/attention_basics_demo.py)<br>[softmax_demo.py](./basics/softmax_demo.py) | [attention_basics.md](./docs/attention_basics.md)<br>[softmax_basics.md](./docs/softmax_basics.md) | アテンションスコアのドット積と行列積（`inputs @ query`）の対比、Softmaxの性質（オーバーフロー/アンダーフロー、`dim`引数の意味） |
| **Step 5: GPTモデルの構築** | - | [gpt_architecture.md](./docs/gpt_architecture.md)<br>[final_normalization_and_output_head.md](./docs/final_normalization_and_output_head.md)<br>[weight_tying_basics.md](./docs/weight_tying_basics.md)<br>[generation_loop_stopping.md](./docs/generation_loop_stopping.md) | GPTモデルの全体設計・実装ロードマップ（プレースホルダ設計）から、入出力テンソルの全体予測フロー、ブロック内部の計算パス、最終出口のLayerNormと線形出力層、「重み共有」、およびEOSトークン・最大長・Stop Sequencesによる生成ループの停止メカニズムの可視化 |

---

## 📄 ライセンスと出典 (Attribution)

本フォルダ内のコードは、学習目的で以下のオープンソースをベースに作成・改変したものです。
*   **オリジナル著者**: Sebastian Raschka 氏
*   **参考リポジトリ**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache License 2.0)
