# Python & LLM 開発学習ロードマップ (Python Learn)

Pythonの基本仕様から、PyTorchを用いたLLM（大規模言語モデル）の基礎構築、さらに外部LLM API（Gemini API）の活用までを段階的に学習・検証するためのリポジトリです。

初めて学ぶ場合、または後から見返す場合は、以下の**学習ロードマップ順**に進めることを推奨します。

---

## 🗺️ 推奨学習ロードマップ (Roadmap)

### 📌 LLM学習の全体像
まず最初に、LLM開発のロードマップ全体と、自作学習スペースの構成を把握することをおすすめします。
*   **LLM自作学習スペース案内図**: [pytorch-llm/README.md](./pytorch-llm/README.md) 🌟**必読 (GitHub上で最も見やすいマップです)**
*   **LLM開発の3つのステージ**: [llm_development_stages.md](./pytorch-llm/docs/llm_development_stages.md) 🌟**最初に見るのがおすすめ**
    *   **概要**: スクラッチからLLMを構築し、特定のアプリケーション（分類器やアシスタント）へファインチューニングするまでの「3つのステージ」と「アテンションの進化プロセス」を独自のMermaid図で整理したロードマップです。
*   **LLM用語・PyTorch仕様の体系的整理**:
    理解を助け、コードの「なぜ」を解消するための3つの専用ガイドです。
    *   **LLM一般概念・AI定義編**: [llm_terminology.md](./pytorch-llm/docs/llm_terminology.md)
        *   アテンションとトランスフォーマーの違い、コンテキスト、エンコーダー/デコーダーの歴史、モデルの表現力など、純粋な「AI概念・理論」の用語集。
    *   **PyTorch基礎・Linear数理編**: [pytorch_basics.md](./pytorch-llm/docs/pytorch_basics.md)
        *   `requires_grad`、`nn.Module` / `forward` の作法、Linear（全結合）層のバイアス（切片）や重みの転置状態での保存仕様、重みの初期化など、PyTorchの「基本クラスと学習メカニズム」の整理。
    *   **テンソル変形・メモリ最適化編**: [pytorch_tensor_operations.md](./pytorch-llm/docs/pytorch_tensor_operations.md)
        *   テンソルのRankとベクトルの次元の違い、三角行列マスク、`stack` / `cat` の違い、`view` と `Storage` のメモリ実体、`contiguous()` による一列再配置、インプレース演算、Multi-Head Attentionの形状変化マップなど、「テンソル変形とメモリハック」の整理。

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
*   **1-D. 動的インポートとモジュール検索パス (sys.path) の仕組み**
    *   **コード**: [import_demo.py](./import-basics/import_demo.py)
    *   **解説ドキュメント**: [README.md](./import-basics/README.md) 🌟**必読**
    *   **概要**: Pythonが import の対象を探し出す `sys.path` リストの仕組みを理解し、フォルダ階層が異なるスクリプトを呼び出す際に `Path(__file__).resolve().parents[N]` と `sys.path.insert(0, ...)` を用いて検索パスを動的に追加・解決するメカニズムを学ぶデモプログラムです。

---

### STEP 2: LLM用トークナイザーとデータセット作成
モデルにデータを入力するための前処理パイプラインの構築について学びます。

*   **2-A. 語彙（Vocab）辞書の作成**
    *   **コード**: [make-vocab.py](./pytorch-llm/src/make-vocab.py) （参照: [sample.txt](./pytorch-llm/src/sample.txt)）
    *   **概要**: 抽出した単語リストから各単語にIDを割り当てる語彙辞書の作成と、Pythonでのイテレーション処理を学びます。
*   **2-B. サブワードトークン化 (BPE)**
    *   **コード**: [Byte-Pair_Encoding.py](./pytorch-llm/src/Byte-Pair_Encoding.py)
    *   **概要**: OpenAIの `tiktoken` ライブラリを使用し、未知語（OOV）に強いByte-Pair Encoding (BPE) によるエンコード・デコードの仕組みを学びます。
*   **2-C. データセットとデータローダーの作成**
    *   **解説ドキュメント**: [dataset_and_dataloader.md](./pytorch-llm/docs/dataset_and_dataloader.md) 🌟**必読**
    *   **概要**: LLMへ入力するためのミニバッチを作成する「スライディングウィンドウ方式」について学びます。重要なパラメータである `max_length` (最大長) と `stride` (歩幅/移動幅) の違いを図解で解説しています。

---

### 💡 PyTorch基礎ワークフローと前提知識の振り返り
本格的なLLMの埋め込みやアテンションに入る前に、PyTorchによる基本的な「データ準備 $\to$ モデル構築 $\to$ 訓練 $\to$ 評価 $\to$ 保存」の全体プロセス、およびテンソル変形やメモリ配置に関する前提知識を復習します。

*   **オッズ比・ロジットの数理デモ (ロジットの逆変換)**
    *   **コード**: [odds_ratio_demo.py](./pytorch-llm/basics/odds_ratio_demo.py)
    *   **概要**: 確率から「オッズ」「オッズ比」を計算し、モデルの出力する生スコア（ロジット）を逆関数であるシグモイド関数に通して再び完全に確率へと復元する、統計学から機械学習へと繋がる数学的背景を実証するデモプログラムです。
*   **PyTorch基礎の一気通貫デモ**
    *   **コード**: [pytorch_basics_demo.py](./pytorch-llm/basics/pytorch_basics_demo.py) 🌟**必読**
    *   **概要**: `nn.Module` を用いた3層ニューラルネットの定義、自作DatasetとDataLoader、訓練ループの3ステップ（ゼログラ、逆伝播、パラメータ更新）、評価時の `no_grad` の意義、モデルの保存と再ロードまで、PyTorchの基礎を一括で学べるコメント付きデモコードです。
*   **勾配降下法・等高線の可視化デモ (パラメータの探索軌跡)**
    *   **コード**: [gradient_descent_demo.py](./pytorch-llm/basics/gradient_descent_demo.py)
    *   **解説ドキュメント**: [gradient_descent.md](./pytorch-llm/docs/gradient_descent.md) 🌟**必読**
    *   **概要**: パラメータが損失関数の等高線図（地形）を、自動微分（Backward）によって得られた勾配をヒントに谷底を目指して下る様子を可視化するデモです。標準的なSGDと慣性を入れたMomentumのジグザグ挙動の違いをグラフ画像にして出力します。
*   **Linear（全結合）層とバイアスの動作デモ**
    *   **コード**: [linear_basics_demo.py](./pytorch-llm/basics/linear_basics_demo.py)
    *   **概要**: ニューラルネットワークの基本要素である `torch.nn.Linear` が内部に持つ重みとバイアス、およびその数式（y = x @ W.T + b）が手動の行列演算と完全に一致することを確認するデモプログラムです。
*   **活性化関数・keepdim・Sequentialデモ (ReLUの検証)**
    *   **コード**: [activation_sequential_demo.py](./pytorch-llm/basics/activation_sequential_demo.py)
    *   **解説ドキュメント**: [activation_and_sequential.md](./pytorch-llm/docs/activation_and_sequential.md) 🌟**必読**
    *   **概要**: 活性化関数（ReLU）の非線形処理の役割、統計量算出における `keepdim=True` の Shape 保持の必要性と引き算エラーの再現、および `nn.Sequential` によるモジュール直列化パイプラインの動作を検証するデモプログラムです。
*   **LayerNormのscale/shift動作デモ (分布制御の検証)**
    *   **コード**: [layernorm_scale_shift_demo.py](./pytorch-llm/basics/layernorm_scale_shift_demo.py)
    *   **解説ドキュメント**: [layernorm_scale_shift.md](./pytorch-llm/docs/layernorm_scale_shift.md) 🌟**必読**
    *   **概要**: LayerNorm内の `scale` と `shift` パラメータが、平均0・分散1に標準化されたデータの分布をいかに自在に再調整（拡大縮小・平行移動）しているのかを、数式通りに平均と分散がシフトする様子から学ぶデモプログラムです。
*   **バッチ正規化 vs 層正規化デモ (バッチ依存性の検証)**
    *   **コード**: [batch_vs_layer_norm_demo.py](./pytorch-llm/basics/batch_vs_layer_norm_demo.py)
    *   **解説ドキュメント**: [batch_vs_layer_normalization.md](./pytorch-llm/docs/batch_vs_layer_normalization.md) 🌟**必読**
    *   **概要**: バッチ次元に沿って正規化する「バッチ正規化（BatchNorm）」と、特徴次元に沿って自己完結させる「層正規化（LayerNorm）」の計算方向の違いを学び、バッチサイズ（周りのデータ構成）の変動が各サンプルの正規化結果にどう影響するかを数値で対比実証するデモプログラムです。
*   **GELU・SwiGLU活性化関数デモ (標準ガウス分布とゲートの検証)**
    *   **コード**: [gelu_swiglu_demo.py](./pytorch-llm/basics/gelu_swiglu_demo.py)
    *   **解説ドキュメント**: [gelu_swiglu_normal_distribution.md](./pytorch-llm/docs/gelu_swiglu_normal_distribution.md) 🌟**必読**
    *   **概要**: 標準ガウス分布の累積確率（CDF）をゲート制御に使用するGELU活性化関数の定義と近似式の検証、最新LLMで主流のSwiGLUモジュールの定義と順伝播の実行、および `matplotlib` によるReLU/GELU/Swishの滑らかな活性化曲線の比較グラフ自動生成を行うデモプログラムです。
*   **フィードフォワード層 vs アテンション層デモ (トークン独立性の検証)**
    *   **コード**: [feed_forward_demo.py](./pytorch-llm/basics/feed_forward_demo.py)
    *   **解説ドキュメント**: [feed_forward_network.md](./pytorch-llm/docs/feed_forward_network.md) 🌟**必読**
    *   **概要**: アテンション層（トークン間で情報をブレンドする）とフィードフォワード層（各トークンを完全に独立して処理する）の決定的な役割の違いを、トークンの値を書き換えた際の影響の有無から数値で対比実証するデモプログラムです。
*   **FFNのKey-Valueメモリ動作デモ (個別知識処理の可視化)**
    *   **コード**: [ffn_key_value_demo.py](./pytorch-llm/basics/ffn_key_value_demo.py)
    *   **解説ドキュメント**: [ffn_knowledge_processing.md](./pytorch-llm/docs/ffn_knowledge_processing.md) 🌟**必読**
    *   **概要**: FFNの「個別知識処理」の内部メカニズム（1層目でKeyと照合し、GELUゲートを経て、2層目で対応するValueを引き出すデータベース動作）を、GELU特有の減衰・情報漏れといった数理特性を含めて数値でシミュレートするデモプログラムです。
*   **スキップ接続・勾配流シミュレーションデモ (勾配消失防止の検証)**
    *   **コード**: [skip_connection_demo.py](./pytorch-llm/basics/skip_connection_demo.py)
    *   **解説ドキュメント**: [skip_connection_history.md](./pytorch-llm/docs/skip_connection_history.md) 🌟**必読**
    *   **概要**: 2015年のResNet誕生に繋がる劣化問題の歴史的背景と、微分式「+1」が敷く勾配のバイパス（高速道路）の数理を学び、20層の深層ネットワークにおいてスキップ接続がないと最下層で勾配が消失するのに対し、スキップ接続があるといかに力強く勾配が維持されるかを数値で対比実証するデモプログラムです。
*   **有偏分散・不偏分散とモデル互換性デモ (Bessel補正と誤差蓄積の検証)**
    *   **コード**: [variance_compatibility_demo.py](./pytorch-llm/basics/variance_compatibility_demo.py)
    *   **解説ドキュメント**: [variance_and_compatibility.md](./pytorch-llm/docs/variance_and_compatibility.md) 🌟**必読**
    *   **概要**: 有偏分散（分母 $n$）と不偏分散（分母 $n-1$）の数学的な差をデータサイズごとに検証し、このわずかな差がモデルの複数レイヤーを通過する間にいかに大きな誤差へと増幅され、ロードした事前学習済みパラメータの文章生成を壊してしまうかをシミュレートするデモプログラムです。
*   **Tensorの形状変更とメモリ解釈デモ (viewメソッド)**
    *   **コード**: [view_basics_demo.py](./pytorch-llm/basics/view_basics_demo.py)
    *   **概要**: テンソルの形状を操作する `view` メソッドの本質（メモリコピーを行わず、メモリ上の実体データへの「読み取り方の枠組み」だけを書き換えている点）を、メモリ実体の同一性検証やLLM用の多次元分割を通して学ぶデモプログラムです。
*   **メモリ連続性と contiguous() おまじないの検証デモ**
    *   **コード**: [contiguous_basics_demo.py](./pytorch-llm/basics/contiguous_basics_demo.py)
    *   **概要**: 転置（transpose）したテンソルのメモリ上の並びが非連続になる物理的仕組みと、非連続のままだと `view` でクラッシュする現象、および `contiguous()` によってメモリ上のデータが物理的に再配置（整頓）されて正常に変形できるようになる解決プロセスを目で見て学ぶデモプログラムです。
*   **keepdim の視覚的形状確認デモ (ブロードキャストの仕組み)**
    *   **コード**: [keepdim_visual_demo.py](./pytorch-llm/basics/keepdim_visual_demo.py)
    *   **解説ドキュメント**: [keepdim_details.md](./pytorch-llm/docs/keepdim_details.md) 🌟**必読**
    *   **概要**: テンソルの統計量（平均値など）を計算する際、次元を潰す `keepdim=False` と、元の次元構造を維持する `keepdim=True`（2行1列のビル型）の違いを、コンソール上のアスキーアート（テキスト図解）を用いて視覚的に比較・学習するデモプログラムです。

---

### STEP 3: 入力埋め込み (Input Embedding) パイプライン
モデルにトークンを渡す直前の「ベクトル表現」について学びます。

*   **3-A. nn.Embeddingの動作デモ**
    *   **コード**: [embedding_demo.py](./pytorch-llm/basics/embedding_demo.py) 🌟**最初に見るのがおすすめ**
    *   **概要**: `torch.nn.Embedding` が内部でどのような数値テーブル（行列）を持ち、IDからベクトルへの変換（ルックアップ）がどのように行われているかを目で見て確認できる、極小サイズのサンプルコードです。
*   **3-B. PyTorch torch.arange の動作デモ**
    *   **コード**: [arange_demo.py](./pytorch-llm/basics/arange_demo.py)
    *   **概要**: 位置インデックスを作成する際に多用される、PyTorchの連番生成関数 `torch.arange` の基本的な使い方を確認するデモプログラムです。
*   **3-C. 入力埋め込みの処理パイプライン**
    *   **コード**: [make-embedding.py](./pytorch-llm/src/make-embedding.py)
    *   **概要**: トークンIDを意味ベクトルに変換するトークン埋め込み（Token Embedding）と、位置情報を表す位置埋め込み（Positional Embedding）を PyTorch で合成する実装です。
*   **3-D. 埋め込みのメカニズム解説**
    *   **解説ドキュメント**: [embedding_mechanism.md](./pytorch-llm/docs/embedding_mechanism.md) 🌟**必読**
    *   **概要**: 入力テキストがベクトルへ変換され、Transformerモデルへ入力されるまでのプロセスを独自のMermaid図を用いてステップバイステップで解説しています。

---

### STEP 4: Attentionメカニズムの基礎
LLMの心臓部であるAttention（注意機構）の仕組みと、その計算に必要な数学的処理を学びます。

*   **4-A. Attentionスコア計算の基礎とデモ**
    *   **コード**: [attention_basics_demo.py](./pytorch-llm/src/attention_basics_demo.py)
    *   **概要**: `torch.empty` と `enumerate` の動作確認から、ループを用いたドット積によるスコア計算、さらに効率的な「行列演算（ループなし）」への移行までを1つにまとめたデモコードです。
*   **4-B. ドット積と行列演算のビジュアル解説**
    *   **解説ドキュメント**: [attention_basics.md](./pytorch-llm/docs/attention_basics.md) 🌟**必読**
    *   **概要**: なぜドット積が「単語の類似度」として機能するのか、そしてループ処理がどのように並列行列演算へ変換されるのかを図解した解説書です。
*   **4-C. ソフトマックス関数の挙動確認デモ**
    *   **コード**: [softmax_demo.py](./pytorch-llm/basics/softmax_demo.py)
    *   **概要**: アテンションスコアが確率分布（重み）に変換されるプロセスを、コンソール上のテキストグラフで可視化して確認するデモプログラムです。
*   **4-D. ソフトマックス関数のグラフ特性と役割解説**
    *   **解説ドキュメント**: [softmax_basics.md](./pytorch-llm/docs/softmax_basics.md) 🌟**必読**
    *   **概要**: 指数関数 $e^x$ が描くグラフの特性がどのように「注意の集中」に役立つのか、なぜ単純な割り算ではダメなのかを数学的・直感的に解説したドキュメントです。

---

### STEP 5: LLMアーキテクチャの構築 (GPTモデル構築)
アテンションや埋め込みパーツを組み合わせ、GPTモデル全体の構築と、データフローの仕組みを学びます。

*   **5-A. GPTモデル構成とデータフロー解説**
    *   **解説ドキュメント**: [gpt_architecture.md](./pytorch-llm/docs/gpt_architecture.md) 🌟**必読**
    *   **概要**: 書籍の「図4-4」をベースにした単語予測の全体データフロー、Transformerブロック内部（LayerNorm、Attention、FFN、残差接続等）のテンソル処理フロー、およびプレースホルダ設計アプローチやEmbeddingの数理を体系的に解説した一冊です。
*   **5-B. 最後の層正規化と線形出力層の役割解説**
    *   **解説ドキュメント**: [final_normalization_and_output_head.md](./pytorch-llm/docs/final_normalization_and_output_head.md) 🌟**必読**
    *   **概要**: 書籍の「図4-15」に示されているモデルの最終出口（LayerNormと線形出力層/LM Head）の役割を学び、加算の繰り返しによって歪んだデータを安定化させてから、768次元の内部ベクトルを語彙数と同じ50,257次元の確率予測へと射影・変換する仕組みを解説したドキュメントです。
*   **5-C. 重み共有（Weight Tying）のメカニズム解説**
    *   **解説ドキュメント**: [weight_tying_basics.md](./pytorch-llm/docs/weight_tying_basics.md) 🌟**必読**
    *   **概要**: 本来1億6,300万パラメータあるモデルが、なぜ実質的に1億2,400万パラメータまで削減できるのか、入力Embeddingと出力Linear層で同じ重みパラメータを共有する「重み共有」のメリット、数理、および直感的な対訳辞書の例えを交えて解説したドキュメントです。
*   **5-D. 損失を計算する6つのステップ（図5-7）の解説と実証**
    *   **コード**: [loss_calculation_demo.py](./pytorch-llm/basics/loss_calculation_demo.py)
    *   **解説ドキュメント**: [cross_entropy_loss_calculation.md](./pytorch-llm/docs/cross_entropy_loss_calculation.md) 🌟**必読**
    *   **概要**: 書籍の「図5-7」に描かれているモデルの最終損失（NLL/Cross Entropy Loss）を計算する6つのステップを解説し、PyTorchでのアドバンスト・インデキシングを用いたターゲット確率の抽出から対数変換、平均化、負数化までの手動ステップと、PyTorch組み込み関数の計算値が完全に一致することを検証・実演するデモプログラムです。
*   **5-E. パープレキシティ（Perplexity）の数理解説と実証**
    *   **コード**: [perplexity_demo.py](./pytorch-llm/basics/perplexity_demo.py)
    *   **解説ドキュメント**: [perplexity_basics.md](./pytorch-llm/docs/perplexity_basics.md) 🌟**必読**
    *   **概要**: 書籍の147ページに記載されている言語モデルの性能指標「パープレキシティ（PPL）」の数理を解説し、あてずっぽう予測（PPL＝語彙数）やN択で迷っている状態（PPL＝N）など、なぜ Loss の指数 $e^{\text{Loss}}$ を取るだけで「実質的な選択肢の個数」に変換されるのかをPyTorchの数値検証を交えて学ぶデモプログラムです。
*   **5-F. PyTorchの訓練ループとエポック・バッチ（図5-11）の解説と実証**
    *   **コード**: [training_loop_demo.py](./pytorch-llm/basics/training_loop_demo.py)
    *   **解説ドキュメント**: [pytorch_training_loop.md](./pytorch-llm/docs/pytorch_training_loop.md) 🌟**必読**
    *   **概要**: 書籍154ページの「図5-11」に示されているPyTorchの一般的な訓練ループ（Training Loop）を解説し、「エポック」「バッチ（ミニバッチ）」「イテレーション」の概念と階層関係を定義するとともに、`zero_grad()` での勾配初期化、順伝播、`backward()` での勾配計算、`step()` での重み更新の一連のステップで各パラメータの値と勾配がどう変動するかを数値的に追従するデモプログラムです。
*   **5-G. オプティマイザー（Adam vs AdamW）の数理解説と実証**
    *   **コード**: [optimizer_demo.py](./pytorch-llm/basics/optimizer_demo.py)
    *   **解説ドキュメント**: [optimizer_adam_vs_adamw.md](./pytorch-llm/docs/optimizer_adam_vs_adamw.md) 🌟**必読**
    *   **概要**: 書籍156ページコラムの「AdamW」を解説し、なぜ従来の「Adam＋L2正則化」では自適応学習率によって重み減衰が歪んでしまうバグがあったのか、そして「AdamW」がそれをどう分離（Decouple）して解決したのかの数理を学び、手動の更新ループ計算とPyTorchの公式関数出力が完全一致することを確認・実演するデモプログラムです。
*   **5-H. multinomialサンプリングと確率比例の数理解説と実証**
    *   **コード**: [multinomial_demo.py](./pytorch-llm/basics/multinomial_demo.py)
    *   **解説ドキュメント**: [multinomial_sampling.md](./pytorch-llm/docs/multinomial_sampling.md) 🌟**必読**
    *   **概要**: 書籍161ページの確率サンプリングを解説し、決定論的な `argmax`（常に最大値を選択）と確率比例の `multinomial`（確率をルーレットの面積とみなして抽選）の違い、およびLLMの応答に人間らしい「揺らぎ」や「創造性（多様性）」を与えるために必須となる数理背景を学び、1,000回サンプリングで確率比例頻度に収束することを実証するデモプログラムです。

---

### STEP 6: 外部LLM APIの活用 (Gemini API)
モデルを自作するだけでなく、最先端のクラウドLLMをAPIから活用する応用スキルを学びます。

*   **学習ディレクトリ**: [gemini-api/](./gemini-api/)
*   **概要**: Google GenAI SDK (`google-genai`) を使った Gemini API の実装例です。
    *   [count_tokens.py](./gemini-api/count_tokens.py): プロンプトの入力・出力・思考（Thoughts）プロセスのトークン数を計測・出力します。
    *   [streaming_response.py](./gemini-api/streaming_response.py): リアルタイムにAIの応答を表示するストリーミング処理について学びます。

---

### STEP 7: Hugging Faceエコシステムと実践的なNLPタスクの解決
「LLM自作入門」で低レイヤーの仕組みを学んだ後、Hugging Faceライブラリを使って実際のNLPタスクの解決やファインチューニングを行う実践スキルを学びます。

*   **学習ディレクトリ**: [transformers-book/](./transformers-book/)
*   **書籍**: 『機械学習エンジニアのためのTransformers』
*   **概要**: Hugging Faceの `transformers` や `datasets` などのエコシステムを用いて、テキスト分類、NER（名前付きエンティティ認識）、要約、質問回答などのタスク解法や、モデルの評価、本番環境での効率化を学習・実装します。

---

## 📄 ライセンスと出典 (License & Attribution)

本リポジトリ内には、学習・検証目的で以下のオープンソースを参考に作成または改変したコードが含まれています。

### Sebastian Raschka 氏 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 由来のコード
*   **対象ファイル**: `regular-expression.py`, `pytorch-llm/src/make-vocab.py`, `pytorch-llm/src/Byte-Pair_Encoding.py`, `pytorch-llm/src/make-embedding.py`, `pytorch-llm/src/attention_basics_demo.py`, `pytorch-llm/basics/softmax_demo.py`
*   **ライセンス**: [Apache License 2.0](./LICENSE) （オリジナルコードの著作権は Sebastian Raschka 氏に帰属します）
