# LLM構築ライフサイクル：開発の3つのステージ

LLM（大規模言語モデル）を一から構築し、特定のアプリケーションに適用するまでの全体像を示す開発ステージマップです。
本リポジトリでの学習・実装は、このロードマップに基づいて進めていきます。

---

## 🗺️ LLM構築の全体図 (図3-13に基づく詳細版)

以下のダイアグラムは、スクラッチからLLMを構築し、最終的に特定のアプリケーション（分類器やアシスタント）へ適用するまでの全プロセスを整理したものです。
アテンション機構（ステップ2）の内部進化プロセスについても詳細化しています。

```mermaid
graph TD
    %% ステージ1
    subgraph Stage1["【ステージ 1】 LLMの構築 (Building the LLM)"]
        S1_1["1) データの準備と前処理<br>(Data Preprocessing & Sampling)"]
        
        %% アテンションの進化
        subgraph AttnFlow["2) Attentionメカニズムの実装 (Attention Mechanism)"]
            A1["① 単純化されたSelf-Attention<br>(訓練可能な重みなし)"]
            A2["② Self-Attention<br>(訓練可能な重みを追加)"]
            A3["③ Causal Attention<br>(未来のトークンをマスク)"]
            A4["④ Multi-head Attention<br>(複数のヘッドで並列計算)"]
            
            A1 --> A2 --> A3 --> A4
        end
        
        S1_3["3) LLMアーキテクチャの設計<br>(Transformer / GPT Architecture)"]
        
        S1_1 --> A1
        A4 --> S1_3
    end

    %% 事前学習
    Pretraining["4) 事前学習 (Pre-training)<br>未ラベルの大規模テキストでの予測学習"]

    %% ステージ2
    subgraph Stage2["【ステージ 2】 ベースモデルと評価 (Base Model & Evaluation)"]
        S2_1["5) 訓練ループの実装<br>(Training Loops)"]
        S2_2["6) モデルの性能評価<br>(Model Evaluation)"]
        S2_3["7) 訓練済み重みのロード<br>(Loading Pre-trained Weights)"]
        
        S2_1 --> S2_2 --> S2_3
    end

    %% ステージ3
    subgraph Stage3["【ステージ 3】 特定タスクへの微調整 (Fine-Tuning)"]
        S3_1["8-A) 分類用ファインチューニング<br>(Classification Fine-tuning)"]
        S3_2["8-B) 指示追従用ファインチューニング<br>(Instruction Fine-tuning)"]
    end
    
    %% 外部データセット
    Dataset_Class["クラスラベルを持つデータセット<br>(例: スパム判定用のラベル付きメール)"]
    Dataset_Instr["指示データセット<br>(例: 質問と回答の対話ペア)"]
    
    %% 最終プロダクト
    Classifier["分類器 (Classifier)<br>スパム判定・感情分析など"]
    Assistant["パーソナルアシスタント<br>対話型チャットAI"]

    %% 接続関係
    Stage1 --> Pretraining
    Pretraining --> Stage2
    
    Stage2 --> S3_1
    Stage2 --> S3_2
    
    Dataset_Class --> S3_1
    Dataset_Instr --> S3_2
    
    S3_1 --> Classifier
    S3_2 --> Assistant

    style Stage1 fill:#f5f5f5,stroke:#333
    style Stage2 fill:#f5f5f5,stroke:#333
    style Stage3 fill:#f5f5f5,stroke:#333
    style AttnFlow fill:#e6f2ff,stroke:#0066cc,stroke-width:1px
```

---

## 📝 各ステージの解説

### 【ステージ 1】 LLMの構築
モデルの「肉体」を作るフェーズです。テキストデータを数値に変換するトークナイザーから、文脈を捉えるアテンション機構、そしてそれらを結合したGPTモデルのアーキテクチャそのものをコードで構築します。
特に**「Attentionメカニズム（2）」**は、重みなしの極小アプローチから始まり、学習可能な重みの追加、生成時の未来情報の制限（Causal）、マルチヘッド化へと段階的に進化させていきます。

### 【ステージ 2】 ベースモデルと評価
モデルを「学習・訓練」させるフェーズです。言語モデルの予測精度を高めるためのループを設計・評価します。また、自分で巨大な学習を行う代わりに、オープンソースの学習済みパラメータを自分で構築したモデル構造に読み込む方法もここで学びます。

### 【ステージ 3】 特定タスクへの微調整 (ファインチューニング)
ベースモデルは「次の単語を予測すること」しかできません。これを、スパムメールを判定する「分類器」や、対話形式でユーザーの指示に従う「チャットアシスタント」へ変化させるため、専用のデータセットを用いてさらに追加学習（チューニング）を行います。

---

## 📄 出典・参考情報

*   **参考図書**: Sebastian Raschka 著『LLMs from Scratch』（邦題：『つくりながら学ぶ！LLM自作入門』）の「図3-13: Self-AttentionメカニズムがLLMの実装という広いコンテキストの中でどのように位置付けられるか」の概念より着想。
*   **本ドキュメントの位置付け**: 上記書籍の全体設計図をベースに、独自のMermaidフロー図と解説を用いて学習マイルストーンを整理したものです。
