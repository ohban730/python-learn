# LLM & PyTorch 学習ドキュメント・Q&A解説ノートインデックス (docs)

本ディレクトリは、LLM（大規模言語モデル）のスクラッチ実装や『機械学習エンジニアのためのTransformers』学習中に生じた質問、数理的背景、アーキテクチャの解説ノート（Markdown形式）を集約したドキュメントライブラリです。

---

## 📚 テーマ別ドキュメント検索目次 (Category Index)

### 🗺️ 1. 全体ガイド・学習ロードマップ & 用語集
LLM開発の全体像、用語、開発ステージを整理した必須ガイドです。

* 🌟 **LLM開発の3ステージとAttentionロードマップ**: [llm_development_stages.md](./llm_development_stages.md)
* 🌟 **LLM一般概念・AI定義用語集**: [llm_terminology.md](./llm_terminology.md)
* **Baseモデル vs Instructモデルの違い**: [base_vs_instruct_models.md](./base_vs_instruct_models.md)
* **LLM量子化（Quantization）の基礎**: [llm_quantization.md](./llm_quantization.md)
* **Hugging Face Trainer習得ロードマップ**: [mastering_huggingface_trainer_roadmap.md](./mastering_huggingface_trainer_roadmap.md)

---

### 🧠 2. PyTorch基礎・テンソル・数理モデル
PyTorchの基本動作、テンソル変形、メモリ最適化、および基礎数学の解説です。

* 🌟 **PyTorch基礎・Linear数理編**: [pytorch_basics.md](./pytorch_basics.md)
* 🌟 **テンソル変形・メモリ最適化編**: [pytorch_tensor_operations.md](./pytorch_tensor_operations.md)
* **`torch.nn.functional` と `nn.Module` の使い分け**: [pytorch_functional_module.md](./pytorch_functional_module.md)
* **PyTorch訓練ループの基本構造**: [pytorch_training_loop.md](./pytorch_training_loop.md)
* **`nn.Linear` 層の内部仕様ガイド**: [nn_linear_guide.md](./nn_linear_guide.md)
* **`keepdim=True/False` の形状変化の詳細**: [keepdim_details.md](./keepdim_details.md)
* **LayerNormの scale / shift の役割**: [layernorm_scale_shift.md](./layernorm_scale_shift.md)
* **バッチ正規化 (BatchNorm) vs 層正規化 (LayerNorm)**: [batch_vs_layer_normalization.md](./batch_vs_layer_normalization.md)
* **GELU / SwiGLU 活性化関数とガウス分布**: [gelu_swiglu_normal_distribution.md](./gelu_swiglu_normal_distribution.md)
* **活性化関数と `nn.Sequential` の解説**: [activation_and_sequential.md](./activation_and_sequential.md)
* **勾配降下法・等高線探索の数理**: [gradient_descent.md](./gradient_descent.md)
* **オッズ比・ロジット・シグモイドの数学接続**: [odds_ratio.md](./odds_ratio.md)
* **Logit / Probit 変換比較**: [logit_probit_transformation.md](./logit_probit_transformation.md)
* **Softmax関数の数値的安定性と基礎**: [softmax_basics.md](./softmax_basics.md)
* **スキップ接続 (Skip Connection) の歴史と勾配流**: [skip_connection_history.md](./skip_connection_history.md)
* **有偏分散 vs 不偏分散と互換性**: [variance_and_compatibility.md](./variance_and_compatibility.md)
* **`torch.bmm` (バッチ行列積) の解説**: [torch_bmm_explanation.md](./torch_bmm_explanation.md)
* **Numpy配列 vs Pythonリストのメモリ比較**: [numpy_vs_python_list.md](./numpy_vs_python_list.md)
* **Pythonのアスタリスク `*` アンパック文法**: [python_asterisk_unpacking.md](./python_asterisk_unpacking.md)
* **PyTorchで逆伝播 (Backprop) はどこで定義されているか**: [where_is_backpropagation_defined.md](./where_is_backpropagation_defined.md)

---

### ⚙️ 3. Transformerアーキテクチャ & アテンション
Self-Attention、Cross-Attention、位置エンコーディング、GPT/BERT等の構造解説です。

* 🌟 **Attentionメカニズムの数理解説**: [attention_basics.md](./attention_basics.md)
* **Scaled Dot-Product Attentionの4ステップ**: [scaled_dot_product_attention_4steps.md](./scaled_dot_product_attention_4steps.md)
* **Cross-Attention vs Self-Attention の違い**: [cross_attention_vs_self_attention.md](./cross_attention_vs_self_attention.md)
* **フィードフォワードネットワーク (FFN) の役割**: [feed_forward_network.md](./feed_forward_network.md)
* **FFNのKey-Valueメモリとしての知識処理**: [ffn_knowledge_processing.md](./ffn_knowledge_processing.md)
* 🌟 **GPTモデル全体のアーキテクチャ図解**: [gpt_architecture.md](./gpt_architecture.md)
* **Encoder Stack アーキテクチャ解説**: [encoder_stack_architecture.md](./encoder_stack_architecture.md)
* **Encoder / Decoder モデル比較 (BERT, GPT, BART, T5)**: [encoder_decoder_bert_gpt.md](./encoder_decoder_bert_gpt.md)
* **Encoder-Decoder間アテンションと中間表現**: [encoder_decoder_attention_and_intermediate_rep.md](./encoder_decoder_attention_and_intermediate_rep.md)
* **最終層正規化と出力ヘッドの役割**: [final_normalization_and_output_head.md](./final_normalization_and_output_head.md)
* **重み共有 (Weight Tying) のメカニズム**: [weight_tying_basics.md](./weight_tying_basics.md)
* **RoPE (Rotary Position Embedding) の原理**: [rope_rotary_position_embedding.md](./rope_rotary_position_embedding.md)
* **RoPE YaRN スケーリング**: [rope_yarn_scaling.md](./rope_yarn_scaling.md)
* **位置エンコーディングとDeBERTa**: [position_embeddings_and_deberta.md](./position_embeddings_and_deberta.md)
* **Transformerモデルの各種分類**: [transformer_model_types.md](./transformer_model_types.md)
* **BERT vs BART の比較**: [bert_vs_bart_comparison.md](./bert_vs_bart_comparison.md)
* **DistilBERT の基礎**: [distilbert_basics.md](./distilbert_basics.md)
* **Style-BERT-VITS2 と BERT の関係**: [style_bert_vits2_and_bert.md](./style_bert_vits2_and_bert.md)
* **Google翻訳の歴史と T5**: [google_translation_and_t5.md](./google_translation_and_t5.md)
* **CNN, RNN, ResNet の基礎比較**: [cnn_rnn_resnet_basics.md](./cnn_rnn_resnet_basics.md)
* **再帰型モデル (RNN) におけるAttention**: [attention_in_recurrent_models.md](./attention_in_recurrent_models.md)

---

### 🔤 4. トークナイズ・データ処理・データセット
BPE、語彙辞書、スライディングウィンドウ、Hugging Face Datasetsの前処理です。

* 🌟 **埋め込みメカニズム (Token + Positional Embedding)**: [embedding_mechanism.md](./embedding_mechanism.md)
* **サブワードトークン化 (BPE, WordPiece, Unigram)**: [subword_tokenization_bpe_wordpiece_unigram.md](./subword_tokenization_bpe_wordpiece_unigram.md)
* **トークナイザーライブラリ比較**: [tokenizer_libraries.md](./tokenizer_libraries.md)
* **バッチトークナイズ処理**: [batched_tokenization.md](./batched_tokenization.md)
* **Truncation (切り捨て) の基礎**: [truncation_basics.md](./truncation_basics.md)
* 🌟 **Dataset と DataLoader (max_length / stride)**: [dataset_and_dataloader.md](./dataset_and_dataloader.md)
* **DataCollator の基礎**: [data_collator_basics.md](./data_collator_basics.md)
* **DataCollator と Padding の関係**: [data_collator_and_padding.md](./data_collator_and_padding.md)
* **Hugging Face Datasets の Memory Mapping 仕組み**: [huggingface_datasets_memory_mapping.md](./huggingface_datasets_memory_mapping.md)
* **`datasets.map` の各種パラメータ**: [datasets_map_parameters.md](./datasets_map_parameters.md)
* **One-Hotベクトルと2Dテンソル**: [one_hot_vectors_2d_tensor.md](./one_hot_vectors_2d_tensor.md)
* **One-Hotエンコーディングのクラス数制限**: [one_hot_num_classes_limit.md](./one_hot_num_classes_limit.md)
* **BERTの特殊トークン `[CLS]` / `[SEP]`**: [bert_special_tokens_cls_sep.md](./bert_special_tokens_cls_sep.md)
* **ステミング (Stemming) vs 形態素解析/レンマ化 (Lemmatization)**: [stemming_vs_lemmatization.md](./stemming_vs_lemmatization.md)
* **`wget` コマンドの基礎**: [wget_basics.md](./wget_basics.md)

---

### 🎯 5. モデルの学習・最適化・推論・評価
損失関数、AdamW、テキスト生成サンプリング、BLEU、Perplexity、評価指標の解説です。

* **交差エントロピー損失計算の6ステップ**: [cross_entropy_loss_calculation.md](./cross_entropy_loss_calculation.md)
* **オプティマイザ (Adam vs AdamW) の数理的差異**: [optimizer_adam_vs_adamw.md](./optimizer_adam_vs_adamw.md)
* **確率サンプリング (`multinomial`) の数理**: [multinomial_sampling.md](./multinomial_sampling.md)
* **テキスト生成ループの停止メカニズム**: [generation_loop_stopping.md](./generation_loop_stopping.md)
* **ビームサーチ (Beam Search) の基礎**: [beam_search_basics.md](./beam_search_basics.md)
* **ビームサーチの内部メカニズム**: [beam_search_mechanism.md](./beam_search_mechanism.md)
* **パープレキシティ (Perplexity / PPL) の数理解説**: [perplexity_basics.md](./perplexity_basics.md)
* 🌟 **BLEUスコアとDownstream Taskの概念**: [bleu_metric_and_downstream_task.md](./bleu_metric_and_downstream_task.md)
* **BLEUスコアの Vanilla Precision 計算**: [bleu_vanilla_precision_explained.md](./bleu_vanilla_precision_explained.md)
* 🌟 **ROUGEスコア（テキスト要約評価）の徹底解説**: [rouge_score_explained.md](./rouge_score_explained.md)
* **`compute_metrics` の解説**: [compute_metrics_explained.md](./compute_metrics_explained.md)
* **`compute_metrics` と評価スコア**: [compute_metrics_and_eval_scores.md](./compute_metrics_and_eval_scores.md)
* **混同行列 (Confusion Matrix) 活用ガイド**: [confusion_matrix_guide.md](./confusion_matrix_guide.md)
* **Batch Size vs Number of Batches の用語対比**: [batch_size_vs_num_batches.md](./batch_size_vs_num_batches.md)
* **データリーク (Data Leakage) とサンプリング**: [data_leakage_and_sampling.md](./data_leakage_and_sampling.md)
* **Causal LM における Logits / Label の Shift 処理**: [causal_lm_logits_label_shift.md](./causal_lm_logits_label_shift.md)
* **Hidden States vs Logits の違い**: [hidden_states_vs_logits.md](./hidden_states_vs_logits.md)
* **線形重み最適化の基礎**: [linear_weight_optimization.md](./linear_weight_optimization.md)
* **多項ロジスティック回帰 (Multinomial Logistic Regression)**: [multinomial_logistic_regression.md](./multinomial_logistic_regression.md)
* **`max_iter` と収束性の解説**: [max_iter_and_convergence.md](./max_iter_and_convergence.md)
* **ヒューリスティクスとベースラインモデル**: [heuristics_and_baseline_models.md](./heuristics_and_baseline_models.md)
* **アンサンブル学習 (Ensemble Learning) の基礎**: [ensemble_learning_basics.md](./ensemble_learning_basics.md)
* **パラメトリック vs ノンパラメトリック**: [parametric_vs_nonparametric.md](./parametric_vs_nonparametric.md)

---

### 🤗 6. Hugging Face & 『Transformers』実践応用
Hugging Face Ecosystem、ファインチューニング、タスク別分類ヘッドの解説です。

* **Hugging Face pipeline (テキスト分類)**: [huggingface_pipeline_text_classification.md](./huggingface_pipeline_text_classification.md)
* **Hugging Face pipeline (要約パラメータ)**: [huggingface_pipeline_summarization_params.md](./huggingface_pipeline_summarization_params.md)
* **Hugging Face Trainer 再利用可能ガイド**: [huggingface_trainer_reusable_guide.md](./huggingface_trainer_reusable_guide.md)
* **Hugging Face Hub へのプッシュガイド**: [huggingface_hub_push_guide.md](./huggingface_hub_push_guide.md)
* **Trainer Checkpoint の仕組み**: [trainer_checkpoint_explanation.md](./trainer_checkpoint_explanation.md)
* **チェックポイント (Checkpoint) とは何か**: [what_is_checkpoint.md](./what_is_checkpoint.md)
* **分類ヘッド (Classification Heads) の基礎**: [classification_heads_basics.md](./classification_heads_basics.md)
* **分類ヘッドにおける Dropout の効果**: [classification_head_dropout.md](./classification_head_dropout.md)
* **Sequence Classification vs Token Classification**: [sequence_classification_vs_token_classification.md](./sequence_classification_vs_token_classification.md)
* **Token Classification Head のデータフロー**: [token_classification_head_flow_guide.md](./token_classification_head_flow_guide.md)
* **Hidden States の抽出方法**: [extract_hidden_states_details.md](./extract_hidden_states_details.md)
* **`last_hidden_state` のテンソルスライシング**: [tensor_slicing_last_hidden_state.md](./tensor_slicing_last_hidden_state.md)
* **なぜファインチューニングで Hidden States が適応されるのか**: [why_finetuning_adapts_hidden_states.md](./why_finetuning_adapts_hidden_states.md)
* **ファインチューニングの手法バリエーション**: [fine_tuning_variations.md](./fine_tuning_variations.md)
* **Domain Adaptation vs Fine-tuning**: [domain_adaptation_vs_finetuning.md](./domain_adaptation_vs_finetuning.md)
* **転移学習 (Transfer Learning) と ULMFiT の歴史**: [transfer_learning_and_ulmfit.md](./transfer_learning_and_ulmfit.md)
* **事前学習 (Pre-training) とアーキテクチャ共有**: [pretraining_and_architecture_sharing.md](./pretraining_and_architecture_sharing.md)
* **Zero-Shot vs Few-Shot 学習**: [zero_shot_vs_few_shot.md](./zero_shot_vs_few_shot.md)
* **Question Answering (質問応答) の各種分類**: [question_answering_types.md](./question_answering_types.md)
* **翻訳と要約のための LLM**: [llm_for_translation_and_summarization.md](./llm_for_translation_and_summarization.md)
* **モデルの可解釈性 (Model Interpretability)**: [model_interpretability.md](./model_interpretability.md)
* **`int2str` と pandas `apply` の活用**: [int2str_and_pandas_apply.md](./int2str_and_pandas_apply.md)
* **PAN-X のデータ構造と `defaultdict`**: [panx_structure_and_defaultdict.md](./panx_structure_and_defaultdict.md)
* **PAN-X ダウンサンプリングパイプライン**: [panx_downsampling_pipeline.md](./panx_downsampling_pipeline.md)
* **XTREME PAN-X データセットガイド**: [xtreme_pan_x_dataset_guide.md](./xtreme_pan_x_dataset_guide.md)
* **なぜ小規模言語モデル (SLM) の多言語NERが重要か**: [why_slm_multilingual_ner_matters.md](./why_slm_multilingual_ner_matters.md)
* **XLM-RoBERTa Token Classification コード解説**: [xlm_roberta_token_classification_code_breakdown.md](./xlm_roberta_token_classification_code_breakdown.md)
* **Transformers 2章 まとめコード**: [chap02_summary_and_final_code.md](./chap02_summary_and_final_code.md)
* **Transformers 4章 データセット準備解説**: [chap04_dataset_preparation_explained.md](./chap04_dataset_preparation_explained.md)
