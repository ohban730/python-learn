# XLMRobertaForTokenClassification コードの1行ごとの解読ガイド

オライリー本第4章（104〜105ページ）に登場する **トークン分類用（NER用）のカスタムモデル定義コード** について、難しく見えている理由とコード1行ずつの動作を噛み砕いて解説します。

---

## 1. 結論：本質的な処理は「たった 3 行」だけ！

コードが長く難しく見える最大の理由は、**「Hugging Face の `Trainer` や `pipeline` と連携するためのフレームワークのルール（お約束コード）」** が周囲にたくさん書かれているからです。

モデル内部で行われている本質的な計算は、実は以下の **たった 3 行** だけです。

```python
# 1. 文章をボディ(XLM-RoBERTa)に通して、全単語のベクトルを取り出す
outputs = self.roberta(input_ids, attention_mask=attention_mask, ...)

# 2. 全単語のベクトルを分類ヘッド (nn.Linear) に流してラベルスコアを計算する
logits = self.classifier(outputs[0])

# 3. 正解ラベルとの誤差 (Loss) を計算する
loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
```

---

## 2. コード1行ずつの完全分解解説

### 部屋①：`__init__` メソッド (パーツの準備部屋)

```python
class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # ① モデル本体（ボディ）の作成
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # ② トークン分類ヘッドの作成
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # ③ 重みの初期化
        self.init_weights()
```

* **`add_pooling_layer=False` の意味**:
  `[CLS]` トークン1つだけにまとめず、**「全単語（全トークン）のベクトルをそのまま残して出力してね！」** という重要な指示です。
* **`self.classifier = nn.Linear(config.hidden_size, config.num_labels)`**:
  これが **分類ヘッド** の実体です。768次元の単語ベクトルを受け取り、7つの感情・カテゴリラベル（`B-PER`, `B-LOC` など）のスコアに変換します。

---

### 部屋②：`forward` メソッド (実際の計算実行部屋)

```python
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
    # ① ボディ（XLM-RoBERTa）にテキストを入力し、全単語のベクトル表現を取得
    outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
    
    # ② ドロップアウトを適用 (outputs[0] は [batch_size, seq_len, 768] の全単語ベクトル)
    sequence_output = self.dropout(outputs[0])
    
    # ③ 分類ヘッド (nn.Linear) に一気に通し、全単語分のラベルスコアを算出
    logits = self.classifier(sequence_output)
    
    # ④ 損失 (Loss) の計算
    loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        # Shapeを [バッチ数×単語数, 7クラス] と [バッチ数×単語数] に平坦化して誤差計算
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
    # ⑤ Hugging Face 専用の出力オブジェクト形式にして返す
    return TokenClassifierOutput(
        loss=loss, 
        logits=logits, 
        hidden_states=outputs.hidden_states, 
        attentions=outputs.attentions
    )
```

---

## 3. 難解に見える `logits.view(-1, self.num_labels)` のトリック

`loss_fct` の行で登場する `.view(-1, ...)` は、**全単語のデータを一列に平らに並べ替える操作** です。

```text
 ［ logits の元の形 ］: [ 2文章,  5単語,  7クラスのスコア ]  (3次元)
                            │  .view(-1, 7) で平らに伸ばす！
                            ▼
 ［ logits.view(-1, 7) ］: [ 10単語分, 7クラスのスコア ]      (2次元)
```

PyTorch の `nn.CrossEntropyLoss` は「単語数 × クラス数」の 2 次元行列を受け取るルールになっているため、`.view(-1, ...)` を使って「2文章×5単語 ＝ 計10単語」を一列に伸ばしてまとめて損失計算を行っています。

---

## 4. まとめ

1. **`RobertaModel(..., add_pooling_layer=False)`**: 全単語のベクトルを取り出すボディ。
2. **`self.classifier = nn.Linear(768, 7)`**: 768次元を 7種類のラベルスコアに変える頭（ヘッド）。
3. **`logits = self.classifier(sequence_output)`**: 全単語のベクトルに分類ヘッドを一斉に適用する部分。
4. **`TokenClassifierOutput(...)`**: Hugging Face の `Trainer` が読み取れるように辞書風オブジェクトに梱包して返す「お約束の戻り値」。
