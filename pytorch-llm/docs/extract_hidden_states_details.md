# `extract_hidden_states` 関数ラップ後の処理の流れと仕組み

書籍の44〜45ページ「関数でラップした後の説明（`extract_hidden_states` の定義から `map` による一括抽出まで）」で何が行われているのか、一連の処理の流れとポイントをわかりやすく解説します。

---

## 1. 全体の流れ

ここで行っているのは、**「これまで手動でテストした『隠れ状態の抽出処理』を自動化し、データセット全体のすべての文章に対して一括で実行する」** という作業です。

```
 [データセット全体 (数万件)] 
        │
        ▼ 1. set_format("torch") で PyTorchテンソル型に変換
 [PyTorchで処理できる形式のデータセット] ── (1000件ずつバッチ化)
        │
        ├──────────────────────────┐ (データセットの map 処理)
        ▼                          ▼
 [バッチ1 (1000件)]         [バッチ2 (1000件)] ...
        │                          │
        ▼                          ▼
 🌟【 extract_hidden_states 】   🌟【 extract_hidden_states 】  (GPUで高速にベクトル抽出)
        │                          │
        ▼                          ▼
  (CPU/NumPy配列に戻す)       (CPU/NumPy配列に戻す)
        │                          │
        └───────────┬──────────────┘
                    ▼
 [元のデータセットに "hidden_state" 列が追加される]
```

---

## 2. ポイント①：`extract_hidden_states` 関数の中身の役割

手動でやった時と比べ、この関数には **Hugging Face Datasets で一括処理（map）を行うための重要なルール**が2つ追加されています。

### ① モデルに入力できる列だけを GPU に送る
```python
inputs = {k: v.to(device) for k, v in batch.items()
          if k in tokenizer.model_input_names}
```
データセットには、モデルの入力に必要のない `"text"`（生の文章）や `"label"`（正解ID）などの列も含まれています。これらをモデルに入力しようとするとエラーになります。
そのため、`tokenizer.model_input_names`（モデルが期待する入力名、すなわち `input_ids` と `attention_mask`）に含まれるキーだけを選別し、GPU（`to(device)`）に転送しています。

### ② 計算したベクトルを CPU と NumPy 配列に戻して返す
```python
return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
```
ここが非常に重要です。
* モデルが計算した結果（`last_hidden_state`）は、**GPU（CUDA）上の PyTorch テンソル**です。
* しかし、データセットの `map()` メソッドは、**GPU上のテンソルをそのまま受け取ってデータセットに書き戻すことができない**仕様になっています。
* そのため、一度データをGPUからCPUに引き戻し（`.cpu()`）、さらに汎用的な **`NumPy` 配列** に変換（`.numpy()`）してからデータを返しています。

---

## 3. ポイント②：なぜ `set_format("torch", ...)` を行うのか？

`map` で一括処理を実行する直前に、以下のコードを実行しています。

```python
emotions_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
```

* **理由**: デフォルト状態のデータセットの中身は、PyTorchのデータではなく単なる「Pythonのリスト（数値の羅列）」や「Arrow形式」で格納されています。
* モデル（ニューラルネットワーク）に入力するためには、これらが **PyTorch の `Tensor` 型** になっている必要があります。
* `set_format("torch")` を実行することで、指定した列（`input_ids` や `attention_mask` など）を、裏で自動的に PyTorch テンソルとして扱えるようにフォーマットを切り替えています。

---

## 4. 補足：`tokenizer.model_input_names` とデータ型の関係

ユーザー様から「すでに Tensor 型なのでは？ `if k in tokenizer.model_input_names` はその判定用？」という疑問が生まれることがありますが、実際には以下のような違いがあります。

### ① `tokenizer.model_input_names` は「名前（文字列）」のリストです
この変数の中身は、単なる文字列のリスト（`['input_ids', 'attention_mask']`）です。
`if k in ...` は、「データの型がTensorかどうか」を調べているのではなく、**「辞書のキー（文字列）が、モデルの受け入れ可能な『名前（キー）』と一致しているか」** という文字列の一致チェックを行っているだけです。

### ② `set_format` を通す前は、ただの「Pythonのリスト(list)」です
データセット（`emotions_encoded`）に保存されている段階では、データはメモリ節約やシリアライズのために、単なる `[101, 1045, 2134, ...]` という**普通の Python のリスト（数値の配列）**として保存されています。

もし `set_format("torch")` を実行せずに、`v.to(device)` を呼び出すと、Python は以下のエラーを吐いてクラッシュします。
```text
AttributeError: 'list' object has no attribute 'to'
（リスト型オブジェクトには 'to' メソッドはありません）
```

そのため、`set_format("torch")` によってデータセットから取り出す際に**自動的に `torch.Tensor` 型に化けさせる（フォーマット変換する）**という前処理が必須になります。テンソル型になって初めて、`v.to(device)`（GPUへの転送メソッド）が実行可能になります。
