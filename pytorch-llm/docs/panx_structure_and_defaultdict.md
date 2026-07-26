# PAN-X データセットの構造と `defaultdict(DatasetDict)` の仕組み

`ds = load_dataset(...)` で返ってくるデータセットの内部構造と、Python の `defaultdict(DatasetDict)` が行っている初期化の仕組みについて解説します。

---

## 1. `ds` (PAN-X データセット) の内部構造

`ds = load_dataset("google/xtreme", name="PAN-X.de")` を実行した際、返ってくる変数 `ds` の中身は以下のような **ネストされた辞書構造 (DatasetDict)** になっています。

```text
 DatasetDict ({
     'train':       Dataset ({ features: ['tokens', 'ner_tags', 'langs'], num_rows: 20000 }),
     'validation':  Dataset ({ features: ['tokens', 'ner_tags', 'langs'], num_rows: 10000 }),
     'test':        Dataset ({ features: ['tokens', 'ner_tags', 'langs'], num_rows: 10000 })
 })
```

### 1件のサンプルデータ (`ds['train'][0]`) の具体的な中身

例えば、訓練データの先頭1件（`ds["train"][0]`）を取り出すと、以下のような 3 つのリストが入っています。

```python
{
    'tokens':   ['als', 'Teil', 'der', 'Savoyer', 'Voralpen', 'im', 'Osten', '.'],
    'ner_tags': [  0,     0,     0,       5,         6,       0,      0,     0 ],
    'langs':    [ 'de',  'de',  'de',    'de',      'de',    'de',   'de',  'de']
}
```

* **`tokens`**: 文章を単語ごとに分けた文字列のリスト。
* **`ner_tags`**: 各単語が「人名・場所・組織」のどれに該当するかを示す数値IDのリスト。
* **`langs`**: 言語コード（ここではすべて `'de'`（ドイツ語））。

---

### `ner_tags` の数値IDとラベル名の対応表 (全 7 種類)

数値の `0` や `5` や `6` は、以下のラベルに対応しています：

| 数値 ID | ラベル名 | 意味 |
| :--- | :--- | :--- |
| **`0`** | **`O`** | 固有表現ではない（普通の単語） |
| **`1`** | **`B-PER`** | **人名 (Person)** の先頭単語 |
| **`2`** | **`I-PER`** | **人名 (Person)** の2単語目以降 |
| **`3`** | **`B-ORG`** | **組織名 (Organization)** の先頭単語 |
| **`4`** | **`I-ORG`** | **組織名 (Organization)** の2単語目以降 |
| **`5`** | **`B-LOC`** | **場所名 (Location)** の先頭単語 （例: `Savoyer`） |
| **`6`** | **`I-LOC`** | **場所名 (Location)** の2単語目以降 （例: `Voralpen`） |

---

## 2. `panx_ch = defaultdict(DatasetDict)` は何を初期化しているのか？

### 💡 一言で言うと：二重の辞書を「KeyErrorを出さずに自動作成する空の箱」です

通常の Python 辞書（`dict`）で以下のように書くとエラーになります：

```python
panx_ch = {}
panx_ch["de"]["train"] = データ  # ★ KeyError! ("de" というキーがまだ作られていないため)
```

しかし、`defaultdict(DatasetDict)` を使って初期化しておくと、**まだ作られていないキー（例: `"de"`）にアクセスした瞬間に、自動的に空の `DatasetDict()` オブジェクトを裏で新しく生成** してくれます。

### 処理が進んだ後の `panx_ch` の完成イメージ

ループ処理が終わると、`panx_ch` は各言語のデータをきれいに整理した **言語別二重辞書** になります。

```python
panx_ch = {
    "de": DatasetDict({'train': ドイツ語訓練データ, 'validation': ..., 'test': ...}),
    "fr": DatasetDict({'train': フランス語訓練データ, 'validation': ..., 'test': ...}),
    "it": DatasetDict({'train': イタリア語訓練データ, 'validation': ..., 'test': ...}),
    "en": DatasetDict({'train': 英語訓練データ, 'validation': ..., 'test': ...}),
}
```

これで `panx_ch["de"]["train"]` と書くだけで、「ドイツ語の訓練データ」へ即座にアクセスできるようになります。
