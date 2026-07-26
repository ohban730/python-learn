# 第4章 4.1節「データセット準備」の目的・全体像・解説

オライリー本第4章 4.1節（93〜98ページ）で **「最終的に何を作りたかったのか」** と **「プログラムが何を行っているのか」** をスッキリ解きほぐして解説します。

一括で動作確認できるスクリプトを作成しました：
* 📄 [chap04_dataset_pipeline.py](file:///C:/Users/owner/Documents/lab/llm-sandbox/huggingface-learn/chap04/chap04_dataset_pipeline.py)

---

## 1. 結論：4.1節で「最終的に作りたかった 2 つのもの」

4.1節のゴールは、モデルをファインチューニングする前準備として、以下の **2 つの成果物** を完成させることでした。

```text
 ［ 最終成果物 1: panx_ch ］
   ・ドイツ語 (de)  ── 12,580 件 (★これをモデルのファインチューニングに使用)
   ・フランス語 (fr) ──  4,580 件 (★学習させず、ゼロショットテストに使用)
   ・イタリア語 (it) ──  1,680 件 (★同上)
   ・英語 (en)      ──  1,180 件 (★同上)

 ［ 最終成果物 2: データセットへの 'tags' カラム追加 ］
   ・元のデータ : ner_tags = [0, 0, 5, 6]  (人間には何のタグか分からない)
   ・追加後     : tags     = ['O', 'O', 'B-LOC', 'I-LOC']  (人間が見て直感的に分かるラベル)
```

---

## 2. プログラムの 4 つのステップ解読

書籍のプログラムは、以下の 4 ステップ順で進行しています。

### 【Step 1】不均等ダウンサンプリング (`panx_ch` の作成)
スイスの人口比率に合わせ、主力のドイツ語を多め、他言語を少なめに抽出します。

```python
langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]
panx_ch = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    ds = load_dataset("google/xtreme", name=f"PAN-X.{lang}")
    for split in ds:
        panx_ch[lang][split] = (
            ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows)))
        )
```

### 【Step 2】数値ID (`ner_tags`) を文字ラベル (`tags`) に変換する関数の追加
元のデータには `[0, 0, 5, 6]` という数値しか入っていません。
`ClassLabel.int2str()` を使って `'B-LOC'` や `'I-LOC'` という文字列に変換する関数 `create_tag_names` を作り、`.map()` でデータセット全件に追加します。

```python
tags = panx_ch["de"]["train"].features["ner_tags"].feature

def create_tag_names(batch):
    return {"tags": [tags.int2str(idx) for idx in batch["ner_tags"]]}

# 全言語・全分割に 'tags' 列を一括追加
for lang in langs:
    panx_ch[lang] = panx_ch[lang].map(create_tag_names)
```

### 【Step 3】タグの出現頻度の確認 (集計)
データセット内に「人名 (PER)」「地名 (LOC)」「組織名 (ORG)」がどれくらいの割合で含まれているか、`Counter` を使って集計・確認します。

### 【Step 4】Pandas による結果の可視化
データセットの1件を取り出し、単語（`Tokens`）と対応するタグ（`Tags`）を上下に並べて綺麗に可視化します。

```python
de_example = panx_ch["de"]["train"][0]
pd.DataFrame([de_example["tokens"], de_example["tags"]], index=["Tokens", "Tags"])
```

**表示結果**:
```text
          0           1   2    3         4      5   6    7           8             9        10 11
Tokens  2.000  Einwohnern  an  der  Danziger  Bucht  in  der  polnischen  Woiwodschaft  Pommern  .
Tags        O           O   O    O     B-LOC  I-LOC   O    O       B-LOC         B-LOC    I-LOC  O
```

---

## 3. なぜこれらが必要だったのか？（理解負債の解消）

1. **`panx_ch`（不均等データ）が必要な理由**:
   「ドイツ語（12,500件）だけで学習させたモデルが、未学習のフランス語（4,500件）や英語（1,100件）でどれくらい固有表現（人名・地名）を言い当てられるか？」という **ゼロショット言語間転移の評価環境を作るため** です。
2. **`tags` カラム（文字ラベル）が必要な理由**:
   モデルの予測結果やデータの中身を評価・分析する際、`5` や `6` という数値のままでは人間がエラー分析できないためです。

---

## 4. まとめ

4.1節は、**「実務を模した多言語テスト用データセット `panx_ch` を作り、人間が見やすいラベル `tags` を付与して準備を完了させる章」** でした。
