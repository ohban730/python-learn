# XTREME / PAN-X データセットのエラー解決と背景知識

オライリー本第4章（多言語の固有表現認識）で発生した `HfUriError` エラーの原因と解決方法、および登場するデータセット **XTREME** や **PAN-X** の正体と目的を解説します。

---

## 1. エラーの原因と修正方法 (`HfUriError: Invalid HF URI`)

### 原因
Hugging Face Hub の仕様変更（セキュリティ強化およびリポジトリ名の `ユーザー名/リポジトリ名` への統一）により、かつての短縮名 `"xtreme"` 直接指定が廃止・無効化されたことが原因です。

### 修正方法
データセットの指定を、提供元であるネームスペース付きの **`"google/xtreme"`** に変更します。

```python
# 旧コード (エラーになる):
xtreme_subsets = get_dataset_config_names("xtreme")

# 新コード (解決):
xtreme_subsets = get_dataset_config_names("google/xtreme")
```

今後のコードでデータセットをダウンロードする際も、同様に `"google/xtreme"` を指定します：

```python
from datasets import load_dataset

# ドイツ語の PAN-X データをロードする場合の例
element = load_dataset("google/xtreme", name="PAN-X.de")
```

---

## 2. PAN-X とは何か？（データの目的）

### 💡 XTREME ベンチマークとは？
Google などが作成した、**多言語Transformerモデルの言語間転移（Cross-lingual Transfer）能力を測定するための総合テスト問題集（ベンチマーク）** です。

### 💡 PAN-X (WikiANN) データセットとは？
XTREME ベンチマークの中に含まれている、**多言語の固有表現抽出（NER: Named Entity Recognition）用データセット** です。

Wikipedia の各言語記事から抽出されており、文章中の単語に以下のタグが付与されています：

* `PER` (Person / 人名)
* `ORG` (Organization / 組織名)
* `LOC` (Location / 地名・場所)

サブセット名として `"PAN-X.de"` (ドイツ語), `"PAN-X.fr"` (フランス語), `"PAN-X.it"` (イタリア語), `"PAN-X.en"` (英語) などが用意されています。

---

## 3. なぜ PAN-X をダウンロードするのか？ (第4章のシナリオ)

第4章では、**「スイスの顧客（ドイツ語・フランス語・イタリア語・英語の4つの言語が飛び交う環境）」** に対応する AI システムの構築を想定しています。

通常であれば、4言語それぞれの学習データを大量に集めて学習させる必要がありますが、多言語モデル（XLM-RoBERTa など）を使うと以下の実験が可能になります：

```text
 ［ ドイツ語 (PAN-X.de) のデータだけでモデルをファインチューニング ］
                         │
                         ▼ (一切学習させていない他言語へ適用)
 ［ フランス語 (PAN-X.fr) や イタリア語 (PAN-X.it) のテスト文章を入力！ ］
                         │
                         ▼
 ［ 結果 ］: 未学習の言語であっても、人名や場所を正しく認識できる！ (ゼロショット転移)
```

この **「1つの言語で学習させて、他の言語へそのまま横流しして機能するか（ゼロショット言語間転移）」** を検証・評価するために、PAN-X の学習データと評価データをダウンロードして実験を行っています。
