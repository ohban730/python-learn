# Elasticsearch はライブラリではなく「独立したサーバー」である理由の解説

> **📄 出典・参考情報**
> * **参考書籍**: 『機械学習エンジニアのためのTransformers』 (オライリー・ジャパン発行, ISBN: 978-4-87311-995-3)  
> * **該当箇所**: 7.1.3.1 節「ドキュメントストアの初期化」 (pp.192-193)  
> * **著者**: Lewis Tunstall, Leandro von Werra, Thomas Wolf / 訳: 中山 光樹  

---

## 💡 1. 結論：ライブラリではなく「独立して動くデータベース（サーバー）」です！

Elasticsearch は、Python の `numpy` や `pandas` のような「Pythonコード内で完結するライブラリ」ではなく、**MySQL や PostgreSQL と同種の「独立したデータベース・サーバーソフトウェア」** です。

```text
 ［ 通常の Python ライブラリ (numpy / pandas) ］
   ・Python のプロセス内部で動くツール群

 ［ Elasticsearch ］
   ・Python とは完全に独立して単独で起動している「専用サーバー (port 9200)」
   ・Python からは HTTP 通信 (API) を使ってリモコン操作する
```

---

## 🔍 2. なぜ 193 ページで `tar.gz` をダウンロードして起動しているのか？

オライリー本 193 ページのコードを見ると、`pip install` ではなく以下のようなダウンロード＆起動処理を行っています。

```python
# 1. サーバー本体のプログラム (tar.gz) をダウンロードして解凍
!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2...
!tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz

# 2. 独立したバックグラウンドプロセスとしてサーバーを起動！
es_server = Popen(args=['elasticsearch-7.9.2/bin/elasticsearch'], ...)
```

これは、**「Python の外側で、検索エンジン専用のデータベースサーバーを起動させている」** ためです。

---

## 🔌 3. 混同しやすい「Python の elasticsearch パッケージ」の正体

Python で `pip install elasticsearch` を実行して以下のようにコードを書くことがあります：

```python
from elasticsearch import Elasticsearch

# 9200番ポートで起動している Elasticsearch サーバーに接続する
es = Elasticsearch("http://localhost:9200")
```

この Python パッケージは、Elasticsearch 本体ではなく **「別で動いている Elasticsearch サーバーと通信するためのリモコン（クライアントライブラリ）」** です。

---

## 📝 まとめ

* **Python ライブラリ**: Python の中で動くプログラムパーツ（例: `scikit-learn`）。
* **Elasticsearch 本体**: 単独で起動してデータを管理する**検索用データベースサーバー**。
* **通信の仕組み**: Python は HTTP リクエスト（`localhost:9200`）を通じて、Elasticsearch サーバーに「データを保存して」「検索して」と指示を出します。
