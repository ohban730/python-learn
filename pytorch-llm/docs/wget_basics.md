# wget コマンドの基礎知識と Windows での扱い

書籍の28ページでデータセットのダウンロードに使用されている **wget コマンド** の概要、基本的な使い方、および Windows 環境における注意点について解説します。

---

## 1. `wget` コマンドとは？

`wget`（ダブリューゲット）は、**インターネット上（WebサーバーやFTPサーバー）からファイルをダウンロードするためのコマンドラインツール**です。

GUI（ブラウザ）を使わずに、コマンドプロンプトやターミナルから直接ファイルをダウンロードする際によく使われます。
名前の由来は **World Wide Web (WWW)** + **get** からきています。

---

## 2. 基本的な使い方

最もシンプルな使い方は、コマンドの後にダウンロードしたいファイルのURLを指定するだけです。

```bash
wget https://example.com/data/train.txt
```
* **結果**: カレントディレクトリに `train.txt` という名前でファイルが保存されます。

### よく使われるオプション

| オプション | 説明 | 使用例 |
| :--- | :--- | :--- |
| **`-O`** (大文字のオー) | **保存するファイル名を指定**してダウンロードします。 | `wget -O my_dataset.txt https://example.com/raw.txt` |
| **`-P`** (大文字のピー) | **保存先のフォルダを指定**してダウンロードします。 | `wget -P ./data/ https://example.com/file.zip` |
| **`-c`** | 接続が切れた際、ダウンロードを**途中から再開（レジューム）**します。 | `wget -c https://example.com/huge_video.mp4` |
| **`-q`** (quiet) | 画面上にダウンロードの進捗ログを表示しません（スクリプト内などで便利）。 | `wget -q https://example.com/file.zip` |

---

## 3. ⚠️ Windows 環境における重要な注意点

`wget` は本来 Linux や macOS などの Unix 系 OS のコマンドであり、**Windows のコマンドプロンプトには標準で入っていません。**

### PowerShell での「偽 wget」に注意
Windows の PowerShell で `wget` と入力すると、一見動くように見えます。しかし、これは本物の `wget` ではなく、PowerShell の `Invoke-WebRequest` という別のコマンドに付けられた **「エイリアス（別名）」** です。

そのため、Linux用のオプション（例: `-O` や `-c`）をつけて実行すると、**エラーが発生して動きません。**

---

## 4. Windows でファイルをダウンロードする代替案

Windows で書籍のコマンド（ファイルのダウンロード）を再現する場合、以下の2つの方法がおすすめです。

### 代替案①：`curl` コマンドを使う（推奨）
現在の Windows 10 / 11 には、標準で本物の **`curl` (カール)** コマンドがインストールされています。`curl` は `wget` とほぼ同じ機能を持つツールです。

* **`wget` の代わりに `curl` を使うコード**:
  ```bash
  # -o (小文字) オプションで保存するファイル名を設定します
  curl -o train.txt https://huggingface.co/datasets/transformersbook/emotion-train-split/raw/main/train.txt
  ```

### 代替案②：Git Bash を使う
もし Git for Windows に付属している **Git Bash** ターミナルを使用している場合、その中では本物の `wget` コマンドが最初から使えるようになっています。
