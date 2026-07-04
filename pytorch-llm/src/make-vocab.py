# Based on code from LLMs-from-scratch by Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Licensed under the Apache License, Version 2.0

import re

# 1. サンプルテキストの読み込み
with open("sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. テキストの前処理（正規表現による分割・トークン化）
# 記号や空白で分割しつつ、キャプチャグループ () で記号も保持
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 3. 重複の除去と整列（語彙リストの作成）
# set() で重複を除き、sorted() でアルファベット順に並べ替え
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"語彙数 (Vocab Size): {vocab_size}\n")

# 4. 単語辞書 (Vocab) の作成
# 【Point: enumerate() の利用 その1】
# enumerate(iterable) は (インデックス番号, 要素) を順番に返す関数。
# ここでは各単語(token)に対して、連番の整数ID(integer)を割り当てた辞書を作成しています。
vocab = {token: integer for integer, token in enumerate(all_words)}

# 5. enumerate() と items() の動作確認
# 【Point: dict.items() と enumerate() の利用 その2】
# ・dict.items() : 辞書の (キー, 値) のペアをタプルとして返すメソッド
# ・enumerate()  : ループの処理回数（インデックス）を付与する関数
print("--- vocab.items() と enumerate() の組み合わせ確認 ---")
for i, item in enumerate(vocab.items()):
    # i    : enumerate が付与したインデックス (0, 1, 2...)
    # item : vocab.items() が返す (単語, ID) のタプル
    print(f"ループインデックス [{i:2d}] -> item: {item} (型: {type(item).__name__})")
    
    # 確認のため最初の10件を表示して終了
    if i >= 9:
        break

# 【参考: アンパック記述】
# 実際にコードを書く際は、item タプルを解体して以下のように書くのが一般的です。
# for i, (word, word_id) in enumerate(vocab.items()):
#     print(f"Index: {i}, Word: {word}, ID: {word_id}")