import tiktoken

# --------------------------------------------------
# Byte-Pair Encoding (BPE) と tiktoken の基本例
# --------------------------------------------------
# BPE (Byte-Pair Encoding) は、頻出する文字や文字列のペアを繰り返し統合していくサブワード分割アルゴリズムです。
# 未知語 (OOV: Out-of-Vocabulary) であっても、より細かい単位（サブワードやバイト）に分割してエンコードできます。

# 1. GPT-2 用の BPE エンコーディングを取得
tokenizer = tiktoken.get_encoding("gpt2")

# 2. テキストの用意
# "pkewets" や "iser" のような辞書に登録されていない未知語も、BPEによって複数のサブワードに分解されます
text = "pkewets iser <|endoftext|>"

# 3. エンコード（テキスト -> トークンIDのリスト）
# allowed_special: 通常、特殊トークン（<|endoftext|>など）はセキュリティ上の理由でエラーになりますが、
#                  許可したい特殊トークンを集合(set)で指定することでエンコード可能になります
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print("--- 1. Token IDs (Encode Result) ---")
print(integers)

# 各トークンIDがどのサブワードに対応しているかを可視化
print("\n--- 2. Subword Breakdown ---")
for token_id in integers:
    # バイト列/文字列として復元して表示
    subword = tokenizer.decode([token_id])
    print(f"ID: {token_id:5d}  ->  '{subword}'")

# 4. デコード（トークンIDのリスト -> 元のテキスト）
strings = tokenizer.decode(integers)

print("\n--- 3. Restored Text (Decode Result) ---")
print(strings)