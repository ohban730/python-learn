# --------------------------------------------------
# Pythonの iter() と next() の挙動を理解するデモ
# --------------------------------------------------

# 1. 元となるデータ（リスト）を用意します
# リストなどの「繰り返し可能なオブジェクト」を Iterable（イテラブル）と呼びます
fruits = ["apple", "banana", "cherry"]
print("=== 1. 元のリスト ===")
print(fruits)
print("-" * 50)

# 2. リストから「イテレータ（データを取り出す道具）」を作成します
# iter() 関数を使うことで、リストからデータを1つずつ取り出せる状態になります
fruits_iterator = iter(fruits)
print("\n=== 2. イテレータの作成 ===")
print("作成されたオブジェクト:", fruits_iterator)  # list_iterator オブジェクトが表示されます
print("-" * 50)

# 3. next() を使って、手動でデータを1つずつ取り出します
print("\n=== 3. next() による手動取り出し ===")

first = next(fruits_iterator)
print("1回目の next():", first)      # "apple"

second = next(fruits_iterator)
print("2回目の next():", second)    # "banana"

third = next(fruits_iterator)
print("3回目の next():", third)      # "cherry"
print("-" * 50)

# 4. データが空になった状態でさらに next() を呼ぶとどうなるか？
print("\n=== 4. データが空の状態で呼び出す ===")
try:
    # 4回目の取り出し（もう要素はありません）
    fourth = next(fruits_iterator)
except StopIteration:
    print("要素がもう無いため、StopIteration 例外（エラー）が発生しました！")
    print("Pythonのシステムは、このエラーを検知して『繰り返しが終了した』と判断します。")
print("-" * 50)

# 5. for ループの正体
# 実は、Pythonの for ループは裏側で自動的に iter() と next() を呼び出し、
# StopIteration が発生するまでループを回す処理を行っています。
print("\n=== 5. 参考: for ループの裏側の挙動 ===")
print("通常の for ループで出力:")
for item in fruits:
    print(" -", item)
