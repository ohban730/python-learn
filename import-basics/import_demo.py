import sys
import os
from pathlib import Path

print("==============================================================")
print("     Python 動的インポート & モジュール検索パス (sys.path) デモ")
print("==============================================================\n")

# このスクリプトがあるディレクトリの絶対パス
current_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# STEP 1: パス未登録状態でのインポート失敗テスト
# ------------------------------------------------------------
print("--- [STEP 1] 検索パス未登録でのインポート試行 ---")

# Pythonが自動で追加する「実行スクリプトのあるフォルダ」のパスを一時的に削除します
# (これにより、モジュールが見つからないエラー状態を意図的に再現します)
if current_dir in sys.path:
    sys.path.remove(current_dir)

try:
    print("[INFO] 'my_package.utils' から関数をインポートしようと試みています...")
    from my_package.utils import hello_from_package
except ModuleNotFoundError as e:
    print(f"[EXPECTED ERROR] インポートに失敗しました！")
    print(f"                 エラー内容: {e}")
    print("                 理由: Pythonの検索パスリスト (sys.path) に、")
    print("                       'my_package' が置かれているフォルダが登録されていないためです。\n")


# ------------------------------------------------------------
# STEP 2: sys.path.insert を用いた動的解決と成功テスト
# ------------------------------------------------------------
print("--- [STEP 2] sys.path.insert(0, ...) による動的解決 ---")

# pathlib.Path を使用して、自分自身から見た親（または目的のルート）ディレクトリを特定します
# __file__ は実行しているファイル自体のパスを表します
# resolve() で絶対パスにし、parents[0] でその1つ上のディレクトリ（import-basics）を取得します
target_path = Path(__file__).resolve().parents[0]

print(f"[INFO] 検索パスに追加するディレクトリ: {target_path}")

# sys.path のインデックス 0 (最優先位置) にパスを追加します
sys.path.insert(0, str(target_path))

try:
    print("[INFO] 再度インポートを試みています...")
    # パスが追加されたため、今度は成功します
    from my_package.utils import hello_from_package
    
    # 関数の呼び出し
    hello_from_package()
    print("--> [SUCCESS] 動的インポートに成功しました！\n")
except ModuleNotFoundError as e:
    print(f"[ERROR] パスを追加したにもかかわらずエラーになりました: {e}\n")


# ------------------------------------------------------------
# STEP 3: 現在の sys.path の検索優先順位リストを表示
# ------------------------------------------------------------
print("--- [STEP 3] 現在のモジュール検索パス (sys.path) の一覧 ---")
print("Pythonは、import文を見つけると以下のリストの『上から順番に』ファイルを探しに行きます。\n")

for i, path in enumerate(sys.path):
    # 見やすさのために先頭10件程度を表示
    if i < 8:
        marker = " <== [最優先 (追加したパス)]" if i == 0 else ""
        print(f"  [{i}] {path}{marker}")
    else:
        print(f"  ... (他 {len(sys.path) - 8} 件のシステムパスがあります)")
        break

print("\n==============================================================")
print("  要約:")
print("  * Pythonは import文があると sys.path のリストの上から順番に探す。")
print("  * 'ModuleNotFoundError' は、インポートしたい相手の『親フォルダ』が")
print("    sys.path に登録されていないために発生する。")
print("  * 'Path(__file__).resolve().parents[N]' でディレクトリの絶対パスを特定し、")
print("    'sys.path.insert(0, str(path))' で検索パスの最優先（先頭）に追加することで、")
print("    フォルダ階層が違っても確実にインポートできるようになる。")
print("==============================================================")
