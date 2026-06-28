# --------------------------------------------------
# assert と raise の違いを確認するサンプルコード
# --------------------------------------------------
# 【概要】
# - assert: 開発時の「デバッグ用・内部前提条件の検証」。
#           条件が False の場合に AssertionError を送出する。
#           ※ python -O (最適化モード) で実行すると無視されるため、
#             プロダクションコードでのユーザー入力チェック等に使ってはいけない。
#
# - raise : 「本番運用・仕様上のエラー処理（通常の例外処理）」。
#           ValueError や TypeError など、状況に応じた明示的な例外を送出する。
#           最適化モードでも無視されず、常に確実に実行される。
# --------------------------------------------------


def divide_with_raise(a: int, b: int) -> float:
    """
    raise を使った入力検証の例
    通常のアプリケーション仕様として「0での割り算」や「不正な型」を制限する
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers (int or float).")
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


def process_scores_with_assert(scores: list):
    """
    assert を使った内部前提条件チェックの例
    「関数内部のロジック上、scores は絶対に空リストではないはずだ」という内部状態を検証
    """
    # 内部前提条件の検証（プログラムのバグを早期発見するためのもの）
    assert len(scores) > 0, "Programming Error: scores list is empty (internal bug)."
    
    total = sum(scores)
    return total / len(scores)


if __name__ == "__main__":
    print("=== 1. raise Exception Check ===")
    try:
        result = divide_with_raise(10, 0)
    except ValueError as e:
        print(f"[Caught] raise Exception:\n  -> {type(e).__name__}: {e}")

    try:
        result = divide_with_raise(10, "5")
    except TypeError as e:
        print(f"[Caught] raise Exception:\n  -> {type(e).__name__}: {e}")

    print("\n" + "=" * 40 + "\n")

    print("=== 2. assert AssertionError Check ===")
    try:
        empty_list = []
        average = process_scores_with_assert(empty_list)
    except AssertionError as e:
        print(f"[Caught] assert Exception:\n  -> {type(e).__name__}: {e}")

    print("\n" + "=" * 40 + "\n")
    print("=== Summary ===")
    print("- raise : Standard error handling (user input errors, runtime validation).")
    print("- assert: Developer debugging check (verify internal invariants/bugs).")
