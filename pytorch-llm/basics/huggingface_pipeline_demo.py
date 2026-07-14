from transformers import pipeline

def main():
    # 1. デフォルトモデル (英語の感情分析) での実行
    print("--- 1. デフォルトモデル（英語の感情分析）のテスト ---")
    # modelを指定しない場合、自動的に distilbert-base-uncased-finetuned-sst-2-english が使われます
    nlp_en = pipeline("text-classification")
    
    res_en1 = nlp_en("I love coding with Python!")
    print(f"Input: 'I love coding with Python!'\n  -> Output: {res_en1}")
    
    res_en2 = nlp_en("I hate waiting in long lines.")
    print(f"Input: 'I hate waiting in long lines.'\n  -> Output: {res_en2}")

    # 2. 複数テキストの一括処理
    print("\n--- 2. 複数テキストの一括処理 ---")
    texts = [
        "This movie was absolute garbage.",
        "It was okay, but a bit too long.",
        "Amazing performances, highly recommended!"
    ]
    results = nlp_en(texts)
    for t, r in zip(texts, results):
        print(f"Text: '{t}'\n  -> Result: {r}")

    # 3. 全クラスのスコア（確率）を取得する例
    print("\n--- 3. 全クラスのスコアを取得 (top_k=None) ---")
    # top_k=None にすることで、上位だけでなくすべてのラベルの確率が出力されます
    nlp_en_all = pipeline("text-classification", top_k=None)
    res_all = nlp_en_all("The service was not bad, but the food was cold.")
    print(f"Input: 'The service was not bad, but the food was cold.'\n  -> Output: {res_all}")

    # 4. 多言語感情分析モデル（日本語対応・追加ライブラリ不要）のテスト
    # モデル: lxyuan/distilbert-base-multilingual-cased-sentiments-student
    print("\n--- 4. 日本語テキストに対する多言語モデルのテスト ---")
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    
    try:
        # model 引数で Hugging Face Hub のモデル名を明示的に指定します
        nlp_ja = pipeline("text-classification", model=model_name)
        ja_texts = [
            "この映画は本当に最高でした！感動しました。",
            "普通の作品ですね。可もなく不可もなく。",
            "最悪の気分です。もう二度と買いません。"
        ]
        ja_results = nlp_ja(ja_texts)
        for t, r in zip(ja_texts, ja_results):
            print(f"Text: '{t}'\n  -> Result: {r}")
    except Exception as e:
        print(f"日本語モデルの実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
