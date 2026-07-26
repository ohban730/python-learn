"""
オライリー本『機械学習エンジニアのためのTransformers』第4章 4.1節
最終的に作成するデータセットパイプラインの完全実行スクリプト
"""

import pandas as pd
from collections import defaultdict
from datasets import load_dataset, DatasetDict

def main():
    print("=== 第4章 4.1節：最終的なデータセット作成パイプライン ===\n")

    # -------------------------------------------------------------------
    # 【ステップ1】 4言語の不均等ダウンサンプリングデータセット (panx_ch) の作成
    # -------------------------------------------------------------------
    langs = ["de", "fr", "it", "en"]
    fracs = [0.629, 0.229, 0.084, 0.059]  # スイスの言語分布
    panx_ch = defaultdict(DatasetDict)

    print("Step 1: 4言語のデータをロードしてスイスの比率でダウンサンプリング中...")
    for lang, frac in zip(langs, fracs):
        # 修正ポイント: 最新の Hugging Face Hub では google/xtreme を指定
        ds = load_dataset("google/xtreme", name=f"PAN-X.{lang}")
        for split in ds:
            panx_ch[lang][split] = (
                ds[split]
                .shuffle(seed=0)
                .select(range(int(frac * ds[split].num_rows)))
            )

    # 件数の確認
    print("\n--- 各言語の訓練データ件数 ---")
    element_counts = {lang: panx_ch[lang]["train"].num_rows for lang in langs}
    df_counts = pd.DataFrame(element_counts, index=["Number of training examples"])
    print(df_counts)
    print("⇒ 主力のドイツ語(de)が12,500件と圧倒的に多く、他言語が少量という不均衡データが完成！\n")

    # -------------------------------------------------------------------
    # 【ステップ2】 数値の ID (ner_tags) を人間が見やすい文字列タグ (tags) に変換
    # -------------------------------------------------------------------
    print("Step 2: 数値ID (ner_tags) を文字列タグ (tags) に変換する処理を追加中...")
    tags = panx_ch["de"]["train"].features["ner_tags"].feature
    print(f"定義されているタグ一覧: {tags.names}")

    # 数値IDを文字列にマッピングする関数
    def create_tag_names(batch):
        return {"tags": [tags.int2str(idx) for idx in batch["ner_tags"]]}

    # panx_ch 内の全言語・全分割に一括適用 (map)
    for lang in langs:
        panx_ch[lang] = panx_ch[lang].map(create_tag_names)

    # -------------------------------------------------------------------
    # 【ステップ3】 最終的なデータセットの1件を確認 (Pandasで可視化)
    # -------------------------------------------------------------------
    print("\nStep 3: 完成したデータセットの1件 (ドイツ語の訓練データ[0]) を表示:")
    de_example = panx_ch["de"]["train"][0]
    
    # 単語 (tokens) と 人間用タグ (tags) を表にして綺麗に表示
    df_sample = pd.DataFrame([de_example["tokens"], de_example["tags"]], index=["Tokens", "Tags"])
    print(df_sample.to_string())

    print("\n" + "=" * 70)
    print("【結論】4.1節で最終的に作ったもの:")
    print("  1. panx_ch: ドイツ語(学習用)と他3言語(テスト用)が詰まった言語別辞書データ")
    print("  2. 'tags' カラム: ['O', 'O', 'B-LOC', 'I-LOC'] などの人間が見やすいラベル列")
    print("=" * 70)

if __name__ == "__main__":
    main()
