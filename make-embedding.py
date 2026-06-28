# Based on code from LLMs-from-scratch by Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Licensed under the Apache License, Version 2.0

import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# 1. テキストデータの準備
# --------------------------------------------------
# サンプルテキスト（sample.txtが存在すれば読み込み、無ければ長めのダミーテキストを使用）
file_path = "sample.txt"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
else:
    raw_text = (
        "In the component-based architecture, each component is responsible for its own state "
        "and user interface representation. When building large scale applications, LLMs "
        "can assist in automating repetitive coding tasks and optimizing data pipelines."
    )


# --------------------------------------------------
# 2. GPT用 PyTorch Dataset の定義
# --------------------------------------------------
class GPTDatasetV1(Dataset):
    """
    テキストを BPE (tiktoken) でトークンID化し、
    スライディングウィンドウ方式で入力 (input_ids) とターゲット (target_ids) のペアを作成する Dataset
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # テキスト全体を BPE でトークンIDのリストに変換
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 入力テキストのトークン数が最小長を満たしているかチェック
        assert len(token_ids) > max_length, f"テキストのトークン数({len(token_ids)})は max_length+1 ({max_length+1}) 以上である必要があります。"

        # スライディングウィンドウで max_length ごとに切り出し
        # target は 1 トークン分未来（右シフト）の系列を作成
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# --------------------------------------------------
# 3. DataLoader 作成関数
# --------------------------------------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Dataset を受け取り、ミニバッチ化して供給する PyTorch DataLoader を作成
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# --------------------------------------------------
# 4. 入力埋め込み (Input Embedding) パイプラインの動作検証
# --------------------------------------------------
if __name__ == "__main__":
    # パラメータ設定
    vocab_size = 50257  # GPT-2 の語彙数
    output_dim = 256    # 埋め込みベクトルの次元数 (本来のGPT-2 Smallは768次元)
    max_length = 4      # 1シーケンスあたりのトークン数 (コンテキスト長)

    # --- Step A: ミニバッチデータの取得 ---
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("=== Step A: トークンID (入力バッチ) ===")
    print("Inputs (Token IDs):\n", inputs)
    print("Inputs shape:", inputs.shape)  # 形状: [batch_size=8, max_length=4]

    # --- Step B: トークン埋め込み (Token Embedding) ---
    # 各トークンIDを output_dim 次元の分散表現ベクトルに変換
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    print("\n=== Step B: トークン埋め込み ===")
    print("Token embeddings shape:", token_embeddings.shape)  # 形状: [batch_size=8, max_length=4, output_dim=256]

    # --- Step C: 位置埋め込み (Positional Embedding) ---
    # シーケンス内の「位置 (0, 1, 2, 3...)」を表すベクトルを取得
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    print("\n=== Step C: 位置埋め込み ===")
    print("Positional embeddings shape:", pos_embeddings.shape)  # 形状: [max_length=4, output_dim=256]

    # --- Step D: 最終的な入力埋め込み (Input Embedding) ---
    # トークン埋め込みと位置埋め込みを足し合わせる（ブロードキャスト加算）
    input_embeddings = token_embeddings + pos_embeddings

    print("\n=== Step D: 最終入力埋め込み (Token + Positional) ===")
    print("Final input embeddings shape:", input_embeddings.shape)  # 形状: [batch_size=8, max_length=4, output_dim=256]