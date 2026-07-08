import torch
import torch.nn as nn

def main():
    print("=== PyTorch nn.RNN 動作確認・手動計算デモ ===\n")

    # 1. 系列データの作成 (例: 3単語からなる文)
    # Shape: (Batch=1, Sequence_Length=3, Input_Size=3)
    # 単語0: [1.0, 2.0, 3.0], 単語1: [4.0, 5.0, 6.0], 単語2: [7.0, 8.0, 9.0]
    x = torch.tensor([[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]])

    print("--- 1. 入力系列 (x) ---")
    print(f"Shape: {x.shape}")
    for t in range(x.shape[1]):
        print(f"時刻 t={t} の入力: {x[0, t]}")
    print()

    # 2. RNNモデルの定義
    # 入力特徴量: 3次元, 隠れ状態: 2次元, 1レイヤー
    rnn = nn.RNN(input_size=3, hidden_size=2, num_layers=1, batch_first=True)

    # 3. 順伝播の実行
    # rnn(x) は「全時刻の出力(output)」と「最終時刻の隠れ状態(hn)」を返します
    output, hn = rnn(x)

    print("--- 2. PyTorchによるRNN出力 ---")
    print(f"output Shape (全時刻の隠れ状態): {output.shape}")
    print(output[0])
    print(f"hn Shape (最終時刻の隠れ状態):     {hn.shape}")
    print(hn[0])
    print()

    # 4. 手動計算による答え合わせ (時刻 t=0 と t=1 の隠れ状態を再現)
    # RNNのパラメータを取得
    w_ih = rnn.weight_ih_l0  # 入力 -> 隠れ層の重み (Shape: 2x3)
    w_hh = rnn.weight_hh_l0  # 隠れ層 -> 隠れ層の重み (Shape: 2x2)
    b_ih = rnn.bias_ih_l0    # 入力 -> 隠れ層のバイアス (Shape: 2)
    b_hh = rnn.bias_hh_l0    # 隠れ層 -> 隠れ層のバイアス (Shape: 2)

    # 初期状態 h_{-1} はすべて 0 (Shape: 2)
    h_prev = torch.zeros(2)

    print("--- 3. 隠れ状態のステップ別・手動計算 ---")
    
    # 時刻 t=0 の計算
    # h_0 = tanh(W_ih * x_0 + b_ih + W_hh * h_{-1} + b_hh)
    x_0 = x[0, 0]
    h_0_manual = torch.tanh(
        torch.mv(w_ih, x_0) + b_ih + 
        torch.mv(w_hh, h_prev) + b_hh
    )
    print(f"時刻 t=0 の手動計算 h_0: {h_0_manual}")
    print(f"PyTorchの出力 output[0, 0]: {output[0, 0].detach()}")
    
    # 時刻 t=1 の計算
    # h_1 = tanh(W_ih * x_1 + b_ih + W_hh * h_0 + b_hh)
    x_1 = x[0, 1]
    h_1_manual = torch.tanh(
        torch.mv(w_ih, x_1) + b_ih + 
        torch.mv(w_hh, h_0_manual) + b_hh
    )
    print(f"時刻 t=1 の手動計算 h_1: {h_1_manual}")
    print(f"PyTorchの出力 output[0, 1]: {output[0, 1].detach()}")
    print()

    # 5. アサーションによる一致確認
    # (浮動小数点の誤差を許容して比較)
    assert torch.allclose(output[0, 0], h_0_manual), "t=0 の計算結果が不一致です"
    assert torch.allclose(output[0, 1], h_1_manual), "t=1 の計算結果が不一致です"
    assert torch.allclose(output[0, 2], hn[0, 0]), "最終ステップとhnが不一致です"

    print("=> すべてのステップで手動計算とPyTorchの出力が一致しました！")
    print("RNNが「過去の隠れ状態」を引き継ぎながら計算している仕組みが確認できます。")

if __name__ == "__main__":
    main()
