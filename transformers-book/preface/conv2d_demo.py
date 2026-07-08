import torch
import torch.nn as nn

def main():
    print("=== PyTorch nn.Conv2d 動作確認デモ ===\n")

    # 1. 画像に見立てた 3x3 の極小テンソルを作成
    # Shape: (Batch=1, Channel=1, Height=3, Width=3)
    x = torch.tensor([[[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]]])

    print("--- 1. 入力テンソル (x) ---")
    print(f"Shape: {x.shape}")
    print(x[0, 0])  # バッチ0, チャネル0 のデータ
    print()

    # 2. Conv2d レイヤーの定義
    # 入力チャネル: 1, 出力チャネル: 1, カーネルサイズ: 2x2, バイアスなし
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)

    # 3. 理解しやすくするために、カーネル（重み）の値を手動で設定
    # フィルターの値: 左上 1.0, 右下 1.0, その他 0.0 (対角和フィルター)
    # Weight Shape: (out_channels=1, in_channels=1, kernel_height=2, kernel_width=2)
    custom_weight = torch.tensor([[[
        [1.0, 0.0],
        [0.0, 1.0]
    ]]])
    conv.weight.data = custom_weight

    print("--- 2. 設定したフィルターの重み (Weight) ---")
    print(f"Shape: {conv.weight.shape}")
    print(conv.weight.data[0, 0])
    print()

    # 4. 順伝播（畳み込み処理の実行）
    out = conv(x)

    print("--- 3. 畳み込み後の出力テンソル (Output) ---")
    print(f"Shape: {out.shape}")
    print(out[0, 0])
    print()

    # 5. 手動での計算確認
    # 入力テンソル x:
    # [ 1.0, 2.0, 3.0 ]
    # [ 4.0, 5.0, 6.0 ]
    # [ 7.0, 8.0, 9.0 ]
    #
    # フィルター:
    # [ 1.0, 0.0 ]
    # [ 0.0, 1.0 ]
    #
    # 左上部分 (2x2) の計算: (1.0 * 1.0) + (2.0 * 0.0) + (4.0 * 0.0) + (5.0 * 1.0) = 1.0 + 5.0 = 6.0
    # 右上部分 (2x2) の計算: (2.0 * 1.0) + (3.0 * 0.0) + (5.0 * 0.0) + (6.0 * 1.0) = 2.0 + 6.0 = 8.0
    # 左下部分 (2x2) の計算: (4.0 * 1.0) + (5.0 * 0.0) + (7.0 * 0.0) + (8.0 * 1.0) = 4.0 + 8.0 = 12.0
    # 右下部分 (2x2) の計算: (5.0 * 1.0) + (6.0 * 0.0) + (8.0 * 0.0) + (9.0 * 1.0) = 5.0 + 9.0 = 14.0
    print("--- 4. 手動計算による答え合わせ ---")
    manual_calc = torch.tensor([[
        [6.0, 8.0],
        [12.0, 14.0]
    ]])
    print(manual_calc)
    
    # PyTorchの出力と手動計算が一致しているかアサーション
    assert torch.allclose(out[0, 0], manual_calc), "計算結果が一致しません！"
    print("\n=> 一致確認成功！PyTorchのConv2dは想定通りに畳み込みを行っています。")

    print("\n==========================================")
    print("--- 5. パディング (Padding) とストライド (Stride) の影響実験 ---")
    print(f"元の画像サイズ: 3x3")
    
    # 実験A: padding=1 を設定（入力の周囲を0で囲むため、サイズが大きくなって出力も大きくなる）
    conv_pad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, bias=False)
    out_pad = conv_pad(x)
    print(f"A) padding=1 の時の出力 Shape (サイズ維持・拡張): {out_pad.shape}") # (1, 1, 4, 4)

    # 実験B: stride=2 を設定（フィルターを2マスずつスライドさせるため、出力サイズが半分程度に縮む）
    conv_stride = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False)
    out_stride = conv_stride(x)
    print(f"B) stride=2 の時の出力 Shape (解像度縮小):      {out_stride.shape}") # (1, 1, 1, 1)

if __name__ == "__main__":
    main()
