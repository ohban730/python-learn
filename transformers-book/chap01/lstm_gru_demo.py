import torch
import torch.nn as nn

def main():
    print("=== PyTorch LSTM vs GRU API 動作比較デモ ===\n")

    # 1. 共通の入力系列データの定義 (Batch=1, Sequence=3, Input_Size=3)
    x = torch.tensor([[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]])

    print("--- 1. 入力データ (x) ---")
    print(f"Shape: {x.shape}")
    print(x[0])
    print()

    # ====================================================
    # 2. LSTM (Long Short-Term Memory) の処理
    # ====================================================
    print("--- 2. nn.LSTM の挙動 ---")
    lstm = nn.LSTM(input_size=3, hidden_size=2, num_layers=1, batch_first=True)
    
    # LSTM の順伝播
    # 返り値は: output, (hn, cn)
    #   hn: 最終時刻の隠れ状態 (Short-term memory)
    #   cn: 最終時刻のセル状態 (Long-term memory)
    output_lstm, (hn_lstm, cn_lstm) = lstm(x)

    print(f"LSTM output Shape (全ステップの隠れ状態): {output_lstm.shape}")
    print(output_lstm[0].detach())
    print(f"LSTM hn Shape     (最終隠れ状態 / 短期記憶): {hn_lstm.shape}")
    print(hn_lstm[0].detach())
    print(f"LSTM cn Shape     (最終セル状態 / 長期記憶): {cn_lstm.shape}")
    print(cn_lstm[0].detach())
    print()

    # ====================================================
    # 3. GRU (Gated Recurrent Unit) の処理
    # ====================================================
    print("--- 3. nn.GRU の挙動 ---")
    gru = nn.GRU(input_size=3, hidden_size=2, num_layers=1, batch_first=True)

    # GRU の順伝播
    # 返り値はシンプルに: output, hn  (セル状態 cn は存在しない)
    output_gru, hn_gru = gru(x)

    print(f"GRU output Shape (全ステップの隠れ状態): {output_gru.shape}")
    print(output_gru[0].detach())
    print(f"GRU hn Shape     (最終隠れ状態):           {hn_gru.shape}")
    print(hn_gru[0].detach())
    print()

    # ====================================================
    # 4. まとめと違いの確認
    # ====================================================
    print("--- 4. API設計上の決定的な違い ---")
    print("【戻り値の違い】")
    print("・LSTM: output, (hn, cn)  <- タプルの中に「隠れ状態」と「セル状態」の2つが返る")
    print("・GRU : output, hn        <- シンプルに「隠れ状態」のみが返る (Cell Stateがないため)")
    print()
    print("【最終ステップの出力との一致確認】")
    
    # LSTMの最終ステップの出力(output_lstm[0, -1])と最終隠れ状態(hn_lstm[0, 0])が一致することを確認
    assert torch.allclose(output_lstm[0, -1], hn_lstm[0, 0]), "LSTMの出力とhnが不一致です"
    # GRUの最終ステップの出力(output_gru[0, -1])と最終隠れ状態(hn_gru[0, 0])が一致することを確認
    assert torch.allclose(output_gru[0, -1], hn_gru[0, 0]), "GRUの出力とhnが不一致です"
    
    print("=> どちらのモデルも、outputの最後の時刻のベクトルは、最終隠れ状態(hn)と完全に一致します。")

if __name__ == "__main__":
    main()
