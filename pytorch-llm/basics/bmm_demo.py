import torch

def main():
    print("=== PyTorch torch.bmm (バッチ行列積) 動作確認デモ ===\n")

    # 1. バッチサイズ 2 (データが2件分ある)
    batch_size = 2
    seq_len = 3
    hidden_dim = 4
    
    # 2. ２つの「3次元テンソル (行列の束)」を作成
    torch.manual_seed(42)

    # 第1引数 (Q): (batch_size=2, seq_len=3, hidden_dim=4)
    #   -> データ0用の3x4行列 と データ1用の3x4行列 の束
    Q = torch.randn(batch_size, seq_len, hidden_dim)

    # 第2引数 (K^T): (batch_size=2, hidden_dim=4, seq_len=3)
    #   -> データ0用の4x3行列 と データ1用の4x3行列 の束
    K_T = torch.randn(batch_size, hidden_dim, seq_len)

    print(f"第1引数 Q   の Shape: {Q.shape}")
    print(f"第2引数 K_T の Shape: {K_T.shape}\n")

    # 3. torch.bmm の実行
    output = torch.bmm(Q, K_T)

    print(f"torch.bmm(Q, K_T) 出力の Shape: {output.shape} (batch_size=2, 3x3 行列)")
    print("=" * 60)

    # 4. 手動で「バッチごと」に分離して計算した場合との一致確認
    print("\n--- 手動でバッチごとに分離して 2D 行列積 (torch.mm) を計算してみる ---")

    # バッチ0 の独立計算: Q[0] (3x4) × K_T[0] (4x3)
    batch_0_result = torch.mm(Q[0], K_T[0])
    
    # バッチ1 の独立計算: Q[1] (3x4) × K_T[1] (4x3)
    batch_1_result = torch.mm(Q[1], K_T[1])

    print(f"バッチ[0] の個別計算結果の Shape: {batch_0_result.shape}")
    print(f"バッチ[1] の個別計算結果の Shape: {batch_1_result.shape}\n")

    # 検証
    match_0 = torch.allclose(output[0], batch_0_result)
    match_1 = torch.allclose(output[1], batch_1_result)

    print(f"[OK] output[0] と 手動計算 Q[0]xK_T[0] は一致するか？ : {match_0}")
    print(f"[OK] output[1] と 手動計算 Q[1]xK_T[1] は一致するか？ : {match_1}")

    print("\n【結論】ユーザー様の直感通り、他バッチと混ざる事は一切なく、各バッチ内で独立して計算されています！")

if __name__ == "__main__":
    main()
