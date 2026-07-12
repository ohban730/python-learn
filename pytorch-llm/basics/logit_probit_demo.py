import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    # グラフの保存先を docs ディレクトリに設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, '..', 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    output_path = os.path.join(docs_dir, 'logit_probit_comparison.png')

    # 確率 p の範囲 (0より大きく1より小さい)
    p = np.linspace(0.001, 0.999, 1000)

    # ロジット変換
    logit_val = np.log(p / (1 - p))

    # プロビット変換 (標準正規分布の累積分布関数の逆関数)
    probit_val = norm.ppf(p)

    # グラフ描画
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(p, logit_val, label='Logit Transformation', color='#1f77b4', linewidth=2)
    plt.plot(p, probit_val, label='Probit Transformation', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.title('Comparison of Logit and Probit Transformations', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Probability (p)', fontsize=12)
    plt.ylabel('Transformed Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    # 軸の装飾
    plt.axvline(0.5, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # 画像として保存
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Graph successfully saved to: {output_path}")

    # 簡単な変換デモ出力
    test_p = 0.8
    test_x = 1.5
    print("\n--- Value transformation examples ---")
    print(f"Logit({test_p})  = {np.log(test_p / (1 - test_p)):.6f}")
    print(f"Probit({test_p}) = {norm.ppf(test_p):.6f}")
    print(f"Inverse Logit({test_x})  = {1 / (1 + np.exp(-test_x)):.6f}")
    print(f"Inverse Probit({test_x}) = {norm.cdf(test_x):.6f}")

if __name__ == "__main__":
    main()
