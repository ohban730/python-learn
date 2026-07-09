import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("=== tanh 関数プロットプログラム ===")

    # 1. 入力値 (x) のデータを作成 (-5.0 から 5.0 まで 200点分割)
    x = np.linspace(-5.0, 5.0, 200)

    # 2. tanh および シグモイド関数の出力を計算
    y_tanh = np.tanh(x)
    y_sigmoid = 1 / (1 + np.exp(-x))  # 比較用のシグモイド関数

    # 3. グラフ描画の設定
    plt.figure(figsize=(8, 6), dpi=100)
    
    # 補助線 (グリッド線と基準線) の描画
    plt.axhline(0, color='black', linewidth=1.0, linestyle='--')  # y = 0
    plt.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')    # y = 1.0 (tanhの上限)
    plt.axhline(-1.0, color='gray', linewidth=0.8, linestyle=':')   # y = -1.0 (tanhの下限)
    plt.axvline(0, color='black', linewidth=1.0, linestyle='--')  # x = 0
    
    # グラフのプロット
    plt.plot(x, y_tanh, label=r'$\tanh(x)$', color='#1f77b4', linewidth=2.5)
    plt.plot(x, y_sigmoid, label=r'$\sigma(x)$ (Sigmoid)', color='#ff7f0e', linewidth=1.5, linestyle='--')

    # タイトルと軸ラベルの設定
    plt.title("Activation Functions: tanh vs Sigmoid", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output (y)", fontsize=12)
    
    # グラフ表示範囲とグリッドの設定
    plt.xlim(-5.0, 5.0)
    plt.ylim(-1.2, 1.2)
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')

    # 4. 画像として保存
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "tanh_curve.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"\n=> グラフを画像ファイルとして保存しました:")
    print(f"   {save_path}")
    print("\n【特徴のまとめ】")
    print("1. 値域: tanh は [-1.0, 1.0] の範囲に値を収めます。(シスモイドは [0.0, 1.0])")
    print("2. 中心点: 入力 0 のとき出力 0 であり、原点対称（Zero-centered）です。")
    print("   これによりニューラルネットの勾配の偏りが防げ、学習が安定しやすくなります。")
    print("3. 傾き: 入力 0 付近で最も変化率（傾き）が大きく、入力が極端に大きい/小さいと傾きが 0 に収束（勾配消失）します。")

if __name__ == "__main__":
    main()
