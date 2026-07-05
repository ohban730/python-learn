# テキスト生成ループの停止メカニズム

LLMのテキスト生成は「次の単語を1つ予測して、それを入力に追加してまた予測する」という繰り返し処理（自己回帰生成）です。

書籍のサンプルコードでは `for _ in range(max_new_tokens)` という固定長ループが使われていますが、実際のプロダクトでは「いつ止めるか」を複数の条件で制御しています。

---

## 固定長ループ vs 停止条件付きループ

```python
# 書籍のサンプルコード（固定長）
for _ in range(max_new_tokens):
    next_id = torch.argmax(logits[:, -1, :], dim=-1)
    tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)

# ──────────────────────────────────────────────────────────
# 実際のプロダクトコード（停止条件付き while ループ）
EOS_TOKEN_ID = 50256  # GPT-2: "<|endoftext|>"
MAX_LENGTH = 2048     # 安全弁

while True:
    logits = model(tokens)
    next_id = torch.argmax(logits[:, -1, :], dim=-1)

    # 停止条件 1: EOS トークンが出力された
    if next_id.item() == EOS_TOKEN_ID:
        break

    # 停止条件 2: 最大長を超えた（安全弁）
    if tokens.shape[1] >= MAX_LENGTH:
        break

    tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
```

---

## 3つの停止メカニズム

### 1. EOSトークン（終端記号）

**EOS（End of Sequence）**とは、モデルが事前学習の中で「文章はここで終わり」という意味として学習した特別なトークンです。

GPT-2 の場合、`<|endoftext|>` という文字列が EOS として定義されており、トークンIDは **`50256`**（全50,257語の最後の1つ）です。

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

print(enc.eot_token)         # → 50256
print(enc.decode([50256]))   # → '<|endoftext|>'
```

#### 生成フローの具体例

```text
入力:  "Every effort"

生成1: " moves"         (ID: 6100)   ─→ 続ける
生成2: " you"           (ID: 345)    ─→ 続ける
生成3: " forward"       (ID: 2651)   ─→ 続ける
生成4: "<|endoftext|>"  (ID: 50256)  ─→ 🛑 終了！

最終出力: "Every effort moves you forward"
```

#### なぜ EOS を学習できるのか？
事前学習データ（書籍全文、Webページなど）の各テキストブロックの末尾に、必ず `<|endoftext|>` が付与された状態で学習されます。
そのため、モデルは「文章が自然に終わる文脈」でEOSトークンに最も高い予測スコアを返すことを自然に身につけます。

---

### 2. 最大トークン長（安全弁）

EOS だけでは、モデルが何らかの理由でEOSを出力しない場合に**無限ループ**になってしまいます。
そのため、現実的な最大長を設けて強制的に打ち切る安全弁が常に設けられています。

```python
# OpenAI API での例
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    max_tokens=1024   # ← この引数が安全弁に相当
)
```

> [!NOTE]
> `max_tokens` はAPIの課金単位でもあるため、コスト管理の観点からも重要な設定です。

---

### 3. 停止シーケンス（Stop Sequences）

チャット形式のLLMでは、EOS以外にも**特定の文字列パターン**が出力された時点で生成を止める仕組みが使われます。

```python
stop_sequences = [
    "\n\nUser:",     # チャット形式でモデルがユーザーの発言を自分で作り始めた
    "###",           # プロンプトの区切り記号として使われるパターン
    "<|im_end|>",    # ChatML形式（GPT-4等）のターン終端記号
]

# 生成テキストにこれらが含まれたら強制終了
if any(seq in generated_text for seq in stop_sequences):
    break
```

これが必要な理由は、事前学習済みモデルはそもそも「会話を続けること」を学習しています。
Stop Sequences がないと、モデルが自分でユーザーの質問を作り上げ、それに自分で答え、また質問して……と際限なく「一人漫才」を続けてしまいます。

---

## まとめ：3つの停止条件の役割分担

| 停止条件 | いつ発動するか | 主な目的 |
|:--|:--|:--|
| **EOS トークン** | モデルが「文章終わり」と判断した時 | 自然なテキストの終端 |
| **最大トークン長** | 出力が一定の長さを超えた時 | 無限ループの防止 / コスト管理 |
| **Stop Sequences** | 特定のパターンが生成テキストに現れた時 | チャット形式の暴走防止 |

---

## 関連する概念：Greedy Decoding と Sampling

上記の停止メカニズムとは別に、「次のトークンをどう選ぶか」にも戦略があります。

| 選択戦略 | 方法 | 特徴 |
|:--|:--|:--|
| **Greedy Decoding** | `argmax` で常に最も確率の高い単語を選ぶ | 決定的・安定・多様性なし |
| **Temperature Sampling** | Softmax の温度を上げてランダム性を加える | 創造的・多様・不安定 |
| **Top-k Sampling** | 上位k個の候補からランダムに選ぶ | 品質と多様性のバランス |
| **Top-p (Nucleus) Sampling** | 累積確率がp以上になるまでの候補から選ぶ | OpenAI APIでも使われる主流手法 |

書籍のサンプルコードは `argmax`（Greedy Decoding）で実装されていますが、実プロダクトでは Temperature Sampling や Top-p Sampling が組み合わせて使われます。
