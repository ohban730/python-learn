from google import genai
from dotenv import load_dotenv

load_dotenv()

# load_dotenv()を実行すれば、genai.Client()は自動的に環境変数GEMINI_API_KEYを読み込む
client = genai.Client()

stream = client.interactions.create(
    model="gemini-3.5-flash",
    input="Explain how AI works",
    stream=True
)

print("--- ストリーミング1回答開始 ---")
# interactins.createを使ったストリーミング
# 自律型AIエージェントを作る時（Computer Useなど）
# テキストだけでなく、AIが「今何を考えているか（思考ステップ）」
# 「今からブラウザを操作しようとしている（ツール実行要求）」といったAIの裏側の状態変化を
# リアルタイムで画面やログに表示したい時に使う。
for event in stream:
    if hasattr(event, "delta") and event.delta and getattr(event.delta, "type", None) == "text":
        print(event.delta.text, end="", flush=True)
print("\n--- ストリーミング1完了 ---")

print("\n")

print("--- ストリーミング2回答開始 ---")
# generate_content_streamを使ったストリーミング
# 生成AIの回答をリアルタイムで受け取り、画面に出力する用途が主
# 文字がパラパラと表示される「ChatGPT風のUX」を最も簡単に実装したい時。
# テキストしか扱わないのでエラーが起きにくく最も安全。
response = client.models.generate_content_stream(
    model="gemini-3.5-flash",
    contents="Explain how AI works"
)
for chunk in response:
    print(chunk.text, end="", flush=True)
print("\n--- ストリーミング2完了 ---")