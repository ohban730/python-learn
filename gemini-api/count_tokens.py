from google import genai
from dotenv import load_dotenv

load_dotenv()

# load_dotenv()を実行すれば、genai.Client()は自動的に環境変数GEMINI_API_KEYを読み込む
client = genai.Client()

def count_tokens(prompt_text, model_name="gemini-3.5-flash"):
    # レスポンス生成してトークン数を取得
    response = client.models.generate_content(model=model_name, contents=prompt_text)

    print("--- AIの回答 ---")
    print(response.text)
    print("----------------\n")

    print(f"prompt_token_count (入力): {response.usage_metadata.prompt_token_count}")
    print(f"candidates_token_count (回答): {response.usage_metadata.candidates_token_count}")
    print(f"thoughts_token_count (思考): {response.usage_metadata.thoughts_token_count}")
    print(f"total_token_count (合計): {response.usage_metadata.total_token_count}")

    return response

if __name__ == "__main__":
    prompt = "今日の大阪の天気を教えて？"
    count_tokens(prompt)
