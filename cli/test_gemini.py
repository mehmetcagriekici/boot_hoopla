import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash-lite"  

def main():
    prompt = "Why is Boot.dev such a great place to learn about RAG?"
    response = client.models.generate_content(model=model, contents=prompt)
    metadata = response.usage_metadata
    print(f"Prompt Tokens: {metadata.prompt_token_count}")
    print(f"Response Tokens: {metadata.candidates_token_count}")
    print(response.text)

if __name__ == "__main__":
    main()
