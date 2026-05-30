from openai import OpenAI
from dotenv import load_dotenv
import os

# Explicitly load the .env file
load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("NVIDIA_API_KEY")

# Debug check
if not api_key:
    print("❌ API key not found. Check your .env file.")
else:
    print(f"✅ API key loaded: {api_key[:8]}...")  # prints only first 8 chars

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Reply with OK if you can hear me."}],
    max_tokens=10
)

print("🤖 Model response:", response.choices[0].message.content)