import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

start_time = time.time()
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)
end_time = time.time()

# Calculate tokens per second
total_tokens = response['usage']['total_tokens']
time_taken = end_time - start_time
tokens_per_second = total_tokens / time_taken

print(f"Tokens per second: {tokens_per_second}")
