
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    print("Listing available models...")
    models = client.models.list()
    gpt_models = [m.id for m in models.data if "gpt" in m.id]
    print(f"Found {len(gpt_models)} GPT models.")
    for m in sorted(gpt_models):
        print(f" - {m}")
        
    print("\nAttempting test generation with 'gpt-4o'...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "say hi"}],
            max_completion_tokens=50
        )
        print("Success with gpt-4o:", resp.choices[0].message.content)
    except Exception as e:
        print("Failed with gpt-4o:", e)

    print("\nAttempting test generation with 'gpt-5'...")
    try:
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "say hi"}],
            max_completion_tokens=50
        )
        print("Success with gpt-5:", resp.choices[0].message.content)
    except Exception as e:
        print("Failed with gpt-5:", e)

except Exception as e:
    print(f"Error listing models: {e}")
