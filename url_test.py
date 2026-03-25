import os
import requests

AZURE_OPENAI_URL = "https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
API_KEY = "6aaa06050695420281492456cc4fca7b"

print(API_KEY)


def test_chat():
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    payload = {
        "messages": [
            {"role": "user", "content": "Hello, say hello back!"}
        ],
        "max_tokens": 100,
    }

    response = requests.post(AZURE_OPENAI_URL, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        print(f"\nAI Response: {content}")
        return content
    else:
        print(f"Error: {response.text}")
        return None


if __name__ == "__main__":
    test_chat()
