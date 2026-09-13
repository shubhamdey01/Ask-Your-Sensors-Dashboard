import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def ask_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


if __name__ == "__main__":

    print("=" * 70)
    print("OLLAMA PYTHON TEST")
    print("=" * 70)

    prompt = """
You are an assistant for a smart-building sensor database.

Answer the following question in exactly two sentences:

What is a sensor database?
"""

    answer = ask_ollama(prompt)

    print("\nQuestion:")
    print("What is a sensor database?")

    print("\nQwen3 answer:")
    print(answer)

    print("\n" + "=" * 70)
    print("OLLAMA TEST COMPLETE")
    print("=" * 70)