import json

import requests

from retrieval.query_engine import execute_query


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def format_evidence(result):
    """Convert retrieved query results into evidence for the LLM."""

    evidence = [
        f"Query source: {result.get('source', 'unknown')}",
        f"Query type: {result.get('query_type', 'unknown')}",
    ]

    structured_results = result.get("structured_results", [])

    if structured_results:
        evidence.append("\nSTRUCTURED DATABASE RESULTS:")

        for row in structured_results[:10]:
            evidence.append(json.dumps(row, default=str))

    semantic_results = result.get("semantic_results", [])

    if semantic_results:
        evidence.append("\nSEMANTIC SEARCH RESULTS:")

        for row in semantic_results[:5]:
            evidence.append(json.dumps(row, default=str))

    return "\n".join(evidence)


def ask_ollama(question, evidence, query_type):
    """Generate an answer using only the retrieved sensor evidence."""

    prompt = f"""
You are Ask the Sensors, an assistant for a smart-building
sensor database.

User question:
{question}

Query type:
{query_type}

Retrieved evidence:
{evidence}

IMPORTANT RULES:

1. Answer ONLY using the retrieved evidence.
2. Never invent sensor values, rooms, dates, or events.
3. Do not introduce information that is not present in the evidence.
4. For exact numerical questions, prioritize structured database results.
5. For absolute questions such as "highest", "lowest",
   "maximum", or "minimum", report the numerical results
   from the structured database.
6. For anomaly questions, explain that the values are unusual
   relative to the normal conditions of the respective room.
7. Do not treat a room-relative anomaly as an absolute extreme.
8. For occupancy questions, use PIR activity as evidence
   for possible occupancy or movement.
9. Include room numbers and time periods when relevant.
10. Keep the answer concise and easy to understand.
11. Use °C for temperature and ppm for CO2.
12. If the evidence is insufficient, say so instead of guessing.

Return only the final answer.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


def ask_sensors(question):
    """Retrieve evidence and generate an answer for a question."""

    result = execute_query(question)

    evidence = format_evidence(result)

    answer = ask_ollama(
        question,
        evidence,
        result["query_type"],
    )

    return result, answer


def main():
    """Run the command-line interface."""

    print("=" * 70)
    print("ASK THE SENSORS")
    print("=" * 70)

    while True:
        question = input(
            "\nAsk a sensor question (type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            _, answer = ask_sensors(question)

            print("\nAnswer:")
            print(answer)

        except Exception as e:
            print(f"\nError while processing query: {e}")


if __name__ == "__main__":
    main()