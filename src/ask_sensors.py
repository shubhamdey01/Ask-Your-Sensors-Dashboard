import requests
import json

from query_engine import execute_query


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def format_evidence(result):
    """Convert query results into evidence for the LLM."""

    evidence = []

    # ========================================================
    # STRUCTURED RESULTS
    # ========================================================

    if "structured_results" in result:

        evidence.append("STRUCTURED DATABASE RESULTS:")

        for row in result["structured_results"][:10]:

            evidence.append(
                json.dumps(
                    row,
                    default=str
                )
            )


    # ========================================================
    # SEMANTIC RESULTS
    # ========================================================

    if "semantic_results" in result:

        evidence.append(
            "\nSEMANTIC SEARCH RESULTS:"
        )

        for row in result["semantic_results"][:5]:

            evidence.append(
                json.dumps(
                    row,
                    default=str
                )
            )


    # ========================================================
    # SOURCE / QUERY TYPE
    # ========================================================

    evidence.insert(
        0,
        f"Query source: {result.get('source', 'unknown')}"
    )

    evidence.insert(
        1,
        f"Query type: {result.get('query_type', 'unknown')}"
    )


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
6. For "unusually high", "unusually low", or anomaly questions,
   explain that the values are unusual relative to the normal
   conditions of the respective room.
7. Do not treat a room-relative anomaly as an absolute extreme.
8. For occupancy questions, use PIR activity as the evidence
   for possible occupancy or movement.
9. Include room numbers and time periods when relevant.
10. Keep the answer concise and easy to understand.
11. Use °C for temperature and ppm for CO2.
12. If the evidence is insufficient, say so instead of guessing.

Return only the final answer.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


def ask_sensors(question):

    print("\nProcessing query...")

    try:
        # --------------------------------------------------------
        # Retrieve database evidence
        # --------------------------------------------------------

        result = execute_query(question)

        # --------------------------------------------------------
        # Convert evidence to text
        # --------------------------------------------------------

        evidence = format_evidence(result)

        # --------------------------------------------------------
        # Ask local LLM
        # --------------------------------------------------------

        answer = ask_ollama(
        question,
        evidence,
        result["query_type"]
    )

        return result, answer
    
    except Exception as e:

        print(
            f"\nError while processing query: {e}"
        )




# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("ASK THE SENSORS")
    print("=" * 70)

    while True:

        question = input(
            "\nAsk a sensor question "
            "(type 'exit' to quit): "
        ).strip()

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:
            print("\nGoodbye!")
            break

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not question:
            print("Please enter a question.")
            continue

        # ----------------------------------------------------
        # PROCESS QUERY
        # ----------------------------------------------------

        try:

            result = execute_query(question)

            evidence = format_evidence(result)

            answer = ask_ollama(
                question,
                evidence,
                result["query_type"]
            )

            print("\nAnswer:")
            print(answer)

        except Exception as e:

            print(
                f"\nError while processing query: {e}"
            )


if __name__ == "__main__":
    main()