import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)

MEMORY_FILE = Path("conversation_memory.json")
MAX_MESSAGES = 20

SYSTEM_PROMPT = """
You are a helpful teaching assistant.

Instructions:
- Explain concepts clearly and accurately.
- Use simple language suitable for beginners.
- Provide examples when useful.
- Use conversation history to understand follow-up questions.
- If you are unsure, say so instead of inventing information.
"""

# Experiment with these values.
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 300
TOP_P = 1.0


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("Invalid memory file. Starting a new conversation.")
        return []


def save_memory(history):
    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)


def get_answer(history):
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=history,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
    )

    return response.output_text


conversation_history = load_memory()

print("Teaching assistant started.")
print("Commands: 'clear' to erase memory, 'settings' to show parameters, 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if question.lower() == "clear":
        conversation_history = []
        save_memory(conversation_history)
        print("Conversation memory cleared.\n")
        continue

    if question.lower() == "settings":
        print("\nCurrent settings:")
        print(f"Model: {MODEL}")
        print(f"Temperature: {TEMPERATURE}")
        print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
        print(f"Top-p: {TOP_P}\n")
        continue

    if not question:
        print("Please enter a question.\n")
        continue

    conversation_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    conversation_history = conversation_history[-MAX_MESSAGES:]

    try:
        answer = get_answer(conversation_history)

        print(f"\nAssistant: {answer}\n")

        conversation_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        conversation_history = conversation_history[-MAX_MESSAGES:]
        save_memory(conversation_history)

    except Exception as error:
        print(f"\nAn error occurred: {error}\n")
