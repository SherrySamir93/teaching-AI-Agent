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
- Use the conversation history to understand follow-up questions.
- If you are unsure, say so instead of inventing information.
"""


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


def get_streaming_answer(history):
    stream = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=history,
        stream=True,
        max_output_tokens=300,
    )

    answer_parts = []

    print("\nAssistant: ", end="", flush=True)

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
            answer_parts.append(event.delta)

    print("\n")

    return "".join(answer_parts)


conversation_history = load_memory()

print("Teaching assistant started.")
print("Commands: 'clear' to erase memory, 'exit' to quit.\n")

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
        answer = get_streaming_answer(conversation_history)

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
