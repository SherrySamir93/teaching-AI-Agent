import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a helpful teaching assistant.
Explain concepts clearly and use simple examples.
"""


def ask_question(question: str) -> str:
    """Send one question to OpenAI and return the answer."""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=question,
        max_output_tokens=300,
    )

    return response.output_text


def run_chatbot():
    """Run the interactive command-line chatbot."""
    print("Teaching assistant started.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            answer = ask_question(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as error:
            print(f"\nAn error occurred: {error}\n")


if __name__ == "__main__":
    run_chatbot()
