import os

from dotenv import load_dotenv
from openai import OpenAI

from prompt_templates import build_teaching_prompt


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)


def ask_openai(prompt):
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "Follow the supplied prompt carefully. "
            "Do not add information that is not supported by the prompt."
        ),
        input=prompt,
        max_output_tokens=500,
    )

    return response.output_text


print("Structured prompt teaching assistant")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    topic = input("Topic: ").strip()

    if topic.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not topic:
        print("Please enter a topic.\n")
        continue

    audience = input("Audience [beginner]: ").strip() or "beginner"
    max_words = input("Maximum words [250]: ").strip() or "250"
    response_format = (
        input("Response format [explanation and example]: ").strip()
        or "Explanation followed by an example"
    )

    prompt = build_teaching_prompt(
        topic=topic,
        audience=audience,
        max_words=max_words,
        response_format=response_format,
    )

    try:
        answer = ask_openai(prompt)

        print("\nGenerated prompt:")
        print(prompt)

        print("\nAssistant:")
        print(answer)
        print()

    except Exception as error:
        print(f"\nAn error occurred: {error}\n")
