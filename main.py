import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)

CATEGORIES = {
    "technical": "Technical problems, errors, bugs, or product functionality",
    "billing": "Invoices, payments, prices, refunds, or subscriptions",
    "account": "Login, registration, permissions, or account settings",
    "general": "Questions that do not fit the other categories",
}


def classify_text(text):
    category_descriptions = "\n".join(
        f"- {name}: {description}"
        for name, description in CATEGORIES.items()
    )

    instructions = f"""
You are a text classification assistant.

Classify the user's text into exactly one of these categories:

{category_descriptions}

Return only valid JSON in this format:
{{
  "category": "technical",
  "confidence": 0.95,
  "reason": "Brief explanation"
}}

Rules:
- The category must be exactly one of: {", ".join(CATEGORIES.keys())}.
- The confidence must be a number between 0 and 1.
- Do not include Markdown or additional text.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=text,
    )

    result_text = response.output_text.strip()

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        raise ValueError(f"The model did not return valid JSON:\n{result_text}")

    if result.get("category") not in CATEGORIES:
        raise ValueError(f"Unexpected category: {result.get('category')}")

    return result


print("Text classifier started.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    text = input("Enter text to classify: ").strip()

    if text.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not text:
        print("Please enter some text.\n")
        continue

    try:
        classification = classify_text(text)

        print("\nClassification result:")
        print(f"Category: {classification['category']}")
        print(f"Confidence: {classification['confidence']}")
        print(f"Reason: {classification['reason']}\n")

    except Exception as error:
        print(f"\nAn error occurred: {error}\n")
