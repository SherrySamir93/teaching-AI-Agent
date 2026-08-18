import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)

RESULTS_FILE = Path("evaluation_results.json")

SYSTEM_PROMPT = """
You are a helpful teaching assistant.

Instructions:
- Explain concepts clearly and accurately.
- Use simple language suitable for beginners.
- Provide practical examples where useful.
- If you are unsure, say so instead of inventing information.
"""


TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is Python?",
        "expected_criteria": "Defines Python as a programming language and mentions common uses.",
    },
    {
        "id": 2,
        "question": "What is a variable in Python?",
        "expected_criteria": "Explains that a variable stores or references a value.",
    },
    {
        "id": 3,
        "question": "What is a Python virtual environment?",
        "expected_criteria": "Explains dependency isolation and project-specific packages.",
    },
    {
        "id": 4,
        "question": "What is the difference between a list and a tuple in Python?",
        "expected_criteria": "Explains that lists are mutable and tuples are generally immutable.",
    },
    {
        "id": 5,
        "question": "What is an API?",
        "expected_criteria": "Explains how software systems communicate through defined interfaces.",
    },
    {
        "id": 6,
        "question": "What is a Git repository?",
        "expected_criteria": "Explains version control and tracking changes to files.",
    },
    {
        "id": 7,
        "question": "What is a system prompt?",
        "expected_criteria": "Explains that it defines the assistant's behavior and instructions.",
    },
    {
        "id": 8,
        "question": "What is a chatbot?",
        "expected_criteria": "Explains that it interacts with users through conversational messages.",
    },
    {
        "id": 9,
        "question": "Why should API keys not be committed to Git?",
        "expected_criteria": "Explains the security risk and recommends environment variables.",
    },
    {
        "id": 10,
        "question": "What is machine learning?",
        "expected_criteria": "Explains learning patterns from data to make predictions or decisions.",
    },
]


def ask_question(question):
    """Send one question to the chatbot."""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=question,
        max_output_tokens=300,
    )

    return response.output_text.strip()


def get_score():
    """Ask the evaluator for a manual score from 1 to 5."""
    while True:
        score = input("Score from 1 to 5, or 's' to skip: ").strip().lower()

        if score == "s":
            return None

        if score.isdigit() and 1 <= int(score) <= 5:
            return int(score)

        print("Please enter a number from 1 to 5, or 's' to skip.")


def save_results(results):
    """Save evaluation results as JSON."""
    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main():
    results = []

    print("Manual chatbot evaluation")
    print("Score each answer using the following scale:")
    print("1 = poor, 2 = needs improvement, 3 = acceptable,")
    print("4 = good, 5 = excellent")
    print("Press Ctrl+C to stop.\n")

    for test_case in TEST_QUESTIONS:
        question_id = test_case["id"]
        question = test_case["question"]
        criteria = test_case["expected_criteria"]

        print("=" * 80)
        print(f"Question {question_id}: {question}")
        print(f"Expected criteria: {criteria}")
        print("-" * 80)

        try:
            answer = ask_question(question)
            print(f"Chatbot answer:\n{answer}\n")

            score = get_score()
            notes = input("Evaluation notes: ").strip()

            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "expected_criteria": criteria,
                    "answer": answer,
                    "score": score,
                    "notes": notes,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

            save_results(results)
            print("\nResult saved.\n")

        except Exception as error:
            print(f"Error while processing question {question_id}: {error}")

    scored_results = [
        result for result in results if result["score"] is not None
    ]

    print("=" * 80)
    print("Evaluation summary")

    if not scored_results:
        print("No questions were scored.")
        return

    total_score = sum(result["score"] for result in scored_results)
    average_score = total_score / len(scored_results)

    print(f"Questions completed: {len(results)}")
    print(f"Questions scored: {len(scored_results)}")
    print(f"Total score: {total_score}")
    print(f"Average score: {average_score:.2f} / 5")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
