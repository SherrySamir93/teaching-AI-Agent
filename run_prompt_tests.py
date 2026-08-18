import os

from dotenv import load_dotenv
from openai import OpenAI

from prompt_templates import build_teaching_prompt


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)


test_cases = [
    {
        "name": "Beginner explanation",
        "topic": "Python virtual environments",
        "audience": "Python beginners",
        "context": "The learner is using Visual Studio Code on Windows.",
        "max_words": 200,
        "response_format": "Use headings, bullet points, and a short command example.",
    },
    {
        "name": "Business explanation",
        "topic": "Artificial intelligence",
        "audience": "Non-technical business employees",
        "context": "Focus on practical workplace use cases.",
        "max_words": 180,
        "response_format": "Use a short explanation followed by three workplace examples.",
    },
    {
        "name": "Advanced explanation",
        "topic": "Vector databases",
        "audience": "Intermediate Python developers",
        "context": "The learner understands embeddings and REST APIs.",
        "max_words": 300,
        "response_format": "Use technical terminology and include implementation considerations.",
    },
]


for test_case in test_cases:
    prompt = build_teaching_prompt(
        topic=test_case["topic"],
        audience=test_case["audience"],
        context=test_case["context"],
        max_words=test_case["max_words"],
        response_format=test_case["response_format"],
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "Follow the supplied teaching prompt exactly. "
            "Return only the requested teaching response."
        ),
        input=prompt,
        max_output_tokens=500,
    )

    print("\n" + "=" * 80)
    print(test_case["name"])
    print("=" * 80)
    print(response.output_text)