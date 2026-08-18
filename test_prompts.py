from prompt_templates import build_teaching_prompt


test_cases = [
    {
        "name": "Beginner technical explanation",
        "topic": "Python virtual environments",
        "audience": "Python beginners",
        "context": "The learner is using Visual Studio Code on Windows.",
        "max_words": 200,
        "response_format": "Use headings, bullet points, and a short command example.",
    },
    {
        "name": "Workplace explanation",
        "topic": "Artificial intelligence",
        "audience": "Non-technical business employees",
        "context": "The explanation should focus on practical workplace use cases.",
        "max_words": 180,
        "response_format": "Use a short explanation followed by three workplace examples.",
    },
    {
        "name": "Advanced explanation",
        "topic": "Vector databases",
        "audience": "Intermediate Python developers",
        "context": "The learner already understands embeddings and REST APIs.",
        "max_words": 300,
        "response_format": "Use technical terminology and include a comparison table in plain text.",
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

    print("=" * 80)
    print(test_case["name"])
    print("=" * 80)
    print(prompt)