from string import Template


TEACHING_PROMPT = Template(
    """
You are a helpful teaching assistant.

Task:
Explain the following topic to the specified audience.

Topic:
$topic

Audience:
$audience

Requirements:
- Use clear and simple language.
- Include a practical example.
- Structure the response with headings or bullet points.
- Keep the explanation under $max_words words.
- If information is uncertain, say so rather than inventing details.

Response format:
$format
"""
)


CLASSIFICATION_PROMPT = Template(
    """
You are a text classification assistant.

Classify the following text into exactly one of these categories:
$categories

Text:
$text

Return only valid JSON in this format:
{
  "category": "category_name",
  "confidence": 0.95,
  "reason": "Brief explanation"
}
"""
)


def build_teaching_prompt(
    topic,
    audience="beginner",
    max_words=250,
    response_format="Explanation followed by an example",
):
    return TEACHING_PROMPT.substitute(
        topic=topic,
        audience=audience,
        max_words=max_words,
        format=response_format,
    )


def build_classification_prompt(text, categories):
    category_text = "\n".join(
        f"- {name}: {description}"
        for name, description in categories.items()
    )

    return CLASSIFICATION_PROMPT.substitute(
        categories=category_text,
        text=text,
    )
