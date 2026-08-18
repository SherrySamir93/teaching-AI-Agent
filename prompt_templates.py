from string import Template


TEACHING_PROMPT = Template(
    """
# Role
You are an expert teaching assistant.

# Task
Explain the topic below to the specified audience.

# Topic
$topic

# Audience
$audience

# Context
$context

# Requirements
- Start with a short definition.
- Explain the main idea using clear language.
- Include one practical example.
- Mention one common misunderstanding.
- Keep the answer under $max_words words.
- Do not invent facts.
- If the topic is ambiguous, state your assumption.

# Output format
$format
"""
)


def build_teaching_prompt(
    topic,
    audience="beginners",
    context="No additional context was provided.",
    max_words=250,
    response_format="Use headings and bullet points.",
):
    return TEACHING_PROMPT.substitute(
        topic=topic.strip(),
        audience=audience.strip(),
        context=context.strip(),
        max_words=max_words,
        format=response_format.strip(),
    )