#!/usr/bin/env python3
"""
MBTI Class Descriptions for Prompt Enhancement

Provides semantic descriptions of each MBTI type to help the LLM
understand the "concept" behind each class, especially useful
when training data is limited.
"""

MBTI_DESCRIPTIONS = {
    "INFP": """INFP stands for Introverted, Intuitive, Feeling, and Perceiving, representing one of the 16 personality types from the Myers-Briggs Type Indicator (MBTI). Often called "the Mediator" or "the Idealist," INFPs are known for being creative, empathetic, and driven by their strong personal values. They are imaginative and often seek deep connections, preferring to recharge in solitude.""",

    "INTJ": """INTJ stands for Introverted, Intuitive, Thinking, and Judging. Known as "the Architect" or "the Mastermind," INTJs are strategic thinkers who value logic, efficiency, and long-term planning. They are independent, analytical, and often prefer working alone on complex problems. INTJs seek to understand systems and improve them.""",

    "INTP": """INTP stands for Introverted, Intuitive, Thinking, and Perceiving. Called "the Logician" or "the Thinker," INTPs are curious, analytical minds who love exploring theories and ideas. They value intellectual independence and are often found diving deep into abstract concepts, preferring flexibility over structure.""",

    "INFJ": """INFJ stands for Introverted, Intuitive, Feeling, and Judging. Known as "the Advocate" or "the Counselor," INFJs are rare idealists who seek meaning and authenticity. They are insightful, empathetic, and driven by a desire to help others. INFJs often have a strong sense of purpose and work toward making a positive impact.""",

    "ENFP": """ENFP stands for Extraverted, Intuitive, Feeling, and Perceiving. Called "the Campaigner" or "the Champion," ENFPs are enthusiastic, creative, and sociable. They are energized by possibilities and connections with others, often bringing warmth and inspiration to their interactions. ENFPs value freedom and authenticity.""",

    "ENTP": """ENTP stands for Extraverted, Intuitive, Thinking, and Perceiving. Known as "the Debater" or "the Visionary," ENTPs are innovative thinkers who love intellectual challenges and debates. They are quick-witted, curious, and enjoy exploring new ideas and possibilities, often questioning conventional wisdom.""",

    "ENFJ": """ENFJ stands for Extraverted, Intuitive, Feeling, and Judging. Called "the Protagonist" or "the Teacher," ENFJs are charismatic leaders who inspire and guide others. They are warm, empathetic, and driven by a desire to help people reach their potential. ENFJs excel at bringing people together.""",

    "ENTJ": """ENTJ stands for Extraverted, Intuitive, Thinking, and Judging. Known as "the Commander" or "the Executive," ENTJs are natural leaders who value efficiency and strategic thinking. They are decisive, organized, and excel at turning vision into reality. ENTJs thrive in leadership roles.""",

    "ISFP": """ISFP stands for Introverted, Sensing, Feeling, and Perceiving. Called "the Adventurer" or "the Artist," ISFPs are gentle, creative souls who live in the present moment. They value authenticity and personal expression, often through artistic or hands-on pursuits. ISFPs seek harmony and beauty.""",

    "ISFJ": """ISFJ stands for Introverted, Sensing, Feeling, and Judging. Known as "the Defender" or "the Protector," ISFJs are caring, reliable individuals who value tradition and stability. They are detail-oriented, responsible, and dedicated to helping and protecting those they care about.""",

    "ISTP": """ISTP stands for Introverted, Sensing, Thinking, and Perceiving. Called "the Virtuoso" or "the Craftsman," ISTPs are practical problem-solvers who excel with tools and mechanics. They are independent, logical, and enjoy hands-on exploration. ISTPs value efficiency and adaptability.""",

    "ISTJ": """ISTJ stands for Introverted, Sensing, Thinking, and Judging. Known as "the Logistician" or "the Inspector," ISTJs are responsible, organized, and detail-oriented. They value tradition, loyalty, and practical solutions. ISTJs excel at creating and maintaining structure and order.""",

    "ESFP": """ESFP stands for Extraverted, Sensing, Feeling, and Perceiving. Called "the Entertainer" or "the Performer," ESFPs are spontaneous, energetic, and love being the center of attention. They live in the moment, value experiences, and bring joy and excitement to social situations.""",

    "ESFJ": """ESFJ stands for Extraverted, Sensing, Feeling, and Judging. Known as "the Consul" or "the Caregiver," ESFJs are warm, sociable, and organized. They value harmony, tradition, and helping others. ESFJs excel at creating welcoming environments and supporting their communities.""",

    "ESTP": """ESTP stands for Extraverted, Sensing, Thinking, and Perceiving. Called "the Entrepreneur" or "the Dynamo," ESTPs are energetic, action-oriented, and love taking risks. They are practical, adaptable, and thrive in dynamic environments. ESTPs excel at solving immediate problems.""",

    "ESTJ": """ESTJ stands for Extraverted, Sensing, Thinking, and Judging. Known as "the Executive" or "the Supervisor," ESTJs are efficient, organized leaders who value tradition and order. They are practical, decisive, and excel at managing people and resources to achieve concrete results.""",
}


def get_class_description(class_name: str) -> str:
    """Get the description for a class, or empty string if not found."""
    return MBTI_DESCRIPTIONS.get(class_name, "")


def enhance_prompt_with_description(
    base_prompt: str,
    class_name: str,
    use_description: bool = True
) -> str:
    """
    Enhance a generation prompt with class description.

    Args:
        base_prompt: Original prompt with examples/keywords
        class_name: MBTI type (e.g., "INFP")
        use_description: Whether to add description (for A/B testing)

    Returns:
        Enhanced prompt with optional description
    """
    if not use_description:
        return base_prompt

    description = get_class_description(class_name)
    if not description:
        return base_prompt

    # Add description before the examples
    enhanced = f"""# Personality Type Context
{description}

# Task
{base_prompt}"""

    return enhanced


def example_usage():
    """Demonstrate prompt enhancement."""

    base_prompt = """Generate a social media post typical of someone with this personality type.

Examples from this cluster:
- "I spent the whole day lost in my imagination..."
- "I feel so deeply connected to..."

Keywords: creative, empathetic, values

Generate a similar post:"""

    print("=" * 80)
    print("PROMPT ENHANCEMENT EXAMPLE")
    print("=" * 80)

    print("\n📝 WITHOUT Description:")
    print("-" * 80)
    prompt_without = enhance_prompt_with_description(
        base_prompt, "INFP", use_description=False
    )
    print(prompt_without)

    print("\n" + "=" * 80)
    print("\n📝 WITH Description:")
    print("-" * 80)
    prompt_with = enhance_prompt_with_description(
        base_prompt, "INFP", use_description=True
    )
    print(prompt_with)

    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("=" * 80)
    print("""
    Adding semantic descriptions may help the LLM:
    1. Better understand the "concept" of the class
    2. Generate more coherent synthetics when examples are limited
    3. Especially useful for minority classes (<50 samples)

    We test this with A/B comparison:
    - Group A: No description (baseline)
    - Group B: With description
    - Measure: Synthetic quality & improvement delta
    """)


if __name__ == "__main__":
    example_usage()
