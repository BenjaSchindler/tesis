"""
MBTI Class Confuser Mapping for Contrastive Prompting

Maps each MBTI type to its most likely confusers (types that differ by 1 letter).
Provides contrastive descriptions to help LLM generate more distinctive samples.

This module addresses the MID tier degradation problem by helping the LLM
understand what makes each type DIFFERENT from similar types.
"""

from typing import List, Dict, Optional

# Each MBTI type maps to types that differ by exactly 1 letter (most confusing)
MBTI_CONFUSERS: Dict[str, List[str]] = {
    # Introverts
    "INFP": ["ENFP", "INTP", "INFJ", "ISFP"],
    "INFJ": ["ENFJ", "INTJ", "INFP", "ISFJ"],
    "INTP": ["ENTP", "INFP", "INTJ", "ISTP"],
    "INTJ": ["ENTJ", "INFJ", "INTP", "ISTJ"],
    "ISFP": ["ESFP", "INFP", "ISTP", "ISFJ"],
    "ISFJ": ["ESFJ", "INFJ", "ISTJ", "ISFP"],
    "ISTP": ["ESTP", "INTP", "ISTJ", "ISFP"],
    "ISTJ": ["ESTJ", "INTJ", "ISTP", "ISFJ"],
    # Extroverts
    "ENFP": ["INFP", "ENTP", "ENFJ", "ESFP"],
    "ENFJ": ["INFJ", "ENTJ", "ENFP", "ESFJ"],
    "ENTP": ["INTP", "ENFP", "ENTJ", "ESTP"],
    "ENTJ": ["INTJ", "ENFJ", "ENTP", "ESTJ"],
    "ESFP": ["ISFP", "ENFP", "ESTP", "ESFJ"],
    "ESFJ": ["ISFJ", "ENFJ", "ESTJ", "ESFP"],
    "ESTP": ["ISTP", "ENTP", "ESTJ", "ESFP"],
    "ESTJ": ["ISTJ", "ENTJ", "ESTP", "ESFJ"],
}

# Contrastive traits: what makes each type DIFFERENT from its confusers
MBTI_CONTRASTIVE_TRAITS: Dict[str, Dict[str, str]] = {
    # === INTROVERT TYPES ===
    "INFP": {
        "vs_ENFP": "More reserved, processes emotions internally, prefers deep 1-on-1 over groups",
        "vs_INTP": "More emotional and value-driven, less analytical, focuses on meaning over logic",
        "vs_INFJ": "More individualistic, resists structure, expresses values through art/creativity",
        "vs_ISFP": "More abstract/theoretical, lives in world of ideas, less present-focused",
        "key_differentiator": "Idealistic dreamer, deeply personal values, creative individualist"
    },
    "INFJ": {
        "vs_ENFJ": "More private, needs alone time, influences through writing over speaking",
        "vs_INTJ": "More empathetic, people-focused, decisions based on harmony over efficiency",
        "vs_INFP": "More structured, plans ahead, focuses on others' growth over self-expression",
        "vs_ISFJ": "More abstract/visionary, sees patterns, future-oriented over tradition",
        "key_differentiator": "Quiet visionary, empathic counselor, seeks to help humanity"
    },
    "INTP": {
        "vs_ENTP": "More reserved, thinks before speaking, prefers solo analysis over debate",
        "vs_INFP": "More logical, detached, focuses on truth over personal meaning",
        "vs_INTJ": "More exploratory, less decisive, enjoys theory for its own sake",
        "vs_ISTP": "More abstract/theoretical, less hands-on, lives in conceptual world",
        "key_differentiator": "Analytical thinker, loves complex systems, seeks logical truth"
    },
    "INTJ": {
        "vs_ENTJ": "More reserved, strategizes privately, leads through ideas over direct command",
        "vs_INFJ": "More logical, efficiency-focused, less emotionally attuned",
        "vs_INTP": "More decisive and goal-oriented, implements plans, less theoretical",
        "vs_ISTJ": "More innovative, questions traditions, sees big picture over details",
        "key_differentiator": "Strategic mastermind, long-term planner, independent visionary"
    },
    "ISFP": {
        "vs_ESFP": "More reserved, expresses through art privately, avoids spotlight",
        "vs_INFP": "More concrete/practical, lives in present moment, less theoretical",
        "vs_ISTP": "More emotional/aesthetic, values harmony, less mechanical",
        "vs_ISFJ": "More spontaneous, resists routine, follows personal impulses",
        "key_differentiator": "Gentle artist, lives authentically, quiet aesthetic sensibility"
    },
    "ISFJ": {
        "vs_ESFJ": "More reserved, serves quietly, uncomfortable with social spotlight",
        "vs_INFJ": "More practical/concrete, focuses on present needs, less visionary",
        "vs_ISTJ": "More people-focused, prioritizes harmony, emotionally supportive",
        "vs_ISFP": "More structured, values tradition, follows established methods",
        "key_differentiator": "Quiet protector, devoted helper, preserves traditions"
    },
    "ISTP": {
        "vs_ESTP": "More reserved, observes before acting, prefers solo problem-solving",
        "vs_INTP": "More hands-on/practical, learns by doing, less theoretical",
        "vs_ISTJ": "More flexible, improvises, less bound by rules/procedures",
        "vs_ISFP": "More logical/mechanical, focuses on how things work, less emotional",
        "key_differentiator": "Practical problem-solver, cool under pressure, mechanical virtuoso"
    },
    "ISTJ": {
        "vs_ESTJ": "More reserved, leads by example, less vocal about expectations",
        "vs_INTJ": "More traditional, follows proven methods, focuses on details over strategy",
        "vs_ISTP": "More structured, follows rules/procedures, less improvisational",
        "vs_ISFJ": "More task-focused, prioritizes efficiency over harmony, less emotional",
        "key_differentiator": "Dutiful organizer, reliable executor, upholds traditions"
    },
    # === EXTROVERT TYPES ===
    "ENFP": {
        "vs_INFP": "More outgoing, energized by people, shares ideas enthusiastically",
        "vs_ENTP": "More emotional/value-driven, focuses on people over abstract debate",
        "vs_ENFJ": "More spontaneous, resists structure, explores possibilities freely",
        "vs_ESFP": "More abstract/theoretical, interested in meanings over experiences",
        "key_differentiator": "Enthusiastic inspirer, champion of possibilities, social butterfly"
    },
    "ENFJ": {
        "vs_INFJ": "More outgoing, leads groups directly, energized by helping others",
        "vs_ENTJ": "More empathetic, prioritizes harmony, leads through inspiration",
        "vs_ENFP": "More structured, follows through on plans, mentors systematically",
        "vs_ESFJ": "More visionary, sees potential in people, focuses on growth",
        "key_differentiator": "Charismatic teacher, natural leader, develops others' potential"
    },
    "ENTP": {
        "vs_INTP": "More outgoing, debates enthusiastically, shares ideas openly",
        "vs_ENFP": "More logical/analytical, enjoys intellectual sparring, less emotional",
        "vs_ENTJ": "More exploratory, loves debate itself, less focused on implementation",
        "vs_ESTP": "More theoretical/abstract, enjoys ideas over physical action",
        "key_differentiator": "Innovative debater, challenges conventions, loves intellectual combat"
    },
    "ENTJ": {
        "vs_INTJ": "More outgoing, commands directly, leads from the front",
        "vs_ENFJ": "More logical, efficiency-focused, prioritizes results over harmony",
        "vs_ENTP": "More decisive, implements plans, less interested in pure debate",
        "vs_ESTJ": "More strategic, challenges status quo, focuses on innovation",
        "key_differentiator": "Decisive commander, natural executive, drives toward goals"
    },
    "ESFP": {
        "vs_ISFP": "More outgoing, seeks spotlight, energized by entertaining others",
        "vs_ENFP": "More concrete/practical, lives in moment, less theoretical",
        "vs_ESTP": "More emotional/social, prioritizes fun and harmony, less competitive",
        "vs_ESFJ": "More spontaneous, lives for now, less focused on duties",
        "key_differentiator": "Natural entertainer, life of the party, lives in the moment"
    },
    "ESFJ": {
        "vs_ISFJ": "More outgoing, organizes social events, vocal about caring",
        "vs_ENFJ": "More practical/concrete, focuses on present needs, less visionary",
        "vs_ESTJ": "More people-focused, prioritizes harmony, emotionally expressive",
        "vs_ESFP": "More structured, plans ahead, values traditions and duties",
        "key_differentiator": "Social organizer, caring host, maintains group harmony"
    },
    "ESTP": {
        "vs_ISTP": "More outgoing, seeks action with others, enjoys spotlight",
        "vs_ENTP": "More practical/concrete, prefers physical action over debate",
        "vs_ESTJ": "More flexible, improvises, less bound by rules/procedures",
        "vs_ESFP": "More logical/competitive, focuses on winning, less emotional",
        "key_differentiator": "Bold risk-taker, lives for action, thrives under pressure"
    },
    "ESTJ": {
        "vs_ISTJ": "More outgoing, commands directly, vocal about expectations",
        "vs_ENTJ": "More traditional, follows proven methods, maintains order",
        "vs_ESTP": "More structured, follows procedures, less improvisational",
        "vs_ESFJ": "More task-focused, prioritizes efficiency, direct communication",
        "key_differentiator": "Natural administrator, enforces standards, organizes efficiently"
    },
}


def get_confusers(mbti_type: str, top_k: int = 3) -> List[str]:
    """
    Get top-k confuser classes for a given MBTI type.

    Args:
        mbti_type: The MBTI type (e.g., "ENFJ")
        top_k: Number of confusers to return (default 3)

    Returns:
        List of confuser MBTI types, ordered by confusion likelihood
    """
    mbti_upper = mbti_type.upper()
    confusers = MBTI_CONFUSERS.get(mbti_upper, [])
    return confusers[:top_k]


def get_primary_confuser(mbti_type: str) -> Optional[str]:
    """
    Get the single most likely confuser for a given MBTI type.
    This is typically the type that differs only in E/I (introvert/extrovert).

    Args:
        mbti_type: The MBTI type (e.g., "ENFJ")

    Returns:
        The primary confuser type, or None if not found
    """
    confusers = get_confusers(mbti_type, top_k=1)
    return confusers[0] if confusers else None


def get_contrastive_description(mbti_type: str) -> str:
    """
    Get a contrastive description for prompting.

    This description highlights what makes the type DIFFERENT from similar types,
    helping the LLM generate more distinctive samples.

    Args:
        mbti_type: The MBTI type (e.g., "ENFJ")

    Returns:
        Multi-line string with contrastive information
    """
    mbti_upper = mbti_type.upper()
    traits = MBTI_CONTRASTIVE_TRAITS.get(mbti_upper, {})

    if not traits:
        return ""

    lines = [f"What makes {mbti_upper} DISTINCT from similar types:"]

    for key, value in traits.items():
        if key.startswith("vs_"):
            confuser = key.replace("vs_", "")
            lines.append(f"  - Unlike {confuser}: {value}")
        elif key == "key_differentiator":
            lines.append(f"  - Core identity: {value}")

    return "\n".join(lines)


def get_contrastive_prompt_section(
    target_class: str,
    top_k_confusers: int = 2
) -> str:
    """
    Generate a contrastive prompt section for synthetic generation.

    This section can be appended to the main prompt to help the LLM
    generate samples that are clearly distinguishable from confuser classes.

    Args:
        target_class: The target MBTI type
        top_k_confusers: Number of confusers to mention

    Returns:
        Formatted prompt section string
    """
    target_upper = target_class.upper()
    confusers = get_confusers(target_upper, top_k=top_k_confusers)

    if not confusers:
        return ""

    traits = MBTI_CONTRASTIVE_TRAITS.get(target_upper, {})
    key_trait = traits.get("key_differentiator", "")

    # Build differentiation points
    diff_points = []
    for confuser in confusers:
        vs_key = f"vs_{confuser}"
        if vs_key in traits:
            diff_points.append(f"- NOT like {confuser}: {traits[vs_key]}")

    prompt_section = f"""
## CRITICAL: Differentiation Guidelines
The generated text MUST clearly belong to {target_upper} and NOT be confused with: {', '.join(confusers)}.

{target_upper} Core Identity: {key_trait}

Key Differences:
{chr(10).join(diff_points)}

Ensure the generated text reflects {target_upper}'s unique characteristics.
"""
    return prompt_section.strip()


def get_confuser_centroids_for_filter(
    target_class: str,
    class_centroids: Dict[str, any],
    top_k: int = 3
) -> List[tuple]:
    """
    Get confuser class centroids for contrastive filtering.

    Used in filter_candidates() to reject synthetics that are
    too similar to confuser classes.

    Args:
        target_class: The target MBTI type
        class_centroids: Dict mapping class names to their centroids
        top_k: Number of confuser centroids to return

    Returns:
        List of (confuser_name, centroid) tuples
    """
    confusers = get_confusers(target_class, top_k=top_k)
    result = []

    for confuser in confusers:
        if confuser in class_centroids:
            result.append((confuser, class_centroids[confuser]))

    return result


# Quick test
if __name__ == "__main__":
    print("=== MBTI Confusers Module Test ===\n")

    test_types = ["ENFJ", "ESFP", "ISFJ", "INTJ"]

    for mbti in test_types:
        print(f"\n--- {mbti} ---")
        print(f"Confusers: {get_confusers(mbti)}")
        print(f"Primary confuser: {get_primary_confuser(mbti)}")
        print(f"\nContrastive description:\n{get_contrastive_description(mbti)}")
        print(f"\nPrompt section:\n{get_contrastive_prompt_section(mbti)}")
        print("-" * 50)
