#!/usr/bin/env python3
"""
NER-Specific LLM Generation Pipeline

Generates synthetic NER training sentences with inline entity annotations,
then parses them into BIO-tagged format compatible with NER training.

Inline format: [entity text](ENTITY_TYPE)
Example: [John Smith](PER) works at [Google](ORG) in [NYC](LOC).

Usage:
    from core.ner_generator import generate_ner_batch, parse_annotated_sentence

    sentences, embeddings = generate_ner_batch(
        provider, "PER", real_sentences, 25, embed_model, "conll2003"
    )
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional


# ============================================================================
# DATASET-SPECIFIC NER PROMPT CONFIGS
# ============================================================================

NER_DATASET_PROMPTS = {
    "multinerd": {
        "domain": "diverse text (Wikipedia, news, web)",
        "entity_descriptions": {
            "PER": "person names (first names, last names, full names, historical figures)",
            "ORG": "organization names (companies, agencies, institutions, teams)",
            "LOC": "location names (countries, cities, states, geographic features)",
            "DIS": "disease and medical condition names (COVID-19, diabetes, cancer, influenza)",
            "EVE": "event names (World War II, Olympics, elections, festivals)",
        },
        "style_notes": "Use varied writing styles from encyclopedic to journalistic.",
        "length_guidance": "Each sentence should be 10-30 words.",
    },
    "conll2003": {
        "domain": "news articles (Reuters)",
        "entity_descriptions": {
            "PER": "person names (first names, last names, full names)",
            "ORG": "organization names (companies, agencies, institutions, teams)",
            "LOC": "location names (countries, cities, states, geographic features)",
            "MISC": "miscellaneous named entities (nationalities, events, languages, works of art)",
        },
        "style_notes": "Use formal news writing style. Sentences should resemble Reuters news wire text.",
        "length_guidance": "Each sentence should be 10-30 words.",
    },
    "wikiann": {
        "domain": "Wikipedia-style encyclopedic text",
        "entity_descriptions": {
            "PER": "person names (historical figures, politicians, artists, scientists)",
            "ORG": "organization names (universities, companies, governments, sports teams)",
            "LOC": "location names (countries, cities, rivers, mountains, regions)",
        },
        "style_notes": "Use encyclopedic, factual writing style similar to Wikipedia articles.",
        "length_guidance": "Each sentence should be 10-25 words.",
    },
    "fewnerd": {
        "domain": "diverse text sources (Wikipedia, news, web)",
        "entity_descriptions": {
            "person": "person names (actors, athletes, politicians, authors, scholars, soldiers)",
            "organization": "organization names (companies, sports teams, governments, media, political parties)",
            "location": "location names (countries, cities, bodies of water, mountains, parks, roads)",
            "building": "building names (airports, hospitals, hotels, libraries, restaurants, theaters)",
            "art": "art and creative work names (films, music, paintings, broadcasts, written works)",
            "product": "product names (cars, software, games, weapons, food, ships, airplanes)",
            "event": "event names (battles, wars, elections, disasters, sports events, protests)",
            "other": "other named entities (awards, diseases, languages, currencies, laws, astronomical objects)",
        },
        "style_notes": "Use varied writing styles from encyclopedic to journalistic.",
        "length_guidance": "Each sentence should be 8-25 words.",
    },
}


# ============================================================================
# PROMPT CREATION
# ============================================================================

def sentences_to_annotated_examples(sentences: List[Dict], max_examples: int = 10) -> List[str]:
    """Convert NER sentences (tokens + BIO tags) into inline annotated format for prompts."""
    examples = []
    for sent in sentences[:max_examples]:
        tokens = sent["tokens"]
        tags = sent["ner_tags"]
        result = _bio_to_inline(tokens, tags)
        if result:
            examples.append(result)
    return examples


def _bio_to_inline(tokens: List[str], tags: List[str]) -> str:
    """Convert BIO-tagged tokens to inline annotation format.

    Input:  tokens=["John", "Smith", "works", "at", "Google"]
            tags=  ["B-PER", "I-PER", "O",     "O",  "B-ORG"]
    Output: "[John Smith](PER) works at [Google](ORG)"
    """
    parts = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type = tag[2:]
            entity_tokens = [tokens[i]]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{entity_type}":
                entity_tokens.append(tokens[j])
                j += 1
            entity_text = " ".join(entity_tokens)
            parts.append(f"[{entity_text}]({entity_type})")
            i = j
        else:
            parts.append(tokens[i])
            i += 1
    return " ".join(parts)


def create_ner_prompt(
    entity_type: str,
    examples: List[str],
    n_generate: int,
    dataset_name: str
) -> str:
    """Create an LLM prompt for generating NER-annotated sentences.

    Args:
        entity_type: Target entity type (e.g., "PER", "ORG")
        examples: Inline-annotated example sentences from real data
        n_generate: Number of sentences to generate
        dataset_name: Base dataset name for style matching
    """
    config = NER_DATASET_PROMPTS.get(dataset_name, NER_DATASET_PROMPTS["conll2003"])

    entity_desc = config["entity_descriptions"].get(
        entity_type, f"named entities of type '{entity_type}'"
    )
    all_types = list(config["entity_descriptions"].keys())

    examples_text = "\n".join(f"- {ex}" for ex in examples)

    prompt = f"""# ROLE
You are a specialist data augmentation system for Named Entity Recognition (NER).

# CONTEXT
- Domain: {config["domain"]}
- Target entity type: {entity_type} — {entity_desc}
- All entity types in this dataset: {", ".join(all_types)}

# ANNOTATION FORMAT
Use inline annotations: [entity text](ENTITY_TYPE)
Example: [John Smith](PER) traveled to [Paris](LOC) for the [UN](ORG) summit.

# REFERENCE EXAMPLES
{examples_text}

# TASK
Generate exactly {n_generate} NEW sentences, each containing at least one [{entity_type}] entity.
Sentences may also contain other entity types ({", ".join(t for t in all_types if t != entity_type)}).

# REQUIREMENTS
1. Each sentence MUST contain at least one [{entity_type}] entity
2. Use DIVERSE entity names — do not repeat the same names from examples
3. {config["style_notes"]}
4. {config["length_guidance"]}
5. Annotate ALL named entities in each sentence, not just {entity_type}
6. Use the exact format: [entity text](ENTITY_TYPE)

# CONSTRAINTS - DO NOT:
- Copy or closely paraphrase the reference examples
- Use placeholder names like "John Doe" or "Company X"
- Generate numbered lists or bullet points
- Include meta-commentary
- Leave any named entity unannotated

# OUTPUT
Generate exactly {n_generate} annotated sentences, one per line:"""

    return prompt


# ============================================================================
# PARSING: INLINE ANNOTATIONS -> BIO FORMAT
# ============================================================================

ANNOTATION_PATTERN = re.compile(r'\[([^\]]+)\]\(([A-Za-z_]+)\)')


def parse_annotated_sentence(text: str, valid_types: Optional[List[str]] = None) -> Optional[Dict]:
    """Parse an inline-annotated sentence into tokens + BIO tags.

    Args:
        text: Annotated sentence, e.g. "[John](PER) works at [Google](ORG)."
        valid_types: If provided, only accept these entity types

    Returns:
        {"tokens": [...], "ner_tags": [...]} or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None

    tokens = []
    tags = []
    last_end = 0

    for match in ANNOTATION_PATTERN.finditer(text):
        # Process text before this annotation
        before = text[last_end:match.start()].strip()
        if before:
            for token in _tokenize(before):
                tokens.append(token)
                tags.append("O")

        entity_text = match.group(1).strip()
        entity_type = match.group(2).strip()

        # Validate entity type
        if valid_types and entity_type not in valid_types:
            # Unknown entity type — treat as plain text
            for token in _tokenize(entity_text):
                tokens.append(token)
                tags.append("O")
        else:
            entity_tokens = _tokenize(entity_text)
            if entity_tokens:
                tokens.append(entity_tokens[0])
                tags.append(f"B-{entity_type}")
                for et in entity_tokens[1:]:
                    tokens.append(et)
                    tags.append(f"I-{entity_type}")

        last_end = match.end()

    # Process remaining text after last annotation
    remaining = text[last_end:].strip()
    if remaining:
        for token in _tokenize(remaining):
            tokens.append(token)
            tags.append("O")

    if not tokens:
        return None

    # Validate: must have at least one entity
    has_entity = any(t.startswith("B-") for t in tags)
    if not has_entity:
        return None

    return {"tokens": tokens, "ner_tags": tags}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    # Split on whitespace first
    raw_tokens = text.split()
    tokens = []
    for raw in raw_tokens:
        # Separate leading/trailing punctuation
        stripped = raw.strip()
        if not stripped:
            continue

        # Split off leading punctuation
        i = 0
        while i < len(stripped) and stripped[i] in '([{"\'':
            tokens.append(stripped[i])
            i += 1

        # Find trailing punctuation
        j = len(stripped)
        trailing = []
        while j > i and stripped[j-1] in '.,:;!?)]\'"':
            trailing.append(stripped[j-1])
            j -= 1

        # Core token
        core = stripped[i:j]
        if core:
            tokens.append(core)

        # Add trailing punctuation
        for t in reversed(trailing):
            tokens.append(t)

    return tokens


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_ner_batch(
    provider,
    entity_type: str,
    real_sentences: List[Dict],
    n_generate: int,
    embed_model,
    dataset_name: str,
    n_examples: int = 10
) -> Tuple[List[Dict], np.ndarray, List[str]]:
    """Generate a batch of synthetic NER sentences for one entity type.

    Args:
        provider: LLM provider instance
        entity_type: Target entity type (e.g., "PER")
        real_sentences: Real NER sentences containing this entity type
        n_generate: Number of sentences to generate
        embed_model: SentenceTransformer model for embedding
        dataset_name: Dataset name for prompt style
        n_examples: Number of few-shot examples to include

    Returns:
        (parsed_sentences, embeddings, raw_texts)
        - parsed_sentences: List of {"tokens": [...], "ner_tags": [...]}
        - embeddings: np.ndarray (N, 768)
        - raw_texts: List of reconstructed plain text (for embedding)
    """
    # Select example sentences that contain this entity type
    type_sentences = [
        s for s in real_sentences
        if any(t.startswith(f"B-{entity_type}") for t in s["ner_tags"])
    ]

    # Convert to inline format for the prompt
    examples = sentences_to_annotated_examples(type_sentences, max_examples=n_examples)
    if not examples:
        return [], np.array([]).reshape(0, 768), []

    # Get valid entity types from dataset config
    config = NER_DATASET_PROMPTS.get(dataset_name, NER_DATASET_PROMPTS["conll2003"])
    valid_types = list(config["entity_descriptions"].keys())

    prompt = create_ner_prompt(entity_type, examples, n_generate, dataset_name)

    try:
        messages = [{"role": "user", "content": prompt}]
        response, _ = provider.generate(messages, temperature=0.8, max_tokens=4000)

        # Parse each line
        parsed = []
        raw_texts = []

        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering prefixes
            line = re.sub(r'^\d+[\.\)\-:\s]+', '', line).strip()
            if len(line) < 10:
                continue

            result = parse_annotated_sentence(line, valid_types=valid_types)
            if result is not None:
                # Verify it contains the target entity type
                has_target = any(
                    t == f"B-{entity_type}" for t in result["ner_tags"]
                )
                if has_target:
                    parsed.append(result)
                    raw_texts.append(" ".join(result["tokens"]))

        if not parsed:
            return [], np.array([]).reshape(0, 768), []

        # Embed the plain text versions
        embeddings = embed_model.encode(raw_texts, show_progress_bar=False)

        return parsed, embeddings, raw_texts

    except Exception as e:
        print(f"        Error generating NER batch: {e}")
        return [], np.array([]).reshape(0, 768), []
