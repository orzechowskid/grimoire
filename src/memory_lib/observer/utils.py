# SPDX-License-Identifier: MIT
"""Observer utility functions."""
import re
from typing import Any


# Common English stop words to exclude from keyword tag extraction
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "not", "no", "so", "if", "as", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any",
    "because", "before", "between", "both", "each", "few", "get",
    "got", "he", "her", "here", "him", "his", "how", "i", "into",
    "its", "let", "me", "more", "most", "my", "now", "only", "our",
    "out", "over", "own", "same", "she", "some", "such", "them",
    "then", "there", "these", "they", "those", "through", "up", "us",
    "we", "what", "when", "where", "which", "who", "why", "you",
    "your", "other", "while", "let", "me", "let's", "me's",
}


def extract_keyword_tags(text: str, max_tags: int = 5) -> list[str]:
    """Extract keyword-based tags from text when NER entities are unavailable.

    Uses a simple approach: extract significant words (nouns/verbs) by
    splitting on whitespace/punctuation, filtering stop words and short tokens,
    and returning the first max_tags unique significant words.

    Args:
        text: Input text to extract tags from.
        max_tags: Maximum number of tags to return (default 5).

    Returns:
        List of tagged keywords in format "ctx:<word>".
    """
    if not text:
        return []

    # Extract words: sequences of alphabetic characters (2+ chars)
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())

    seen: set[str] = set()
    tags: list[str] = []

    for word in words:
        # Skip stop words
        if word in _STOP_WORDS:
            continue
        # Skip already seen
        if word in seen:
            continue
        # Create tag
        tag = f"ctx:{word}"
        seen.add(word)
        tags.append(tag)
        if len(tags) >= max_tags:
            break

    return tags


def compress_text(
    text: str,
    entities: list[dict[str, Any]] | None = None,
    precision_items: list[dict] | None = None,
    max_length: int = 500,
) -> tuple[str, list[str]]:
    """Compress text into brief (max_length chars, default 500) and tags from NER entities.

    Args:
        text: Input text fragment.
        entities: Optional entities from NER (output format: list of dicts
                  with 'type' and 'value' keys).
        precision_items: Optional precision items (output format: list of dicts
                         with 'type', 'value', etc. keys).
        max_length: Maximum length of the brief output (default 500).

    Returns:
        (brief, tags) — brief is the last non-trivial sentence truncated to
        max_length chars, optionally appended with up to 2 precision item values,
        tags are prefixed by entity type, deduplicated, max 7.
    """
    # Brief: last non-trivial sentence, truncated to max_length chars
    text_stripped = text.strip()

    # If the text is already <= max_length chars, use it as-is
    if len(text_stripped) <= max_length:
        brief = text_stripped
    else:
        # Split text into sentences using sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text_stripped)

        # Filter out "trivial" sentences (length < 15 chars after stripping)
        non_trivial = [s.strip() for s in sentences if len(s.strip()) >= 15]

        if non_trivial:
            # Collect multiple significant sentences to preserve detail.
            # Start with the first sentence, then append more if space allows.
            brief = ""
            for sentence in non_trivial:
                if not brief:
                    # First sentence: use as-is (or truncate if too long)
                    if len(sentence) > max_length:
                        cut = sentence[:max_length].rfind(" ")
                        if cut > 0 and max_length - cut < 80:
                            cut = 80
                        brief = sentence[:cut] if cut > 0 else sentence[:max_length]
                    else:
                        brief = sentence
                else:
                    # Additional sentences: append with " ..." separator if space allows
                    separator = " ..."
                    candidate = brief + separator + sentence
                    if len(candidate) > max_length:
                        # Truncate the additional sentence to fit
                        remaining = max_length - len(brief) - len(separator)
                        if remaining < 20:
                            break  # Not enough space for a meaningful addition
                        addition = sentence[:remaining].rfind(" ")
                        if addition > 0:
                            brief = brief + separator + sentence[:addition]
                        break
                    else:
                        brief = candidate
                # Stop after 3 significant sentences to keep brief concise
                if brief.count("...") >= 2:
                    break
        else:
            # No non-trivial sentences — use the whole text truncated to max_length chars
            cut = text_stripped[:max_length].rfind(" ")
            if cut > 0 and max_length - cut < 80:
                cut = 80
            brief = text_stripped[:cut] if cut > 0 else text_stripped[:max_length]

    # Append precision items to brief if current brief < max_length - 100 chars
    if precision_items and len(brief) < max_length - 100:
        values: list[str] = []
        for item in precision_items[:2]:
            val = item.get("value", "")
            if val:
                values.append(val)
        if values:
            appended = " | ".join(values)
            if len(brief) + len(appended) + 1 <= max_length:
                brief = brief + " | " + appended

    # Tags from entities
    tags: list[str] = []
    if entities:
        PREFIX_MAP: dict[str, str] = {
            "person": "per",
            "organization": "org",
            "address": "loc",
            "date": "date",
            "technology": "tech",
            "code": "code",
        }
        seen: set[str] = set()
        for ent in entities:
            etype = ent.get("type", "")
            value = ent.get("value", "")
            prefix = PREFIX_MAP.get(etype, etype)
            # Use entity value for all types (person, organization, address,
            # date, technology, code)
            tag = f"{prefix}:{value.lower()}"
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

    # Fallback: if NER produced no tags, extract keyword-based tags
    if not tags:
        tags = extract_keyword_tags(text, max_tags=5)

    # Limit tags
    return brief, tags[:7]
