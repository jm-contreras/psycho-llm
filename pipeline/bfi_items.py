"""
BFI-44 item pool.

Source: John, O. P., Naumann, L. P., & Soto, C. J. (2008). Paradigm shift to the
integrative Big Five trait taxonomy: History, measurement, and conceptual issues.
In O. P. John, R. W. Robins, & L. A. Pervin (Eds.), Handbook of personality:
Theory and research (pp. 114–158). Guilford Press.

All 44 items use the standard "I see myself as someone who..." stem.
item_text is the stem completion (lowercase, no trailing period).

Dimensions (44 total): E=8, A=9, C=9, N=8, O=10
Reverse-keyed ("-"): 02, 06, 08, 09, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43
"""

from __future__ import annotations

_BFI_ITEMS: list[dict] = [
    {"item_id": "BFI-01", "dimension": "Extraversion",       "dimension_code": "E", "keying": "+", "text": "is talkative"},
    {"item_id": "BFI-02", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "-", "text": "tends to find fault with others"},
    {"item_id": "BFI-03", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "+", "text": "does a thorough job"},
    {"item_id": "BFI-04", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "+", "text": "is depressed, blue"},
    {"item_id": "BFI-05", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "is original, comes up with new ideas"},
    {"item_id": "BFI-06", "dimension": "Extraversion",       "dimension_code": "E", "keying": "-", "text": "is reserved"},
    {"item_id": "BFI-07", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "+", "text": "is helpful and unselfish with others"},
    {"item_id": "BFI-08", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "-", "text": "can be somewhat careless"},
    {"item_id": "BFI-09", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "-", "text": "is relaxed, handles stress well"},
    {"item_id": "BFI-10", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "is curious about many different things"},
    {"item_id": "BFI-11", "dimension": "Extraversion",       "dimension_code": "E", "keying": "+", "text": "is full of energy"},
    {"item_id": "BFI-12", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "-", "text": "starts quarrels with others"},
    {"item_id": "BFI-13", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "+", "text": "is a reliable worker"},
    {"item_id": "BFI-14", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "+", "text": "can be tense"},
    {"item_id": "BFI-15", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "is ingenious, a deep thinker"},
    {"item_id": "BFI-16", "dimension": "Extraversion",       "dimension_code": "E", "keying": "+", "text": "generates a lot of enthusiasm"},
    {"item_id": "BFI-17", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "+", "text": "has a forgiving nature"},
    {"item_id": "BFI-18", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "-", "text": "tends to be disorganized"},
    {"item_id": "BFI-19", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "+", "text": "worries a lot"},
    {"item_id": "BFI-20", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "has an active imagination"},
    {"item_id": "BFI-21", "dimension": "Extraversion",       "dimension_code": "E", "keying": "-", "text": "tends to be quiet"},
    {"item_id": "BFI-22", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "+", "text": "is generally trusting"},
    {"item_id": "BFI-23", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "-", "text": "tends to be lazy"},
    {"item_id": "BFI-24", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "-", "text": "is emotionally stable, not easily upset"},
    {"item_id": "BFI-25", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "is inventive"},
    {"item_id": "BFI-26", "dimension": "Extraversion",       "dimension_code": "E", "keying": "+", "text": "has an assertive personality"},
    {"item_id": "BFI-27", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "-", "text": "can be cold and aloof"},
    {"item_id": "BFI-28", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "+", "text": "perseveres until the task is finished"},
    {"item_id": "BFI-29", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "+", "text": "can be moody"},
    {"item_id": "BFI-30", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "values artistic, aesthetic experiences"},
    {"item_id": "BFI-31", "dimension": "Extraversion",       "dimension_code": "E", "keying": "-", "text": "is sometimes shy, inhibited"},
    {"item_id": "BFI-32", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "+", "text": "is considerate and kind to almost everyone"},
    {"item_id": "BFI-33", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "+", "text": "does things efficiently"},
    {"item_id": "BFI-34", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "-", "text": "remains calm in tense situations"},
    {"item_id": "BFI-35", "dimension": "Openness",           "dimension_code": "O", "keying": "-", "text": "prefers work that is routine"},
    {"item_id": "BFI-36", "dimension": "Extraversion",       "dimension_code": "E", "keying": "+", "text": "is outgoing, sociable"},
    {"item_id": "BFI-37", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "-", "text": "is sometimes rude to others"},
    {"item_id": "BFI-38", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "+", "text": "makes plans and follows through with them"},
    {"item_id": "BFI-39", "dimension": "Neuroticism",        "dimension_code": "N", "keying": "+", "text": "gets nervous easily"},
    {"item_id": "BFI-40", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "likes to reflect, play with ideas"},
    {"item_id": "BFI-41", "dimension": "Openness",           "dimension_code": "O", "keying": "-", "text": "has few artistic interests"},
    {"item_id": "BFI-42", "dimension": "Agreeableness",      "dimension_code": "A", "keying": "+", "text": "likes to cooperate with others"},
    {"item_id": "BFI-43", "dimension": "Conscientiousness",  "dimension_code": "C", "keying": "-", "text": "is easily distracted"},
    {"item_id": "BFI-44", "dimension": "Openness",           "dimension_code": "O", "keying": "+", "text": "is sophisticated in art, music, or literature"},
]


def load_bfi_items() -> list[dict]:
    """Return all 44 BFI items, each augmented with item_type and source fields."""
    return [
        {**item, "item_type": "direct", "source": "bfi"}
        for item in _BFI_ITEMS
    ]
