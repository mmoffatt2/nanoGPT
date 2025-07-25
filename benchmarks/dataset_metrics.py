import re
from typing import Dict

try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
except Exception:
    SpellChecker = None
    spell = None


def spelling_correctness(text: str) -> float:
    """Return the fraction of correctly spelled English words."""
    if spell is None:
        return 0.0
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    misspelled = spell.unknown(words)
    return 1.0 - len(misspelled) / len(words)


def average_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.strip().split()) for s in sentences if s.strip()]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def lexical_diversity(text: str) -> float:
    """Ratio of unique words to total words."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def run_all(text: str) -> Dict[str, float]:
    """Compute all dataset metrics on provided text."""
    return {
        "spelling_correctness": spelling_correctness(text),
        "avg_sentence_length": average_sentence_length(text),
        "lexical_diversity": lexical_diversity(text),
    }
