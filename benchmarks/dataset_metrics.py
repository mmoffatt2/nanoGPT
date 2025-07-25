import re
import warnings
from typing import Dict

# Regex pattern to remove ANSI escape codes used for colorization
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text."""
    return ANSI_ESCAPE.sub("", text)


_warned_spellchecker_missing = False


def _load_spellchecker():
    """Attempt to load pyspellchecker and return an instance or None."""
    global SpellChecker, spell, _warned_spellchecker_missing
    if spell is not None:
        return spell
    try:
        from spellchecker import SpellChecker as _SC
        SpellChecker = _SC
        spell = SpellChecker()
    except Exception:
        SpellChecker = None
        spell = None
        if not _warned_spellchecker_missing:
            warnings.warn(
                "pyspellchecker not installed; spelling correctness will be 0.0",
                RuntimeWarning,
            )
            _warned_spellchecker_missing = True
    return spell

spell = None
SpellChecker = None


def spelling_correctness(text: str) -> float:
    """Return the fraction of correctly spelled English words."""
    if _load_spellchecker() is None:
        return 0.0
    clean = _strip_ansi(text)
    words = re.findall(r"[A-Za-z']+", clean.lower())
    if not words:
        return 0.0
    misspelled = spell.unknown(words)
    return 1.0 - len(misspelled) / len(words)


def average_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    clean = _strip_ansi(text)
    sentences = re.split(r"[.!?]+", clean)
    lengths = [len(s.strip().split()) for s in sentences if s.strip()]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def lexical_diversity(text: str) -> float:
    """Ratio of unique words to total words."""
    clean = _strip_ansi(text)
    words = re.findall(r"[A-Za-z']+", clean.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def run_all(text: str) -> Dict[str, float]:
    """Compute all dataset metrics on provided text."""
    clean = _strip_ansi(text)
    return {
        "spelling_correctness": spelling_correctness(clean),
        "avg_sentence_length": average_sentence_length(clean),
        "lexical_diversity": lexical_diversity(clean),
    }
