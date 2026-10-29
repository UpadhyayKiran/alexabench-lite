"""Script diagnostics are transparent heuristics, not language identification."""
import re
import unicodedata
from .text import normalize


def transcript_diagnostics(text, language):
    if not isinstance(text, str):
        raise TypeError("model inference must return a transcript string")
    normalized = normalize(text)
    letters = [c for c in normalized if unicodedata.category(c).startswith("L")]
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    devanagari = sum("DEVANAGARI" in unicodedata.name(c, "") for c in letters)
    # This only nominates outputs for human review, never declares their language.
    english_cues = set(re.findall(r"[a-z]+", normalized)) & {
        "the", "and", "is", "was", "were", "of", "to", "in", "for", "with", "that", "a"}
    latin_only = bool(letters) and latin == len(letters)
    return dict(empty_transcript=not bool(normalized), latin_letters=latin,
                devanagari_letters=devanagari,
                hindi_latin_only=language == "hi" and latin_only,
                hindi_no_devanagari=language == "hi" and bool(normalized) and not devanagari,
                unexpected_english_candidate=language == "hi" and latin_only and len(english_cues) >= 3)
