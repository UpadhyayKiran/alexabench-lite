"""Versioned normalization and edit-distance metrics, without third-party dependencies."""
import unicodedata

NORMALIZATION_VERSION = "nfc-casefold-punctuation-space-v1"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    # Preserve combining marks (essential for Hindi), digits, symbols and accents.
    # Discard only invisible joiners/BOM; punctuation becomes a word boundary.
    text = "".join(" " if unicodedata.category(c).startswith("P") else c
                   for c in text if c not in "\u200c\u200d\ufeff")
    return " ".join(text.split())


def distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("empty reference after normalization")
    words = ref.split()
    # CER uses Unicode code points and excludes whitespace, not grapheme clusters.
    chars, predicted_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    we, ce = distance(words, hyp.split()), distance(chars, predicted_chars)
    return dict(reference_normalized=ref, hypothesis_normalized=hyp,
                word_errors=we, reference_words=len(words),
                character_errors=ce, reference_characters=len(chars),
                wer=we / len(words), cer=ce / len(chars))
