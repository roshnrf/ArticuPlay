from phonemizer import phonemize

_ESPEAK_LANGUAGE = {
    "en": "en-us",
    "ar": "ar",
    "hi": "hi",
    "zh": "cmn",
}


def to_ipa(text: str, language: str = "en") -> str:
    """Convert text to IPA via espeak-ng. Single source of truth for target/child IPA,
    used at word-content seed time now and by compare_ipa() in Week 2."""
    espeak_lang = _ESPEAK_LANGUAGE.get(language, "en-us")
    return phonemize(text, language=espeak_lang, backend="espeak", strip=True)


# espeak-ng's English (en-us) IPA symbol inventory. Multi-character symbols first
# (longest match wins) so e.g. "tʃ" is never split into "t" + "ʃ".
_MULTI_CHAR_PHONEMES = [
    # affricates
    "tʃ", "dʒ",
    # diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪə", "eə", "ʊə", "əʊ",
    # long vowels (vowel + length mark)
    "iː", "uː", "ɑː", "ɔː", "ɜː",
]
_STRESS_MARKS = ("ˈ", "ˌ")
_SYLLABIC_MARK = "̩"  # combining vertical line below, e.g. button -> bʌtn̩


def tokenize_ipa(ipa: str) -> list[str]:
    """Split a raw espeak-ng IPA string into individual phoneme tokens.

    Greedy longest-match against the known multi-character symbol table; anything
    left over is a single character. Stress marks are positional, not sounds, so
    they're stripped rather than tokenized. A syllabic-consonant diacritic attaches
    to the preceding consonant instead of becoming its own token."""
    for mark in _STRESS_MARKS:
        ipa = ipa.replace(mark, "")

    tokens: list[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] == " ":
            i += 1
            continue
        for symbol in _MULTI_CHAR_PHONEMES:
            if ipa.startswith(symbol, i):
                tokens.append(symbol)
                i += len(symbol)
                break
        else:
            if ipa[i] == _SYLLABIC_MARK and tokens:
                tokens[-1] += _SYLLABIC_MARK
            else:
                tokens.append(ipa[i])
            i += 1
    return tokens
