import re

NON_ENGLISH_SCRIPT_PATTERNS = [
    r"[\u0900-\u097F]",  # Hindi / Devanagari / Marathi
    r"[\u0C00-\u0C7F]",  # Telugu
    r"[\u0B80-\u0BFF]",  # Tamil
    r"[\u0C80-\u0CFF]",  # Kannada
    r"[\u0D00-\u0D7F]",  # Malayalam
    r"[\u0980-\u09FF]",  # Bengali
    r"[\u0A80-\u0AFF]",  # Gujarati
    r"[\u0A00-\u0A7F]",  # Punjabi
    r"[\u0600-\u06FF]",  # Arabic / Urdu
    r"[\u4E00-\u9FFF]",  # Chinese
    r"[\u3040-\u30FF]",  # Japanese
    r"[\uAC00-\uD7AF]",  # Korean
]


def contains_non_english_script(text: str) -> bool:
    return any(re.search(pattern, text or "") for pattern in NON_ENGLISH_SCRIPT_PATTERNS)
