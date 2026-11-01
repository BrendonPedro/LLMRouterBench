"""capability-profile-v1: frozen, deterministic query-capability feature extractor.

Defined BEFORE any experiment was run (CAPABILITY_PROFILE_MISSION.md, "Frozen
Capability Feature Set"). Do not add, remove or edit features after results are
observed; any change must ship as a new version (capability-profile-v2).

Input: RAW QUERY TEXT ONLY. The extractor never sees dataset name / ID,
benchmark metadata, model outcomes, winner labels or fold information.

The only non-regex component is the MiniLM tokenizer
(sentence-transformers/all-MiniLM-L6-v2), used for minilm_token_count. The count
includes the [CLS]/[SEP] special tokens, i.e. it is the length the encoder would
see with truncation disabled, so exceeds_256_tokens == "truncated by the
deployed max_length=256 MiniLM pipeline".

Feature types (used for preprocessing in the router):
  CONTINUOUS_FEATURES -> standardized with outer-training statistics
  BINARY_FEATURES     -> left as 0/1
"""

import re

VERSION = "capability-profile-v1"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MAX_TOKENS = 256

FEATURES = [
    "character_count",
    "whitespace_word_count",
    "minilm_token_count",
    "exceeds_256_tokens",
    "contains_digits",
    "numeric_density",
    "contains_math_expression",
    "contains_code",
    "requests_code",
    "multiple_choice_structure",
    "explicit_answer_format",
    "reasoning_cue",
    "constraint_count",
    "has_embedded_context",
    "structural_line_count",
]
CONTINUOUS_FEATURES = [
    "character_count",
    "whitespace_word_count",
    "minilm_token_count",
    "numeric_density",
    "constraint_count",
    "structural_line_count",
]
BINARY_FEATURES = [f for f in FEATURES if f not in CONTINUOUS_FEATURES]

_I = re.IGNORECASE
_M = re.MULTILINE

# 7. math expressions / equations
_MATH = [
    re.compile(r"\d\s*[+\-*/^=×÷]\s*\d"),                    # 3 + 4, 2^5, 6/7, x=3 style digit ops
    re.compile(r"\b[a-zA-Z]\s*(?:\^|\*\*)\s*[\d{(a-zA-Z]"),  # x^2, n**k
    re.compile(r"(?<![A-Za-z])[a-zA-Z]\s*=\s*[-\d(a-zA-Z\\]"),  # y = 2x, f = ...
    re.compile(r"\d\s*%"),                                  # percentages
    re.compile(r"\bsqrt\s*\(|\\sqrt|\\frac|\\dfrac|\\cdot|\\times|\\sum|\\int|\\pi\b|\\le\b|\\ge\b|\\neq|\\infty|\\binom|\\log|\\sin|\\cos"),
    re.compile(r"\$[^$\n]{1,200}\$|\\\(|\\\["),             # LaTeX inline/display math
    re.compile(r"[√π≤≥≠∑∫∞±]"),
    re.compile(r"\b\d+\s*/\s*\d+\b"),                       # fractions
]

# 8. code blocks / programming syntax present in the query
_CODE = [
    re.compile(r"```"),
    re.compile(r"^\s*def\s+\w+\s*\(", _M),
    re.compile(r"^\s*class\s+\w+\s*[(:{]", _M),
    re.compile(r"^\s*(?:import\s+[\w.]+|from\s+[\w.]+\s+import\b)", _M),
    re.compile(r"#include\s*<"),
    re.compile(r"\bfunction\s*\w*\s*\([^)]*\)\s*\{"),
    re.compile(r"\b(?:public|private|protected)\s+(?:static\s+)?\w+[\w<>\[\]]*\s+\w+\s*\("),
    re.compile(r"^>>>\s", _M),
    re.compile(r"\bprint\s*\(|\bconsole\.log\s*\(|\bSystem\.out\.print"),
    re.compile(r"^\s{2,}(?:return|for|while|if|elif|else)\b.*[:;{]?\s*$", _M),
    re.compile(r"\bself\.\w+"),
    re.compile(r"\blambda\s+\w*\s*:"),
    re.compile(r"\bSELECT\b[\s\S]{1,200}\bFROM\b"),
]

# 9. instructions requesting code
_REQ_CODE = [
    re.compile(r"\b(?:write|implement|complete|create|define|debug|fix|code)\b[^.\n]{0,60}"
               r"\b(?:code|program|function|script|class|method|algorithm|python|java|javascript|c\+\+|sql)\b", _I),
    re.compile(r"\b(?:python|javascript|java|c\+\+|rust|go)\s+(?:function|code|program|script|implementation)\b", _I),
    re.compile(r"\byour\s+(?:code|function|program|solution\s+code)\b", _I),
]

# 10. multiple-choice answer options
_MC_LETTER = re.compile(r"(?:^|\n|\s)\(?([A-E])[).:]\s+\S")
_MC_KEYWORDS = re.compile(
    r"\b(?:options?\s*:|choices?\s*:|answer choices|which of the following|select one|choose one|"
    r"choose the (?:correct|best)|select the (?:correct|best))", _I)

# 11. explicit answer-format instructions
_FORMAT = re.compile(
    r"answer only|only the answer|only answer|letter only|only the letter|just the letter|"
    r"\bjson\b|\\boxed|\bboxed\b|one word|single word|yes/no|yes or no|true or false|true/false|"
    r"output format|answer format|in the format|format:|your answer should|your response should|"
    r"respond with|return only|output only|final answer|answer with|end your response with|"
    r"in the form of|round(?:ed)? to|decimal places?|the answer is", _I)

# 12. reasoning cues
_REASON = re.compile(
    r"\bexplain\w*|\breason\w*|\bderiv\w*|\bprove\b|\bproof\b|\bcalculat\w*|\bcompute\b|"
    r"step[\s-]by[\s-]step|\bwhy\b|\bjustif\w*|\banaly[sz]\w*|\bshow that\b|\bshow your work\b", _I)

# 13. explicit constraint words / phrases (counted)
_CONSTRAINT = re.compile(
    r"\bmust\b|\bonly\b|\bexactly\b|\bdo not\b|\bdon't\b|\bat least\b|\bat most\b|\brequired?\b|"
    r"\brequires\b|\bwithout\b|\binclude\w*|\bexclud\w*|\bnever\b|\bshould not\b|"
    r"\bno more than\b|\bno less than\b", _I)

# 14. embedded context
_CONTEXT_KW = re.compile(
    r"\b(?:passage|context|document|article|paragraph|table|conversation|dialogue|story|premise|"
    r"excerpt|report|text below|following text|transcript)\b", _I)
_TABLE = re.compile(r"^\s*\|.*\|.*\|\s*$", _M)
_EMBED_MIN_WORDS = 100


def _any(patterns, text):
    return int(any(p.search(text) for p in patterns))


def multiple_choice(text):
    letters = set(_MC_LETTER.findall(text))
    if {"A", "B"} <= letters and (len(letters) >= 3 or "(B)" in text):
        return 1
    return int(bool(_MC_KEYWORDS.search(text)))


def embedded_context(text, n_words):
    if n_words < _EMBED_MIN_WORDS:
        return 0
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return int(bool(_CONTEXT_KW.search(text)) or bool(_TABLE.search(text)) or len(paragraphs) >= 3)


def extract(text, n_tokens):
    """Return the 15 capability-profile-v1 features for one raw query.

    n_tokens: MiniLM token count (special tokens included, no truncation).
    """
    n_chars = len(text)
    n_words = len(text.split())
    n_digits = sum(ch.isdigit() for ch in text)
    return {
        "character_count": n_chars,
        "whitespace_word_count": n_words,
        "minilm_token_count": int(n_tokens),
        "exceeds_256_tokens": int(n_tokens > MINILM_MAX_TOKENS),
        "contains_digits": int(n_digits > 0),
        "numeric_density": n_digits / n_chars if n_chars else 0.0,
        "contains_math_expression": _any(_MATH, text),
        "contains_code": _any(_CODE, text),
        "requests_code": _any(_REQ_CODE, text),
        "multiple_choice_structure": multiple_choice(text),
        "explicit_answer_format": int(bool(_FORMAT.search(text))),
        "reasoning_cue": int(bool(_REASON.search(text))),
        "constraint_count": len(_CONSTRAINT.findall(text)),
        "has_embedded_context": embedded_context(text, n_words),
        "structural_line_count": sum(1 for ln in text.splitlines() if ln.strip()),
    }


def load_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot_download(MINILM_MODEL, local_files_only=True))


def token_counts(texts, tokenizer):
    enc = tokenizer(list(texts), add_special_tokens=True, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def extract_many(texts, tokenizer):
    return [extract(t, n) for t, n in zip(texts, token_counts(texts, tokenizer))]
