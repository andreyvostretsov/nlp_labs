# -*- coding: utf-8 -*-
"""`lab_1` — анализ тональности русскоязычных отзывов.

Публичный API пакета (INTERFACES.md §7).
"""
from .tokenizer import tokenize, sent_tokenize
from .normalization import normalize_text, normalize_token, canonical_tokens, NormalizedToken
from .vocabulary import build_vocabulary
from .sentiment import SentimentAnalyzer, SentimentResult

__version__ = "0.1.0"

__all__ = [
    "tokenize",
    "sent_tokenize",
    "normalize_text",
    "normalize_token",
    "canonical_tokens",
    "NormalizedToken",
    "build_vocabulary",
    "SentimentAnalyzer",
    "SentimentResult",
    "__version__",
]
