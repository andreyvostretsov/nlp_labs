# -*- coding: utf-8 -*-
"""`lab_2` — прототип семантического анализатора (Latent Semantic Analysis).

Публичный API пакета (INTERFACES.md §5). Нормализация переиспользуется из
`lab_1` через мост :mod:`lab_2.src._lab1`.
"""
from .corpus import load_text_files, load_lines, load_reviews, doc_tokens
from .vectorizer import TfidfVectorizer
from .lsa import LsaModel, cosine_similarity
from .analyzer import SemanticAnalyzer, Match

__version__ = "0.1.0"

__all__ = [
    "load_text_files",
    "load_lines",
    "load_reviews",
    "doc_tokens",
    "TfidfVectorizer",
    "LsaModel",
    "cosine_similarity",
    "SemanticAnalyzer",
    "Match",
    "__version__",
]
