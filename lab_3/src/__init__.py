# -*- coding: utf-8 -*-
"""`lab_3` — семантический анализатор на Doc2Vec/Word2Vec (расширение `lab_2`).

Публичный API пакета (INTERFACES.md §6). Нормализация переиспользуется из `lab_1`
через мост :mod:`lab_3.src._lab1`; LSA-анализатор и загрузчики корпусов — из
`lab_2` через мост :mod:`lab_3.src._lab2`. Собственные shallow-модели (skip-gram /
PV-DBOW) и фасад :class:`DocVecAnalyzer` реализованы на numpy.
"""
from .embeddings import Word2Vec, Doc2Vec
from .analyzer import DocVecAnalyzer, Match
from ._lab2 import (  # noqa: F401
    SemanticAnalyzer,
    cosine_similarity,
    load_text_files,
    load_lines,
    load_reviews,
    doc_tokens,
)

__version__ = "0.1.0"

__all__ = [
    "Word2Vec",
    "Doc2Vec",
    "DocVecAnalyzer",
    "Match",
    "SemanticAnalyzer",
    "cosine_similarity",
    "load_text_files",
    "load_lines",
    "load_reviews",
    "doc_tokens",
    "__version__",
]
