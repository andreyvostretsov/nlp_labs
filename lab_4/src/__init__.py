# -*- coding: utf-8 -*-
"""`lab_4` — семантический анализатор на нейросетях CNN/RNN (расширение `lab_3`).

Публичный API пакета (INTERFACES.md §6). Нормализация переиспользуется из `lab_1`
через мост :mod:`lab_4.src._lab1`; LSA-анализатор и загрузчики корпусов — из
`lab_2` через мост :mod:`lab_4.src._lab2`. Собственные нейросети (TextCNN/GRU) и
фасад :class:`NeuralAnalyzer` реализованы на PyTorch.
"""
from .models import TextCNN, GRUEncoder
from .analyzer import NeuralAnalyzer, Prediction, Match
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
    "TextCNN",
    "GRUEncoder",
    "NeuralAnalyzer",
    "Prediction",
    "Match",
    "SemanticAnalyzer",
    "cosine_similarity",
    "load_text_files",
    "load_lines",
    "load_reviews",
    "doc_tokens",
    "__version__",
]
