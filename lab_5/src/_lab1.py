# -*- coding: utf-8 -*-
"""Мост к результатам `lab_1` (INTERFACES.md §1).

`lab_5` переиспользует из `lab_1` конвейер нормализации (токенизация, стемминг,
лемматизация, синонимия, стоп-слова), разбиение на предложения и загрузчик
корпуса RuReviews. Единственный разрешённый способ обращения к `lab_1` — через
этот модуль: он добавляет корень проекта в ``sys.path`` и реэкспортирует нужные
имена из namespace-пакета `lab_1.src`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта: /data/Projects/NLP (parents[2] от src/_lab1.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lab_1.src.tokenizer import tokenize, sent_tokenize  # noqa: E402,F401
from lab_1.src.normalization import (  # noqa: E402,F401
    STOPWORDS,
    normalize_text,
    normalize_token,
    canonical_tokens,
    NormalizedToken,
)
from lab_1.src.data import load_corpus  # noqa: E402,F401
from lab_1.src.sentiment import SentimentAnalyzer, SentimentResult  # noqa: E402,F401

__all__ = [
    "STOPWORDS",
    "tokenize",
    "sent_tokenize",
    "normalize_text",
    "normalize_token",
    "canonical_tokens",
    "NormalizedToken",
    "load_corpus",
    "SentimentAnalyzer",
    "SentimentResult",
]
