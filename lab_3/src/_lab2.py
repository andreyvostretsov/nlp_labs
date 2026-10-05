# -*- coding: utf-8 -*-
"""Мост к результатам `lab_2` (INTERFACES.md §2).

`lab_3` расширяет семантический анализатор, полученный в `lab_2` (LSA), новым
методом на Doc2Vec/Word2Vec. LSA-анализатор, загрузчики корпусов и косинусная
близость переиспользуются из `lab_2` и не дублируются. Единственный разрешённый
способ обращения к `lab_2` — через этот модуль: он добавляет корень проекта в
`sys.path` и реэкспортирует нужные объекты из `lab_2.src`.

Остальные модули `lab_3` импортируют lab_2-объекты только отсюда:
``from ._lab2 import doc_tokens, cosine_similarity, Match``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта: /data/Projects/NLP (parents[2] от src/_lab2.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lab_2.src.analyzer import SemanticAnalyzer, Match  # noqa: E402,F401
from lab_2.src.lsa import cosine_similarity  # noqa: E402,F401
from lab_2.src.corpus import (  # noqa: E402,F401
    load_text_files,
    load_lines,
    load_reviews,
    doc_tokens,
)

__all__ = [
    "SemanticAnalyzer",
    "Match",
    "cosine_similarity",
    "load_text_files",
    "load_lines",
    "load_reviews",
    "doc_tokens",
]
