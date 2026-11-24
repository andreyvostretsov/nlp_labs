# -*- coding: utf-8 -*-
"""Мост к результатам `lab_4` (INTERFACES.md §4).

`lab_4` — семантический анализатор на нейросетях CNN/GRU. Он является
**семантическим ядром** генератора `lab_5`: генератор обучается только на его
корпусе и использует его представления документов, темы и поиск. Обращение к
`lab_4` — только через этот модуль.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта: /data/Projects/NLP (parents[2] от src/_lab4.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lab_4.src.analyzer import NeuralAnalyzer, Prediction, Match  # noqa: E402,F401
from lab_4.src.models import TextCNN, GRUEncoder  # noqa: E402,F401

__all__ = [
    "NeuralAnalyzer",
    "Prediction",
    "Match",
    "TextCNN",
    "GRUEncoder",
]
