# -*- coding: utf-8 -*-
"""Мост к результатам `lab_3` (INTERFACES.md §3).

Doc2Vec-анализатор `lab_3` нужен `lab_5` для сравнительного описания четырёх
семантических анализаторов. Обращение к `lab_3` — только через этот модуль.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта: /data/Projects/NLP (parents[2] от src/_lab3.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lab_3.src.analyzer import DocVecAnalyzer  # noqa: E402,F401

__all__ = ["DocVecAnalyzer"]
