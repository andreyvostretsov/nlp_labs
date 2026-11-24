# -*- coding: utf-8 -*-
"""Bootstrap тестов lab_5 (INTERFACES.md §9).

Добавляет корень проекта `/data/Projects/NLP` в `sys.path`, чтобы тесты могли
импортировать модули как `from lab_5.src.<module> import ...` (и через мосты —
`lab_1.src` … `lab_4.src`).
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
