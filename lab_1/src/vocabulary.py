# -*- coding: utf-8 -*-
"""Построение словаря (частотного) по корпусу документов для проекта `lab_1`."""
from __future__ import annotations

from collections import Counter

from .normalization import canonical_tokens


def build_vocabulary(documents) -> dict:
    """Частотный словарь канонических форм по итерируемому корпусу строк.

    Каждый документ проходит конвейер ``tokenize + normalize``; стоп-слова
    исключаются, словоформы и синонимы сводятся к канонической лемме.

    :param documents: итерируемое строк (``None``/пустое -> ``{}``).
    :return: ``{каноническая_лемма: частота}`` — по убыванию частоты,
        при равенстве частот — лексикографически по возрастанию.
    """
    if documents is None:
        return {}
    counter = Counter()
    for document in documents:
        if not document:
            continue
        counter.update(canonical_tokens(document))
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered)
