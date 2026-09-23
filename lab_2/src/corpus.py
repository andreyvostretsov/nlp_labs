# -*- coding: utf-8 -*-
"""Загрузка корпусов и представление документов токенами (INTERFACES.md §1).

Модуль отвечает только за ввод-вывод и выбор представления документа. Вся
нормализация (токенизация, стемминг, лемматизация, синонимия, стоп-слова)
переиспользуется из `lab_1` через мост :mod:`lab_2.src._lab1` и здесь не
дублируется.

Все функции устойчивы к пустым/мусорным данным: исключение ``ValueError``
возникает только там, где это прямо оговорено контрактом (пустой результат
загрузки, отсутствующий файл, неизвестный режим нормализации).
"""
from __future__ import annotations

from pathlib import Path

from ._lab1 import tokenize, normalize_text, canonical_tokens
from ._lab1 import load_corpus

__all__ = ["load_text_files", "load_lines", "load_reviews", "doc_tokens"]

#: Кодировка и стратегия обработки ошибок для пользовательских текстовых файлов.
_FILE_ENCODING = "utf-8"
_FILE_ERRORS = "replace"

#: Режимы, извлекающие поле из :class:`lab_1.src.normalization.NormalizedToken`.
_TOKEN_FIELDS = {
    "stem": "stem",
    "lemma": "lemma",
}

_KNOWN_MODES = ("canonical", "stem", "lemma", "raw")


def _read_text(path) -> str:
    """Прочитать текстовый файл как UTF-8 с ``errors='replace'``.

    Ведущий BOM (``\\ufeff``) считается артефактом файла и отбрасывается — так же
    поступает декодер корпусов `lab_1` (``utf-8-sig`` там пробуется первым).

    :return: содержимое файла или ``None``, если путь отсутствует/нечитаем
        (несуществующий файл, каталог, ``None``, неверный тип пути).
    """
    if path is None:
        return None
    try:
        text = Path(path).read_text(encoding=_FILE_ENCODING, errors=_FILE_ERRORS)
    except (OSError, TypeError, ValueError):
        return None
    return text.lstrip("\ufeff")


def _as_path_list(paths):
    """Привести аргумент к итерируемому списку путей.

    Одиночная строка или :class:`pathlib.Path` трактуется как один путь;
    ``None`` и неитерируемые значения — как пустой список.
    """
    if isinstance(paths, (str, bytes, Path)):
        return [paths]
    try:
        return list(paths)
    except TypeError:
        return []


def _coerce_text(text) -> str:
    """Привести вход к строке: ``None`` -> ``""``, прочее -> ``str(...)``."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def load_text_files(paths) -> list[str]:
    """Загрузить документы: один документ на файл.

    Файлы читаются как UTF-8 с ``errors='replace'`` (битые байты заменяются,
    исключение не возникает) и ``strip``-аются. Несуществующие/нечитаемые файлы
    и документы, пустые после ``strip``, молча пропускаются; порядок документов
    совпадает с порядком ``paths``.

    :param paths: итерируемое путей к файлам (одиночная строка тоже допустима).
    :return: список непустых текстов документов.
    :raises ValueError: ни одного непустого документа получить не удалось.
    """
    documents = []
    total = 0
    for path in _as_path_list(paths):
        total += 1
        text = _read_text(path)
        if text is None:
            continue
        text = text.strip()
        if text:
            documents.append(text)

    if not documents:
        raise ValueError(
            "не удалось загрузить ни одного документа: "
            "из {} переданных путей ни один не оказался читаемым непустым файлом".format(total)
        )
    return documents


def load_lines(path: str) -> list[str]:
    """Загрузить документы: один документ на непустую строку файла.

    Файл читается как UTF-8 с ``errors='replace'``, каждая строка
    ``strip``-ается, пустые строки пропускаются.

    :param path: путь к файлу.
    :return: список непустых строк в исходном порядке.
    :raises ValueError: файл отсутствует/нечитаем или не содержит ни одной
        непустой строки.
    """
    text = _read_text(path)
    if text is None:
        raise ValueError(
            "не удалось прочитать файл {}: файл отсутствует или недоступен".format(path)
        )

    documents = [line.strip() for line in text.splitlines()]
    documents = [line for line in documents if line]
    if not documents:
        raise ValueError("в файле {} нет ни одной непустой строки".format(path))
    return documents


def load_reviews(csv_path: str, limit=None) -> list[str]:
    """Загрузить тексты отзывов из корпуса `lab_1`.

    Формат (RuReviews TSV ``review``/``sentiment`` или пользовательский CSV
    ``label,text``) определяется функцией ``lab_1.src.data.load_corpus``; метки
    отбрасываются, возвращаются ТОЛЬКО тексты в порядке следования.

    :param csv_path: путь к файлу корпуса.
    :param limit: если задан и больше нуля — вернуть первые ``limit`` текстов;
        иначе возвращаются все тексты.
    :return: список текстов отзывов.
    :raises ValueError: путь не задан, файл отсутствует/пуст или в корпусе нет
        ни одной пригодной строки (``ValueError`` из ``load_corpus``).
    """
    if not csv_path:
        raise ValueError("не указан путь к корпусу отзывов")

    rows = load_corpus(csv_path)

    texts = []
    for row in rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and row[1] is not None:
            texts.append(str(row[1]))

    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 0
        if limit_value > 0:
            texts = texts[:limit_value]

    return texts


def doc_tokens(text: str, mode: str = "canonical") -> list[str]:
    """Токены документа в выбранном режиме нормализации.

    Режимы:

    * ``'canonical'`` (по умолчанию) — лемма + синонимы без стоп-слов
      (тот же результат, что ``canonical_tokens``);
    * ``'stem'`` — стеммы без стоп-слов;
    * ``'lemma'`` — леммы без стоп-слов;
    * ``'raw'`` — ``tokenize(text)``: нижний регистр, без нормализации и без
      удаления стоп-слов.

    :param text: текст документа; ``None`` и пустой текст дают ``[]``.
    :param mode: режим нормализации.
    :return: список строк-токенов (возможно пустой).
    :raises ValueError: неизвестный ``mode``.
    """
    key = str(mode).strip().lower() if mode is not None else ""
    if key not in _KNOWN_MODES:
        raise ValueError(
            "неизвестный режим нормализации {!r}; ожидается один из {}".format(mode, _KNOWN_MODES)
        )

    text = _coerce_text(text)

    if key == "raw":
        return tokenize(text)
    if key == "canonical":
        return canonical_tokens(text)

    field = _TOKEN_FIELDS[key]
    return [getattr(token, field) for token in normalize_text(text) if not token.is_stopword]
