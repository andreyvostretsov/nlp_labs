# -*- coding: utf-8 -*-
"""Загрузка и подготовка корпусов отзывов для проекта `lab_1` (INTERFACES.md §5).

Поддерживаются два формата входных данных:

* RuReviews TSV — колонки ``review`` / ``sentiment`` (разделитель ``\\t``),
  метка уже записана словом, но в исходном файле нейтральный класс ошибочно
  назван ``neautral`` — он отображается в канонический ``neutral`` (Ruling 1);
* пользовательский CSV — колонки ``label,text`` (разделитель ``,``).

Разделитель и наличие заголовка определяются автоматически, поэтому оба
варианта читаются одним и тем же вызовом :func:`load_corpus`.
"""
from __future__ import annotations

import csv
import io
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

__all__ = [
    "CLASS_LABELS",
    "LABEL_ALIASES",
    "RUREVIEWS_URL",
    "download_rureviews",
    "load_corpus",
    "stratified_split",
    "make_seed_sample",
]

#: Канонический порядок классов (INTERFACES.md, «Метки классов»).
CLASS_LABELS = ["negative", "neutral", "positive"]

#: Синонимы/опечатки меток реальных корпусов -> каноническая метка.
LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "neautral": "neutral",  # опечатка в RuReviews (Ruling 1)
    "neutral": "neutral",
    "neut": "neutral",
    "positive": "positive",
    "pos": "positive",
}

RUREVIEWS_URL = (
    "https://raw.githubusercontent.com/sismetanin/rureviews/master/"
    "women-clothing-accessories.3-class.balanced.csv"
)

#: Слова, по которым узнаётся строка заголовка.
_HEADER_KEYWORDS = frozenset(
    {
        "review",
        "sentiment",
        "label",
        "text",
        "отзыв",
        "тональность",
        "метка",
        "текст",
    }
)

_SENTIMENT_HEADER_KEYWORDS = frozenset({"review", "sentiment", "отзыв", "тональность"})
_LABEL_HEADER_KEYWORDS = frozenset({"label", "text", "метка", "текст"})

_CANDIDATE_DELIMITERS = ("\t", ",", ";")


def _normalize_label(value) -> str:
    """Привести метку к канонической; неизвестная метка -> ``""``."""
    if value is None:
        return ""
    label = str(value).strip().strip("\ufeff").lower()
    return LABEL_ALIASES.get(label, "")


def _normalize_text(value) -> str:
    """Нормализовать текст отзыва: убрать переводы строк и лишние пробелы."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _decode(raw: bytes) -> str:
    """Декодировать файл корпуса (UTF-8, UTF-8 с BOM, cp1251 как запасной вариант)."""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _detect_delimiter(sample: str) -> str:
    """Определить разделитель по первой непустой строке (голосование по счётчикам)."""
    for line in sample.splitlines():
        if not line.strip():
            continue
        counts = {delimiter: line.count(delimiter) for delimiter in _CANDIDATE_DELIMITERS}
        best = max(counts, key=lambda delimiter: counts[delimiter])
        if counts[best] > 0:
            return best
        break
    return ","


def _is_header(parts: list) -> bool:
    """Является ли строка заголовком: все непустые ячейки — известные ключевые слова."""
    cells = [str(cell).strip().strip("\ufeff").lower() for cell in parts]
    non_empty = [cell for cell in cells if cell]
    if not non_empty:
        return False
    return all(cell in _HEADER_KEYWORDS for cell in non_empty)


def _row_orientation(header: list) -> str:
    """Определить порядок колонок по заголовку: ``sentiment_first`` или ``label_first``."""
    cells = {str(cell).strip().lower() for cell in header}
    sentiment_hits = len(cells & _SENTIMENT_HEADER_KEYWORDS)
    label_hits = len(cells & _LABEL_HEADER_KEYWORDS)
    if sentiment_hits > label_hits:
        return "sentiment_first"
    return "label_first"


def _split_row(parts: list, delimiter: str) -> list:
    """Разбить строку на ячейки; при одноколоночном вводе — по ведущему разделителю."""
    if len(parts) >= 2:
        return parts
    if len(parts) == 1 and delimiter not in ("\t",):
        return parts[0].split(delimiter)
    return parts


def load_corpus(path: str) -> list:
    """Загрузить корпус отзывов: список пар ``(label, text)``.

    Автоопределение формата:

    * разделитель — ``\\t`` (RuReviews) или ``,`` (пользовательский CSV);
    * заголовок — первая строка, все ячейки которой являются известными
      ключевыми словами (``review``/``sentiment``/``label``/``text``), пропускается;
    * RuReviews TSV — порядок колонок ``text,sentiment``; CSV — ``label,text``;
    * метки нормализуются, в том числе ``neautral`` -> ``neutral``;
    * пустые и битые строки молча пропускаются.

    :param path: путь к файлу корпуса.
    :return: ``[(label, text), ...]`` в порядке следования в файле.
    :raises ValueError: файл отсутствует, пуст или не содержит ни одной
        пригодной строки с известной меткой.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("не удалось прочитать корпус {}: {}".format(path, exc)) from exc

    if not raw.strip():
        raise ValueError("корпус пуст: {}".format(path))

    text_data = _decode(raw)
    delimiter = _detect_delimiter(text_data)

    rows = []
    orientation = "label_first"
    got_header = False

    for parts in csv.reader(io.StringIO(text_data, newline=""), delimiter=delimiter):
        parts = [part for part in parts]
        if not any(str(part).strip() for part in parts):
            continue
        if not got_header and _is_header(parts):
            orientation = _row_orientation(parts)
            got_header = True
            continue

        cells = _split_row(parts, delimiter)
        if len(cells) < 2:
            continue

        if orientation == "sentiment_first":
            raw_label, raw_text = cells[-1], cells[0]
        else:
            raw_label, raw_text = cells[0], cells[1] if len(cells) == 2 else ",".join(cells[1:])

        label = _normalize_label(raw_label)
        text = _normalize_text(raw_text)
        if not label or not text:
            continue
        rows.append((label, text))

    if not rows:
        raise ValueError(
            "в корпусе {} не найдено ни одной строки с известной меткой "
            "{}".format(path, CLASS_LABELS)
        )
    return rows


def stratified_split(rows: list, test_size: float = 0.2, seed: int = 42):
    """Разделить строки на обучающую и тестовую части со стратификацией по классам.

    Порядок строк внутри каждой части перемешивается детерминированно
    (генератор ``random.Random(seed)``), поэтому результат воспроизводим.

    :param rows: список пар ``(label, text)``.
    :param test_size: доля тестовой части (0.0 <= test_size < 1.0).
    :param seed: зерно генератора случайных чисел.
    :return: ``(train_rows, test_rows)``.
    """
    rows = list(rows or [])
    if not rows:
        return [], []

    try:
        test_size = float(test_size)
    except (TypeError, ValueError):
        test_size = 0.2
    if not 0.0 <= test_size < 1.0:
        test_size = min(max(test_size, 0.0), 0.99)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    rng = random.Random(seed)
    train_rows = []
    test_rows = []
    for label in sorted(grouped):
        group = list(grouped[label])
        rng.shuffle(group)
        n_test = int(round(len(group) * test_size))
        if test_size > 0.0 and n_test == 0 and len(group) > 1:
            n_test = 1
        if n_test >= len(group):
            n_test = max(len(group) - 1, 0)
        test_rows.extend(group[:n_test])
        train_rows.extend(group[n_test:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def make_seed_sample(source_path: str, out_path: str, n: int = 1000, seed: int = 42) -> None:
    """Сделать сбалансированную выборку ``n`` строк и записать её как CSV ``label,text``.

    Классы берутся поровну (остаток распределяется по первым, лексикографически
    отсортированным, классам); строки перемешиваются детерминированно.

    :param source_path: исходный корпус (любой формат, который понимает :func:`load_corpus`).
    :param out_path: путь для записи CSV с заголовком ``label,text``.
    :param n: желаемое число строк (не больше размера корпуса).
    :param seed: зерно генератора случайных чисел.
    """
    rows = load_corpus(source_path)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1000
    n = max(n, 0)

    grouped = defaultdict(list)
    for label, text in rows:
        grouped[label].append((label, text))

    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    labels = sorted(grouped)
    sample = []
    if labels and n:
        per_class = n // len(labels)
        remainder = n % len(labels)
        for index, label in enumerate(labels):
            quota = per_class + (1 if index < remainder else 0)
            sample.extend(grouped[label][:quota])
        rng.shuffle(sample)

    destination = Path(out_path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "text"])
        for label, text in sample:
            writer.writerow([label, text])


def download_rureviews(dest: str) -> str:
    """Скачать RuReviews TSV в ``dest`` (если файла нет или он пуст).

    :param dest: путь назначения.
    :return: путь ``dest``.
    :raises RuntimeError: сеть недоступна — используйте встроенную
        ``lab_1/resources/train_data.csv``.
    """
    destination = Path(dest)
    if destination.exists() and destination.stat().st_size > 0:
        return str(dest)

    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(RUREVIEWS_URL, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            "не удалось скачать RuReviews ({}) из {}: {}. "
            "Скачайте файл вручную или используйте встроенную "
            "lab_1/resources/train_data.csv".format(dest, RUREVIEWS_URL, exc)
        ) from exc

    if not payload:
        raise RuntimeError(
            "получен пустой ответ при скачивании RuReviews в {}. "
            "Используйте встроенную lab_1/resources/train_data.csv".format(dest)
        )

    destination.write_bytes(payload)
    return str(dest)


def corpus_stats(rows: list) -> Counter:  # pragma: no cover - вспомогательная утилита
    """Число документов по классам (удобно для отчётов и ноутбука)."""
    return Counter(label for label, _text in rows or [])
