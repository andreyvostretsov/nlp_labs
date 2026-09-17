# -*- coding: utf-8 -*-
"""Нормализация русских токенов для проекта `lab_1`.

Конвейер: токенизация -> нижний регистр -> стемминг (Snowball) -> лемматизация
(pymorphy3) -> канонизация по словарю синонимов -> отметка стоп-слова.

Модуль спроектирован так, чтобы импорт никогда не падал: тяжёлые зависимости и
файлы ресурсов поднимаются лениво, а при их отсутствии используются безопасные
заглушки с предупреждением (``warnings.warn``).
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESOURCES_DIR = _HERE / "resources"
STOPWORDS_PATH = RESOURCES_DIR / "stopwords_ru.txt"
SYNONYMS_PATH = RESOURCES_DIR / "synonyms_ru.json"

# Встроенный минимальный набор — используется, если файл стоп-слов недоступен.
_BUILTIN_STOPWORDS = frozenset(
    """
    и в во не что он она оно они на я мы ты вы с со к ко у за по от до из о об для без
    про как а то все весь вся это этот эта эти так его ему ее ей их но да же бы ли
    только мне меня тебя тебе нас вас нам вам себя себе было была были был быть будет
    вот от меня еще еще нет теперь когда даже ну вдруг если уже или ни него нее них до
    вас опять уж вам ведь там тут потом ничего может где здесь есть надо ней для мы
    тебя их чем сам чтоб без будто чего раз тоже себе под тогда кто того потому этого
    какой совсем ним этом один почти мой моя тем чтобы нее сейчас куда зачем всех
    никогда можно при другой хоть после над больше тот через эти нас про всего много
    разве эту впрочем хорошо свою этой перед иногда лучше том нельзя такой им более
    всегда конечно всю между
    """.split()
)


def _load_stopwords(path: Path) -> frozenset:
    """Прочитать стоп-слова: по одному слову в строке, '#'-комментарии и пустые пропускаются."""
    words = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.lower())
    return frozenset(words)


try:
    STOPWORDS = _load_stopwords(STOPWORDS_PATH)
except (OSError, UnicodeError) as exc:  # pragma: no cover - зависит от окружения
    warnings.warn(
        "lab_1.normalization: не удалось прочитать {} ({}); "
        "используется встроенный минимальный набор стоп-слов".format(STOPWORDS_PATH, exc),
        RuntimeWarning,
        stacklevel=2,
    )
    STOPWORDS = _BUILTIN_STOPWORDS


class RussianStemmer:
    """Обёртка над ``nltk.stem.SnowballStemmer('russian')``."""

    def __init__(self):
        from nltk.stem import SnowballStemmer

        self._stemmer = SnowballStemmer("russian")

    def stem(self, word: str) -> str:
        """Стем слова; для пустого/нечитаемого ввода возвращает его же."""
        if word is None:
            return ""
        try:
            return self._stemmer.stem(word)
        except (TypeError, ValueError, AttributeError):
            return word


class RussianLemmatizer:
    """Обёртка над ``pymorphy3.MorphAnalyzer``: нормальная форма первого разбора."""

    def __init__(self):
        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()

    def lemmatize(self, word: str) -> str:
        """Нормальная форма первого разбора; при ошибке/пустом разборе — исходное слово."""
        if word is None:
            return ""
        try:
            parses = self._morph.parse(word)
        except (TypeError, ValueError, AttributeError):
            return word
        if not parses:
            return word
        return parses[0].normal_form or word


class SynonymNormalizer:
    """Канонизация лемм по словарю синонимов.

    Формат файла: ``{"каноническая_лемма": ["синоним_лемма", ...], ...}``.
    Каждый синоним и сам канон отображаются в канон.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else SYNONYMS_PATH
        self._index = {}
        self._load()

    def _load(self):
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.warn(
                "lab_1.normalization: словарь синонимов {} недоступен ({}); "
                "канонизация выполняться не будет".format(self.path, exc),
                RuntimeWarning,
                stacklevel=2,
            )
            self._index = {}
            return
        index = {}
        if isinstance(data, dict):
            for canonical, synonyms in data.items():
                if not isinstance(canonical, str):
                    continue
                canonical = canonical.strip().lower()
                if canonical:
                    index[canonical] = canonical
                if not isinstance(synonyms, (list, tuple, set)):
                    continue
                for synonym in synonyms:
                    if isinstance(synonym, str) and synonym.strip():
                        index[synonym.strip().lower()] = canonical
        self._index = index

    def canonical(self, word: str) -> str:
        """Каноническая форма синонима; если слово не найдено — слово как есть."""
        if word is None:
            return ""
        return self._index.get(word, word)


@dataclass
class NormalizedToken:
    """Все представления одного токена."""

    original: str
    lower: str
    stem: str
    lemma: str
    canonical: str
    is_stopword: bool


_STEMMER = RussianStemmer()
_LEMMATIZER = RussianLemmatizer()
_SYNONYMS = SynonymNormalizer()

# Резервный токенизатор — используется только если lab_1.tokenizer недоступен.
_FALLBACK_TOKEN_RE = re.compile(r"[а-яёa-z]+(?:[-'][а-яёa-z]+)*", re.IGNORECASE)


def _tokenize(text):
    """Токенизация через ``lab_1.tokenizer.tokenize``; при недоступности — локальный split."""
    try:
        from .tokenizer import tokenize
    except ImportError:  # pragma: no cover - tokenizer.py есть в проекте
        warnings.warn(
            "lab_1.normalization: модуль lab_1.tokenizer недоступен, "
            "используется встроенное упрощённое разбиение",
            RuntimeWarning,
            stacklevel=2,
        )

        def tokenize(text):
            if not text:
                return []
            return _FALLBACK_TOKEN_RE.findall(str(text).lower())

    return tokenize(text)


def normalize_token(token: str) -> NormalizedToken:
    """Построить :class:`NormalizedToken` для одного токена."""
    if token is None:
        token = ""
    original = str(token).lower()
    stem = _STEMMER.stem(original)
    lemma = _LEMMATIZER.lemmatize(original)
    canonical = _SYNONYMS.canonical(lemma)
    is_stopword = lemma in STOPWORDS or original in STOPWORDS
    return NormalizedToken(
        original=original,
        lower=original,
        stem=stem,
        lemma=lemma,
        canonical=canonical,
        is_stopword=is_stopword,
    )


def normalize_text(text: str) -> list:
    """Полный конвейер по всем токенам; стоп-слова ОСТАВЛЕНЫ с ``is_stopword=True``."""
    if not text:
        return []
    return [normalize_token(token) for token in _tokenize(text)]


def canonical_tokens(text: str) -> list:
    """Канонические формы без стоп-слов — признаки для наивного байесовского классификатора."""
    return [token.canonical for token in normalize_text(text) if not token.is_stopword]
