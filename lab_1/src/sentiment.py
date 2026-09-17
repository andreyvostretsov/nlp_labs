# -*- coding: utf-8 -*-
"""Анализ тональности русскоязычных отзывов (INTERFACES.md §6).

Признаки документа — канонические токены :func:`lab_1.normalization.canonical_tokens`
(лемматизация + синонимы, стоп-слова исключены). Классификатор —
:class:`lab_1.naive_bayes.MultinomialNB` на стандартной библиотеке.

Перед обучением токены, которые встречаются менее чем в ``MIN_DF = 3`` документах
обучающей части, отбрасываются — это убирает шум и заметно уменьшает размер модели.
Порог доступен как параметр ``train(..., min_df=3)`` и адаптивен: фактически
применяется ``max(1, min(min_df, n_train // 10))``, поэтому на крошечных корпусах
словарь не схлопывается, а при ``vocab_size < 10`` выдаётся ``RuntimeWarning``.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .data import CLASS_LABELS, load_corpus, stratified_split
from .naive_bayes import MultinomialNB, load_model, save_model
from .normalization import canonical_tokens

__all__ = ["SentimentResult", "SentimentAnalyzer", "MIN_DF", "CLASS_LABELS"]

#: Минимальная документная частота токена, чтобы он попал в признаки (min_df).
#: Порог адаптивный: применяется ``max(1, min(MIN_DF, n_train // 10))``.
MIN_DF = 3

#: Ниже этого размера словаря обучение сопровождается RuntimeWarning.
MIN_VOCAB_WARNING_THRESHOLD = 10

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _HERE / "models" / "sentiment_model.json"
DEFAULT_SEED_PATH = _HERE / "resources" / "train_data.csv"

_NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Результат анализа одного текста."""

    label: str
    probabilities: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)


class SentimentAnalyzer:
    """Обёртка над MultinomialNB: обучение, сохранение и анализ текстов."""

    def __init__(self, model_path=None):
        """Загрузить модель.

        :param model_path: путь к JSON модели; по умолчанию
            ``<пакет>/models/sentiment_model.json``. Если файла нет — модель
            обучается на ``<пакет>/resources/train_data.csv`` и сохраняется
            по указанному пути (с предупреждением).
        :raises ValueError: модель несовместима или в ней нет трёх классов.
        """
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.model = None

        if self.model_path.exists():
            self.model = load_model(str(self.model_path))
        else:
            warnings.warn(
                "lab_1.sentiment: модель {} не найдена, обучение на {} "
                "(результат будет сохранён)".format(self.model_path, DEFAULT_SEED_PATH),
                RuntimeWarning,
                stacklevel=2,
            )
            stats = self.train(
                corpus_csv=str(DEFAULT_SEED_PATH), out_model=str(self.model_path)
            )
            self.model = load_model(str(self.model_path))
            self._report_training(stats)

        self._validate_model()

    # -- внутреннее --------------------------------------------------------------------

    @staticmethod
    def _report_training(stats: dict) -> None:
        print(
            "Модель обучена на встроенной выборке: accuracy={:.3f}, train={}, test={}".format(
                stats["accuracy"], stats["n_train"], stats["n_test"]
            )
        )

    def _validate_model(self) -> None:
        """Проверить, что в модели ровно три канонических класса."""
        classes = list(getattr(self.model, "classes", []) or [])
        if set(classes) != set(CLASS_LABELS) or len(classes) != len(CLASS_LABELS):
            raise ValueError(
                "несовместимая модель: ожидались классы {}, получены {}; "
                "переобучите модель".format(CLASS_LABELS, classes)
            )

    @staticmethod
    def _features(text: str) -> list:
        """Канонические токены текста — признаки для NB."""
        return canonical_tokens(text)

    @staticmethod
    def _token_document_frequency(documents) -> Counter:
        """Число документов, в которых встретился каждый токен (document frequency)."""
        counter = Counter()
        for document in documents:
            counter.update(set(document))
        return counter

    # -- анализ ------------------------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        """Определить тональность текста.

        Пустой или нечитаемый текст (нет ни одного канонического токена) даёт
        ``label='neutral'``, ``probabilities={}`` и ``tokens=[]``.
        """
        tokens = self._features(text)
        if not tokens:
            return SentimentResult(label=_NEUTRAL, probabilities={}, tokens=[])

        probabilities = self.model.predict_proba(tokens)
        label = self.model.predict(tokens)
        ordered = {
            class_label: float(probabilities.get(class_label, 0.0))
            for class_label in CLASS_LABELS
        }
        return SentimentResult(label=label or _NEUTRAL, probabilities=ordered, tokens=tokens)

    def analyze_file(self, path: str) -> SentimentResult:
        """Прочитать текст из файла и проанализировать его."""
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.analyze(text)

    # -- обучение ----------------------------------------------------------------------

    def train(
        self,
        corpus_csv: str,
        out_model: str,
        limit=None,
        alpha: float = 1.0,
        test_size: float = 0.2,
        min_df: int = MIN_DF,
    ) -> dict:
        """Обучить модель на корпусе и сохранить её.

        :param corpus_csv: путь к корпусу (любой формат из :func:`lab_1.data.load_corpus`).
        :param out_model: путь для сохранения JSON модели.
        :param limit: ограничить объём корпуса ``limit`` строками (сбалансированно
            по классам); ``None`` — использовать весь корпус.
        :param alpha: параметр сглаживания (add-alpha), строго больше нуля.
        :param test_size: доля отложенной выборки для оценки accuracy (0 < test_size < 1).
        :param min_df: минимальная документная частота токена. Порог адаптивный —
            фактически применяется ``max(1, min(min_df, n_train // 10))``, поэтому
            словарь не может схлопнуться в пустой.
        :return: ``{'accuracy': float, 'n_train': int, 'n_test': int}``;
            ``n_test == 0`` означает, что корпус слишком мал для отложенной
            выборки и accuracy посчитана на обучающей части.
        :raises ValueError: корпус пуст/нечитаем, не содержит всех трёх классов,
            либо ``alpha``/``test_size`` вне допустимого диапазона.
        """
        alpha = self._check_alpha(alpha)
        test_size = self._check_test_size(test_size)
        rows = load_corpus(str(corpus_csv))
        if not rows:
            raise ValueError("корпус пуст: {}".format(corpus_csv))

        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = None
        if limit is not None and limit > 0:
            rows = self._balanced_subset(rows, limit)

        labels = {label for label, _text in rows}
        if labels != set(CLASS_LABELS):
            raise ValueError(
                "для обучения нужны все три класса {}, в корпусе {} найдены {}; "
                "модель с неполным набором классов нельзя загрузить для анализа".format(
                    CLASS_LABELS, corpus_csv, sorted(labels)
                )
            )

        train_rows, test_rows = stratified_split(rows, test_size=test_size, seed=42)
        if not train_rows:
            raise ValueError(
                "не удалось выделить обучающую часть из корпуса {} (строк: {})".format(
                    corpus_csv, len(rows)
                )
            )

        train_documents = [self._features(text) for _label, text in train_rows]
        train_labels = [label for label, _text in train_rows]
        test_documents = [self._features(text) for _label, text in test_rows]
        test_labels = [label for label, _text in test_rows]

        # min_df адаптивен: порог не может вычистить словарь даже на крошечном корпусе
        effective_min_df = self._effective_min_df(min_df, len(train_documents))

        document_frequency = self._token_document_frequency(train_documents)
        vocabulary = {
            token for token, df in document_frequency.items() if df >= effective_min_df
        }
        train_documents = [
            [token for token in document if token in vocabulary] for document in train_documents
        ]
        test_documents = [
            [token for token in document if token in vocabulary] for document in test_documents
        ]

        model = MultinomialNB(alpha=float(alpha))
        model.fit(train_documents, train_labels)

        if model.vocab_size < MIN_VOCAB_WARNING_THRESHOLD:
            warnings.warn(
                "lab_1.sentiment: итоговый словарь модели содержит всего {} токен(ов) "
                "(min_df={} при {} обучающих документах) — качество анализа будет низким, "
                "добавьте данные в {}".format(
                    model.vocab_size, effective_min_df, len(train_documents), corpus_csv
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        destination = Path(out_model)
        if destination.parent and not destination.parent.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(destination))

        self.model_path = destination
        self.model = model

        if test_documents:
            accuracy = float(model.score(test_documents, test_labels))
        else:
            # корпус слишком мал для отложенной выборки: оцениваем на обучающей части
            warnings.warn(
                "lab_1.sentiment: тестовая часть пуста (корпус {} слишком мал, строк: {}); "
                "accuracy посчитана на обучающей части".format(corpus_csv, len(rows)),
                RuntimeWarning,
                stacklevel=2,
            )
            accuracy = float(model.score(train_documents, train_labels))

        return {
            "accuracy": accuracy,
            "n_train": int(len(train_documents)),
            "n_test": int(len(test_documents)),
        }

    @staticmethod
    def _effective_min_df(min_df, n_train: int) -> int:
        """Адаптивный порог min_df: ``max(1, min(min_df, n_train // 10))``.

        Гарантирует, что порог никогда не вычистит словарь целиком: на крошечных
        корпусах (меньше 10 документов) он опускается до 1.
        """
        try:
            min_df = int(min_df)
        except (TypeError, ValueError):
            min_df = MIN_DF
        min_df = max(1, min_df)
        return max(1, min(min_df, max(n_train, 0) // 10))

    @staticmethod
    def _check_alpha(alpha) -> float:
        """Проверить параметр сглаживания: строго больше нуля."""
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            raise ValueError("alpha должен быть числом больше нуля, получено {!r}".format(alpha))
        if not alpha > 0.0:
            raise ValueError(
                "alpha должен быть строго больше нуля (получено {}): иначе вероятности "
                "вырождаются и в JSON модели попадают значения -Infinity".format(alpha)
            )
        return alpha

    @staticmethod
    def _check_test_size(test_size) -> float:
        """Проверить долю отложенной выборки: ``0 < test_size < 1``."""
        try:
            test_size = float(test_size)
        except (TypeError, ValueError):
            raise ValueError(
                "test_size должен быть числом в интервале (0, 1), получено {!r}".format(test_size)
            )
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                "test_size должен удовлетворять 0 < test_size < 1 (получено {}); "
                "например, test_size=0.2".format(test_size)
            )
        return test_size


    @staticmethod
    def _balanced_subset(rows: list, limit: int) -> list:
        """Взять из корпуса ``limit`` строк, поровну по классам (детерминированно)."""
        grouped = {}
        for label, text in rows:
            grouped.setdefault(label, []).append((label, text))

        labels = sorted(grouped)
        per_class = limit // len(labels) if labels else 0
        remainder = limit % len(labels) if labels else 0
        subset = []
        for index, label in enumerate(labels):
            quota = per_class + (1 if index < remainder else 0)
            subset.extend(grouped[label][:quota])
        return subset
