# -*- coding: utf-8 -*-
"""Сводный доступ к четырём семантическим анализаторам лабораторных 1–4
(INTERFACES.md §7).

`SemanticSuite` загружает готовые модели (`lab_1` — наивный Байес, `lab_2` — LSA,
`lab_3` — Doc2Vec, `lab_4` — CNN/GRU) и прогоняет один и тот же запрос через все
четыре. Это нужно для сравнительного описания в README и ноутбуке `lab_5`;
сами анализаторы не модифицируются и не переобучаются.

Единый результат :class:`AnalyzerView` содержит:

* ``label`` — тема (для CNN/GRU — предсказание классификатора, для LSA/Doc2Vec —
  тема ближайшего документа, для Байеса — тональность);
* ``probabilities`` — вероятности классов (там, где они есть);
* ``top`` — ближайшие документы ``[(источник, косинус), ...]``;
* ``note`` — пояснение задачи анализатора или текст ошибки загрузки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ._lab1 import SentimentAnalyzer
from ._lab2 import SemanticAnalyzer
from ._lab3 import DocVecAnalyzer
from ._lab4 import NeuralAnalyzer

__all__ = ["AnalyzerView", "SemanticSuite"]

#: Имена четырёх анализаторов в порядке лабораторных.
ANALYZER_NAMES = ("nb", "lsa", "doc2vec", "cnn", "rnn")

#: Человекочитаемые подписи.
ANALYZER_TITLES = {
    "nb": "Наивный Байес (lab_1)",
    "lsa": "LSA (lab_2)",
    "doc2vec": "Doc2Vec (lab_3)",
    "cnn": "TextCNN (lab_4)",
    "rnn": "GRU (lab_4)",
}


@dataclass
class AnalyzerView:
    """Результат одного анализатора по одному запросу (INTERFACES.md §7)."""

    name: str
    title: str
    label: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    top: List[Tuple[str, float]] = field(default_factory=list)
    note: str = ""

    def top_sources(self, limit: int = 3) -> List[str]:
        """Источники top-N без числовых оценок."""
        return [source for source, _ in self.top[:limit]]


class SemanticSuite:
    """Загрузить четыре анализатора и сравнивать их на общих запросах."""

    def __init__(self, project_root=None, corpus: str = "demo"):
        """Создать сводку; модели загружаются лениво при первом запросе.

        :param project_root: корень проекта; по умолчанию вычисляется от файла.
        :param corpus: ``"demo"`` или ``"custom"`` — набор готовых моделей.
        :raises ValueError: неизвестный ``corpus``.
        """
        key = str(corpus or "demo").strip().lower()
        if key not in ("demo", "custom"):
            raise ValueError(
                "corpus должен быть 'demo' или 'custom', получено {!r}".format(corpus)
            )
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.corpus = key
        self._loaded = False
        self._analyzers: Dict[str, object] = {}
        self._errors: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #

    def _model_dir(self, lab: str, name: str) -> Path:
        """Путь к готовой модели ``<lab>/src/models/<name>_<corpus>``."""
        return self.project_root / lab / "src" / "models" / "{}_{}".format(name, self.corpus)

    def _load(self) -> None:
        """Ленивая загрузка всех доступных моделей (ошибки не бросаются)."""
        if self._loaded:
            return
        self._loaded = True

        sentiment_path = self.project_root / "lab_1" / "src" / "models" / "sentiment_model.json"
        try:
            self._analyzers["nb"] = SentimentAnalyzer(str(sentiment_path))
        except Exception as exc:  # noqa: BLE001 - сводка должна пережить отсутствие модели
            self._errors["nb"] = str(exc)

        loaders = {
            "lsa": (SemanticAnalyzer, "lab_2", "lsa"),
            "doc2vec": (DocVecAnalyzer, "lab_3", "d2v"),
            "cnn": (NeuralAnalyzer, "lab_4", "cnn"),
            "rnn": (NeuralAnalyzer, "lab_4", "rnn"),
        }
        for key, (factory, lab, prefix) in loaders.items():
            directory = self._model_dir(lab, prefix)
            try:
                if key in ("cnn", "rnn"):
                    analyzer = factory(str(directory), method=key)
                else:
                    analyzer = factory(str(directory))
                if not getattr(analyzer, "is_fitted", True):
                    message = getattr(analyzer, "load_error", None) or "модель не загружена"
                    self._errors[key] = message
                else:
                    self._analyzers[key] = analyzer
            except Exception as exc:  # noqa: BLE001
                self._errors[key] = str(exc)

    @property
    def available(self) -> Dict[str, bool]:
        """Какие анализаторы удалось загрузить."""
        self._load()
        return {name: name in self._analyzers for name in ANALYZER_NAMES}

    @property
    def errors(self) -> Dict[str, str]:
        """Тексты ошибок загрузки по имени анализатора."""
        self._load()
        return dict(self._errors)

    # ------------------------------------------------------------------ #
    # Анализ
    # ------------------------------------------------------------------ #

    @staticmethod
    def _topic_of(source: str) -> str:
        """Тема из имени источника: префикс до ``_`` или ``.``."""
        text = str(source)
        for separator in ("_", "."):
            if separator in text:
                return text.split(separator, 1)[0]
        return text

    def analyze(self, query: str, top_n: int = 3) -> Dict[str, AnalyzerView]:
        """Прогнать запрос через все четыре анализатора.

        :return: словарь ``{имя: AnalyzerView}`` для доступных анализаторов.
        """
        self._load()
        try:
            limit = max(1, int(top_n))
        except (TypeError, ValueError):
            limit = 3
        text = "" if query is None else str(query)
        views: Dict[str, AnalyzerView] = {}

        # 1. lab_1: тональность (не поиск по темам).
        analyzer = self._analyzers.get("nb")
        if analyzer is not None:
            result = analyzer.analyze(text)
            views["nb"] = AnalyzerView(
                name="nb",
                title=ANALYZER_TITLES["nb"],
                label=result.label,
                probabilities=dict(result.probabilities),
                note="тональность (negative/neutral/positive), а не тема",
            )

        # 2–3. LSA и Doc2Vec: поиск по смыслу.
        for key in ("lsa", "doc2vec"):
            analyzer = self._analyzers.get(key)
            if analyzer is None:
                continue
            try:
                matches = analyzer.query(text, top_n=limit)
            except Exception:  # noqa: BLE001 - пустой/OOV запрос не должен ломать сводку
                matches = []
            top = [(match.source, float(match.score)) for match in matches]
            views[key] = AnalyzerView(
                name=key,
                title=ANALYZER_TITLES[key],
                label=self._topic_of(top[0][0]) if top else None,
                top=top,
                note="тема ближайшего документа",
            )

        # 4. lab_4: CNN и GRU — тема классификатора + поиск.
        for key in ("cnn", "rnn"):
            analyzer = self._analyzers.get(key)
            if analyzer is None:
                continue
            label = None
            probabilities: Dict[str, float] = {}
            try:
                prediction = analyzer.predict(text)
                label = prediction.label
                probabilities = dict(prediction.probabilities)
            except Exception:  # noqa: BLE001
                label = None
            try:
                matches = analyzer.query(text, top_n=limit)
            except Exception:  # noqa: BLE001
                matches = []
            top = [(match.source, float(match.score)) for match in matches]
            views[key] = AnalyzerView(
                name=key,
                title=ANALYZER_TITLES[key],
                label=label,
                probabilities=probabilities,
                top=top,
                note="тема классификатора + поиск",
            )

        # Недоступные анализаторы тоже попадают в ответ с пояснением.
        for key in ANALYZER_NAMES:
            if key not in views:
                views[key] = AnalyzerView(
                    name=key,
                    title=ANALYZER_TITLES[key],
                    note="недоступен: {}".format(self._errors.get(key, "модель не загружена")),
                )
        return views

    def compare(self, queries: Sequence[str], top_n: int = 3) -> List[dict]:
        """Сравнить анализаторы на списке запросов.

        :return: список строк ``{'query', 'nb', 'lsa', 'doc2vec', 'cnn', 'rnn'}``;
            значения — :class:`AnalyzerView`.
        """
        rows: List[dict] = []
        for query in queries:
            row = {"query": str(query)}
            row.update(self.analyze(query, top_n=top_n))
            rows.append(row)
        return rows

    def topic_agreement(self, queries: Sequence[str]) -> Dict[str, float]:
        """Доля запросов, где тема анализатора совпала с темой LSA (ориентир).

        Для Байеса сравнивается тональность, поэтому он в согласие не входит;
        возвращаются доли для ``lsa`` (1.0 по определению), ``doc2vec``, ``cnn``,
        ``rnn``.
        """
        rows = self.compare(queries)
        reference = [row["lsa"].label for row in rows]
        result: Dict[str, float] = {}
        for key in ("lsa", "doc2vec", "cnn", "rnn"):
            labels = [row[key].label for row in rows]
            matches = sum(
                1 for expected, actual in zip(reference, labels) if expected is not None and expected == actual
            )
            result[key] = matches / len(rows) if rows else 0.0
        return result
