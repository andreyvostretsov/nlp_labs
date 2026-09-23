# -*- coding: utf-8 -*-
"""TF-IDF векторизатор на ``scipy.sparse`` (INTERFACES.md §2).

Реализация не использует ``sklearn``: словарь, документные частоты, idf и
L2-нормировка считаются вручную, разреженная матрица собирается как
``scipy.sparse.csr_matrix``.

Формулы:
    tf(t, d)  = 1 + ln(count(t, d))   при ``sublinear_tf=True``, иначе count(t, d)
    idf(t)    = ln((1 + n_docs) / (1 + df)) + 1   при ``smooth_idf=True``,
                иначе ln(n_docs / df) + 1
    w(t, d)   = tf * idf, строки L2-нормируются (ненулевые строки -> норма 1,
                нулевая строка остаётся нулевой)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse

__all__ = ["TfidfVectorizer"]


class TfidfVectorizer:
    """TF-IDF векторизатор (аналог sklearn-интерфейса на scipy.sparse).

    Параметры
    ---------
    sublinear_tf:
        ``True`` — логарифмическое сглаживание tf (``1 + ln(count)``);
        ``False`` — сырой ``count``.
    smooth_idf:
        ``True`` — ``ln((1 + n_docs) / (1 + df)) + 1``;
        ``False`` — ``ln(n_docs / df) + 1``.
    norm:
        ``"l2"`` — L2-нормировка строк (по умолчанию); ``None``/``""`` — без
        нормировки. Иные значения — ``ValueError``.
    min_df:
        Минимальная абсолютная документная частота: термин с ``df < min_df``
        удаляется. Должно быть >= 1.
    max_df:
        Максимальная ДОЛЯ документов в ``(0, 1]``: термин с
        ``df > max_df * n_docs`` удаляется.

    Атрибуты после :meth:`fit`
    --------------------------
    vocabulary_ : dict[str, int]
        Термин -> индекс. Порядок терминов лексикографический, поэтому индексы
        детерминированы.
    idf_ : numpy.ndarray
        Значения idf в порядке индексов словаря (``float64``).
    n_docs_ : int
        Число документов, поданных в :meth:`fit`.
    n_features_ : int
        Размер словаря после обрезки.
    """

    def __init__(
        self,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        norm: str = "l2",
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> None:
        normalized_norm: Optional[str]
        if norm is None or norm == "":
            normalized_norm = None
        elif norm == "l2":
            normalized_norm = "l2"
        else:
            raise ValueError(
                "norm должен быть 'l2' или None, получено: {!r}".format(norm)
            )

        if isinstance(min_df, bool) or not isinstance(min_df, (int, float)):
            raise ValueError("min_df должен быть числом >= 1, получено: {!r}".format(min_df))
        if not isinstance(max_df, bool) and isinstance(max_df, (int, float)):
            max_df_value = float(max_df)
        else:
            raise ValueError(
                "max_df должен быть числом в (0, 1], получено: {!r}".format(max_df)
            )

        if min_df < 1:
            raise ValueError("min_df должен быть >= 1, получено: {!r}".format(min_df))
        if not (0.0 < max_df_value <= 1.0):
            raise ValueError(
                "max_df должен быть долей документов в (0, 1], получено: {!r}".format(max_df)
            )

        self.sublinear_tf = bool(sublinear_tf)
        self.smooth_idf = bool(smooth_idf)
        self.norm = normalized_norm
        self.min_df = min_df
        self.max_df = max_df_value

        # Состояние «не обучен» — валидные пустые значения (см. контракт §2).
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.n_docs_: int = 0
        self.n_features_: int = 0
        self._df: Dict[str, int] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Внутреннее: подготовка документов к подсчёту
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_tokens(document: Any) -> List[str]:
        """Привести один документ к списку строковых токенов.

        Принимает ``None``/пустое -> ``[]``; ``str`` -> токены по пробелам в
        нижнем регистре; любую итерируемую последовательность токенов -> как
        есть (пустые токены отбрасываются).
        """
        if document is None:
            return []
        if isinstance(document, str):
            return [tok for tok in document.lower().split() if tok]

        try:
            items = list(document)
        except TypeError:
            return []

        tokens: List[str] = []
        for item in items:
            if item is None:
                continue
            token = item if isinstance(item, str) else str(item)
            if token:
                tokens.append(token)
        return tokens

    @classmethod
    def _prepare_documents(cls, documents: Any) -> List[List[str]]:
        """Нормализовать вход в список документов (список токенов на документ)."""
        if documents is None:
            return []
        if isinstance(documents, str):
            return [cls._doc_tokens(documents)]
        try:
            raw_documents = list(documents)
        except TypeError:
            return []
        return [cls._doc_tokens(doc) for doc in raw_documents]

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def fit(self, documents) -> "TfidfVectorizer":
        """Построить словарь и посчитать idf. Обрезка min_df/max_df — до idf."""
        docs = self._prepare_documents(documents)
        n_docs = len(docs)
        self.n_docs_ = n_docs

        # документная частота: сколько документов содержат термин
        df: Dict[str, int] = {}
        for tokens in docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        max_df_count = self.max_df * n_docs
        # Контракт §2: термин удаляется при df > max_df*n_docs (строгое сравнение).
        kept = [
            term
            for term, count in df.items()
            if count >= self.min_df and count <= max_df_count
        ]

        # лексикографический порядок -> детерминированные индексы
        vocabulary = {term: index for index, term in enumerate(sorted(kept))}
        self.vocabulary_ = vocabulary
        self.n_features_ = len(vocabulary)
        self._df = {term: df[term] for term in vocabulary}

        if self.n_features_ == 0:
            self.idf_ = np.zeros(0, dtype=np.float64)
        else:
            idf = np.empty(self.n_features_, dtype=np.float64)
            for term, index in vocabulary.items():
                term_df = df[term]
                if self.smooth_idf:
                    idf[index] = math.log((1.0 + n_docs) / (1.0 + term_df)) + 1.0
                else:
                    idf[index] = math.log(n_docs / term_df) + 1.0
            self.idf_ = idf

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Применение
    # ------------------------------------------------------------------
    def transform(self, documents) -> "scipy.sparse.csr_matrix":
        """Построить матрицу TF-IDF ``(n_docs, n_features_)`` по обученному словарю.

        Термины вне словаря игнорируются; пустой документ даёт нулевую строку.
        """
        if not self._fitted:
            raise RuntimeError(
                "TfidfVectorizer не обучен: вызовите fit(...) или fit_transform(...) "
                "перед transform(...)"
            )

        docs = self._prepare_documents(documents)
        n_docs = len(docs)
        n_features = self.n_features_

        if n_docs == 0 or n_features == 0:
            # (n, 0) при пустом словаре — по контракту §2
            return scipy.sparse.csr_matrix((n_docs, n_features), dtype=np.float64)

        rows: List[int] = []
        cols: List[int] = []
        values: List[float] = []

        for row_index, tokens in enumerate(docs):
            counts: Dict[str, int] = {}
            for token in tokens:
                if token in self.vocabulary_:
                    counts[token] = counts.get(token, 0) + 1
            if not counts:
                continue  # пустой документ / все токены OOV -> нулевая строка

            indices: List[int] = []
            weights: List[float] = []
            for term, count in counts.items():
                if self.sublinear_tf and count > 0:
                    tf = 1.0 + math.log(count)
                else:
                    tf = float(count)
                column = self.vocabulary_[term]
                indices.append(column)
                weights.append(tf * float(self.idf_[column]))

            row_values = np.asarray(weights, dtype=np.float64)
            if self.norm == "l2":
                row_norm = float(np.sqrt(np.dot(row_values, row_values)))
                if row_norm > 0.0:
                    row_values = row_values / row_norm
                # нулевая строка остаётся нулевой (без деления на 0)

            rows.extend([row_index] * len(indices))
            cols.extend(indices)
            values.extend(row_values.tolist())

        matrix = scipy.sparse.csr_matrix(
            (values, (rows, cols)), shape=(n_docs, n_features), dtype=np.float64
        )
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def fit_transform(self, documents) -> "scipy.sparse.csr_matrix":
        """``fit`` + ``transform`` на одном и том же (однократно прочитанном) входе."""
        docs = self._prepare_documents(documents)
        self.fit(docs)
        return self.transform(docs)

    # ------------------------------------------------------------------
    # Имена признаков
    # ------------------------------------------------------------------
    def get_feature_names(self) -> "list[str]":
        """Термины в порядке индексов словаря (копия, лексикографический порядок)."""
        names: List[str] = [""] * self.n_features_
        for term, index in self.vocabulary_.items():
            if 0 <= index < self.n_features_:
                names[index] = term
        return names

    def __repr__(self) -> str:  # pragma: no cover - отладочное представление
        return (
            "{}(sublinear_tf={}, smooth_idf={}, norm={!r}, min_df={!r}, max_df={!r}, "
            "n_features_={})".format(
                type(self).__name__,
                self.sublinear_tf,
                self.smooth_idf,
                self.norm,
                self.min_df,
                self.max_df,
                self.n_features_,
            )
        )
