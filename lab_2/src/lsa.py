# -*- coding: utf-8 -*-
"""Усечённое SVD (латентно-семантический анализ) — контракт INTERFACES.md §3.

Модуль предоставляет:

* :func:`cosine_similarity` — косинусная близость вектора к строкам матрицы;
* :class:`LsaModel` — усечённое SVD матрицы документ-термин с канонизацией знака
  базиса тем (доминирующий термин каждой темы положителен), проецированием новых
  документов (fold-in) и сериализацией в один ``.npz``.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

__all__ = ["LsaModel", "cosine_similarity"]

#: Обязательные ключи файла модели (INTERFACES.md §3, §6).
NPZ_KEYS = ("vt", "s", "doc_vectors")


def _as_matrix(X, name="X"):
    """Приводит вход к 2-D матрице: csr_matrix (float64) либо плотный ndarray.

    Пустой или не двумерный вход -> ``ValueError``.
    """
    if sp.issparse(X):
        matrix = X.tocsr().astype(np.float64)
        shape = matrix.shape
    else:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                "{} должен быть двумерной матрицей (docs x terms), получено ndim={}".format(
                    name, matrix.ndim
                )
            )
        shape = matrix.shape
    if len(shape) != 2:
        raise ValueError("{} должен быть двумерной матрицей, получена форма {}".format(name, shape))
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("{} пуста: форма {}".format(name, shape))
    return matrix


def cosine_similarity(query, matrix) -> "np.ndarray":
    """Косинусная близость 1-D вектора ``query`` (форма ``(k,)``) к строкам
    2-D ``matrix`` (форма ``(n, k)``). Возвращает 1-D массив длины ``n``.
    Вектор нулевой нормы даёт сходство ``0.0``. ``matrix`` может быть
    ``ndarray`` или ``csr_matrix``. Query формы ``(1, k)``/``(k, 1)``
    трактуется как 1-D вектор (fold-in одного документа).
    """
    q = np.asarray(query, dtype=np.float64)
    if q.ndim == 2 and 1 in q.shape:
        # Лояльность к fold-in одного документа: строка/столбец (1, k)/(k, 1)
        # трактуется как 1-D вектор.
        q = q.reshape(-1)
    if q.ndim != 1:
        raise ValueError("query должен быть одномерным вектором, получено ndim={}".format(q.ndim))
    n_features = q.shape[0]

    if sp.issparse(matrix):
        rows = matrix.tocsr().astype(np.float64)
        if rows.ndim != 2:
            raise ValueError("matrix должен быть двумерной, получено ndim={}".format(rows.ndim))
        if rows.shape[1] != n_features:
            raise ValueError(
                "несовпадение размерностей: query ({}), matrix {}".format(n_features, rows.shape)
            )
        row_norms = np.sqrt(np.asarray(rows.multiply(rows).sum(axis=1), dtype=np.float64).ravel())
        dots = np.asarray(rows @ q, dtype=np.float64).ravel()
    else:
        rows = np.asarray(matrix, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("matrix должен быть двумерной, получено ndim={}".format(rows.ndim))
        if rows.shape[1] != n_features:
            raise ValueError(
                "несовпадение размерностей: query ({}), matrix {}".format(n_features, rows.shape)
            )
        row_norms = np.linalg.norm(rows, axis=1)
        dots = rows @ q

    denominators = float(np.linalg.norm(q)) * row_norms
    similarities = np.zeros(dots.shape[0], dtype=np.float64)
    nonzero = denominators > 0.0
    similarities[nonzero] = dots[nonzero] / denominators[nonzero]
    return similarities


class LsaModel:
    """Усечённое SVD матрицы документ-термин (INTERFACES.md §3).

    Атрибуты после :meth:`fit` / :meth:`load`: ``k_``, ``components_`` (``Vt``,
    ``k x m``), ``singular_values_`` (по убыванию), ``explained_variance_``,
    ``explained_variance_ratio_``, ``doc_vectors_`` (``n x k``).
    """

    def __init__(self, k: int = 50):
        self.k = int(k)
        self.k_ = 0
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.doc_vectors_ = None

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(self, X, k: int = None) -> "LsaModel":
        """Считает усечённое SVD матрицы ``X`` (n_docs x n_terms).

        Эффективный ранг: ``k = min(k, min(X.shape) - 1)``, минимум 1.
        ``svds`` возвращает сингулярные значения по возрастанию — они приводятся
        к убыванию вместе с перестановкой столбцов ``U`` и строк ``Vt``, после чего
        выполняется канонизация знака (доминирующий по модулю термин каждой темы
        получает положительный коэффициент), и только затем считаются
        ``doc_vectors_ = X @ Vt.T``.
        """
        if k is None:
            k = self.k
        matrix = _as_matrix(X)
        try:
            requested = int(k)
        except (TypeError, ValueError) as exc:
            raise ValueError("k должен быть целым числом, получено {!r}".format(k)) from exc

        if min(matrix.shape) < 2:
            # svds требует 0 < k < min(X.shape); при min(X.shape) == 1 контрактное
            # k = max(1, min(k, min(shape) - 1)) = 1 недопустимо для ARPACK.
            raise ValueError(
                "матрица слишком мала для усечённого SVD: нужны минимум 2 документа и "
                "2 термина, получена форма {}".format(matrix.shape)
            )

        k_eff = max(1, min(requested, min(matrix.shape) - 1))
        try:
            u, s, vt = svds(matrix, k=k_eff, which="LM")
        except Exception as exc:  # ArpackError/ArpackNoConvError/LinAlgError/...
            raise ValueError(
                "усечённое SVD (k={}) не сошлось: {}".format(k_eff, exc)
            ) from exc

        # Контракт: сингулярные значения -> по убыванию, U/Vt переставляются
        # в том же порядке, что и s.
        order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
        s = np.asarray(s, dtype=np.float64)[order]
        u = np.asarray(u)[:, order]
        vt = np.asarray(vt, dtype=np.float64)[order, :]

        # Контракт: канонизация знака. Знак сингулярных векторов svds выбирает
        # ARPACK по стартовому вектору из глобального numpy.random, поэтому без
        # канонизации components_/doc_vectors_/topic_weights меняли бы знаки между
        # прогонами. Доминирующий (наибольший по модулю) термин каждой темы
        # делается положительным; U[:, j] умножается на тот же -1.
        # Нулевая строка (вырожденный нулевой спектр) остаётся без изменений.
        dominant = vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)]
        signs = np.where(dominant < 0.0, -1.0, 1.0)
        vt = vt * signs[:, np.newaxis]
        u = u * signs[np.newaxis, :]  # U не хранится, но знак согласован с Vt

        self.k_ = int(k_eff)
        self.components_ = vt
        self.singular_values_ = s
        self.explained_variance_ = s ** 2
        total = float(self.explained_variance_.sum())
        if total > 0.0:
            self.explained_variance_ratio_ = self.explained_variance_ / total
        else:
            # Вырожденная (нулевая) матрица: дисперсия не объяснена ничем.
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
        self.doc_vectors_ = np.asarray(matrix @ vt.T, dtype=np.float64)
        return self

    # ------------------------------------------------------------------ #
    # Проецирование
    # ------------------------------------------------------------------ #

    def transform(self, X) -> "np.ndarray":
        """Fold-in: ``X @ components_.T`` — плотный ``ndarray`` формы ``(n, k)``."""
        components = self._require_fitted()
        matrix = _as_matrix(X)
        if matrix.shape[1] != components.shape[1]:
            raise ValueError(
                "несовпадение числа признаков: X имеет {}, components_ ожидает {}".format(
                    matrix.shape[1], components.shape[1]
                )
            )
        return np.asarray(matrix @ components.T, dtype=np.float64)

    def topics(self, terms, n_terms: int = 10) -> list:
        """Топ-``n_terms`` терминов темы (по убыванию нагрузки в строке
        ``components_``). ``terms`` — список имён признаков той же длины, что и
        ``components_.shape[1]`` (иначе ``ValueError``).

        Ранжирование идёт по убыванию МОДУЛЯ коэффициента (INTERFACES.md §3).
        Знак сингулярных векторов ``svds`` данными не определяется (ARPACK
        выбирает его по стартовому вектору), поэтому сортировка по знаковому
        коэффициенту невоспроизводима и выводит на первое место термины с
        околонулевой нагрузкой. Сортировка по ``|коэффициент|`` инвариантна к
        знаку, воспроизводима и совпадает со знаковой там, где все нагрузки
        темы одного знака.
        """
        components = self._require_fitted()
        term_names = [] if terms is None else list(terms)
        if len(term_names) != components.shape[1]:
            raise ValueError(
                "длина terms ({}) не совпадает с числом признаков components_ ({})".format(
                    len(term_names), components.shape[1]
                )
            )
        limit = max(0, int(n_terms))
        result = []
        for row in components:
            loadings = np.abs(np.asarray(row, dtype=np.float64))
            order = np.argsort(-loadings, kind="stable")[:limit]
            result.append([term_names[int(index)] for index in order])
        return result

    # ------------------------------------------------------------------ #
    # Сериализация
    # ------------------------------------------------------------------ #

    def save(self, npz_path: str) -> None:
        """Сохраняет модель в один ``.npz``: ``vt`` (k x m float32),
        ``s`` (k float64), ``doc_vectors`` (n x k float32)."""
        components = self._require_fitted()
        path = str(npz_path)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez(
            path,
            vt=np.asarray(components, dtype=np.float32),
            s=np.asarray(self.singular_values_, dtype=np.float64),
            doc_vectors=np.asarray(self.doc_vectors_, dtype=np.float32),
        )

    @classmethod
    def load(cls, npz_path: str) -> "LsaModel":
        """Загружает модель из ``.npz``, проверяя согласованность форм.

        Несогласованные формы (``vt.shape[0] != s.shape[0]`` либо
        ``doc_vectors.shape[1] != s.shape[0]``) -> ``ValueError``.
        """
        path = str(npz_path)
        if not os.path.exists(path) and not path.endswith(".npz"):
            candidate = path + ".npz"
            if os.path.exists(candidate):
                path = candidate
        with np.load(path) as data:
            missing = [key for key in NPZ_KEYS if key not in data.files]
            if missing:
                raise ValueError(
                    "несовместимый файл модели {}: отсутствуют ключи {}".format(
                        path, ", ".join(missing)
                    )
                )
            vt = np.asarray(data["vt"])
            s = np.asarray(data["s"], dtype=np.float64)
            doc_vectors = np.asarray(data["doc_vectors"])

        if vt.ndim != 2 or s.ndim != 1 or doc_vectors.ndim != 2 or s.shape[0] == 0:
            raise ValueError(
                "несовместимый файл модели {}: ожидаются vt (k, m), s (k,), "
                "doc_vectors (n, k), получено vt {}, s {}, doc_vectors {}".format(
                    path, vt.shape, s.shape, doc_vectors.shape
                )
            )
        if vt.shape[0] != s.shape[0] or doc_vectors.shape[1] != s.shape[0]:
            raise ValueError(
                "несогласованные формы в {}: vt {}, s {}, doc_vectors {}".format(
                    path, vt.shape, s.shape, doc_vectors.shape
                )
            )

        model = cls(k=int(s.shape[0]))
        model.k_ = int(s.shape[0])
        model.components_ = vt
        model.singular_values_ = s
        model.explained_variance_ = s ** 2
        total = float(model.explained_variance_.sum())
        if total > 0.0:
            model.explained_variance_ratio_ = model.explained_variance_ / total
        else:
            model.explained_variance_ratio_ = np.zeros_like(model.explained_variance_)
        model.doc_vectors_ = doc_vectors
        return model

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> "np.ndarray":
        if self.components_ is None:
            raise ValueError("модель не обучена: вызовите fit(X, k) или LsaModel.load(npz_path)")
        return self.components_

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "LsaModel(k={}, k_={})".format(self.k, self.k_)
