# -*- coding: utf-8 -*-
"""Сквозной фасад семантического анализатора (INTERFACES.md §4 и §6).

Модуль связывает три слоя прототипа LSA в один пользовательский сценарий:

* :mod:`lab_2.src.corpus` — нормализация документов (``doc_tokens``);
* :mod:`lab_2.src.vectorizer` — матрица TF-IDF (``TfidfVectorizer``);
* :mod:`lab_2.src.lsa` — усечённое SVD, fold-in и косинусная близость
  (``LsaModel``, ``cosine_similarity``).

Публичный API:

* :class:`Match` — результат поиска (индекс документа, близость, сниппет, метка);
* :class:`SemanticAnalyzer` — обучение на корпусе, поиск (``query``), поиск
  похожих документов (``similar``), интерпретация тем (``topics``,
  ``topic_weights``) и персистенция в каталог модели из трёх файлов
  (``meta.json``, ``svd.npz``, ``docs.json``).

Особенности контракта:

* ``SemanticAnalyzer(model_dir=None)`` при отсутствии модели возвращает
  НЕобученный экземпляр (без исключения);
* ``query``/``similar``/``topics``/``topic_weights`` до ``fit`` бросают
  ``RuntimeError`` с сообщением :data:`NOT_FITTED_MESSAGE` (наружу не
  просачивается ``ValueError`` из ``LsaModel.transform``/``topics``).
"""
from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .corpus import doc_tokens
from .lsa import LsaModel, cosine_similarity
from .vectorizer import TfidfVectorizer

__all__ = ["Match", "SemanticAnalyzer"]

#: Версия схемы модели (INTERFACES.md §6).
MODEL_VERSION = 1

#: Имена трёх файлов каталога модели (INTERFACES.md §6).
META_FILENAME = "meta.json"
SVD_FILENAME = "svd.npz"
DOCS_FILENAME = "docs.json"

#: Сообщение о вызове анализатора до ``fit`` (INTERFACES.md §4).
NOT_FITTED_MESSAGE = "модель не обучена — вызовите `fit(...)` или выполните ноутбук"

#: Маркер несовместимой модели (INTERFACES.md §6).
INCOMPATIBLE_MARKER = "несовместимая версия модели, переобучите"

#: Ограничение длины сниппета документа в результатах поиска (INTERFACES.md §4).
SNIPPET_LIMIT = 160

#: Порог «численно нулевой нормы» латентного вектора (см. :func:`_is_degenerate`).
_ZERO_NORM_TOL = 1e-12

#: Подкаталог модели по умолчанию: ``<пакет>/models/lsa_demo`` (INTERFACES.md §4).
#: ``<пакет>`` — каталог самого пакета ``lab_2.src`` (ровно там же, где lab_1
#: держит ``lab_1/src/models/sentiment_model.json``, и куда указывают
#: ``lab_2/.gitignore`` со сценарием ноутбука).
_DEFAULT_MODEL_PARTS = ("models", "lsa_demo")

#: Значение ``top_n`` по умолчанию для ``query``/``similar``.
_DEFAULT_TOP_N = 10


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def _default_model_dir() -> str:
    """``<пакет>/models/lsa_demo`` — каталог модели по умолчанию.

    Основной путь — каталог пакета ``lab_2/src`` (как в lab_1). Если его нет, но
    существует альтернативный ``lab_2/models/lsa_demo`` (трактовка «пакет =
    lab_2»), используется он — так готовая модель находится при любой из двух
    допустимых трактовок формулировки §4.
    """
    package_dir = Path(__file__).resolve().parent  # .../lab_2/src
    primary = package_dir.joinpath(*_DEFAULT_MODEL_PARTS)
    if not primary.is_dir():
        alternative = package_dir.parent.joinpath(*_DEFAULT_MODEL_PARTS)
        if alternative.is_dir():
            return str(alternative)
    return str(primary)


def _as_text(value) -> str:
    """Привести элемент корпуса к тексту (``None`` -> ``""``).

    Последовательность строк трактуется как уже готовые токены документа и
    склеивается пробелами — это позволяет передавать в ``fit`` как тексты, так и
    заранее токенизированные документы.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def _normalize_ws(text) -> str:
    """Нормализовать пробелы: любые последовательности пробельных -> один пробел."""
    return " ".join(_as_text(text).split())


def _snippet(text, limit: int = SNIPPET_LIMIT) -> str:
    """Сниппет документа: нормализованные пробелы, не длиннее ``limit`` символов."""
    flat = _normalize_ws(text)
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _limit(value, default: int = _DEFAULT_TOP_N) -> int:
    """Привести ``top_n``/``n_terms`` к неотрицательному int (мусор -> default)."""
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError — ``inf``/``Decimal('Infinity')``: это мусор, а не сбой.
        return default
    return max(0, number)


def _is_degenerate(vector) -> bool:
    """Численно нулевой вектор (норма ниже допуска) или вектор с NaN/inf.

    Строки TF-IDF нормированы (норма 1), строки ``Vt`` единичны, поэтому норма
    латентной проекции не превосходит 1; всё, что ниже :data:`_ZERO_NORM_TOL`, —
    шум округления (наблюдались нормы ~1e-16), а не сигнал. Косинус такого
    вектора не определён, и §3 предписывает считать его равным 0.0.
    """
    array = np.asarray(vector, dtype=np.float64).ravel()
    if array.size == 0:
        return True
    if not bool(np.all(np.isfinite(array))):
        return True
    return float(np.linalg.norm(array)) <= _ZERO_NORM_TOL


def _as_documents(texts) -> Optional[List[str]]:
    """Привести ``texts`` к списку текстов.

    ``None`` -> ``None`` (признак отсутствия корпуса); одиночная строка
    трактуется как один документ; неитерируемый объект -> ``None``.
    """
    if texts is None:
        return None
    if isinstance(texts, str):
        return [texts]
    try:
        items = list(texts)
    except TypeError:
        return None
    return [_as_text(item) for item in items]


def _sparsity(matrix) -> float:
    """Доля нулевых элементов разреженной матрицы."""
    rows, columns = matrix.shape
    total = int(rows) * int(columns)
    if total <= 0:
        return 1.0
    return float(1.0 - matrix.nnz / float(total))


def _has_model_files(model_dir) -> bool:
    """Есть ли в каталоге все три файла модели (INTERFACES.md §6)."""
    base = Path(model_dir)
    return all(
        (base / name).is_file()
        for name in (META_FILENAME, SVD_FILENAME, DOCS_FILENAME)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    """Прочитать JSON-объект файла модели; ошибки чтения/разбора -> ``ValueError``.

    Сообщение содержит маркер несовместимой модели (§6), потому что испорченный
    или отсутствующий файл означает, что модель надо переобучить.
    """
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise _incompatible("не удалось прочитать {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise _incompatible("неожиданный формат {}: ожидался объект JSON".format(path))
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Записать JSON-объект в UTF-8 (создаёт родительские каталоги)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _incompatible(detail: str) -> ValueError:
    """``ValueError`` с контрактным маркером несовместимой модели (§6)."""
    return ValueError("{}: {}".format(INCOMPATIBLE_MARKER, detail))


# --------------------------------------------------------------------------- #
# Результат поиска
# --------------------------------------------------------------------------- #

@dataclass
class Match:
    """Один найденный документ (INTERFACES.md §4).

    :param index: индекс документа в корпусе обучения;
    :param score: косинусная близость запроса и документа;
    :param text: сниппет документа (нормализованные пробелы, до ~160 символов);
    :param source: метка документа (``doc_0`` по умолчанию либо имя файла).
    """

    index: int
    score: float
    text: str
    source: str = ""


# --------------------------------------------------------------------------- #
# Анализатор
# --------------------------------------------------------------------------- #

class SemanticAnalyzer:
    """Сквозной фасад LSA: обучение, поиск, темы и персистенция (§4, §6).

    Атрибуты состояния (после ``fit`` или ``load``):

    ``model_dir``
        Каталог модели (строка). По умолчанию ``<пакет>/models/lsa_demo``.
    ``load_error``
        Текст ошибки, если модель в каталоге оказалась несовместимой: конструктор
        в этом случае не падает, а отдаёт необученный экземпляр (§4).
    """

    # ------------------------------------------------------------------ #
    # Создание и загрузка
    # ------------------------------------------------------------------ #

    def __init__(self, model_dir=None):
        self.model_dir = str(model_dir) if model_dir is not None else _default_model_dir()
        self.load_error: Optional[str] = None
        self._reset()

        if _has_model_files(self.model_dir):
            try:
                loaded = type(self)._load_from_dir(self.model_dir)
            except (ValueError, OSError) as exc:
                # Контракт §4: при отсутствии ПРИГОДНОЙ модели конструктор не
                # падает, а возвращает необученный экземпляр. Причина доступна
                # в ``load_error``; сам вызов ``load`` такую модель отвергает.
                self.load_error = str(exc)
            else:
                self._adopt(loaded)

    @classmethod
    def load(cls, model_dir) -> "SemanticAnalyzer":
        """Загрузить модель из каталога (INTERFACES.md §4, §6).

        :raises ValueError: каталог/файлы отсутствуют, ``version != 1`` или
            нарушена согласованность длин (``len(vocabulary) == vt.shape[1]``,
            ``len(texts) == doc_vectors.shape[0] == n_docs``).
        """
        return cls._load_from_dir(model_dir)

    @classmethod
    def _load_from_dir(cls, model_dir) -> "SemanticAnalyzer":
        """Собрать анализатор из файлов модели, минуя авто-загрузку ``__init__``."""
        base = Path(model_dir)

        meta_path = base / META_FILENAME
        if not meta_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(META_FILENAME, base))
        meta = _read_json(meta_path)

        version = meta.get("version")
        try:
            version_value = int(version)
        except (TypeError, ValueError, OverflowError):
            version_value = -1
        if version_value != MODEL_VERSION:
            raise _incompatible(
                "version={!r} в {}, ожидается {}".format(version, meta_path, MODEL_VERSION)
            )

        for name in (SVD_FILENAME, DOCS_FILENAME):
            if not (base / name).is_file():
                raise _incompatible("нет файла {} в каталоге {}".format(name, base))

        try:
            n_docs = int(meta["n_docs"])
            k_value = int(meta["k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _incompatible("в {} нет целых полей k/n_docs".format(meta_path)) from exc

        vocabulary = meta.get("vocabulary")
        idf = meta.get("idf")
        if not isinstance(vocabulary, list) or not isinstance(idf, list):
            raise _incompatible(
                "в {} поля vocabulary/idf должны быть списками".format(meta_path)
            )
        if len(vocabulary) != len(idf):
            raise _incompatible(
                "длина vocabulary ({}) не совпадает с длиной idf ({})".format(
                    len(vocabulary), len(idf)
                )
            )

        try:
            lsa = LsaModel.load(str(base / SVD_FILENAME))
        except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
            raise _incompatible("не удалось прочитать {}: {}".format(SVD_FILENAME, exc)) from exc

        n_components, n_terms = lsa.components_.shape
        if len(vocabulary) != int(n_terms):
            raise _incompatible(
                "длина vocabulary ({}) не совпадает с vt.shape[1] ({})".format(
                    len(vocabulary), int(n_terms)
                )
            )
        if k_value != int(n_components):
            raise _incompatible(
                "k из {} ({}) не совпадает с vt.shape[0] ({})".format(
                    meta_path, k_value, int(n_components)
                )
            )

        docs = _read_json(base / DOCS_FILENAME)
        texts = docs.get("texts")
        sources = docs.get("sources")
        if not isinstance(texts, list) or not isinstance(sources, list):
            raise _incompatible(
                "в {} поля texts/sources должны быть списками".format(DOCS_FILENAME)
            )

        stored_docs = int(lsa.doc_vectors_.shape[0])
        if not (len(texts) == stored_docs == n_docs):
            raise _incompatible(
                "len(texts)={}, doc_vectors.shape[0]={}, n_docs={}".format(
                    len(texts), stored_docs, n_docs
                )
            )
        if len(sources) != len(texts):
            raise _incompatible(
                "длина sources ({}) не совпадает с числом документов ({})".format(
                    len(sources), len(texts)
                )
            )

        vectorizer = cls._rebuild_vectorizer(meta, vocabulary, idf, n_docs)

        instance = cls.__new__(cls)
        instance.model_dir = str(base)
        instance.load_error = None
        instance._reset()
        instance._vectorizer = vectorizer
        instance._lsa = lsa
        instance._terms = [str(term) for term in vocabulary]
        instance._texts = [str(text) for text in texts]
        instance._sources = [str(source) for source in sources]
        instance._is_fitted = True
        return instance

    @staticmethod
    def _rebuild_vectorizer(meta, vocabulary, idf, n_docs) -> TfidfVectorizer:
        """Восстановить обученный векторизатор из ``meta.json``."""
        params = meta.get("vectorizer")
        if not isinstance(params, dict):
            params = {}
        try:
            vectorizer = TfidfVectorizer(
                sublinear_tf=bool(params.get("sublinear_tf", True)),
                smooth_idf=bool(params.get("smooth_idf", True)),
                norm=params.get("norm", "l2"),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
            )
        except ValueError as exc:
            raise _incompatible("некорректные параметры векторизатора: {}".format(exc)) from exc

        # Восстанавливаем состояние ровно так, как его оставил бы ``fit``:
        # словарь (лексикографический порядок = порядок индексов), idf, n_docs_.
        vectorizer.vocabulary_ = {
            str(term): index for index, term in enumerate(vocabulary)
        }
        vectorizer.n_features_ = len(vocabulary)
        vectorizer.idf_ = np.asarray([float(value) for value in idf], dtype=np.float64)
        vectorizer.n_docs_ = int(n_docs)
        vectorizer._df = {}
        vectorizer._fitted = True
        return vectorizer

    def _reset(self) -> None:
        """Сбросить состояние в «не обучен»."""
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._lsa: Optional[LsaModel] = None
        self._terms: List[str] = []
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._is_fitted: bool = False

    def _adopt(self, other: "SemanticAnalyzer") -> None:
        """Скопировать состояние обученной модели (для авто-загрузки)."""
        self._vectorizer = other._vectorizer
        self._lsa = other._lsa
        self._terms = list(other._terms)
        self._texts = list(other._texts)
        self._sources = list(other._sources)
        self._is_fitted = other._is_fitted

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Обучен ли анализатор (``False`` — методы поиска бросят ``RuntimeError``)."""
        return bool(self._is_fitted)

    @property
    def n_docs(self) -> int:
        """Число документов корпуса обучения (0 до ``fit``)."""
        return len(self._texts)

    @property
    def n_features(self) -> int:
        """Размер словаря после обрезки (0 до ``fit``)."""
        return len(self._terms)

    @property
    def k(self) -> int:
        """Эффективное число тем LSA (0 до ``fit``)."""
        if self._lsa is None:
            return 0
        return int(self._lsa.k_)

    def _require_fitted(self) -> None:
        """Проверить собственное состояние и не пустить ``ValueError`` из LSA."""
        if not self._is_fitted or self._vectorizer is None or self._lsa is None:
            raise RuntimeError(NOT_FITTED_MESSAGE)

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(
        self,
        texts,
        *,
        k: int = 50,
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        norm: str = "l2",
        sources=None,
    ) -> dict:
        """Обучить анализатор на корпусе и сохранить модель в ``model_dir``.

        :param texts: итерируемое текстов документов (обязательно непустое).
        :param k: запрошенное число тем (эффективное уточняет ``LsaModel``).
        :param min_df/max_df/sublinear_tf/smooth_idf/norm: параметры TF-IDF (§2).
        :param sources: метки документов той же длины, что и ``texts``; по
            умолчанию ``["doc_0", "doc_1", ...]``.
        :return: ``{'n_docs', 'n_features', 'k', 'sparsity',
            'explained_variance_ratio', 'seconds'}`` (``seconds`` — время SVD).
        :raises ValueError: корпус пуст/``None``, документов меньше двух, в
            словаре меньше двух терминов (``svds`` не работает), длина
            ``sources`` не совпадает с числом документов либо ``k`` не является
            целым числом (``k=None``/``k='abc'``/``k=inf``).
        """
        documents = _as_documents(texts)
        if not documents:
            raise ValueError(
                "texts пуст или None: для обучения нужен непустой итерируемый набор документов"
            )
        n_docs = len(documents)
        if n_docs < 2:
            raise ValueError(
                "для усечённого SVD нужно минимум 2 документа, получено {}".format(n_docs)
            )

        # ``k`` проверяется здесь: LsaModel.__init__ делает ``int(k)`` до своей
        # защиты, поэтому мусорный ``k`` иначе выпустил бы TypeError наружу.
        try:
            k_value = int(k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "k должен быть целым числом, получено {!r}".format(k)
            ) from exc

        source_labels = self._resolve_sources(sources, n_docs)

        tokenized = [doc_tokens(text, "canonical") for text in documents]

        vectorizer = TfidfVectorizer(
            sublinear_tf=sublinear_tf,
            smooth_idf=smooth_idf,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
        )
        matrix = vectorizer.fit_transform(tokenized)

        n_features = int(vectorizer.n_features_)
        if n_features < 2:
            raise ValueError(
                "словарь содержит {} термин(ов): для усечённого SVD нужно минимум "
                "2 термина — увеличьте корпус или ослабьте min_df/max_df".format(n_features)
            )

        started = time.perf_counter()
        lsa = LsaModel(k=k_value).fit(matrix, k_value)
        seconds = float(time.perf_counter() - started)

        self._vectorizer = vectorizer
        self._lsa = lsa
        self._terms = list(vectorizer.get_feature_names())
        self._texts = list(documents)
        self._sources = list(source_labels)
        self._is_fitted = True
        self.load_error = None

        stats = {
            "n_docs": n_docs,
            "n_features": n_features,
            "k": int(lsa.k_),
            "sparsity": _sparsity(matrix),
            "explained_variance_ratio": [
                float(value)
                for value in np.asarray(lsa.explained_variance_ratio_, dtype=np.float64).ravel()
            ],
            "seconds": seconds,
        }

        self.save()
        return stats

    @staticmethod
    def _resolve_sources(sources, n_docs: int) -> List[str]:
        """Метки документов: по умолчанию ``doc_0..doc_{n-1}``."""
        if sources is None:
            return ["doc_{}".format(index) for index in range(n_docs)]
        if isinstance(sources, str):
            raw = [sources]
        else:
            try:
                raw = list(sources)
            except TypeError as exc:
                raise ValueError(
                    "sources должен быть итерируемым строк, получено {!r}".format(sources)
                ) from exc
        labels = ["" if item is None else str(item) for item in raw]
        if len(labels) != n_docs:
            raise ValueError(
                "длина sources ({}) не совпадает с числом документов ({})".format(
                    len(labels), n_docs
                )
            )
        return labels

    # ------------------------------------------------------------------ #
    # Поиск
    # ------------------------------------------------------------------ #

    def query(self, text, top_n: int = 10) -> "list[Match]":
        """Найти ``top_n`` документов, ближайших к тексту запроса (§4).

        Пустой текст или запрос, все токены которого вне словаря (OOV), дают
        пустой список.
        """
        self._require_fitted()
        vector = self._fold_in(text)
        if vector is None:
            return []
        similarities = cosine_similarity(vector, self._lsa.doc_vectors_)
        return self._top_matches(similarities, top_n)

    def similar(self, text_or_index, top_n: int = 10, exclude_self: bool = True) -> "list[Match]":
        """Найти документы, похожие на документ корпуса или на текст (§4).

        :param text_or_index: ``int`` — индекс документа корпуса (готовый
            ``doc_vectors_[index]``); ``str`` — текст, проецируемый тем же
            конвейером (fold-in).
        :param exclude_self: исключить сам документ (для ``int`` — этот индекс;
            для строки — индекс документа с точно таким же текстом, если он есть).
        :return: ``top_n`` ближайших документов; невалидный индекс -> ``[]``.
        """
        self._require_fitted()

        excluded = set()
        if isinstance(text_or_index, bool):
            return []
        if isinstance(text_or_index, (int, np.integer)):
            index = int(text_or_index)
            if index < 0 or index >= self.n_docs:
                return []
            vector = np.asarray(self._lsa.doc_vectors_[index], dtype=np.float64).ravel()
            if _is_degenerate(vector):
                # Латентного представления у документа нет (норма ~1e-16):
                # косинус не определён, ближайших документов не существует.
                return []
            if exclude_self:
                excluded.add(index)
        elif isinstance(text_or_index, str):
            vector = self._fold_in(text_or_index)
            if vector is None:
                return []
            if exclude_self:
                twin = self._find_document(text_or_index)
                if twin is not None:
                    excluded.add(twin)
        else:
            return []

        similarities = cosine_similarity(vector, self._lsa.doc_vectors_)
        return self._top_matches(similarities, top_n, exclude=excluded)

    def topics(self, n_terms: int = 10) -> list:
        """Топ-``n_terms`` терминов каждой темы (``LsaModel.topics``, §4)."""
        self._require_fitted()
        return self._lsa.topics(self._terms, _limit(n_terms))

    def topic_weights(self, text) -> dict:
        """Латентные координаты текста ``{индекс темы: float}`` (§4).

        Пустой текст или все токены вне словаря -> ``{}``.
        """
        self._require_fitted()
        vector = self._fold_in(text)
        if vector is None:
            return {}
        return {index: float(value) for index, value in enumerate(vector)}

    # ------------------------------------------------------------------ #
    # Персистенция
    # ------------------------------------------------------------------ #

    def save(self, model_dir=None) -> str:
        """Сохранить модель в ``model_dir`` (по умолчанию — текущий) и вернуть путь.

        Каталог модели состоит ровно из трёх файлов (INTERFACES.md §6):
        ``meta.json``, ``svd.npz``, ``docs.json``. Родительские каталоги создаются.
        """
        self._require_fitted()
        target = Path(model_dir) if model_dir is not None else Path(self.model_dir)
        target.mkdir(parents=True, exist_ok=True)

        vectorizer = self._vectorizer
        meta = {
            "version": MODEL_VERSION,
            "k": int(self._lsa.k_),
            "n_docs": len(self._texts),
            "vectorizer": {
                "sublinear_tf": bool(vectorizer.sublinear_tf),
                "smooth_idf": bool(vectorizer.smooth_idf),
                "norm": vectorizer.norm,
                "min_df": (
                    int(vectorizer.min_df)
                    if isinstance(vectorizer.min_df, int)
                    else float(vectorizer.min_df)
                ),
                "max_df": float(vectorizer.max_df),
            },
            "vocabulary": list(self._terms),
            "idf": [
                float(value)
                for value in np.asarray(vectorizer.idf_, dtype=np.float64).ravel()
            ],
        }
        _write_json(target / META_FILENAME, meta)
        self._lsa.save(str(target / SVD_FILENAME))
        _write_json(
            target / DOCS_FILENAME,
            {"texts": list(self._texts), "sources": list(self._sources)},
        )
        # ``model_dir`` меняется только после успешной записи всех трёх файлов,
        # иначе неудачный save увёл бы последующие save() в битый каталог.
        self.model_dir = str(target)
        return str(target)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _fold_in(self, text) -> Optional[np.ndarray]:
        """Спроецировать текст в латентное пространство.

        :return: вектор формы ``(k,)`` либо ``None``, если токенов нет, все они
            вне словаря (OOV) или проекция численно нулевая (норма ниже
            :data:`_ZERO_NORM_TOL`): у такого запроса нет латентного сигнала, и
            косинусная близость к нему была бы чистым шумом округления.
        """
        tokens = doc_tokens(text, "canonical")
        if not tokens:
            return None
        matrix = self._vectorizer.transform([tokens])
        if matrix.nnz == 0:
            return None
        vectors = np.asarray(self._lsa.transform(matrix), dtype=np.float64)
        if vectors.size == 0:
            return None
        vector = vectors.reshape(-1, vectors.shape[-1])[0].ravel()
        if _is_degenerate(vector):
            return None
        return vector

    def _find_document(self, text) -> Optional[int]:
        """Индекс документа с совпадающим текстом (сравнение после ``strip``)."""
        needle = _as_text(text).strip()
        if not needle:
            return None
        flat = _normalize_ws(needle)
        for index, document in enumerate(self._texts):
            if document.strip() == needle or _normalize_ws(document) == flat:
                return index
        return None

    def _top_matches(self, similarities, top_n, exclude=()) -> "list[Match]":
        """Собрать ``Match`` по убыванию близости, исключая заданные индексы.

        Близости документов с численно нулевым латентным вектором принудительно
        обнуляются: ``cosine_similarity`` делит на ``|q| * row_norm`` и при
        ``row_norm ~ 1e-16`` даёт шум в ``[-1, 1]`` (наблюдался score ровно 1.0 у
        документа, не имеющего отношения к запросу). §3 предписывает для вектора
        нулевой нормы сходство ``0.0``.
        """
        scores = np.asarray(similarities, dtype=np.float64).ravel()
        limit = _limit(top_n)
        if limit == 0 or scores.size == 0:
            return []

        scores = scores.copy()
        scores[~np.isfinite(scores)] = 0.0
        norms = np.linalg.norm(
            np.asarray(self._lsa.doc_vectors_, dtype=np.float64), axis=1
        )
        if norms.size == scores.size:
            negligible = ~np.isfinite(norms) | (norms <= _ZERO_NORM_TOL)
            if negligible.any():
                scores[negligible] = 0.0

        excluded = {int(index) for index in exclude}
        order = np.argsort(-scores, kind="stable")

        results: List[Match] = []
        for position in order:
            index = int(position)
            if index in excluded or not (0 <= index < len(self._texts)):
                continue
            results.append(
                Match(
                    index=index,
                    score=float(scores[index]),
                    text=_snippet(self._texts[index]),
                    source=self._sources[index],
                )
            )
            if len(results) >= limit:
                break
        return results

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "SemanticAnalyzer(model_dir={!r}, n_docs={}, n_features={}, k={})".format(
            self.model_dir, self.n_docs, self.n_features, self.k
        )
