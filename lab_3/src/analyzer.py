# -*- coding: utf-8 -*-
"""Сквозной фасад анализатора на Doc2Vec/Word2Vec (INTERFACES.md §4, §5).

Модуль связывает три слоя прототипа в один пользовательский сценарий:

* :mod:`lab_3.src._lab2` — нормализация документов (``doc_tokens``) и косинусная
  близость (``cosine_similarity``), переиспользуемые из `lab_2`;
* :mod:`lab_3.src.embeddings` — shallow-модели ``Word2Vec`` (SGNS) и ``Doc2Vec``
  (PV-DBOW) на numpy;
* каталог модели из трёх файлов (``meta.json``, ``vectors.npz``, ``docs.json``) —
  персистенция без повторного обучения.

Публичный API:

* :class:`Match` — результат поиска (реэкспорт из `lab_2`, §2);
* :class:`DocVecAnalyzer` — обучение на корпусе (``fit``), поиск (``query``),
  поиск похожих (``similar``), векторы слов (``word_vector``,
  ``most_similar``), векторы документов (``doc_vector``) и персистенция
  (``save``/``load``).

Особенности контракта (стиль защиты повторяет `lab_2.src.analyzer`):

* ``DocVecAnalyzer(model_dir=None)`` при отсутствии каталога модели возвращает
  НЕобученный экземпляр (без исключения), а несовместимую модель — необученный
  экземпляр с текстом ошибки в :attr:`DocVecAnalyzer.load_error`;
* ``query``/``similar``/``most_similar``/``word_vector``/``doc_vector`` до ``fit``
  бросают ``RuntimeError`` с сообщением :data:`NOT_FITTED_MESSAGE`;
* пустой/мусорный ввод (пустой текст, все токены OOV, вырожденный вектор,
  мусорный ``top_n``, невалидный индекс) не приводит к исключениям, а даёт
  пустой результат.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._lab2 import Match, cosine_similarity, doc_tokens
from .embeddings import Doc2Vec, Word2Vec

__all__ = [
    "Match",
    "DocVecAnalyzer",
    "MODEL_VERSION",
    "NOT_FITTED_MESSAGE",
    "INCOMPATIBLE_MARKER",
]

#: Версия схемы модели (INTERFACES.md §5).
MODEL_VERSION = 1

#: Имена трёх файлов каталога модели (INTERFACES.md §5).
META_FILENAME = "meta.json"
VECTORS_FILENAME = "vectors.npz"
DOCS_FILENAME = "docs.json"

#: Сообщение о вызове анализатора до ``fit`` (INTERFACES.md §4).
NOT_FITTED_MESSAGE = "модель не обучена — вызовите `fit(...)` или выполните ноутбук"

#: Маркер несовместимой модели (INTERFACES.md §4).
INCOMPATIBLE_MARKER = "несовместимая версия модели, переобучите"

#: Ограничение длины сниппета документа в результатах поиска (INTERFACES.md §4).
SNIPPET_LIMIT = 160

#: Порог «численно нулевой нормы» вектора (см. :func:`_is_degenerate`).
_ZERO_NORM_TOL = 1e-12

#: Подкаталог модели по умолчанию: ``<пакет>/models/d2v_demo`` (INTERFACES.md §4).
_DEFAULT_MODEL_PARTS = ("models", "d2v_demo")

#: Значение ``top_n`` по умолчанию для ``query``/``similar``/``most_similar``.
_DEFAULT_TOP_N = 10

#: Режим нормализации по умолчанию (INTERFACES.md §4: лемма + синонимы).
_DEFAULT_MODE = "canonical"

#: Допустимые методы обучения.
_METHODS = ("doc2vec", "word2vec")

#: Число эпох по умолчанию, если ``epochs=None`` (INTERFACES.md §4).
_DEFAULT_EPOCHS = {"doc2vec": 10, "word2vec": 5}


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def _default_model_dir() -> str:
    """``<пакет>/models/d2v_demo`` — каталог модели по умолчанию.

    Основной путь — каталог пакета ``lab_3/src`` (как в `lab_2`). Если его нет,
    но существует альтернативный ``lab_3/models/d2v_demo`` (трактовка «пакет =
    lab_3»), используется он — так готовая модель находится при любой из двух
    допустимых трактовок формулировки §4.
    """
    package_dir = Path(__file__).resolve().parent  # .../lab_3/src
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
    """Привести ``top_n`` к неотрицательному int.

    ``None`` — это «как принято по умолчанию» (-> ``default``), а вот мусор
    (``"abc"``, ``inf``, объект без ``__int__``) — «запрашивать нечего» (-> ``0``,
    то есть пустой результат); §4 требует обрабатывать такой ``top_n`` без
    исключений. Отрицательное значение также даёт ``0``.
    """
    if value is None:
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _effective_epochs(value, default: int) -> int:
    """Привести ``epochs`` к положительному int; ``None``/``0``/мусор -> ``default``.

    Контракт §4 записывает выбор эпох как ``epochs or 10`` / ``epochs or 5``:
    ``0`` и любой некорректный ввод означают «по умолчанию».
    """
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number > 0 else default


def _is_degenerate(vector) -> bool:
    """Численно нулевой вектор (норма ниже допуска) или вектор с NaN/inf.

    Косинусная близость такого вектора не определена: §4 предписывает возвращать
    пустой результат. Нормы обученных векторов составляют ~1e-2…1e0, поэтому всё
    ниже :data:`_ZERO_NORM_TOL` — шум округления, а не сигнал.
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


def _has_model_files(model_dir) -> bool:
    """Есть ли в каталоге все три файла модели (INTERFACES.md §5)."""
    base = Path(model_dir)
    return all(
        (base / name).is_file()
        for name in (META_FILENAME, VECTORS_FILENAME, DOCS_FILENAME)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    """Прочитать JSON-объект файла модели; ошибки чтения/разбора -> ``ValueError``.

    Сообщение содержит маркер несовместимой модели (§4), потому что испорченный
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
    """``ValueError`` с контрактным маркером несовместимой модели (§4)."""
    return ValueError("{}: {}".format(INCOMPATIBLE_MARKER, detail))


def _as_int(value, default: int) -> int:
    """Привести значение из ``meta.json`` к ``int`` (мусор -> ``default``)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value, default: float) -> float:
    """Привести значение из ``meta.json`` к ``float`` (мусор -> ``default``)."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


# --------------------------------------------------------------------------- #
# Анализатор
# --------------------------------------------------------------------------- #

class DocVecAnalyzer:
    """Фасад Word2Vec/Doc2Vec: обучение, поиск, векторы и персистенция (§4, §5).

    Атрибуты состояния (после ``fit`` или ``load``):

    ``model_dir``
        Каталог модели (строка). По умолчанию ``<пакет>/models/d2v_demo``.
    ``load_error``
        Текст ошибки, если модель в каталоге оказалась несовместимой: конструктор
        в этом случае не падает, а отдаёт необученный экземпляр (§4).
    """

    #: Версия схемы модели (INTERFACES.md §5); доступна и как атрибут класса.
    MODEL_VERSION = MODEL_VERSION

    #: Сообщение о вызове до ``fit`` (INTERFACES.md §4).
    NOT_FITTED_MESSAGE = NOT_FITTED_MESSAGE

    #: Маркер несовместимой модели (INTERFACES.md §4).
    INCOMPATIBLE_MARKER = INCOMPATIBLE_MARKER

    # ------------------------------------------------------------------ #
    # Создание и загрузка
    # ------------------------------------------------------------------ #

    def __init__(self, model_dir=None, method: str = "doc2vec"):
        """Создать анализатор, при наличии готовой модели — загрузить её (§4).

        :param model_dir: каталог модели; ``None`` -> ``<пакет>/models/d2v_demo``.
        :param method: ``"doc2vec"`` или ``"word2vec"``.
        :raises ValueError: ``method`` вне допустимого набора.
        """
        if method not in _METHODS:
            raise ValueError(
                "method должен быть одним из {}, получено {!r}".format(
                    ", ".join(repr(name) for name in _METHODS), method
                )
            )
        self.model_dir = str(model_dir) if model_dir is not None else _default_model_dir()
        self.method = str(method)
        self.load_error: Optional[str] = None
        self._reset()

        if _has_model_files(self.model_dir):
            try:
                loaded = type(self)._load_from_dir(self.model_dir)
            except (ValueError, OSError) as exc:
                # Контракт §4: при отсутствии ПРИГОДНОЙ модели конструктор не
                # падает, а возвращает необученный экземпляр. Причина доступна в
                # ``load_error``; сам вызов ``load`` такую модель отвергает.
                self.load_error = str(exc)
            else:
                self._adopt(loaded)

    @classmethod
    def load(cls, model_dir) -> "DocVecAnalyzer":
        """Загрузить модель из каталога (INTERFACES.md §4, §5).

        :raises ValueError: каталог/файлы отсутствуют, ``version != 1`` или
            нарушена согласованность длин (``len(vocabulary) ==
            word_vectors.shape[0] == word_counts.shape[0]``, ``n_docs ==
            doc_vectors.shape[0] == len(texts)``).
        """
        return cls._load_from_dir(model_dir)

    @classmethod
    def _load_from_dir(cls, model_dir) -> "DocVecAnalyzer":
        """Собрать анализатор из файлов модели, минуя авто-загрузку ``__init__``."""
        base = Path(model_dir)

        meta_path = base / META_FILENAME
        if not meta_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(META_FILENAME, base))
        meta = _read_json(meta_path)

        version = meta.get("version")
        version_value = _as_int(version, -1)
        if version_value != MODEL_VERSION:
            raise _incompatible(
                "version={!r} в {}, ожидается {}".format(version, meta_path, MODEL_VERSION)
            )

        method = meta.get("method")
        if method not in _METHODS:
            raise _incompatible(
                "method={!r} в {}, ожидается один из {}".format(method, meta_path, _METHODS)
            )

        for name in (VECTORS_FILENAME, DOCS_FILENAME):
            if not (base / name).is_file():
                raise _incompatible("нет файла {} в каталоге {}".format(name, base))

        n_docs = _as_int(meta.get("n_docs"), -1)
        dim = _as_int(meta.get("dim"), -1)
        if n_docs < 0 or dim <= 0:
            raise _incompatible(
                "в {} должны быть целые поля dim > 0 и n_docs >= 0".format(meta_path)
            )

        vocabulary = meta.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise _incompatible("в {} поле vocabulary должно быть списком".format(meta_path))
        vocabulary = [str(term) for term in vocabulary]

        params = meta.get("params")
        if not isinstance(params, dict):
            params = {}

        try:
            with np.load(str(base / VECTORS_FILENAME), allow_pickle=False) as archive:
                doc_vectors = np.asarray(archive["doc_vectors"], dtype=np.float32)
                word_vectors = np.asarray(archive["word_vectors"], dtype=np.float32)
                word_counts = np.asarray(archive["word_counts"], dtype=np.int64)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
            raise _incompatible(
                "не удалось прочитать {}: {}".format(VECTORS_FILENAME, exc)
            ) from exc

        docs = _read_json(base / DOCS_FILENAME)
        texts = docs.get("texts")
        sources = docs.get("sources")
        if not isinstance(texts, list) or not isinstance(sources, list):
            raise _incompatible(
                "в {} поля texts/sources должны быть списками".format(DOCS_FILENAME)
            )
        texts = [str(text) for text in texts]
        sources = [str(source) for source in sources]

        # Согласованность длин (INTERFACES.md §5). Проверяется до присваивания
        # атрибутов, чтобы ни один индекс словаря не вышел за границы матриц.
        if doc_vectors.ndim != 2 or word_vectors.ndim != 2 or word_counts.ndim != 1:
            raise _incompatible(
                "неожиданные формы массивов в {}: doc_vectors {}, word_vectors {}, "
                "word_counts {}".format(
                    VECTORS_FILENAME, doc_vectors.shape, word_vectors.shape, word_counts.shape
                )
            )
        m_words = int(word_vectors.shape[0])
        if not (len(vocabulary) == m_words == int(word_counts.shape[0])):
            raise _incompatible(
                "len(vocabulary)={}, word_vectors.shape[0]={}, word_counts.shape[0]={}".format(
                    len(vocabulary), m_words, int(word_counts.shape[0])
                )
            )
        if int(doc_vectors.shape[0]) != n_docs or len(texts) != n_docs:
            raise _incompatible(
                "n_docs={}, doc_vectors.shape[0]={}, len(texts)={}".format(
                    n_docs, int(doc_vectors.shape[0]), len(texts)
                )
            )
        if len(sources) != len(texts):
            raise _incompatible(
                "длина sources ({}) не совпадает с числом документов ({})".format(
                    len(sources), len(texts)
                )
            )
        if int(doc_vectors.shape[1]) != dim or int(word_vectors.shape[1]) != dim:
            raise _incompatible(
                "dim из {} ({}) не совпадает с формой матриц ({}, {})".format(
                    meta_path, dim, doc_vectors.shape, word_vectors.shape
                )
            )

        model = cls._rebuild_model(
            method, meta, params, vocabulary, dim, doc_vectors, word_vectors, word_counts
        )

        instance = cls.__new__(cls)
        instance.model_dir = str(base)
        instance.method = method
        instance.load_error = None
        instance._reset()
        instance._model = model
        instance._doc_vectors_ = doc_vectors
        instance._word_vectors_ = word_vectors
        instance._word_counts_ = word_counts
        instance._vocabulary_ = vocabulary
        instance._texts = texts
        instance._sources = sources
        instance._mode = str(
            params.get("mode", _DEFAULT_MODE)
            if isinstance(params.get("mode", _DEFAULT_MODE), str)
            else _DEFAULT_MODE
        )
        instance._tokenized = None
        instance._is_fitted = True
        return instance

    @classmethod
    def _rebuild_model(cls, method, meta, params, vocabulary, dim, doc_vectors,
                       word_vectors, word_counts):
        """Восстановить внутренний ``Doc2Vec``/``Word2Vec`` из файлов модели (§5).

        Обучение не повторяется: все атрибуты, нужные ``query``/``similar``/
        ``infer_vector``/``most_similar``/``word_vector``/``doc_vector``,
        заполняются напрямую из ``meta.json`` и ``vectors.npz``.
        """
        common = {
            "dim": dim,
            "min_count": max(1, _as_int(params.get("min_count"), 1)),
            "epochs": max(1, _as_int(params.get("epochs"), _DEFAULT_EPOCHS[method])),
            "negative": max(0, _as_int(params.get("negative"), 5)),
            "alpha": _as_float(params.get("alpha"), 0.025),
            "min_alpha": _as_float(params.get("min_alpha"), 0.0001),
            "seed": _as_int(params.get("seed"), 42),
        }

        vocabulary_map = {term: index for index, term in enumerate(vocabulary)}
        if method == "doc2vec":
            model = Doc2Vec(**common)
            model.vocabulary_ = vocabulary_map
            model.doc_vectors_ = doc_vectors
            model.word_vectors_ = word_vectors
            model.counts_ = word_counts
            model.n_docs_ = int(doc_vectors.shape[0])
            model.n_words_ = len(vocabulary)
        else:
            model = Word2Vec(**common, window=max(0, _as_int(params.get("window"), 5)))
            model.vocabulary_ = vocabulary_map
            # У Word2Vec публичная матрица — входная (``vectors_``); выходная
            # (контекстная) после обучения фасаду не нужна и остаётся ``None``.
            model.vectors_ = word_vectors
            model.counts_ = word_counts
            model.n_words_ = len(vocabulary)
            model._context_vectors_ = None

        model.epochs_ = common["epochs"]
        model.negative_ = common["negative"]
        model.alpha_ = common["alpha"]
        model.min_alpha_ = common["min_alpha"]
        model.min_count_ = common["min_count"]
        model.seed_ = common["seed"]
        if method == "word2vec":
            model.window_ = max(0, _as_int(params.get("window"), 5))
        return model

    def _reset(self) -> None:
        """Сбросить состояние в «не обучен»."""
        self._model = None
        self._doc_vectors_: Optional[np.ndarray] = None
        self._word_vectors_: Optional[np.ndarray] = None
        self._word_counts_: Optional[np.ndarray] = None
        self._vocabulary_: List[str] = []
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._tokenized: Optional[List[List[str]]] = None
        self._mode: str = _DEFAULT_MODE
        self._is_fitted: bool = False

    def _adopt(self, other: "DocVecAnalyzer") -> None:
        """Скопировать состояние загруженной модели (для авто-загрузки в ``__init__``)."""
        self._model = other._model
        self._doc_vectors_ = other._doc_vectors_
        self._word_vectors_ = other._word_vectors_
        self._word_counts_ = other._word_counts_
        self._vocabulary_ = list(other._vocabulary_)
        self._texts = list(other._texts)
        self._sources = list(other._sources)
        self._tokenized = None
        self._mode = other._mode
        self.method = other.method
        self._is_fitted = other._is_fitted

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Обучен ли анализатор (``False`` — методы векторов бросят ``RuntimeError``)."""
        return bool(self._is_fitted)

    @property
    def n_docs(self) -> int:
        """Число документов корпуса обучения (0 до ``fit``)."""
        return len(self._texts)

    @property
    def n_features(self) -> int:
        """Размер словаря модели (0 до ``fit``)."""
        return len(self._vocabulary_)

    @property
    def dim(self) -> int:
        """Размерность векторов (0 до ``fit``)."""
        if self._model is None:
            return 0
        return int(getattr(self._model, "dim", 0) or 0)

    def _require_fitted(self) -> None:
        """Проверить собственное состояние и не пустить ошибки моделей наружу."""
        if not self._is_fitted or self._model is None or self._doc_vectors_ is None:
            raise RuntimeError(NOT_FITTED_MESSAGE)

    def _word_matrix_of(self, model) -> "np.ndarray":
        """Матрица слов в схеме §5: ``word_vectors_`` (doc2vec) или ``vectors_`` (word2vec)."""
        if self.method == "word2vec":
            return np.asarray(model.vectors_, dtype=np.float32)
        return np.asarray(model.word_vectors_, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(
        self,
        texts,
        *,
        dim: int = 100,
        epochs=None,
        window: int = 5,
        negative: int = 5,
        min_count: int = 1,
        mode: str = _DEFAULT_MODE,
        sources=None,
        seed: int = 42,
    ) -> dict:
        """Обучить анализатор на корпусе и сохранить модель в ``model_dir`` (§4).

        :param texts: итерируемое текстов документов (элементы приводятся как в
            ``lab_2._as_text``: ``None`` -> ``""``, список токенов -> строка).
        :param dim: размерность векторов.
        :param epochs: число эпох; ``None`` -> 10 для ``doc2vec`` и 5 для
            ``word2vec``.
        :param window: полуширина окна контекста (только ``word2vec``).
        :param negative: число негативов на положительную пару.
        :param min_count: минимальная частота токена.
        :param mode: режим нормализации для ``doc_tokens`` (§4).
        :param sources: метки документов той же длины, что и ``texts``; по
            умолчанию ``["doc_0", "doc_1", ...]``.
        :param seed: сид локального ``numpy.random.RandomState``.
        :return: ``{'n_docs', 'n_features', 'dim', 'epochs', 'negative', 'method',
            'seconds'}`` (``n_features`` — размер словаря, ``seconds`` — время
            обучения).
        :raises ValueError: корпус пуст/``None``, длина ``sources`` не совпадает с
            числом документов либо параметры модели некорректны (``embeddings``).
        """
        documents = _as_documents(texts)
        if not documents:
            raise ValueError(
                "texts пуст или None: для обучения нужен непустой итерируемый набор документов"
            )
        n_docs = len(documents)
        source_labels = self._resolve_sources(sources, n_docs)
        normalization_mode = mode if isinstance(mode, str) and mode else _DEFAULT_MODE
        tokenized = [doc_tokens(text, normalization_mode) for text in documents]

        if self.method == "word2vec":
            effective_epochs = _effective_epochs(epochs, _DEFAULT_EPOCHS["word2vec"])
            model = Word2Vec(
                dim=dim,
                window=window,
                min_count=min_count,
                epochs=effective_epochs,
                negative=negative,
                seed=seed,
            )
        else:
            effective_epochs = _effective_epochs(epochs, _DEFAULT_EPOCHS["doc2vec"])
            model = Doc2Vec(
                dim=dim,
                min_count=min_count,
                epochs=effective_epochs,
                negative=negative,
                seed=seed,
            )

        started = time.perf_counter()
        model.fit(tokenized)
        seconds = float(time.perf_counter() - started)

        if self.method == "word2vec":
            doc_vectors = np.vstack(
                [np.asarray(model.doc_vector(tokens), dtype=np.float32) for tokens in tokenized]
            ).astype(np.float32)
        else:
            doc_vectors = np.asarray(model.doc_vectors_, dtype=np.float32)

        self._model = model
        self._doc_vectors_ = doc_vectors
        self._word_vectors_ = self._word_matrix_of(model)
        self._word_counts_ = np.asarray(model.counts_, dtype=np.int64)
        self._vocabulary_ = [
            token for token, _ in sorted(model.vocabulary_.items(), key=lambda item: item[1])
        ]
        self._texts = list(documents)
        self._sources = list(source_labels)
        self._tokenized = [list(tokens) for tokens in tokenized]
        self._mode = normalization_mode
        self._is_fitted = True
        self.load_error = None

        stats = {
            "n_docs": n_docs,
            "n_features": int(model.n_words_),
            "dim": int(model.dim),
            "epochs": int(model.epochs_),
            "negative": int(model.negative_),
            "method": self.method,
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
        similarities = cosine_similarity(vector, self._doc_vectors_)
        return self._top_matches(similarities, top_n)

    def similar(self, text_or_index, top_n: int = 10, exclude_self: bool = True) -> "list[Match]":
        """Найти документы, похожие на документ корпуса или на текст (§4).

        :param text_or_index: ``int`` — индекс документа корпуса (готовый
            ``doc_vectors_[index]``); ``str`` — текст, сворачиваемый тем же
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
            vector = np.asarray(self._doc_vectors_[index], dtype=np.float32).ravel()
            if _is_degenerate(vector):
                # Вектора у документа нет (норма ~0): косинус не определён,
                # ближайших документов не существует.
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

        similarities = cosine_similarity(vector, self._doc_vectors_)
        return self._top_matches(similarities, top_n, exclude=excluded)

    def most_similar(self, word, top_n: int = 10) -> "list[tuple]":
        """``list[(слово, косинус)]`` ближайших слов; OOV -> ``[]`` (§4).

        Для ``doc2vec`` используется выходная матрица слов
        (``Doc2Vec.most_similar_words``), для ``word2vec`` — входная
        (``Word2Vec.most_similar``).
        """
        self._require_fitted()
        limit = _limit(top_n)
        if limit <= 0:
            return []
        if self.method == "doc2vec":
            return list(self._model.most_similar_words(word, top_n=limit))
        return list(self._model.most_similar(word, top_n=limit))

    def word_vector(self, word):
        """Вектор слова (``numpy.ndarray``) или ``None`` для OOV (§4)."""
        self._require_fitted()
        return self._model.word_vector(word)

    def doc_vector(self, text_or_index):
        """Вектор документа: ``int`` — ``doc_vectors_[index]``; ``str`` — fold-in (§4).

        :raises IndexError: индекс вне ``[0, n_docs)`` (как ``Doc2Vec.doc_vector``).
        """
        self._require_fitted()
        if isinstance(text_or_index, bool):
            raise IndexError(
                "index документа должен быть целым числом, получено {!r}".format(text_or_index)
            )
        if isinstance(text_or_index, (int, np.integer)):
            return self._model.doc_vector(int(text_or_index))
        vector = self._fold_in(text_or_index)
        if vector is None:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(vector, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Персистенция
    # ------------------------------------------------------------------ #

    def save(self, model_dir=None) -> str:
        """Сохранить модель в ``model_dir`` (по умолчанию — текущий) и вернуть путь.

        Каталог модели состоит ровно из трёх файлов (INTERFACES.md §5):
        ``meta.json``, ``vectors.npz``, ``docs.json``. Родительские каталоги
        создаются.
        """
        self._require_fitted()
        target = Path(model_dir) if model_dir is not None else Path(self.model_dir)
        target.mkdir(parents=True, exist_ok=True)

        model = self._model
        vocab_items = sorted(model.vocabulary_.items(), key=lambda item: item[1])
        vocabulary = [str(token) for token, _ in vocab_items]

        meta = {
            "version": MODEL_VERSION,
            "method": self.method,
            "dim": int(model.dim),
            "n_docs": len(self._texts),
            "vocabulary": vocabulary,
            "params": {
                "epochs": int(getattr(model, "epochs_", model.epochs)),
                "negative": int(getattr(model, "negative_", model.negative)),
                "min_count": int(getattr(model, "min_count_", model.min_count)),
                "window": int(getattr(model, "window_", getattr(model, "window", 5)) or 0),
                "mode": self._mode,
                "seed": int(getattr(model, "seed_", model.seed)),
                "alpha": float(getattr(model, "alpha_", model.alpha)),
                "min_alpha": float(getattr(model, "min_alpha_", model.min_alpha)),
            },
        }
        _write_json(target / META_FILENAME, meta)
        np.savez(
            str(target / VECTORS_FILENAME),
            doc_vectors=np.asarray(self._doc_vectors_, dtype=np.float32),
            word_vectors=np.asarray(self._word_vectors_, dtype=np.float32),
            word_counts=np.asarray(self._word_counts_, dtype=np.int64),
        )
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
        """Свернуть текст в вектор пространства модели.

        :return: вектор формы ``(dim,)`` либо ``None``, если токенов нет, все они
            вне словаря (OOV) или вектор численно нулевой (норма ниже
            :data:`_ZERO_NORM_TOL`): у такого запроса нет сигнала, и косинусная
            близость к нему была бы чистым шумом округления.
        """
        tokens = doc_tokens(text, self._mode or _DEFAULT_MODE)
        if not tokens:
            return None
        if self.method == "word2vec":
            vector = self._model.doc_vector(tokens)
        else:
            vector = self._model.infer_vector(tokens)
        array = np.asarray(vector, dtype=np.float32).ravel()
        if _is_degenerate(array):
            return None
        return array

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

        Близости документов с численно нулевым вектором принудительно обнуляются:
        ``cosine_similarity`` уже отдаёт ``0.0`` при нулевой норме строки, но
        страховка нужна и на случай NaN/inf в сохранённой модели.
        """
        scores = np.asarray(similarities, dtype=np.float64).ravel()
        limit = _limit(top_n)
        if limit == 0 or scores.size == 0:
            return []

        scores = scores.copy()
        scores[~np.isfinite(scores)] = 0.0
        norms = np.linalg.norm(
            np.asarray(self._doc_vectors_, dtype=np.float64), axis=1
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
        return "DocVecAnalyzer(model_dir={!r}, method={!r}, n_docs={}, n_features={}, dim={})".format(
            self.model_dir, self.method, self.n_docs, self.n_features, self.dim
        )
