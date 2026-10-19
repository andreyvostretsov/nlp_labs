# -*- coding: utf-8 -*-
"""Сквозной фасад анализатора на CNN/GRU (INTERFACES.md §4, §5).

Модуль связывает три слоя прототипа в один пользовательский сценарий:

* :mod:`lab_4.src._lab2` — нормализация документов (``doc_tokens``) и косинусная
  близость (``cosine_similarity``), переиспользуемые из `lab_2`;
* :mod:`lab_4.src.models` — нейросети ``TextCNN`` (Kim, 2014) и ``GRUEncoder``
  (двунаправленный GRU) на PyTorch;
* каталог модели из трёх файлов (``meta.json``, ``weights.pt``, ``docs.json``) —
  персистенция без повторного обучения.

Публичный API:

* :class:`Match` — результат поиска (реэкспорт из `lab_2`);
* :class:`Prediction` — результат классификации документа;
* :class:`NeuralAnalyzer` — обучение на корпусе (``fit``), поиск (``query``),
  поиск похожих (``similar``), классификация (``predict``), векторы слов
  (``word_vector``, ``most_similar``), векторы документов (``doc_vector``) и
  персистенция (``save``/``load``).

Особенности контракта (стиль защиты повторяет `lab_2.src.analyzer`):

* ``NeuralAnalyzer(model_dir=None)`` при отсутствии каталога модели возвращает
  НЕобученный экземпляр (без исключения), а несовместимую модель — необученный
  экземпляр с текстом ошибки в :attr:`NeuralAnalyzer.load_error`;
* ``query``/``similar``/``most_similar``/``word_vector``/``doc_vector`` до ``fit``
  бросают ``RuntimeError`` с сообщением :data:`NOT_FITTED_MESSAGE`;
* пустой/мусорный ввод (пустой текст, все токены OOV, вырожденный вектор,
  мусорный ``top_n``, невалидный индекс) не приводит к исключениям, а даёт
  пустой результат (кроме ``predict`` — см. контракт).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ._lab2 import Match, cosine_similarity, doc_tokens
from .models import TextCNN, GRUEncoder

__all__ = [
    "Prediction",
    "NeuralAnalyzer",
    "Match",
    "MODEL_VERSION",
    "NOT_FITTED_MESSAGE",
    "INCOMPATIBLE_MARKER",
]

#: Версия схемы модели (INTERFACES.md §5).
MODEL_VERSION = 1

#: Имена трёх файлов каталога модели (INTERFACES.md §5).
META_FILENAME = "meta.json"
WEIGHTS_FILENAME = "weights.pt"
DOCS_FILENAME = "docs.json"

#: Сообщение о вызове анализатора до ``fit`` (INTERFACES.md §4).
NOT_FITTED_MESSAGE = "модель не обучена — вызовите fit(...) или выполните ноутбук"

#: Маркер несовместимой модели (INTERFACES.md §4).
INCOMPATIBLE_MARKER = "несовместимая версия модели, переобучите"

#: Ограничение длины сниппета документа в результатах поиска (INTERFACES.md §4).
SNIPPET_LIMIT = 160

#: Порог «численно нулевой нормы» вектора (см. :func:`_is_degenerate`).
_ZERO_NORM_TOL = 1e-12

#: Значение ``top_n`` по умолчанию для ``query``/``similar``/``most_similar``.
_DEFAULT_TOP_N = 10

#: Режим нормализации по умолчанию (INTERFACES.md §4: лемма + синонимы).
_DEFAULT_MODE = "canonical"

#: Допустимые методы обучения.
_METHODS = ("cnn", "rnn")

#: Подкаталоги модели по умолчанию (по методу): ``<пакет>/models/<имя>``.
_DEFAULT_MODEL_PARTS = {"cnn": ("models", "cnn_demo"), "rnn": ("models", "rnn_demo")}

#: Значения по умолчанию для гиперпараметров (совпадают с сигнатурой ``fit``).
_DEFAULTS = {
    "embedding_dim": 64,
    "hidden_dim": 64,
    "num_filters": 100,
    "kernel_sizes": (3, 4, 5),
    "num_layers": 1,
    "dropout": 0.0,
    "max_len": 64,
    "epochs": 30,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "seed": 42,
}


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def _default_model_dir(method: str) -> str:
    """``<пакет>/models/<имя>`` — каталог модели по умолчанию для метода.

    Основной путь — каталог пакета ``lab_4/src``. Если его нет, но существует
    альтернативный ``lab_4/models/<имя>`` (трактовка «пакет = lab_4»), используется
    он — так готовая модель находится при любой из двух трактовок.
    """
    package_dir = Path(__file__).resolve().parent  # .../lab_4/src
    primary = package_dir.joinpath(*_DEFAULT_MODEL_PARTS[method])
    if not primary.is_dir():
        alternative = package_dir.parent.joinpath(*_DEFAULT_MODEL_PARTS[method])
        if alternative.is_dir():
            return str(alternative)
    return str(primary)


def _as_text(value) -> str:
    """Привести элемент корпуса к тексту (``None`` -> ``""``)."""
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
    """Привести ``top_n`` к неотрицательному int; мусор -> ``0`` (без исключений)."""
    if value is None:
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _is_degenerate(vector) -> bool:
    """Численно нулевой вектор (норма ниже допуска) или вектор с NaN/inf."""
    array = np.asarray(vector, dtype=np.float64).ravel()
    if array.size == 0:
        return True
    if not bool(np.all(np.isfinite(array))):
        return True
    return float(np.linalg.norm(array)) <= _ZERO_NORM_TOL


def _as_documents(texts) -> Optional[List[str]]:
    """Привести ``texts`` к списку текстов; ``None``/неитерируемое -> ``None``."""
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
        for name in (META_FILENAME, WEIGHTS_FILENAME, DOCS_FILENAME)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    """Прочитать JSON-объект файла модели; ошибки чтения/разбора -> ``ValueError``."""
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
# Результат классификации
# --------------------------------------------------------------------------- #

@dataclass
class Prediction:
    """Результат классификации документа (INTERFACES.md §4).

    :param label: предсказанная тема (класс);
    :param probabilities: вероятности по всем классам, сумма 1.0.
    """

    label: str
    probabilities: Dict[str, float]


# --------------------------------------------------------------------------- #
# Анализатор
# --------------------------------------------------------------------------- #

class NeuralAnalyzer:
    """Фасад CNN/GRU: обучение, поиск, классификация и персистенция (§4, §5).

    Атрибуты состояния (после ``fit`` или ``load``):

    ``model_dir``
        Каталог модели (строка). По умолчанию ``<пакет>/models/cnn_demo`` (для
        ``method="cnn"``) или ``<пакет>/models/rnn_demo`` (для ``method="rnn"``).
    ``load_error``
        Текст ошибки, если модель в каталоге оказалась несовместимой.
    """

    MODEL_VERSION = MODEL_VERSION
    NOT_FITTED_MESSAGE = NOT_FITTED_MESSAGE
    INCOMPATIBLE_MARKER = INCOMPATIBLE_MARKER

    # ------------------------------------------------------------------ #
    # Создание и загрузка
    # ------------------------------------------------------------------ #

    def __init__(self, model_dir=None, method: str = "cnn"):
        """Создать анализатор, при наличии готовой модели — загрузить её (§4).

        :param model_dir: каталог модели; ``None`` -> каталог по умолчанию для метода.
        :param method: ``"cnn"`` или ``"rnn"``.
        :raises ValueError: ``method`` вне допустимого набора.
        """
        if method not in _METHODS:
            raise ValueError(
                "method должен быть одним из {}, получено {!r}".format(
                    ", ".join(repr(name) for name in _METHODS), method
                )
            )
        self.model_dir = str(model_dir) if model_dir is not None else _default_model_dir(method)
        self.method = str(method)
        self.load_error: Optional[str] = None
        self._reset()

        if _has_model_files(self.model_dir):
            try:
                loaded = type(self).load(self.model_dir)
            except (ValueError, OSError) as exc:
                self.load_error = str(exc)
            else:
                self._adopt(loaded)

    @classmethod
    def load(cls, model_dir) -> "NeuralAnalyzer":
        """Загрузить модель из каталога (INTERFACES.md §4, §5).

        :raises ValueError: каталог/файлы отсутствуют, ``version != 1`` или нарушена
            согласованность длин.
        """
        base = Path(model_dir)
        meta_path = base / META_FILENAME
        if not meta_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(META_FILENAME, base))
        meta = _read_json(meta_path)

        version = _as_int(meta.get("version"), -1)
        if version != MODEL_VERSION:
            raise _incompatible(
                "version={!r} в {}, ожидается {}".format(meta.get("version"), meta_path, MODEL_VERSION)
            )

        method = meta.get("method")
        if method not in _METHODS:
            raise _incompatible(
                "method={!r} в {}, ожидается один из {}".format(method, meta_path, _METHODS)
            )

        weights_path = base / WEIGHTS_FILENAME
        if not weights_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(WEIGHTS_FILENAME, base))

        vocabulary = meta.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise _incompatible("в {} поле vocabulary должно быть списком".format(meta_path))
        vocabulary = [str(term) for term in vocabulary]

        label_order = meta.get("label_order")
        if not isinstance(label_order, list):
            raise _incompatible("в {} поле label_order должно быть списком".format(meta_path))
        label_order = [str(label) for label in label_order]

        n_docs = _as_int(meta.get("n_docs"), -1)
        n_classes = _as_int(meta.get("n_classes"), -1)
        if n_docs < 0 or n_classes < 2:
            raise _incompatible("в {} должны быть целые поля n_docs >= 0 и n_classes >= 2".format(meta_path))
        if len(label_order) != n_classes:
            raise _incompatible(
                "len(label_order)={}, n_classes={}".format(len(label_order), n_classes)
            )

        params = meta.get("params")
        if not isinstance(params, dict):
            params = {}

        docs = _read_json(base / DOCS_FILENAME)
        texts = docs.get("texts")
        sources = docs.get("sources")
        labels = docs.get("labels")
        if not isinstance(texts, list) or not isinstance(sources, list) or not isinstance(labels, list):
            raise _incompatible("в {} поля texts/sources/labels должны быть списками".format(DOCS_FILENAME))
        texts = [str(text) for text in texts]
        sources = [str(source) for source in sources]
        labels = [str(label) for label in labels]
        if len(texts) != n_docs or len(sources) != n_docs or len(labels) != n_docs:
            raise _incompatible(
                "n_docs={}, len(texts)={}, len(sources)={}, len(labels)={}".format(
                    n_docs, len(texts), len(sources), len(labels)
                )
            )

        # Загрузка весов и восстановление модуля.
        try:
            state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - любой сбой чтения = несовместимость
            raise _incompatible("не удалось прочитать {}: {}".format(WEIGHTS_FILENAME, exc)) from exc
        state_dict = state.get("state_dict") if isinstance(state, dict) else state

        embedding_dim = _as_int(params.get("embedding_dim"), _DEFAULTS["embedding_dim"])
        hidden_dim = _as_int(params.get("hidden_dim"), _DEFAULTS["hidden_dim"])
        num_filters = _as_int(params.get("num_filters"), _DEFAULTS["num_filters"])
        num_layers = _as_int(params.get("num_layers"), _DEFAULTS["num_layers"])
        dropout = _as_float(params.get("dropout"), _DEFAULTS["dropout"])
        bidirectional = bool(params.get("bidirectional", True))
        raw_kernels = params.get("kernel_sizes", list(_DEFAULTS["kernel_sizes"]))
        try:
            kernel_sizes = tuple(int(k) for k in raw_kernels)
        except (TypeError, ValueError) as exc:
            raise _incompatible("kernel_sizes в {} некорректен".format(meta_path)) from exc

        vocab_size = len(vocabulary) + 1
        try:
            if method == "cnn":
                model = TextCNN(
                    vocab_size, embedding_dim=embedding_dim, num_filters=num_filters,
                    kernel_sizes=kernel_sizes, num_classes=n_classes, dropout=dropout,
                )
            else:
                model = GRUEncoder(
                    vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                    num_layers=num_layers, num_classes=n_classes, dropout=dropout,
                    bidirectional=bidirectional,
                )
            model.load_state_dict(state_dict)
        except Exception as exc:  # noqa: BLE001
            raise _incompatible("не удалось восстановить модель из {}: {}".format(WEIGHTS_FILENAME, exc)) from exc
        model.eval()

        if int(model.embedding.weight.shape[0]) != vocab_size:
            raise _incompatible(
                "len(vocabulary)+1={}, embedding.weight.shape[0]={}".format(
                    vocab_size, int(model.embedding.weight.shape[0])
                )
            )

        mode = params.get("mode") if isinstance(params.get("mode"), str) and params["mode"] else _DEFAULT_MODE
        max_len_eff = max(1, _as_int(params.get("max_len"), _DEFAULTS["max_len"]))

        token_to_idx = {token: index + 1 for index, token in enumerate(vocabulary)}
        tokenized = [doc_tokens(text, mode) for text in texts]
        doc_token_indices = [
            [token_to_idx[token] for token in doc if token in token_to_idx]
            for doc in tokenized
        ]

        arr, lengths = cls._pad_indices(doc_token_indices, max_len_eff)
        with torch.no_grad():
            doc_vectors = model.encode(
                torch.from_numpy(arr), torch.from_numpy(lengths)
            ).cpu().numpy().astype(np.float32)
        doc_vector_dim = int(doc_vectors.shape[1])

        instance = cls.__new__(cls)
        instance.model_dir = str(base)
        instance.method = method
        instance.load_error = None
        instance._reset()
        instance._model = model
        instance._vocabulary_ = vocabulary
        instance._token_to_idx_ = token_to_idx
        instance._label_order_ = label_order
        instance._label_to_idx_ = {label: index for index, label in enumerate(label_order)}
        instance._doc_token_indices_ = doc_token_indices
        instance._doc_vectors_ = doc_vectors
        instance._texts = texts
        instance._sources = sources
        instance._labels = labels
        instance._mode = mode
        instance._max_len = max_len_eff
        instance._doc_vector_dim = doc_vector_dim
        instance._params = dict(params)
        instance._is_fitted = True
        return instance

    def _reset(self) -> None:
        """Сбросить состояние в «не обучен»."""
        self._model = None
        self._vocabulary_: List[str] = []
        self._token_to_idx_: Dict[str, int] = {}
        self._label_order_: List[str] = []
        self._label_to_idx_: Dict[str, int] = {}
        self._doc_token_indices_: List[List[int]] = []
        self._doc_vectors_: Optional[np.ndarray] = None
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._labels: List[str] = []
        self._mode: str = _DEFAULT_MODE
        self._max_len: int = _DEFAULTS["max_len"]
        self._doc_vector_dim: int = 0
        self._params: Dict[str, Any] = {}
        self._is_fitted: bool = False

    def _adopt(self, other: "NeuralAnalyzer") -> None:
        """Скопировать состояние загруженной модели (для авто-загрузки в ``__init__``)."""
        self._model = other._model
        self._vocabulary_ = list(other._vocabulary_)
        self._token_to_idx_ = dict(other._token_to_idx_)
        self._label_order_ = list(other._label_order_)
        self._label_to_idx_ = dict(other._label_to_idx_)
        self._doc_token_indices_ = [list(seq) for seq in other._doc_token_indices_]
        self._doc_vectors_ = other._doc_vectors_
        self._texts = list(other._texts)
        self._sources = list(other._sources)
        self._labels = list(other._labels)
        self._mode = other._mode
        self._max_len = other._max_len
        self._doc_vector_dim = other._doc_vector_dim
        self._params = dict(other._params)
        self.method = other.method
        self._is_fitted = other._is_fitted

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Обучен ли анализатор (``False`` — методы бросят ``RuntimeError``)."""
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
    def n_classes(self) -> int:
        """Число классов классификации (0 до ``fit``)."""
        return len(self._label_order_)

    @property
    def dim(self) -> int:
        """Размерность вектора документа (0 до ``fit``)."""
        return self._doc_vector_dim

    def _require_fitted(self) -> None:
        """Проверить собственное состояние и не пустить ошибки моделей наружу."""
        if not self._is_fitted or self._model is None or self._doc_vectors_ is None:
            raise RuntimeError(NOT_FITTED_MESSAGE)

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(
        self,
        texts,
        *,
        labels=None,
        sources=None,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_filters: int = 100,
        kernel_sizes=(3, 4, 5),
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        max_len: int = 64,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        mode: str = _DEFAULT_MODE,
        seed: int = 42,
    ) -> dict:
        """Обучить анализатор на корпусе и сохранить модель в ``model_dir`` (§4).

        :param texts: итерируемое текстов документов.
        :param labels: темы документов (той же длины, что и ``texts``); если ``None`` —
            тема выводится из ``sources`` (префикс до первого ``_``/``.``).
        :param sources: метки документов (по умолчанию ``["doc_0", ...]``).
        :param embedding_dim: размерность эмбеддинга слов.
        :param hidden_dim: размерность скрытого состояния GRU (только ``rnn``).
        :param num_filters: число фильтров CNN на каждый размер ядра (только ``cnn``).
        :param kernel_sizes: размеры ядер CNN (только ``cnn``).
        :param num_layers: число слоёв (только ``rnn``).
        :param bidirectional: двунаправленный GRU (только ``rnn``).
        :param dropout: вероятность дропаута.
        :param max_len: максимальная длина документа в токенах.
        :param epochs: число эпох обучения.
        :param learning_rate: learning rate оптимизатора Adam.
        :param batch_size: размер батча.
        :param mode: режим нормализации для ``doc_tokens``.
        :param seed: сид воспроизводимости.
        :return: ``{'n_docs', 'n_features', 'n_classes', 'dim', 'method', 'epochs',
            'accuracy', 'seconds'}``.
        :raises ValueError: корпус пуст/``None``, < 2 классов, длина ``labels``/
            ``sources`` не совпадает, либо параметры модели некорректны.
        """
        documents = _as_documents(texts)
        if not documents:
            raise ValueError(
                "texts пуст или None: для обучения нужен непустой итерируемый набор документов"
            )
        n_docs = len(documents)
        source_labels = self._resolve_sources(sources, n_docs)
        class_labels = self._resolve_labels(labels, source_labels, n_docs)
        normalization_mode = mode if isinstance(mode, str) and mode else _DEFAULT_MODE
        tokenized = [doc_tokens(text, normalization_mode) for text in documents]

        vocabulary = sorted({token for doc in tokenized for token in doc})
        if not vocabulary:
            raise ValueError("словарь пуст: в документах нет ни одного токена после нормализации")
        token_to_idx = {token: index + 1 for index, token in enumerate(vocabulary)}

        label_order = sorted(set(class_labels))
        if len(label_order) < 2:
            raise ValueError(
                "нужны минимум 2 класса, получено {}: {!r}".format(len(label_order), label_order)
            )
        label_to_idx = {label: index for index, label in enumerate(label_order)}

        try:
            kernels = tuple(int(k) for k in kernel_sizes)
        except (TypeError, ValueError) as exc:
            raise ValueError("kernel_sizes должен быть итерируемым целых чисел") from exc

        try:
            max_len_value = int(max_len)
        except (TypeError, ValueError):
            max_len_value = _DEFAULTS["max_len"]
        if self.method == "cnn":
            needed = max(kernels) + 1 if kernels else 1
            max_len_eff = max(1, max_len_value, needed)
        else:
            max_len_eff = max(1, max_len_value)

        doc_token_indices = [[token_to_idx[token] for token in doc] for doc in tokenized]
        arr, lengths = self._pad_indices(doc_token_indices, max_len_eff)
        y = np.array([label_to_idx[label] for label in class_labels], dtype=np.int64)

        vocab_size = len(vocabulary) + 1

        # Сид выставляется ДО создания модели: инициализация весов и обучение
        # должны черпать случайность из одного и того же воспроизводимого потока.
        seed_value = _as_int(seed, _DEFAULTS["seed"])
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)

        if self.method == "cnn":
            model = TextCNN(
                vocab_size, embedding_dim=embedding_dim, num_filters=num_filters,
                kernel_sizes=kernels, num_classes=len(label_order), dropout=dropout,
            )
        else:
            model = GRUEncoder(
                vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                num_layers=num_layers, num_classes=len(label_order), dropout=dropout,
                bidirectional=bidirectional,
            )

        epochs_value = _as_int(epochs, _DEFAULTS["epochs"])
        if epochs_value <= 0:
            epochs_value = _DEFAULTS["epochs"]

        X_t = torch.from_numpy(arr)
        len_t = torch.from_numpy(lengths)
        y_t = torch.from_numpy(y)

        started = time.perf_counter()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        total = int(X_t.size(0))
        bs = max(1, int(batch_size))
        for _ in range(epochs_value):
            order = np.random.permutation(total)
            for start in range(0, total, bs):
                indices = order[start:start + bs]
                xb = X_t[indices]
                lb = len_t[indices]
                yb = y_t[indices]
                optimizer.zero_grad()
                logits = model(xb, lb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
        seconds = float(time.perf_counter() - started)

        model.eval()
        with torch.no_grad():
            doc_vectors = model.encode(X_t, len_t).cpu().numpy().astype(np.float32)
            logits = model(X_t, len_t)
            predictions = logits.argmax(dim=1)
            accuracy = float((predictions == y_t).float().mean().item())
        doc_vector_dim = int(doc_vectors.shape[1])

        self._model = model
        self._vocabulary_ = vocabulary
        self._token_to_idx_ = token_to_idx
        self._label_order_ = label_order
        self._label_to_idx_ = label_to_idx
        self._doc_token_indices_ = doc_token_indices
        self._doc_vectors_ = doc_vectors
        self._texts = list(documents)
        self._sources = list(source_labels)
        self._labels = list(class_labels)
        self._mode = normalization_mode
        self._max_len = max_len_eff
        self._doc_vector_dim = doc_vector_dim
        self._params = {
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "num_filters": int(num_filters),
            "kernel_sizes": list(kernels),
            "num_layers": int(num_layers),
            "bidirectional": bool(bidirectional),
            "dropout": float(dropout),
            "max_len": max_len_eff,
            "mode": normalization_mode,
            "seed": seed_value,
            "epochs": epochs_value,
            "learning_rate": float(learning_rate),
            "batch_size": bs,
            "accuracy": accuracy,
        }
        self._is_fitted = True
        self.load_error = None

        stats = {
            "n_docs": n_docs,
            "n_features": len(vocabulary),
            "n_classes": len(label_order),
            "dim": doc_vector_dim,
            "method": self.method,
            "epochs": epochs_value,
            "accuracy": accuracy,
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

    @staticmethod
    def _resolve_labels(labels, sources: List[str], n_docs: int) -> List[str]:
        """Темы документов: явные ``labels`` либо префиксы ``sources`` (до ``_``/``.``)."""
        if labels is None:
            return [NeuralAnalyzer._topic_of(source) for source in sources]
        if isinstance(labels, str):
            raw = [labels]
        else:
            try:
                raw = list(labels)
            except TypeError as exc:
                raise ValueError(
                    "labels должен быть итерируемым строк, получено {!r}".format(labels)
                ) from exc
        result = ["" if item is None else str(item) for item in raw]
        if len(result) != n_docs:
            raise ValueError(
                "длина labels ({}) не совпадает с числом документов ({})".format(
                    len(result), n_docs
                )
            )
        return result

    @staticmethod
    def _topic_of(source) -> str:
        """Тема из метки источника: префикс до первого ``_`` или ``.``."""
        text = str(source)
        for separator in ("_", "."):
            if separator in text:
                return text.split(separator, 1)[0]
        return text

    @staticmethod
    def _pad_indices(indices_list, max_len: int):
        """Собрать батч токенов: ``(массив (n, max_len), длины (n,))``.

        Индексы токенов 1-based; ``0`` = PAD. Последовательности длиннее ``max_len``
        обрезаются.
        """
        n = len(indices_list)
        arr = np.zeros((n, max_len), dtype=np.int64)
        lengths = np.zeros(n, dtype=np.int64)
        for i, seq in enumerate(indices_list):
            length = min(len(seq), max_len)
            if length:
                arr[i, :length] = seq[:length]
            lengths[i] = length
        return arr, lengths

    # ------------------------------------------------------------------ #
    # Поиск
    # ------------------------------------------------------------------ #

    def query(self, text, top_n: int = 10) -> "List[Match]":
        """Найти ``top_n`` документов, ближайших к тексту запроса (§4)."""
        self._require_fitted()
        vector = self._fold_in(text)
        if vector is None:
            return []
        similarities = cosine_similarity(vector, self._doc_vectors_)
        return self._top_matches(similarities, top_n)

    def similar(self, text_or_index, top_n: int = 10, exclude_self: bool = True) -> "List[Match]":
        """Найти документы, похожие на документ корпуса или на текст (§4)."""
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

    def predict(self, text_or_index) -> "Prediction":
        """Классифицировать документ: ``int`` — индекс корпуса, ``str`` — текст (§4).

        :raises ValueError: невалидный индекс, либо пустой текст/все токены OOV.
        """
        self._require_fitted()
        if isinstance(text_or_index, bool):
            raise ValueError("аргумент predict должен быть текстом или индексом документа")
        if isinstance(text_or_index, (int, np.integer)):
            index = int(text_or_index)
            if not 0 <= index < self.n_docs:
                raise ValueError(
                    "индекс документа {} вне диапазона [0, {})".format(index, self.n_docs)
                )
            indices = self._doc_token_indices_[index]
            if not indices:
                raise ValueError("не удалось извлечь признаки из документа")
            logits = self._logits_of_indices([indices])[0]
        elif isinstance(text_or_index, str):
            tokens = doc_tokens(text_or_index, self._mode or _DEFAULT_MODE)
            indices = [self._token_to_idx_[token] for token in tokens if token in self._token_to_idx_]
            if not indices:
                raise ValueError("не удалось извлечь признаки из текста")
            logits = self._logits_of_indices([indices])[0]
        else:
            raise ValueError("аргумент predict должен быть текстом или индексом документа")
        return self._prediction_from_logits(logits)

    def most_similar(self, word, top_n: int = 10) -> "List[tuple]":
        """``list[(слово, косинус)]`` ближайших слов по матрице эмбеддинга; OOV -> ``[]`` (§4)."""
        self._require_fitted()
        limit = _limit(top_n)
        if limit <= 0:
            return []
        index = self._token_to_idx_.get(word)
        if index is None:
            return []
        weight = self._model.embedding.weight.detach().cpu().numpy().astype(np.float64)
        vectors = weight[1:]  # без PAD-строки
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = vectors / norms
        query_vector = normalized[index - 1]
        scores = normalized @ query_vector
        scores[index - 1] = -np.inf
        order = np.argsort(-scores, kind="stable")[:limit]
        result = []
        for other in order:
            other = int(other)
            if not np.isfinite(scores[other]):
                continue
            result.append((self._vocabulary_[other], float(scores[other])))
        return result

    def word_vector(self, word):
        """Строка матрицы эмбеддинга слова (float32) или ``None`` при OOV (§4)."""
        self._require_fitted()
        index = self._token_to_idx_.get(word)
        if index is None:
            return None
        weight = self._model.embedding.weight.detach().cpu().numpy().astype(np.float32)
        return np.array(weight[index], dtype=np.float32, copy=True)

    def doc_vector(self, text_or_index):
        """Вектор документа: ``int`` — ``doc_vectors_[index]``; ``str`` — fold-in (§4).

        :raises IndexError: индекс вне ``[0, n_docs)``.
        """
        self._require_fitted()
        if isinstance(text_or_index, bool):
            raise IndexError(
                "index документа должен быть целым числом, получено {!r}".format(text_or_index)
            )
        if isinstance(text_or_index, (int, np.integer)):
            index = int(text_or_index)
            if not 0 <= index < self.n_docs:
                raise IndexError(
                    "индекс документа {} вне диапазона [0, {})".format(index, self.n_docs)
                )
            return np.array(self._doc_vectors_[index], dtype=np.float32, copy=True)
        vector = self._fold_in(text_or_index)
        if vector is None:
            return np.zeros(self.dim, dtype=np.float32)
        return vector

    # ------------------------------------------------------------------ #
    # Персистенция
    # ------------------------------------------------------------------ #

    def save(self, model_dir=None) -> str:
        """Сохранить модель в ``model_dir`` (по умолчанию — текущий) и вернуть путь (§5)."""
        self._require_fitted()
        target = Path(model_dir) if model_dir is not None else Path(self.model_dir)
        target.mkdir(parents=True, exist_ok=True)

        meta = {
            "version": MODEL_VERSION,
            "method": self.method,
            "n_docs": len(self._texts),
            "n_classes": len(self._label_order_),
            "doc_vector_dim": self._doc_vector_dim,
            "vocabulary": list(self._vocabulary_),
            "label_order": list(self._label_order_),
            "params": dict(self._params),
        }
        _write_json(target / META_FILENAME, meta)
        torch.save({"state_dict": self._model.state_dict()}, str(target / WEIGHTS_FILENAME))
        _write_json(
            target / DOCS_FILENAME,
            {"texts": list(self._texts), "sources": list(self._sources), "labels": list(self._labels)},
        )
        self.model_dir = str(target)
        return str(target)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _fold_in(self, text) -> Optional[np.ndarray]:
        """Свернуть текст в вектор пространства модели.

        :return: вектор формы ``(dim,)`` либо ``None``, если токенов нет, все они
            вне словаря (OOV) или вектор численно нулевой.
        """
        tokens = doc_tokens(text, self._mode or _DEFAULT_MODE)
        if not tokens:
            return None
        indices = [self._token_to_idx_[token] for token in tokens if token in self._token_to_idx_]
        if not indices:
            return None
        vector = self._encode_vectors([indices])[0]
        if _is_degenerate(vector):
            return None
        return vector

    def _encode_vectors(self, indices_list) -> np.ndarray:
        """``encode`` модели по списку последовательностей индексов токенов."""
        arr, lengths = self._pad_indices(indices_list, self._max_len)
        with torch.no_grad():
            return self._model.encode(
                torch.from_numpy(arr), torch.from_numpy(lengths)
            ).cpu().numpy().astype(np.float32)

    def _logits_of_indices(self, indices_list) -> np.ndarray:
        """Логиты модели по списку последовательностей индексов токенов."""
        arr, lengths = self._pad_indices(indices_list, self._max_len)
        with torch.no_grad():
            return self._model(
                torch.from_numpy(arr), torch.from_numpy(lengths)
            ).cpu().numpy()

    def _prediction_from_logits(self, logits) -> "Prediction":
        """Softmax по логитам -> ``Prediction`` (вероятности в порядке ``label_order``)."""
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - logits.max()
        exponentials = np.exp(logits)
        probabilities = exponentials / exponentials.sum()
        best = int(np.argmax(probabilities))
        return Prediction(
            label=self._label_order_[best],
            probabilities={
                self._label_order_[i]: float(probabilities[i])
                for i in range(len(self._label_order_))
            },
        )

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

    def _top_matches(self, similarities, top_n, exclude=()) -> "List[Match]":
        """Собрать ``Match`` по убыванию близости, исключая заданные индексы."""
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
        return "NeuralAnalyzer(model_dir={!r}, method={!r}, n_docs={}, n_features={}, dim={})".format(
            self.model_dir, self.method, self.n_docs, self.n_features, self.dim
        )
