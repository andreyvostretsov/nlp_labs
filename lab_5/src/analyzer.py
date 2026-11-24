# -*- coding: utf-8 -*-
"""Сквозной фасад генератора текста на основе семантического анализатора
(INTERFACES.md §6).

`lab_5` расширяет анализатор `lab_4` (`NeuralAnalyzer`, CNN/GRU) средствами
генерации осмысленного текста. Генерация строится **только на основе**
анализатора:

* корпус, темы и словарь берутся из того же набора документов, на котором
  обучен `NeuralAnalyzer`;
* LSTM-языковая модель обусловлена темой анализатора (номер темы — вход модели);
* seq2seq-модель учится на предложениях того же корпуса и решает две задачи:
  «документ → первый абзац» (реферирование) и «предложение → следующее
  предложение» (продолжение);
* гибридный режим ранжирует кандидатов по семантической близости к запросу
  (векторы `NeuralAnalyzer`), уверенности классификатора темы и «гладкости»
  LSTM-модели.

Публичный API:

* :class:`Candidate` — кандидат генерации с компонентами оценки;
* :class:`Generation` — результат генерации (лучший текст + все кандидаты);
* :class:`GeneratingAnalyzer` — обучение (``fit``), генерация (``generate``),
  реферирование (``summarize``), продолжение (``continue_text``), шаблонный
  режим (``template``), оценка качества (``evaluate``) и персистенция
  (``save``/``load``).

Контракт ошибок повторяет стиль `lab_2`–`lab_4`:

* конструктор без готовой модели возвращает **необученный** экземпляр (без
  исключения); несовместимая модель — необученный экземпляр с текстом ошибки в
  :attr:`GeneratingAnalyzer.load_error`;
* ``generate``/``summarize``/``continue_text``/``template``/``evaluate`` до
  ``fit`` бросают ``RuntimeError`` с :data:`NOT_FITTED_MESSAGE`;
* пустой/мусорный ввод не приводит к исключениям: пустой запрос генерирует текст
  по теме, пустой текст для ``summarize`` даёт ``""``, а неизвестная тема —
  ``ValueError`` (единственное явное исключение аргумента).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ._lab1 import canonical_tokens, sent_tokenize
from ._lab2 import doc_tokens
from ._lab4 import NeuralAnalyzer, Prediction
from .generators import (
    BOS,
    EOS,
    FIRST_WORD_INDEX,
    PAD,
    SPECIAL_TOKENS,
    UNK,
    LSTMLanguageModel,
    Seq2SeqLSTM,
    TemplateGenerator,
    build_vocabulary,
    decode_ids,
    encode_tokens,
    format_tokens_as_text,
    mean_log_probability,
    pad_token_sequences,
)

__all__ = [
    "Candidate",
    "Generation",
    "GeneratingAnalyzer",
    "MODEL_VERSION",
    "NOT_FITTED_MESSAGE",
    "INCOMPATIBLE_MARKER",
]

#: Версия схемы модели (INTERFACES.md §7).
MODEL_VERSION = 1

#: Имена файлов каталога модели генератора.
META_FILENAME = "meta.json"
WEIGHTS_FILENAME = "weights.pt"
CORPUS_FILENAME = "corpus.json"

#: Подкаталог семантического анализатора внутри каталога модели.
SEMANTIC_DIRNAME = "semantic"

#: Сообщение о вызове до ``fit``.
NOT_FITTED_MESSAGE = "модель не обучена — вызовите fit(...) или выполните ноутбук"

#: Маркер несовместимой модели.
INCOMPATIBLE_MARKER = "несовместимая версия модели, переобучите"

#: Допустимые семантические ядра.
_METHODS = ("cnn", "rnn")

#: Допустимые режимы генерации.
GEN_METHODS = ("template", "lstm", "seq2seq", "hybrid")

#: Каталоги моделей по умолчанию: ``<пакет>/models/gen_cnn_demo`` и т. п.
_DEFAULT_MODEL_PARTS = {"cnn": ("models", "gen_cnn_demo"), "rnn": ("models", "gen_rnn_demo")}

#: Режим токенизации генератора по умолчанию: словоформы без нормализации.
_DEFAULT_MODE = "raw"

#: Значения гиперпараметров по умолчанию (совпадают с сигнатурой ``fit``).
_DEFAULTS = {
    "semantic_epochs": 40,
    "semantic_batch_size": 8,
    "semantic_max_len": 48,
    "semantic_embedding_dim": 64,
    "semantic_hidden_dim": 64,
    "semantic_num_filters": 64,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "topic_dim": 16,
    "max_len": 32,
    "seq2seq_hidden_dim": 96,
    "seq2seq_num_layers": 1,
    "epochs": 40,
    "seq2seq_epochs": 60,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "seq2seq_learning_rate": 1e-3,
    "max_src_len": 48,
    "max_tgt_len": 24,
    "seed": 42,
}

#: Значение ``max_words`` по умолчанию.
_DEFAULT_MAX_WORDS = 24


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def _default_model_dir(method: str) -> str:
    """``<пакет>/models/gen_<method>_demo`` — каталог модели по умолчанию."""
    package_dir = Path(__file__).resolve().parent  # .../lab_5/src
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
    """Любые последовательности пробельных символов -> один пробел."""
    return " ".join(_as_text(text).split())


def _as_documents(texts) -> Optional[List[str]]:
    """Привести ``texts`` к списку строк; ``None``/неитерируемое -> ``None``."""
    if texts is None:
        return None
    if isinstance(texts, str):
        return [texts]
    try:
        items = list(texts)
    except TypeError:
        return None
    return [_as_text(item) for item in items]


def _as_int(value, default: int) -> int:
    """Привести значение к ``int``; мусор -> ``default``."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value, default: float) -> float:
    """Привести значение к ``float``; мусор -> ``default``."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    """Прочитать JSON-объект; ошибки -> ``ValueError`` с маркером несовместимости."""
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise _incompatible("не удалось прочитать {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise _incompatible("неожиданный формат {}: ожидался объект JSON".format(path))
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Записать JSON-объект в UTF-8 (создаёт каталоги)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _incompatible(detail: str) -> ValueError:
    """``ValueError`` с контрактным маркером несовместимой модели."""
    return ValueError("{}: {}".format(INCOMPATIBLE_MARKER, detail))


def _has_model_files(model_dir) -> bool:
    """Есть ли в каталоге все три файла модели генератора."""
    base = Path(model_dir)
    return all(
        (base / name).is_file()
        for name in (META_FILENAME, WEIGHTS_FILENAME, CORPUS_FILENAME)
    )


def _topic_of(source) -> str:
    """Тема из метки источника: префикс до первого ``_`` или ``.``."""
    text = str(source)
    for separator in ("_", "."):
        if separator in text:
            return text.split(separator, 1)[0]
    return text


def _cosine(left, right) -> float:
    """Косинус двух векторов; вырожденный/несовместимый ввод -> ``0.0``."""
    try:
        a = np.asarray(left, dtype=np.float64).ravel()
        b = np.asarray(right, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return 0.0
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    if not (np.isfinite(norm_a) and np.isfinite(norm_b)):
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _sentence_split(text: str) -> List[str]:
    """Разбить текст на предложения; пустой текст -> ``[]``."""
    sentences = [s.strip() for s in sent_tokenize(_as_text(text)) if s.strip()]
    return sentences


def _ensure_sentence(text: str) -> str:
    """Нормализовать пробелы и добавить точку, если её нет."""
    flat = _normalize_ws(text)
    if flat and flat[-1] not in ".!?…":
        flat += "."
    return flat


def _snippet(text: str, limit: int = 220) -> str:
    """Сниппет фрагмента корпуса не длиннее ``limit`` символов."""
    flat = _normalize_ws(text)
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    for separator in (". ", "! ", "? ", "… "):
        position = cut.rfind(separator)
        if position > limit // 3:
            return cut[: position + 1].strip()
    return cut.rstrip() + "…"


# --------------------------------------------------------------------------- #
# Результаты генерации
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """Один кандидат генерации с компонентами оценки (INTERFACES.md §6).

    :param text: сгенерированный текст.
    :param method: источник кандидата — ``"template"``, ``"lstm"`` или
        ``"seq2seq"``.
    :param topic: тема, которой обусловлена генерация.
    :param score: итоговая оценка (0..1), по ней ранжируются кандидаты.
    :param semantic: косинусная близость к запросу/центроиду темы (0..1).
    :param topic_confidence: вероятность темы по классификатору анализатора.
    :param fluency: ``exp(средний log P)`` по LSTM-модели (0..1).
    :param coherence: доля биграмм текста, встречающихся в корпусе (0..1);
        прокси связности «текст похож на предложения корпуса».
    :param uniqueness: доля уникальных слов (защита от повторов).
    :param completeness: завершённость фразы (нет обрыва на предлоге/союзе).
    :param source: документ-источник фрагмента (для шаблонного режима).
    """

    text: str
    method: str
    topic: str
    score: float = 0.0
    semantic: float = 0.0
    topic_confidence: float = 0.0
    fluency: float = 0.0
    coherence: float = 0.0
    uniqueness: float = 0.0
    completeness: float = 0.0
    source: str = ""


@dataclass
class Generation:
    """Результат генерации (INTERFACES.md §6).

    :param text: лучший сгенерированный текст.
    :param method: запрошенный режим (``template``/``lstm``/``seq2seq``/``hybrid``).
    :param topic: тема генерации.
    :param score: итоговая оценка лучшего кандидата.
    :param prompt: исходный запрос (может быть пустым).
    :param candidates: все кандидаты, отсортированные по убыванию ``score``.
    """

    text: str
    method: str
    topic: str
    score: float
    prompt: str = ""
    candidates: List[Candidate] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Фасад
# --------------------------------------------------------------------------- #

class GeneratingAnalyzer:
    """Генератор текста поверх семантического анализатора `lab_4`.

    Атрибуты состояния (после ``fit``/``load``): ``model_dir``, ``semantic_method``,
    ``load_error``. Обученная модель занимает каталог с файлами
    ``meta.json``, ``weights.pt``, ``corpus.json`` и подкаталогом ``semantic/``
    (модель `lab_4`).
    """

    MODEL_VERSION = MODEL_VERSION
    NOT_FITTED_MESSAGE = NOT_FITTED_MESSAGE
    INCOMPATIBLE_MARKER = INCOMPATIBLE_MARKER
    GEN_METHODS = GEN_METHODS

    # ------------------------------------------------------------------ #
    # Создание / загрузка
    # ------------------------------------------------------------------ #

    def __init__(self, model_dir=None, semantic_method: str = "cnn"):
        """Создать генератор; при наличии готовой модели — загрузить её.

        :param model_dir: каталог модели; ``None`` -> каталог по умолчанию для
            выбранного семантического ядра.
        :param semantic_method: ``"cnn"`` или ``"rnn"`` (ядро `lab_4`).
        :raises ValueError: ``semantic_method`` вне допустимого набора.
        """
        if semantic_method not in _METHODS:
            raise ValueError(
                "semantic_method должен быть одним из {}, получено {!r}".format(
                    ", ".join(repr(name) for name in _METHODS), semantic_method
                )
            )
        self.model_dir = str(model_dir) if model_dir is not None else _default_model_dir(semantic_method)
        self.semantic_method = str(semantic_method)
        self.load_error: Optional[str] = None
        self._reset()

        if _has_model_files(self.model_dir):
            try:
                loaded = type(self).load(self.model_dir)
            except (ValueError, OSError) as exc:
                self.load_error = str(exc)
            else:
                self._adopt(loaded)

    def _reset(self) -> None:
        """Сбросить состояние в «не обучен»."""
        self._semantic: Optional[NeuralAnalyzer] = None
        self._lm: Optional[LSTMLanguageModel] = None
        self._lead: Optional[Seq2SeqLSTM] = None
        self._next: Optional[Seq2SeqLSTM] = None
        self._vocabulary: Dict[str, int] = {}
        self._topic_order: List[str] = []
        self._topic_to_idx: Dict[str, int] = {}
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._labels: List[str] = []
        self._topic_centroids: Dict[str, np.ndarray] = {}
        self._bigrams: set = set()
        self._bigram_next: Dict[str, set] = {}
        self._trigram_next: Dict[Tuple[str, str], set] = {}
        self._templates = TemplateGenerator()
        self._mode = _DEFAULT_MODE
        self._max_len = _DEFAULTS["max_len"]
        self._max_src_len = _DEFAULTS["max_src_len"]
        self._max_tgt_len = _DEFAULTS["max_tgt_len"]
        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._is_fitted = False

    def _adopt(self, other: "GeneratingAnalyzer") -> None:
        """Скопировать состояние загруженной модели."""
        self._semantic = other._semantic
        self._lm = other._lm
        self._lead = other._lead
        self._next = other._next
        self._vocabulary = dict(other._vocabulary)
        self._topic_order = list(other._topic_order)
        self._topic_to_idx = dict(other._topic_to_idx)
        self._texts = list(other._texts)
        self._sources = list(other._sources)
        self._labels = list(other._labels)
        self._topic_centroids = dict(other._topic_centroids)
        self._bigrams = set(other._bigrams)
        self._bigram_next = {key: set(value) for key, value in other._bigram_next.items()}
        self._trigram_next = {key: set(value) for key, value in other._trigram_next.items()}
        self._templates = other._templates
        self._mode = other._mode
        self._max_len = other._max_len
        self._max_src_len = other._max_src_len
        self._max_tgt_len = other._max_tgt_len
        self._params = dict(other._params)
        self._metrics = dict(other._metrics)
        self.semantic_method = other.semantic_method
        self._is_fitted = other._is_fitted

    # ------------------------------------------------------------------ #
    # Свойства
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Обучен ли генератор."""
        return bool(self._is_fitted)

    @property
    def semantic_analyzer(self) -> "NeuralAnalyzer":
        """Семантическое ядро `lab_4` (для поиска и классификации)."""
        self._require_fitted()
        return self._semantic

    @property
    def n_docs(self) -> int:
        """Число документов корпуса."""
        return len(self._texts)

    @property
    def n_features(self) -> int:
        """Число настоящих слов словаря (без специальных токенов)."""
        return max(0, len(self._vocabulary) - FIRST_WORD_INDEX)

    @property
    def dim(self) -> int:
        """Размерность вектора документа семантического ядра (0 до ``fit``)."""
        if self._semantic is None or not self._semantic.is_fitted:
            return 0
        return int(self._semantic.dim)

    @property
    def n_topics(self) -> int:
        """Число тем."""
        return len(self._topic_order)

    @property
    def topics(self) -> List[str]:
        """Темы в каноническом (лексикографическом) порядке."""
        return list(self._topic_order)

    @property
    def params(self) -> Dict[str, Any]:
        """Гиперпараметры обученной модели."""
        return dict(self._params)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Метрики обучения (потери, перплексия, точность)."""
        return dict(self._metrics)

    def _require_fitted(self) -> None:
        """Проверить, что модель обучена."""
        if not self._is_fitted or self._semantic is None or self._lm is None:
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
        semantic_method: Optional[str] = None,
        semantic_epochs: int = _DEFAULTS["semantic_epochs"],
        semantic_batch_size: int = _DEFAULTS["semantic_batch_size"],
        semantic_max_len: int = _DEFAULTS["semantic_max_len"],
        semantic_embedding_dim: int = _DEFAULTS["semantic_embedding_dim"],
        semantic_hidden_dim: int = _DEFAULTS["semantic_hidden_dim"],
        semantic_num_filters: int = _DEFAULTS["semantic_num_filters"],
        embedding_dim: int = _DEFAULTS["embedding_dim"],
        hidden_dim: int = _DEFAULTS["hidden_dim"],
        num_layers: int = _DEFAULTS["num_layers"],
        dropout: float = _DEFAULTS["dropout"],
        topic_dim: int = _DEFAULTS["topic_dim"],
        max_len: int = _DEFAULTS["max_len"],
        seq2seq_hidden_dim: int = _DEFAULTS["seq2seq_hidden_dim"],
        seq2seq_num_layers: int = _DEFAULTS["seq2seq_num_layers"],
        epochs: int = _DEFAULTS["epochs"],
        seq2seq_epochs: int = _DEFAULTS["seq2seq_epochs"],
        batch_size: int = _DEFAULTS["batch_size"],
        learning_rate: float = _DEFAULTS["learning_rate"],
        seq2seq_learning_rate: float = _DEFAULTS["seq2seq_learning_rate"],
        max_src_len: int = _DEFAULTS["max_src_len"],
        max_tgt_len: int = _DEFAULTS["max_tgt_len"],
        mode: str = _DEFAULT_MODE,
        seed: int = _DEFAULTS["seed"],
    ) -> dict:
        """Обучить семантическое ядро, LSTM-модель и seq2seq и сохранить модель.

        :param texts: итерируемое текстов документов.
        :param labels: темы документов; если ``None`` — из префиксов ``sources``.
        :param sources: метки документов (по умолчанию ``doc_0..doc_{n-1}``).
        :param semantic_method: ``"cnn"``/``"rnn"`` (по умолчанию текущий).
        :param semantic_epochs: эпохи обучения ядра `lab_4`.
        :param epochs: эпохи обучения LSTM-языковой модели.
        :param seq2seq_epochs: эпохи обучения каждой seq2seq-модели.
        :param max_len: максимальная длина последовательности LM.
        :param max_src_len: максимальная длина источника seq2seq.
        :param max_tgt_len: максимальная длина цели seq2seq.
        :param mode: режим токенизации генератора (по умолчанию ``"raw"``).
        :param seed: сид воспроизводимости.
        :return: словарь метрик и размеров модели.
        :raises ValueError: пустой корпус, менее двух тем или пустой словарь.
        """
        started = time.perf_counter()
        documents = _as_documents(texts)
        if not documents:
            raise ValueError(
                "texts пуст или None: для обучения нужен непустой итерируемый набор документов"
            )
        n_docs = len(documents)
        source_labels = self._resolve_sources(sources, n_docs)
        class_labels = self._resolve_labels(labels, source_labels, n_docs)
        topic_order = sorted(set(class_labels))
        if len(topic_order) < 2:
            raise ValueError(
                "нужны минимум 2 темы, получено {}: {!r}; передайте labels или "
                "назовите файлы как <тема>_<номер>.txt".format(len(topic_order), topic_order)
            )
        topic_to_idx = {topic: index + 1 for index, topic in enumerate(topic_order)}

        method = str(semantic_method or self.semantic_method)
        if method not in _METHODS:
            raise ValueError(
                "semantic_method должен быть одним из {}, получено {!r}".format(_METHODS, method)
            )
        mode_key = str(mode or _DEFAULT_MODE).strip().lower()
        if mode_key not in ("raw", "canonical", "lemma", "stem"):
            raise ValueError(
                "mode должен быть одним из ('raw', 'canonical', 'lemma', 'stem'), "
                "получено {!r}".format(mode)
            )

        seed_value = _as_int(seed, _DEFAULTS["seed"])
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)

        # 1. Семантическое ядро lab_4. Оно же сохраняет себя в semantic/.
        semantic_dir = Path(self.model_dir) / SEMANTIC_DIRNAME
        semantic = NeuralAnalyzer(str(semantic_dir), method=method)
        semantic_stats = semantic.fit(
            documents,
            labels=class_labels,
            sources=source_labels,
            embedding_dim=_as_int(semantic_embedding_dim, _DEFAULTS["semantic_embedding_dim"]),
            hidden_dim=_as_int(semantic_hidden_dim, _DEFAULTS["semantic_hidden_dim"]),
            num_filters=_as_int(semantic_num_filters, _DEFAULTS["semantic_num_filters"]),
            epochs=_as_int(semantic_epochs, _DEFAULTS["semantic_epochs"]),
            batch_size=_as_int(semantic_batch_size, _DEFAULTS["semantic_batch_size"]),
            max_len=_as_int(semantic_max_len, _DEFAULTS["semantic_max_len"]),
            mode="canonical",
            seed=seed_value,
        )

        # 2. Словарь и последовательности генератора (словоформы, режим raw).
        raw_docs = [doc_tokens(text, mode_key) for text in documents]
        vocabulary = build_vocabulary(raw_docs)
        if len(vocabulary) <= FIRST_WORD_INDEX:
            raise ValueError(
                "словарь генератора пуст: в документах нет ни одного токена после токенизации"
            )
        max_len_value = max(4, _as_int(max_len, _DEFAULTS["max_len"]))
        # LM учится на отдельных предложениях: короткие зависимости и больше
        # примеров, чем при обучении на целых документах.
        lm_sequences: List[List[int]] = []
        lm_topics: List[int] = []
        for text, label in zip(documents, class_labels):
            sentences = _sentence_split(text) or [text]
            sentence_tokens = [doc_tokens(sentence, mode_key) for sentence in sentences]
            sentence_tokens = [tokens for tokens in sentence_tokens if tokens]
            if not sentence_tokens:
                fallback = doc_tokens(text, mode_key)
                if fallback:
                    sentence_tokens = [fallback]
            if not sentence_tokens:
                continue
            for tokens in sentence_tokens:
                lm_sequences.append(
                    encode_tokens(
                        tokens,
                        vocabulary,
                        add_bos=True,
                        add_eos=True,
                        max_len=max_len_value,
                    )
                )
                lm_topics.append(topic_to_idx[label])
        if not lm_sequences:
            raise ValueError("не удалось построить ни одной последовательности для LSTM")

        # 3. LSTM-языковая модель.
        lm = LSTMLanguageModel(
            len(vocabulary),
            embedding_dim=_as_int(embedding_dim, _DEFAULTS["embedding_dim"]),
            hidden_dim=_as_int(hidden_dim, _DEFAULTS["hidden_dim"]),
            num_layers=_as_int(num_layers, _DEFAULTS["num_layers"]),
            dropout=_as_float(dropout, _DEFAULTS["dropout"]),
            n_topics=len(topic_order),
            topic_dim=_as_int(topic_dim, _DEFAULTS["topic_dim"]),
        )
        lm_stats = self._train_lm(
            lm,
            lm_sequences,
            lm_topics,
            epochs=max(1, _as_int(epochs, _DEFAULTS["epochs"])),
            batch_size=max(1, _as_int(batch_size, _DEFAULTS["batch_size"])),
            learning_rate=_as_float(learning_rate, _DEFAULTS["learning_rate"]),
            seed=seed_value,
        )

        # 4. Пары seq2seq: документ -> первый абзац и предложение -> следующее.
        max_src = max(2, _as_int(max_src_len, _DEFAULTS["max_src_len"]))
        max_tgt = max(2, _as_int(max_tgt_len, _DEFAULTS["max_tgt_len"]))
        lead_pairs, next_pairs = self._build_seq2seq_pairs(
            documents, class_labels, topic_to_idx, vocabulary, max_src, max_tgt
        )

        lead, lead_stats = self._train_seq2seq_task(
            lead_pairs,
            vocabulary,
            embedding_dim=_as_int(embedding_dim, _DEFAULTS["embedding_dim"]),
            hidden_dim=_as_int(seq2seq_hidden_dim, _DEFAULTS["seq2seq_hidden_dim"]),
            num_layers=_as_int(seq2seq_num_layers, _DEFAULTS["seq2seq_num_layers"]),
            dropout=_as_float(dropout, _DEFAULTS["dropout"]),
            epochs=max(1, _as_int(seq2seq_epochs, _DEFAULTS["seq2seq_epochs"])),
            batch_size=max(1, _as_int(batch_size, _DEFAULTS["batch_size"])),
            learning_rate=_as_float(seq2seq_learning_rate, _DEFAULTS["seq2seq_learning_rate"]),
            max_src_len=max_src,
            max_tgt_len=max_tgt,
            n_topics=len(topic_order),
            topic_dim=_as_int(topic_dim, _DEFAULTS["topic_dim"]),
            seed=seed_value + 1,
        )
        next_model, next_stats = self._train_seq2seq_task(
            next_pairs,
            vocabulary,
            embedding_dim=_as_int(embedding_dim, _DEFAULTS["embedding_dim"]),
            hidden_dim=_as_int(seq2seq_hidden_dim, _DEFAULTS["seq2seq_hidden_dim"]),
            num_layers=_as_int(seq2seq_num_layers, _DEFAULTS["seq2seq_num_layers"]),
            dropout=_as_float(dropout, _DEFAULTS["dropout"]),
            epochs=max(1, _as_int(seq2seq_epochs, _DEFAULTS["seq2seq_epochs"])),
            batch_size=max(1, _as_int(batch_size, _DEFAULTS["batch_size"])),
            learning_rate=_as_float(seq2seq_learning_rate, _DEFAULTS["seq2seq_learning_rate"]),
            max_src_len=max_src,
            max_tgt_len=max_tgt,
            n_topics=len(topic_order),
            topic_dim=_as_int(topic_dim, _DEFAULTS["topic_dim"]),
            seed=seed_value + 2,
        )

        # 5. Состояние и метрики.
        self._semantic = semantic
        self._lm = lm
        self._lead = lead
        self._next = next_model
        self._vocabulary = vocabulary
        self._topic_order = topic_order
        self._topic_to_idx = topic_to_idx
        self._texts = list(documents)
        self._sources = list(source_labels)
        self._labels = list(class_labels)
        self._mode = mode_key
        self._max_len = max_len_value
        self._max_src_len = max_src
        self._max_tgt_len = max_tgt
        self.semantic_method = method
        self._is_fitted = True
        self.load_error = None
        self._compute_topic_centroids()
        self._bigrams, self._bigram_next, self._trigram_next = self._collect_ngrams(self._texts)

        lm_loss = float(lm_stats["loss"])
        self._params = {
            "semantic_method": method,
            "semantic_epochs": _as_int(semantic_epochs, _DEFAULTS["semantic_epochs"]),
            "semantic_batch_size": _as_int(semantic_batch_size, _DEFAULTS["semantic_batch_size"]),
            "semantic_max_len": _as_int(semantic_max_len, _DEFAULTS["semantic_max_len"]),
            "semantic_embedding_dim": _as_int(semantic_embedding_dim, _DEFAULTS["semantic_embedding_dim"]),
            "semantic_hidden_dim": _as_int(semantic_hidden_dim, _DEFAULTS["semantic_hidden_dim"]),
            "semantic_num_filters": _as_int(semantic_num_filters, _DEFAULTS["semantic_num_filters"]),
            "embedding_dim": _as_int(embedding_dim, _DEFAULTS["embedding_dim"]),
            "hidden_dim": _as_int(hidden_dim, _DEFAULTS["hidden_dim"]),
            "num_layers": _as_int(num_layers, _DEFAULTS["num_layers"]),
            "dropout": _as_float(dropout, _DEFAULTS["dropout"]),
            "topic_dim": _as_int(topic_dim, _DEFAULTS["topic_dim"]),
            "max_len": max_len_value,
            "seq2seq_hidden_dim": _as_int(seq2seq_hidden_dim, _DEFAULTS["seq2seq_hidden_dim"]),
            "seq2seq_num_layers": _as_int(seq2seq_num_layers, _DEFAULTS["seq2seq_num_layers"]),
            "epochs": _as_int(epochs, _DEFAULTS["epochs"]),
            "seq2seq_epochs": _as_int(seq2seq_epochs, _DEFAULTS["seq2seq_epochs"]),
            "batch_size": _as_int(batch_size, _DEFAULTS["batch_size"]),
            "learning_rate": _as_float(learning_rate, _DEFAULTS["learning_rate"]),
            "seq2seq_learning_rate": _as_float(seq2seq_learning_rate, _DEFAULTS["seq2seq_learning_rate"]),
            "max_src_len": max_src,
            "max_tgt_len": max_tgt,
            "mode": mode_key,
            "seed": seed_value,
        }
        self._metrics = {
            "semantic_accuracy": float(semantic_stats.get("accuracy", 0.0)),
            "lm_loss": lm_loss,
            "lm_perplexity": float(math.exp(min(lm_loss, 50.0))) if math.isfinite(lm_loss) else None,
            "lm_token_accuracy": float(lm_stats.get("accuracy", 0.0)),
            "lead_loss": lead_stats.get("loss"),
            "lead_token_accuracy": lead_stats.get("accuracy"),
            "next_loss": next_stats.get("loss"),
            "next_token_accuracy": next_stats.get("accuracy"),
            "lead_pairs": len(lead_pairs),
            "next_pairs": len(next_pairs),
            "lm_history": list(lm_stats.get("history", [])),
            "lead_history": list(lead_stats.get("history", [])),
            "next_history": list(next_stats.get("history", [])),
            "seconds": float(time.perf_counter() - started),
        }
        self.save()
        return self.fit_stats()

    def fit_stats(self) -> dict:
        """Краткие метрики обучения (используются в README и ноутбуке)."""
        self._require_fitted()
        return {
            "n_docs": self.n_docs,
            "n_features": self.n_features,
            "n_classes": self.n_topics,
            "dim": self._semantic.dim,
            "semantic_method": self.semantic_method,
            "semantic_accuracy": self._metrics.get("semantic_accuracy"),
            "vocab_size": len(self._vocabulary),
            "topics": list(self._topic_order),
            "lm_loss": self._metrics.get("lm_loss"),
            "lm_perplexity": self._metrics.get("lm_perplexity"),
            "lm_token_accuracy": self._metrics.get("lm_token_accuracy"),
            "lead_pairs": self._metrics.get("lead_pairs"),
            "next_pairs": self._metrics.get("next_pairs"),
            "lead_loss": self._metrics.get("lead_loss"),
            "next_loss": self._metrics.get("next_loss"),
            "epochs": self._params.get("epochs"),
            "seq2seq_epochs": self._params.get("seq2seq_epochs"),
            "seconds": self._metrics.get("seconds"),
        }

    # ------------------------------------------------------------------ #
    # Генерация
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str = "",
        *,
        topic: Optional[str] = None,
        method: str = "hybrid",
        max_words: int = _DEFAULT_MAX_WORDS,
        temperature: float = 0.8,
        top_k: int = 20,
        num_candidates: int = 3,
        no_repeat_ngram_size: int = 2,
        constrain: bool = True,
        min_words: int = 4,
        seed: int = 42,
    ) -> "Generation":
        """Сгенерировать осмысленный текст по запросу и/или теме.

        :param prompt: запрос своими словами; пустой — генерация по теме.
        :param topic: тема; ``None`` — тема предсказывается анализатором.
        :param method: ``template``, ``lstm``, ``seq2seq`` или ``hybrid``.
        :param max_words: максимум слов в сгенерированном фрагменте.
        :param temperature: температура сэмплирования (``0`` — greedy).
        :param top_k: top-k фильтрация (``0`` — без фильтра).
        :param num_candidates: число кандидатов LSTM/seq2seq.
        :param no_repeat_ngram_size: запрет повторяющихся n-грамм (``0`` — выкл.).
        :param seed: сид сэмплирования.
        :return: :class:`Generation` с лучшим текстом и всеми кандидатами.
        :raises ValueError: неизвестный режим или тема.
        """
        self._require_fitted()
        if method not in GEN_METHODS:
            raise ValueError(
                "method должен быть одним из {}, получено {!r}".format(GEN_METHODS, method)
            )
        prompt_text = _normalize_ws(prompt)
        resolved_topic = self._resolve_topic(topic, prompt_text)
        limit = max(1, min(_as_int(max_words, _DEFAULT_MAX_WORDS), 80))
        candidates_count = max(1, min(_as_int(num_candidates, 3), 6))
        ngram = max(0, _as_int(no_repeat_ngram_size, 2))
        seed_value = _as_int(seed, 42)

        candidates: List[Candidate] = []
        if method in ("template", "hybrid"):
            for variant in range(2):
                candidates.append(
                    self._template_candidate(prompt_text, resolved_topic, variant=variant)
                )
        if method in ("lstm", "hybrid"):
            candidates.extend(
                self._lm_candidates(
                    prompt_text,
                    resolved_topic,
                    count=candidates_count,
                    max_words=limit,
                    temperature=temperature,
                    top_k=top_k,
                    ngram=ngram,
                    seed=seed_value,
                    constrain=constrain,
                    min_words=min_words,
                )
            )
        if method in ("seq2seq", "hybrid"):
            candidates.extend(
                self._seq2seq_candidates(
                    prompt_text,
                    resolved_topic,
                    count=candidates_count,
                    max_words=limit,
                    temperature=temperature,
                    top_k=top_k,
                    ngram=ngram,
                    seed=seed_value + 100,
                    constrain=constrain,
                    min_words=min_words,
                )
            )

        if method == "seq2seq" and self._next is None and self._lead is None:
            raise RuntimeError("seq2seq-модель не обучена: в корпусе нет пар предложений")

        ranked = self._rank(candidates, prompt_text, resolved_topic)
        if not ranked:
            # Последний рубеж: шаблон всегда даёт непустой текст.
            fallback = self._template_candidate(prompt_text, resolved_topic, variant=0)
            ranked = self._rank([fallback], prompt_text, resolved_topic)
        if not ranked:
            return Generation(text="", method=method, topic=resolved_topic, score=0.0, prompt=prompt_text)
        best = ranked[0]
        return Generation(
            text=best.text,
            method=method,
            topic=resolved_topic,
            score=best.score,
            prompt=prompt_text,
            candidates=ranked,
        )

    def summarize(self, text: str, *, max_words: int = 32, temperature: float = 0.0,
                  min_words: int = 4, seed: int = 42) -> str:
        """Seq2seq-реферирование: документ → первый абзац.

        :return: сгенерированный текст; для пустого входа — ``""``.
        :raises RuntimeError: seq2seq-модель реферирования не обучена.
        """
        self._require_fitted()
        if self._lead is None:
            raise RuntimeError("seq2seq-модель реферирования не обучена")
        source = doc_tokens(text, self._mode)
        if not source:
            return ""
        summary_topic = self._predict_topic(text)
        if summary_topic not in self._topic_to_idx and self._labels:
            summary_topic = Counter(self._labels).most_common(1)[0][0]
        tokens = self._seq2seq_generate_tokens(
            source[: self._max_src_len],
            model=self._lead,
            max_words=max(1, min(_as_int(max_words, 32), 80)),
            temperature=temperature,
            top_k=0,
            ngram=2,
            seed=_as_int(seed, 42),
            constrain=True,
            min_words=max(0, _as_int(min_words, 4)),
            topic=summary_topic,
        )
        return format_tokens_as_text(tokens)

    def continue_text(self, prompt: str, *, max_words: int = _DEFAULT_MAX_WORDS,
                      temperature: float = 0.8, top_k: int = 20,
                      no_repeat_ngram_size: int = 2, min_words: int = 4,
                      seed: int = 42) -> str:
        """Продолжить текст LSTM-моделью и вернуть запрос вместе с продолжением."""
        self._require_fitted()
        prompt_text = _normalize_ws(prompt)
        resolved_topic = self._resolve_topic(None, prompt_text)
        tokens = self._lm_generate_tokens(
            prompt_text,
            resolved_topic,
            max_words=max(1, min(_as_int(max_words, _DEFAULT_MAX_WORDS), 80)),
            temperature=temperature,
            top_k=top_k,
            ngram=max(0, _as_int(no_repeat_ngram_size, 2)),
            seed=_as_int(seed, 42),
            constrain=True,
            min_words=max(0, _as_int(min_words, 4)),
        )
        prompt_tokens = doc_tokens(prompt_text, self._mode)
        full = format_tokens_as_text(prompt_tokens + tokens)
        return full or format_tokens_as_text(tokens)

    def template(self, prompt: str = "", *, topic: Optional[str] = None, variant: int = 0) -> str:
        """Шаблонный режим: текст только из результатов анализатора."""
        self._require_fitted()
        prompt_text = _normalize_ws(prompt)
        resolved_topic = self._resolve_topic(topic, prompt_text)
        return self._template_candidate(prompt_text, resolved_topic, variant=variant).text

    def evaluate(self, text: str, *, prompt: str = "", topic: Optional[str] = None) -> dict:
        """Оценить текст по компонентам гибридного ранжирования.

        :return: ``{'score', 'semantic', 'topic_confidence', 'fluency',
            'predicted_topic', 'topic'}``.
        """
        self._require_fitted()
        prompt_text = _normalize_ws(prompt)
        resolved_topic = self._resolve_topic(topic, prompt_text)
        candidate = Candidate(text=_normalize_ws(text), method="evaluate", topic=resolved_topic)
        self._score_candidate(candidate, prompt_text, resolved_topic)
        predicted = self._predict_topic(candidate.text)
        return {
            "score": candidate.score,
            "semantic": candidate.semantic,
            "topic_confidence": candidate.topic_confidence,
            "fluency": candidate.fluency,
            "coherence": candidate.coherence,
            "uniqueness": candidate.uniqueness,
            "completeness": candidate.completeness,
            "predicted_topic": predicted,
            "topic": resolved_topic,
        }

    # ------------------------------------------------------------------ #
    # Персистенция
    # ------------------------------------------------------------------ #

    def save(self, model_dir=None) -> str:
        """Сохранить модель и вернуть путь к каталогу."""
        self._require_fitted()
        target = Path(model_dir) if model_dir is not None else Path(self.model_dir)
        target.mkdir(parents=True, exist_ok=True)

        # Семантическое ядро сохраняется в подкаталог semantic/.
        self._semantic.save(str(target / SEMANTIC_DIRNAME))

        vocabulary_list = [None] * len(self._vocabulary)
        for token, index in self._vocabulary.items():
            vocabulary_list[int(index)] = token
        meta = {
            "version": MODEL_VERSION,
            "semantic_method": self.semantic_method,
            "n_docs": len(self._texts),
            "n_topics": len(self._topic_order),
            "topics": list(self._topic_order),
            "vocabulary": vocabulary_list,
            "params": dict(self._params),
            "metrics": dict(self._metrics),
        }
        _write_json(target / META_FILENAME, meta)
        torch.save(
            {
                "lm": self._lm.state_dict(),
                "lead": self._lead.state_dict() if self._lead is not None else None,
                "next": self._next.state_dict() if self._next is not None else None,
            },
            str(target / WEIGHTS_FILENAME),
        )
        _write_json(
            target / CORPUS_FILENAME,
            {
                "texts": list(self._texts),
                "sources": list(self._sources),
                "labels": list(self._labels),
            },
        )
        self.model_dir = str(target)
        return str(target)

    @classmethod
    def load(cls, model_dir) -> "GeneratingAnalyzer":
        """Загрузить генератор из каталога (INTERFACES.md §7).

        :raises ValueError: файлы отсутствуют/несовместимы, нарушены длины или
            семантическое ядро не обучено.
        """
        base = Path(model_dir)
        meta_path = base / META_FILENAME
        if not meta_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(META_FILENAME, base))
        meta = _read_json(meta_path)
        if _as_int(meta.get("version"), -1) != MODEL_VERSION:
            raise _incompatible(
                "version={!r} в {}, ожидается {}".format(meta.get("version"), meta_path, MODEL_VERSION)
            )

        method = meta.get("semantic_method")
        if method not in _METHODS:
            raise _incompatible(
                "semantic_method={!r} в {}, ожидается один из {}".format(method, meta_path, _METHODS)
            )

        vocabulary_list = meta.get("vocabulary")
        if not isinstance(vocabulary_list, list) or len(vocabulary_list) <= FIRST_WORD_INDEX:
            raise _incompatible("в {} поле vocabulary должно быть непустым списком".format(meta_path))
        if tuple(vocabulary_list[:FIRST_WORD_INDEX]) != SPECIAL_TOKENS:
            raise _incompatible("в {} нарушен порядок специальных токенов".format(meta_path))
        vocabulary = {str(token): index for index, token in enumerate(vocabulary_list)}

        topics = meta.get("topics")
        if not isinstance(topics, list) or len(topics) < 2:
            raise _incompatible("в {} поле topics должно содержать >= 2 тем".format(meta_path))
        topics = [str(topic) for topic in topics]
        if _as_int(meta.get("n_topics"), -1) != len(topics):
            raise _incompatible("n_topics не совпадает с len(topics) в {}".format(meta_path))
        topic_to_idx = {topic: index + 1 for index, topic in enumerate(topics)}

        params = meta.get("params")
        if not isinstance(params, dict):
            params = {}
        metrics = meta.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        corpus = _read_json(base / CORPUS_FILENAME)
        texts = corpus.get("texts")
        sources = corpus.get("sources")
        labels = corpus.get("labels")
        if not isinstance(texts, list) or not isinstance(sources, list) or not isinstance(labels, list):
            raise _incompatible("в {} поля texts/sources/labels должны быть списками".format(CORPUS_FILENAME))
        texts = [str(text) for text in texts]
        sources = [str(source) for source in sources]
        labels = [str(label) for label in labels]
        n_docs = _as_int(meta.get("n_docs"), -1)
        if n_docs < 1 or not (len(texts) == len(sources) == len(labels) == n_docs):
            raise _incompatible(
                "n_docs={}, len(texts)={}, len(sources)={}, len(labels)={}".format(
                    n_docs, len(texts), len(sources), len(labels)
                )
            )

        semantic_dir = base / SEMANTIC_DIRNAME
        semantic = NeuralAnalyzer(str(semantic_dir), method=method)
        if not semantic.is_fitted:
            raise _incompatible(
                "семантическое ядро в {} не загружено: {}".format(
                    semantic_dir, semantic.load_error or "нет модели"
                )
            )

        weights_path = base / WEIGHTS_FILENAME
        if not weights_path.is_file():
            raise _incompatible("нет файла {} в каталоге {}".format(WEIGHTS_FILENAME, base))
        try:
            state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise _incompatible("не удалось прочитать {}: {}".format(WEIGHTS_FILENAME, exc)) from exc
        if not isinstance(state, dict) or "lm" not in state:
            raise _incompatible("в {} нет состояния lm".format(WEIGHTS_FILENAME))

        embedding_dim = _as_int(params.get("embedding_dim"), _DEFAULTS["embedding_dim"])
        hidden_dim = _as_int(params.get("hidden_dim"), _DEFAULTS["hidden_dim"])
        num_layers = _as_int(params.get("num_layers"), _DEFAULTS["num_layers"])
        dropout = _as_float(params.get("dropout"), _DEFAULTS["dropout"])
        topic_dim = _as_int(params.get("topic_dim"), _DEFAULTS["topic_dim"])
        seq2seq_hidden = _as_int(params.get("seq2seq_hidden_dim"), _DEFAULTS["seq2seq_hidden_dim"])
        seq2seq_layers = _as_int(params.get("seq2seq_num_layers"), _DEFAULTS["seq2seq_num_layers"])

        try:
            lm = LSTMLanguageModel(
                len(vocabulary),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                n_topics=len(topics),
                topic_dim=topic_dim,
            )
            lm.load_state_dict(state["lm"])
            lm.eval()
        except Exception as exc:  # noqa: BLE001
            raise _incompatible("не удалось восстановить LSTM-модель: {}".format(exc)) from exc

        def _load_seq2seq(key):
            saved = state.get(key)
            if saved is None:
                return None
            model = Seq2SeqLSTM(
                len(vocabulary),
                embedding_dim=embedding_dim,
                hidden_dim=seq2seq_hidden,
                num_layers=seq2seq_layers,
                dropout=dropout,
                n_topics=len(topics),
                topic_dim=topic_dim,
            )
            model.load_state_dict(saved)
            model.eval()
            return model

        try:
            lead = _load_seq2seq("lead")
            next_model = _load_seq2seq("next")
        except Exception as exc:  # noqa: BLE001
            raise _incompatible("не удалось восстановить seq2seq-модель: {}".format(exc)) from exc

        instance = cls.__new__(cls)
        instance.model_dir = str(base)
        instance.semantic_method = method
        instance.load_error = None
        instance._semantic = semantic
        instance._lm = lm
        instance._lead = lead
        instance._next = next_model
        instance._vocabulary = vocabulary
        instance._topic_order = topics
        instance._topic_to_idx = topic_to_idx
        instance._texts = texts
        instance._sources = sources
        instance._labels = labels
        instance._topic_centroids = {}
        instance._templates = TemplateGenerator()
        instance._mode = str(params.get("mode") or _DEFAULT_MODE)
        instance._max_len = max(4, _as_int(params.get("max_len"), _DEFAULTS["max_len"]))
        instance._max_src_len = max(2, _as_int(params.get("max_src_len"), _DEFAULTS["max_src_len"]))
        instance._max_tgt_len = max(2, _as_int(params.get("max_tgt_len"), _DEFAULTS["max_tgt_len"]))
        instance._params = dict(params)
        instance._metrics = dict(metrics)
        instance._is_fitted = True
        instance._compute_topic_centroids()
        instance._bigrams, instance._bigram_next, instance._trigram_next = instance._collect_ngrams(instance._texts)
        return instance

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "GeneratingAnalyzer(model_dir={!r}, semantic_method={!r}, n_docs={}, topics={})".format(
            self.model_dir, self.semantic_method, self.n_docs, self.n_topics
        )

    # ------------------------------------------------------------------ #
    # Внутреннее: данные
    # ------------------------------------------------------------------ #

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
        """Темы документов: явные ``labels`` либо префиксы ``sources``."""
        if labels is None:
            return [_topic_of(source) for source in sources]
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

    def _build_seq2seq_pairs(self, documents: Sequence[str], labels: Sequence[str],
                             topic_to_idx: Dict[str, int], vocabulary: Dict[str, int],
                             max_src: int, max_tgt: int):
        """Построить пары ``(источник, цель, номер темы)`` для seq2seq."""
        lead_pairs: List[Tuple[List[int], List[int], int]] = []
        next_pairs: List[Tuple[List[int], List[int], int]] = []
        for text, label in zip(documents, labels):
            topic_id = int(topic_to_idx.get(label, 0))
            document_tokens = doc_tokens(text, self._mode)
            sentences = [
                doc_tokens(sentence, self._mode)
                for sentence in _sentence_split(text)
            ]
            sentences = [sentence for sentence in sentences if sentence]
            if not sentences:
                sentences = [document_tokens] if document_tokens else []
            if not sentences:
                continue
            source_ids = encode_tokens(
                document_tokens[:max_src], vocabulary, keep_unknown=True
            )
            target_ids = encode_tokens(
                sentences[0][:max_tgt], vocabulary, keep_unknown=False
            )
            if source_ids and target_ids:
                lead_pairs.append((source_ids, target_ids, topic_id))
            for index in range(len(sentences) - 1):
                src = encode_tokens(
                    sentences[index][:max_src], vocabulary, keep_unknown=True
                )
                tgt = encode_tokens(
                    sentences[index + 1][:max_tgt], vocabulary, keep_unknown=False
                )
                if src and tgt:
                    next_pairs.append((src, tgt, topic_id))
        return lead_pairs, next_pairs

    def _compute_topic_centroids(self) -> None:
        """Пересчитать центроиды тем по векторам документов анализатора."""
        centroids: Dict[str, np.ndarray] = {}
        if self._semantic is None or not self._semantic.is_fitted:
            self._topic_centroids = centroids
            return
        vectors = np.asarray(
            [self._semantic.doc_vector(index) for index in range(len(self._texts))],
            dtype=np.float32,
        )
        for topic in self._topic_order:
            indices = [i for i, label in enumerate(self._labels) if label == topic]
            if indices:
                centroids[topic] = vectors[indices].mean(axis=0).astype(np.float32)
        self._topic_centroids = centroids

    def _collect_ngrams(self, documents: Sequence[str]):
        """Собрать биграммы и триграммы корпуса.

        Возвращает ``(bigrams, bigram_next, trigram_next)``:

        * ``bigrams`` — множество пар (для оценки связности кандидата);
        * ``bigram_next`` — ``{слово: {следующие слова}}``;
        * ``trigram_next`` — ``{(слово1, слово2): {следующие слова}}``.

        Карты используются при декодировании LSTM/seq2seq: следующий токен
        выбирается только из переходов, встречавшихся в корпусе анализатора.
        """
        bigrams = set()
        bigram_next: Dict[str, set] = {}
        trigram_next: Dict[Tuple[str, str], set] = {}
        for text in documents:
            sentences = _sentence_split(text) or [text]
            for sentence in sentences:
                tokens = doc_tokens(sentence, self._mode)
                for left, right in zip(tokens, tokens[1:]):
                    bigrams.add((left, right))
                    bigram_next.setdefault(left, set()).add(right)
                for first, second, third in zip(tokens, tokens[1:], tokens[2:]):
                    trigram_next.setdefault((first, second), set()).add(third)
        return bigrams, bigram_next, trigram_next

    # ------------------------------------------------------------------ #
    # Внутреннее: обучение
    # ------------------------------------------------------------------ #

    @staticmethod
    def _train_lm(model: LSTMLanguageModel, sequences: Sequence[Sequence[int]],
                  topics: Sequence[int], *, epochs: int, batch_size: int,
                  learning_rate: float, seed: int) -> dict:
        """Обучить LSTM-языковую модель; вернуть ``{'loss', 'accuracy'}``."""
        inputs = [list(sequence[:-1]) for sequence in sequences]
        targets = [list(sequence[1:]) for sequence in sequences]
        input_tensor, _ = pad_token_sequences(inputs)
        target_tensor, _ = pad_token_sequences(targets, max_len=input_tensor.size(1))
        topic_tensor = torch.tensor(topics, dtype=torch.long)
        n_docs = input_tensor.size(0)
        if n_docs == 0 or input_tensor.size(1) == 0:
            return {"loss": float("nan"), "accuracy": 0.0}

        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        rng = np.random.default_rng(seed)
        model.train()
        history = []
        for _ in range(max(1, epochs)):
            epoch_loss = 0.0
            epoch_tokens = 0
            order = rng.permutation(n_docs)
            for start in range(0, n_docs, max(1, batch_size)):
                indices = order[start:start + max(1, batch_size)]
                xb = input_tensor[indices]
                tb = target_tensor[indices]
                topic_b = topic_tensor[indices]
                optimizer.zero_grad()
                logits, _ = model(xb, topic_b)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tb.reshape(-1),
                    ignore_index=PAD,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                tokens = int(tb.ne(PAD).sum().item())
                epoch_loss += float(loss.item()) * tokens
                epoch_tokens += tokens
            history.append(epoch_loss / epoch_tokens if epoch_tokens else float("nan"))

        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        with torch.no_grad():
            for start in range(0, n_docs, max(1, batch_size)):
                xb = input_tensor[start:start + max(1, batch_size)]
                tb = target_tensor[start:start + max(1, batch_size)]
                topic_b = topic_tensor[start:start + max(1, batch_size)]
                logits, _ = model(xb, topic_b)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tb.reshape(-1),
                    ignore_index=PAD,
                    reduction="sum",
                )
                mask = tb.ne(PAD)
                total_loss += float(loss.item())
                total_tokens += int(mask.sum().item())
                predictions = logits.argmax(dim=-1)
                correct += int((predictions.eq(tb) & mask).sum().item())
        average_loss = total_loss / total_tokens if total_tokens else float("nan")
        accuracy = correct / total_tokens if total_tokens else 0.0
        return {"loss": average_loss, "accuracy": accuracy, "history": history}

    @staticmethod
    def _train_seq2seq_task(pairs, vocabulary: Dict[str, int], *, embedding_dim: int,
                            hidden_dim: int, num_layers: int, dropout: float,
                            epochs: int, batch_size: int, learning_rate: float,
                            max_src_len: int, max_tgt_len: int, n_topics: int,
                            topic_dim: int, seed: int):
        """Обучить одну seq2seq-модель; вернуть ``(model, {'loss', 'accuracy'})``.

        Пары имеют вид ``(источник, цель, номер темы)``; тема подаётся и в
        энкодер, и в декодер, чтобы генерация следовала теме анализатора.
        """
        if not pairs:
            return None, {"loss": None, "accuracy": None}
        torch.manual_seed(seed)
        model = Seq2SeqLSTM(
            len(vocabulary),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            n_topics=n_topics,
            topic_dim=topic_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        rng = np.random.default_rng(seed)
        model.train()
        history = []
        for _ in range(max(1, epochs)):
            epoch_loss = 0.0
            epoch_tokens = 0
            order = rng.permutation(len(pairs))
            for start in range(0, len(pairs), max(1, batch_size)):
                batch = [pairs[int(index)] for index in order[start:start + max(1, batch_size)]]
                src_batch = [src for src, _, _ in batch]
                tgt_batch = [tgt for _, tgt, _ in batch]
                topic_batch = torch.tensor(
                    [topic for _, _, topic in batch], dtype=torch.long
                )
                src_tensor, src_lengths = pad_token_sequences(src_batch, max_len=max_src_len)
                tgt_in = [[BOS] + tgt for tgt in tgt_batch]
                tgt_out = [tgt + [EOS] for tgt in tgt_batch]
                tgt_in_tensor, _ = pad_token_sequences(tgt_in, max_len=max_tgt_len + 1)
                tgt_out_tensor, _ = pad_token_sequences(tgt_out, max_len=max_tgt_len + 1)
                optimizer.zero_grad()
                logits = model(src_tensor, src_lengths, tgt_in_tensor, topic_batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out_tensor.reshape(-1),
                    ignore_index=PAD,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                tokens = int(tgt_out_tensor.ne(PAD).sum().item())
                epoch_loss += float(loss.item()) * tokens
                epoch_tokens += tokens
            history.append(epoch_loss / epoch_tokens if epoch_tokens else float("nan"))

        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        with torch.no_grad():
            for start in range(0, len(pairs), max(1, batch_size)):
                batch = pairs[start:start + max(1, batch_size)]
                src_batch = [src for src, _, _ in batch]
                tgt_batch = [tgt for _, tgt, _ in batch]
                topic_batch = torch.tensor(
                    [topic for _, _, topic in batch], dtype=torch.long
                )
                src_tensor, src_lengths = pad_token_sequences(src_batch, max_len=max_src_len)
                tgt_in = [[BOS] + tgt for tgt in tgt_batch]
                tgt_out = [tgt + [EOS] for tgt in tgt_batch]
                tgt_in_tensor, _ = pad_token_sequences(tgt_in, max_len=max_tgt_len + 1)
                tgt_out_tensor, _ = pad_token_sequences(tgt_out, max_len=max_tgt_len + 1)
                logits = model(src_tensor, src_lengths, tgt_in_tensor, topic_batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out_tensor.reshape(-1),
                    ignore_index=PAD,
                    reduction="sum",
                )
                mask = tgt_out_tensor.ne(PAD)
                total_loss += float(loss.item())
                total_tokens += int(mask.sum().item())
                predictions = logits.argmax(dim=-1)
                correct += int((predictions.eq(tgt_out_tensor) & mask).sum().item())
        average_loss = total_loss / total_tokens if total_tokens else None
        accuracy = correct / total_tokens if total_tokens else None
        return model, {"loss": average_loss, "accuracy": accuracy, "history": history}

    # ------------------------------------------------------------------ #
    # Внутреннее: шаблонный режим
    # ------------------------------------------------------------------ #

    def _resolve_topic(self, topic: Optional[str], prompt: str) -> str:
        """Определить тему: явная, предсказанная или самая частая."""
        if topic is not None:
            topic_text = str(topic)
            if topic_text not in self._topic_to_idx:
                raise ValueError(
                    "неизвестная тема {!r}; доступны {}".format(topic_text, self._topic_order)
                )
            return topic_text
        predicted = self._predict_topic(prompt)
        if predicted in self._topic_to_idx:
            return predicted
        if self._labels:
            return Counter(self._labels).most_common(1)[0][0]
        return self._topic_order[0] if self._topic_order else ""

    def _predict_topic(self, text: str) -> str:
        """Тема текста по классификатору анализатора; ошибки -> ``""``."""
        if not text or self._semantic is None:
            return ""
        try:
            prediction = self._semantic.predict(text)
        except (ValueError, RuntimeError):
            return ""
        return str(prediction.label)

    def _template_candidate(self, prompt: str, topic: str, *, variant: int = 0) -> Candidate:
        """Кандидат шаблонного генератора по результатам анализатора."""
        evidence, source, _ = self._best_evidence(prompt, topic)
        keywords = self._keywords(evidence, topic)
        text = self._templates.generate(
            prompt=prompt,
            topic=topic,
            keywords=keywords,
            evidence=evidence,
            source=source,
            variant=variant,
        )
        return Candidate(text=text, method="template", topic=topic, source=source)

    def _topic_document_indices(self, topic: str, limit: int = 3) -> List[int]:
        """Индексы документов темы, ближайших к её центроиду."""
        indices = [i for i, label in enumerate(self._labels) if label == topic]
        if not indices:
            return list(range(min(limit, len(self._texts))))
        centroid = self._topic_centroids.get(topic)
        if centroid is None:
            return indices[:limit]
        scored = [
            (i, _cosine(centroid, self._semantic.doc_vector(i)))
            for i in indices
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [index for index, _ in scored[:limit]]

    def _best_evidence(self, prompt: str, topic: str) -> Tuple[str, str, float]:
        """Наиболее близкий к запросу/теме фрагмент корпуса (предложение).

        Фрагмент выбирается только среди документов запрошенной темы: шаблонный
        генератор не должен цитировать чужую тему, даже если поиск анализатора
        на маленьком корпусе ошибся.
        """
        prompt_tokens = doc_tokens(prompt, self._mode) if prompt else []
        prompt_vector = None
        if prompt_tokens:
            prompt_vector = self._semantic.doc_vector(prompt)
        topic_indices = [
            index for index, label in enumerate(self._labels) if label == topic
        ]
        indices: List[int] = []
        if topic_indices:
            if prompt_tokens:
                try:
                    matches = self._semantic.query(prompt, top_n=max(6, len(topic_indices)))
                except RuntimeError:
                    matches = []
                permitted = set(topic_indices)
                indices = [match.index for match in matches if match.index in permitted]
            if not indices:
                centroid = self._topic_centroids.get(topic)
                scored = [
                    (index, _cosine(centroid, self._semantic.doc_vector(index)))
                    for index in topic_indices
                ]
                scored.sort(key=lambda pair: (-pair[1], pair[0]))
                indices = [index for index, _ in scored[:3]]
        else:
            if prompt_tokens:
                try:
                    matches = self._semantic.query(prompt, top_n=3)
                except RuntimeError:
                    matches = []
                indices = [match.index for match in matches]
            if not indices:
                indices = self._topic_document_indices(topic, limit=3)
        if prompt_vector is None:
            prompt_vector = self._topic_centroids.get(topic)
        if not indices:
            return "", "", 0.0

        best_text = ""
        best_source = ""
        best_score = -1.0
        for index in indices:
            if not (0 <= index < len(self._texts)):
                continue
            source = self._sources[index] if index < len(self._sources) else "doc_{}".format(index)
            sentences = _sentence_split(self._texts[index]) or [self._texts[index]]
            for sentence in sentences:
                score = 0.0
                if prompt_vector is not None:
                    score = _cosine(prompt_vector, self._semantic.doc_vector(sentence))
                if score > best_score:
                    best_score = score
                    best_text = _snippet(sentence)
                    best_source = source
        if not best_text:
            index = indices[0]
            best_text = _snippet(self._texts[index])
            best_source = self._sources[index] if index < len(self._sources) else ""
            best_score = 0.0
        return _ensure_sentence(best_text), best_source, max(0.0, best_score)

    #: Частотные местоимения/наречия/служебные слова, не являющиеся понятиями.
    _BAD_KEYWORDS = frozenset(
        "сколько который которая которое которые этот эта это эти такой такая такое "
        "такие свой своя своё свои весь вся всё все наш наша наше наши ваш ваша ваше "
        "ваши его её их они мы вы он она оно что чтобы чем это если когда где куда "
        "откуда почему зачем как так там тут здесь уже ещё еще очень почти тоже также "
        "только даже можно нужно надо более менее каждый любой другой сам самый".split()
    )

    @classmethod
    def _is_keyword(cls, token: str, topic: str) -> bool:
        """Отфильтровать служебные слова и инфинитивы из ключевых понятий."""
        token = str(token).strip()
        if len(token) < 3 or token == topic:
            return False
        if token in cls._BAD_KEYWORDS:
            return False
        if token.endswith(("ться", "ть", "ти", "чь")):
            return False
        return True

    def _keywords(self, evidence: str, topic: str, limit: int = 4) -> List[str]:
        """Ключевые понятия темы: частотные леммы близких документов."""
        counter: Counter = Counter()
        order: List[str] = []
        for token in canonical_tokens(evidence):
            token = token.strip()
            if not self._is_keyword(token, topic):
                continue
            if token not in counter:
                order.append(token)
            counter[token] += 1
        for index in self._topic_document_indices(topic, limit=2):
            for token in canonical_tokens(self._texts[index]):
                token = token.strip()
                if not self._is_keyword(token, topic):
                    continue
                if token not in counter:
                    order.append(token)
                counter[token] += 1
        if not counter:
            return [topic] if topic else []
        position = {token: index for index, token in enumerate(order)}
        ranked = sorted(counter, key=lambda token: (-counter[token], position[token]))
        return ranked[:limit]

    # ------------------------------------------------------------------ #
    # Внутреннее: LSTM/seq2seq генерация
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_next(logits: torch.Tensor, *, temperature: float, top_k: int,
                     banned: Sequence[int], generator: torch.Generator,
                     allowed: Optional[Sequence[int]] = None,
                     allow_eos: bool = True) -> int:
        """Сэмплировать следующий токен; ``temperature <= 0`` — greedy.

        ``allowed`` — индексы слов, разрешённых n-граммными ограничениями корпуса
        (``None`` — без ограничений). Если ограничения отсекают все слова, они
        игнорируются, чтобы генерация не зависла.
        """
        scores = logits.detach().clone()
        scores[PAD] = -1e9
        scores[BOS] = -1e9
        scores[UNK] = -1e9
        if not allow_eos:
            scores[EOS] = -1e9
        if allowed is not None:
            permitted = {int(token) for token in allowed}
            for index in range(FIRST_WORD_INDEX, scores.numel()):
                if index not in permitted:
                    scores[index] = -1e9
            if float(scores.max().item()) <= -1e8:
                scores = logits.detach().clone()
                scores[PAD] = -1e9
                scores[BOS] = -1e9
                scores[UNK] = -1e9
                if not allow_eos:
                    scores[EOS] = -1e9
        for token in banned:
            token = int(token)
            if 0 <= token < scores.numel():
                scores[token] = -1e9
        if float(scores.max().item()) <= -1e8:
            scores = logits.detach().clone()
            scores[PAD] = -1e9
            scores[BOS] = -1e9
            scores[UNK] = -1e9
            if not allow_eos:
                scores[EOS] = -1e9
        if temperature <= 0.0:
            return int(torch.argmax(scores).item())
        scores = scores / max(float(temperature), 1e-6)
        if top_k and top_k > 0:
            k = min(int(top_k), scores.numel())
            values, indices = torch.topk(scores, k)
            filtered = torch.full_like(scores, -1e9)
            filtered[indices] = values
            scores = filtered
        probabilities = torch.softmax(scores, dim=0)
        if not bool(torch.isfinite(probabilities).all()) or float(probabilities.sum()) <= 0.0:
            return int(torch.argmax(scores).item())
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    @staticmethod
    def _banned_ngram_tokens(generated: Sequence[int], ngram: int) -> List[int]:
        """Токены, которые завершили бы уже встречавшуюся n-грамму."""
        if ngram <= 1 or len(generated) < ngram - 1:
            return []
        prefix = list(generated[-(ngram - 1):])
        banned: List[int] = []
        for start in range(len(generated) - ngram + 1):
            window = list(generated[start:start + ngram - 1])
            if window == prefix:
                banned.append(int(generated[start + ngram - 1]))
        return banned

    def _allowed_ids(self, context_tokens: Sequence[str], constrain: bool = True) -> Optional[List[int]]:
        """Разрешённые следующим токены по триграммам/биграммам корпуса.

        Сначала используется триграммный контекст (два последних слова), затем
        биграммный (одно последнее слово). ``<eos>`` разрешён всегда, чтобы модель
        могла завершить предложение.
        """
        if not constrain:
            return None
        tokens = [str(token) for token in context_tokens if str(token)]
        candidates = None
        if len(tokens) >= 2:
            candidates = self._trigram_next.get((tokens[-2], tokens[-1]))
        if not candidates and tokens:
            candidates = self._bigram_next.get(tokens[-1])
        if not candidates:
            return None
        allowed = [
            int(self._vocabulary[token])
            for token in candidates
            if token in self._vocabulary
        ]
        if not allowed:
            return None
        allowed.append(EOS)
        return allowed

    def _lm_generate_tokens(self, prompt: str, topic: str, *, max_words: int,
                            temperature: float, top_k: int, ngram: int,
                            seed: int, constrain: bool = True,
                            min_words: int = 0) -> List[str]:
        """Сгенерировать продолжение LSTM-моделью (без текста запроса)."""
        topic_id = self._topic_to_idx.get(topic, 0)
        prompt_tokens = doc_tokens(prompt, self._mode) if prompt else []
        prompt_ids = encode_tokens(
            prompt_tokens,
            self._vocabulary,
            max_len=max(1, self._max_len - 2),
            keep_unknown=True,
        )
        input_ids = [BOS] + prompt_ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
        generator = torch.Generator().manual_seed(int(seed))

        with torch.no_grad():
            logits, hidden = self._lm(input_tensor, topic_tensor)
        next_logits = logits[0, -1]
        generated: List[int] = []
        generated_words: List[str] = []
        reverse = {index: token for token, index in self._vocabulary.items()}
        for _ in range(max(1, max_words)):
            banned = self._banned_ngram_tokens(generated, ngram)
            allowed = self._allowed_ids(prompt_tokens + generated_words, constrain)
            token = self._sample_next(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                banned=banned,
                generator=generator,
                allowed=allowed,
                allow_eos=self._can_end(generated_words, min_words),
            )
            if token == EOS:
                break
            if token < FIRST_WORD_INDEX:
                continue
            generated.append(token)
            generated_words.append(reverse.get(token, ""))
            with torch.no_grad():
                next_logits, hidden = self._lm.step(
                    torch.tensor([token], dtype=torch.long), topic_tensor, hidden
                )
                next_logits = next_logits[0]
        return decode_ids(generated, self._vocabulary)

    def _lm_candidates(self, prompt: str, topic: str, *, count: int, max_words: int,
                       temperature: float, top_k: int, ngram: int, seed: int,
                       constrain: bool = True, min_words: int = 0) -> List[Candidate]:
        """Несколько кандидатов LSTM-модели."""
        candidates: List[Candidate] = []
        for mode in (constrain, False):
            for attempt in range(count + 2):
                tokens = self._lm_generate_tokens(
                    prompt,
                    topic,
                    max_words=max_words,
                    temperature=temperature,
                    top_k=top_k,
                    ngram=ngram,
                    seed=seed + attempt,
                    constrain=mode,
                    min_words=min_words,
                )
                text = format_tokens_as_text(tokens)
                if len(tokens) >= 2:
                    candidates.append(Candidate(text=text, method="lstm", topic=topic))
                if len(candidates) >= count:
                    return candidates
            if candidates:
                return candidates
        return candidates

    def _seq2seq_source(self, prompt: str, topic: str) -> List[str]:
        """Источник для seq2seq: запрос или предложение темы."""
        prompt_tokens = doc_tokens(prompt, self._mode) if prompt else []
        if prompt_tokens:
            return prompt_tokens[: self._max_src_len]
        for index in self._topic_document_indices(topic, limit=1):
            sentences = _sentence_split(self._texts[index])
            if sentences:
                tokens = doc_tokens(sentences[0], self._mode)
                if tokens:
                    return tokens[: self._max_src_len]
        return []

    def _seq2seq_generate_tokens(self, source_tokens: Sequence[str], *, model: Seq2SeqLSTM,
                                 max_words: int, temperature: float, top_k: int,
                                 ngram: int, seed: int, constrain: bool = True,
                                 min_words: int = 0, topic: str = "") -> List[str]:
        """Сгенерировать целевую последовательность seq2seq-моделью."""
        source_ids = encode_tokens(source_tokens, self._vocabulary, keep_unknown=True)
        source_ids = source_ids[: self._max_src_len]
        if not source_ids:
            return []
        topic_id = int(self._topic_to_idx.get(topic, 0))
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
        source_tensor, source_lengths = pad_token_sequences([source_ids], max_len=self._max_src_len)
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            encoder_outputs, hidden, cell, mask = model.encode(
                source_tensor, source_lengths, topic_tensor
            )
        token = torch.tensor([BOS], dtype=torch.long)
        generated: List[int] = []
        generated_words: List[str] = []
        reverse = {index: token for token, index in self._vocabulary.items()}
        for _ in range(max(1, max_words)):
            with torch.no_grad():
                logits, hidden, cell, _ = model.decode_step(
                    token, hidden, cell, encoder_outputs, mask, topic_tensor
                )
            banned = self._banned_ngram_tokens(generated, ngram)
            allowed = self._allowed_ids(generated_words, constrain)
            token_id = self._sample_next(
                logits[0],
                temperature=temperature,
                top_k=top_k,
                banned=banned,
                generator=generator,
                allowed=allowed,
                allow_eos=self._can_end(generated_words, min_words),
            )
            if token_id == EOS:
                break
            if token_id < FIRST_WORD_INDEX:
                continue
            generated.append(token_id)
            generated_words.append(reverse.get(token_id, ""))
            token = torch.tensor([token_id], dtype=torch.long)
        return decode_ids(generated, self._vocabulary)

    def _seq2seq_candidates(self, prompt: str, topic: str, *, count: int, max_words: int,
                            temperature: float, top_k: int, ngram: int, seed: int,
                            constrain: bool = True, min_words: int = 0) -> List[Candidate]:
        """Кандидаты seq2seq: продолжение предложения (или реферирование)."""
        model = self._next if self._next is not None else self._lead
        if model is None:
            return []
        source_tokens = self._seq2seq_source(prompt, topic)
        if not source_tokens:
            return []
        candidates: List[Candidate] = []
        for mode in (constrain, False):
            for attempt in range(count + 2):
                tokens = self._seq2seq_generate_tokens(
                    source_tokens,
                    model=model,
                    max_words=max_words,
                    temperature=temperature,
                    top_k=top_k,
                    ngram=ngram,
                    seed=seed + attempt,
                    constrain=mode,
                    min_words=min_words,
                    topic=topic,
                )
                text = format_tokens_as_text(tokens)
                if len(tokens) >= 2:
                    candidates.append(Candidate(text=text, method="seq2seq", topic=topic))
                if len(candidates) >= count:
                    return candidates
            if candidates:
                return candidates
        return candidates

    # ------------------------------------------------------------------ #
    # Внутреннее: оценка и ранжирование
    # ------------------------------------------------------------------ #

    def _lm_fluency(self, text: str, topic: str) -> float:
        """``exp(средний log P)`` текста по LSTM-модели (0..1)."""
        tokens = doc_tokens(text, self._mode)
        ids = encode_tokens(
            tokens,
            self._vocabulary,
            add_bos=True,
            add_eos=True,
            max_len=self._max_len,
            keep_unknown=False,
        )
        if len(ids) < 3:
            return 0.0
        input_tensor = torch.tensor([ids[:-1]], dtype=torch.long)
        target_tensor = torch.tensor([ids[1:]], dtype=torch.long)
        topic_tensor = torch.tensor([self._topic_to_idx.get(topic, 0)], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self._lm(input_tensor, topic_tensor)
        mean_log_prob = mean_log_probability(logits, target_tensor)
        if not math.isfinite(mean_log_prob):
            return 0.0
        return float(math.exp(max(-20.0, min(0.0, mean_log_prob))))

    #: Слова, на которых фраза выглядит оборванной.
    _BAD_ENDINGS = frozenset(
        "и а но на в с к у по из от до за для что чтобы или же ли не ни при о об "
        "над под через после перед между без чем как так".split()
    )

    def _completeness(self, text: str) -> float:
        """Оценка завершённости: штраф за обрыв на предлоге/союзе и за длину < 4."""
        tokens = doc_tokens(text, self._mode)
        if not tokens:
            return 0.0
        if len(tokens) < 4:
            return 0.3
        if tokens[-1] in self._BAD_ENDINGS:
            return 0.4
        return 1.0

    def _can_end(self, generated_words: Sequence[str], min_words: int) -> bool:
        """Можно ли завершать последовательность: длина и отсутствие «плохого» конца."""
        if len(generated_words) < max(0, min_words):
            return False
        if generated_words and generated_words[-1] in self._BAD_ENDINGS:
            return False
        return True

    def _coherence(self, text: str) -> float:
        """Доля биграмм текста, встречающихся в корпусе (максимум по предложениям)."""
        sentences = _sentence_split(text) or [text]
        best = 0.0
        for sentence in sentences:
            tokens = doc_tokens(sentence, self._mode)
            if len(tokens) < 2:
                continue
            bigrams = list(zip(tokens, tokens[1:]))
            if not bigrams:
                continue
            overlap = sum(1 for bigram in bigrams if bigram in self._bigrams) / len(bigrams)
            best = max(best, float(overlap))
        return best

    def _uniqueness(self, text: str) -> float:
        """Доля уникальных слов; защищает от вырожденных повторов."""
        tokens = doc_tokens(text, self._mode)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _score_candidate(self, candidate: Candidate, prompt: str, topic: str) -> Candidate:
        """Заполнить компоненты оценки и итоговый ``score`` кандидата."""
        try:
            text_vector = self._semantic.doc_vector(candidate.text)
        except (RuntimeError, ValueError):
            text_vector = None
        reference = None
        if prompt:
            try:
                reference = self._semantic.doc_vector(prompt)
            except (RuntimeError, ValueError):
                reference = None
        if reference is None:
            reference = self._topic_centroids.get(topic)
        similarity = _cosine(reference, text_vector) if reference is not None and text_vector is not None else 0.0
        semantic = max(0.0, min(1.0, similarity))

        confidence = 0.0
        try:
            prediction: Prediction = self._semantic.predict(candidate.text)
            confidence = float(prediction.probabilities.get(topic, 0.0))
        except (ValueError, RuntimeError):
            confidence = 0.0

        fluency = self._lm_fluency(candidate.text, topic)
        coherence = self._coherence(candidate.text)
        uniqueness = self._uniqueness(candidate.text)
        completeness = self._completeness(candidate.text)
        candidate.semantic = semantic
        candidate.topic_confidence = max(0.0, min(1.0, confidence))
        candidate.fluency = max(0.0, min(1.0, fluency))
        candidate.coherence = max(0.0, min(1.0, coherence))
        candidate.uniqueness = max(0.0, min(1.0, uniqueness))
        candidate.completeness = max(0.0, min(1.0, completeness))
        # Веса подобраны так, чтобы «осмысленность» (близость к теме + связность
        # с корпусом + завершённость) доминировала над свободной генерацией.
        candidate.score = (
            0.28 * candidate.semantic
            + 0.22 * candidate.topic_confidence
            + 0.12 * candidate.fluency
            + 0.20 * candidate.coherence
            + 0.10 * candidate.uniqueness
            + 0.08 * candidate.completeness
        )
        return candidate

    def _rank(self, candidates: Sequence[Candidate], prompt: str, topic: str) -> List[Candidate]:
        """Оценить и отсортировать кандидатов по убыванию итоговой оценки."""
        scored = [
            self._score_candidate(candidate, prompt, topic)
            for candidate in candidates
            if candidate is not None and candidate.text
        ]
        scored.sort(
            key=lambda candidate: (-candidate.score, -len(candidate.text), candidate.method),
        )
        return scored
