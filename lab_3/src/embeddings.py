# -*- coding: utf-8 -*-
"""Word2Vec (SGNS) и Doc2Vec (PV-DBOW) с negative sampling — INTERFACES.md §3.

Модуль реализует две shallow-модели на чистом `numpy` (без `gensim`/`sklearn`):

* :class:`Word2Vec` — skip-gram с negative sampling (SGNS): обучаются входная
  матрица слов ``vectors_`` и внутренняя выходная (контекстная) матрица;
* :class:`Doc2Vec` — PV-DBOW: обучаются векторы документов ``doc_vectors_``
  (вход) и выходные векторы слов ``word_vectors_``, плюс fold-in нового
  документа (:meth:`Doc2Vec.infer_vector`).

Общая механика negative sampling (INTERFACES.md §1):

* unigram-распределение негативов — вес слова ``count**0.75``, нормированный в
  кумулятивную функцию; индекс негатива берётся как
  ``np.searchsorted(cumdist, rng.random())`` с клампом в ``[0, vocab-1]``;
  негатив, совпавший с положительным словом, пересэмплируется;
* SGD по логистической функции: для пары «вход ``vin`` × выход ``vout``» с
  меткой ``label ∈ {1, 0}`` — ``g = lr * (label - sigmoid(vin·vout))``;
  ``vin += g*vout``; ``vout += g*vin``. Для негативов обновляется ТОЛЬКО
  выходной вектор;
* learning rate — линейный спад по эпохам
  ``lr = max(min_alpha, alpha * (1 - epoch/epochs))``;
* все векторы хранятся как ``float32``.

Воспроизводимость: источники случайности — только локальный
``numpy.random.RandomState(seed)`` (глобальный ``np.random`` не затрагивается),
порядок обхода документов/слов фиксирован, словарь строится в лексикографическом
порядке. Два :meth:`fit` с одинаковым ``seed`` дают одинаковые векторы.

Сериализация в этом модуле отсутствует: сохранение/загрузка модели — задача
фасада (INTERFACES.md §5).
"""
from __future__ import annotations

import numpy as np

__all__ = ["Word2Vec", "Doc2Vec"]

#: Порог клампа логита: ``exp(±20)`` умещается в float32 без переполнения.
_LOGIT_CLIP = 20.0


# --------------------------------------------------------------------------- #
# Внутренние утилиты
# --------------------------------------------------------------------------- #

def _as_int(value, name: str) -> int:
    """Приводит ``value`` к ``int``; неудачный ввод -> ``ValueError`` (по-русски)."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} должен быть целым числом, получено {!r}".format(name, value)
        ) from exc


def _sigmoid(z: float) -> float:
    """Логистическая функция с клампом аргумента (защита от переполнения exp)."""
    z = min(max(float(z), -_LOGIT_CLIP), _LOGIT_CLIP)
    return 1.0 / (1.0 + float(np.exp(-z)))


def _row_norms(matrix: "np.ndarray") -> "np.ndarray":
    """L2-нормы строк матрицы (``float64``); нулевые строки остаются нулевыми."""
    return np.asarray(np.linalg.norm(matrix, axis=1), dtype=np.float64)


def _l2_normalized(matrix: "np.ndarray") -> "np.ndarray":
    """L2-нормировка строк; строки нулевой нормы остаются нулевыми (без NaN)."""
    norms = _row_norms(matrix)
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(matrix, dtype=np.float64) / safe[:, np.newaxis]


def _coerce_top_n(top_n) -> int:
    """``top_n`` -> неотрицательный ``int``; «мусор» -> ``0`` (без исключений)."""
    try:
        limit = int(top_n)
    except (TypeError, ValueError):
        return 0
    return max(0, limit)


def _sorted_vocabulary(docs) -> dict:
    """Словарь ``token -> index`` в лексикографическом порядке по всем документам.

    Обход документов/токенов стабилен, сортировка ключей задаёт детерминированные
    индексы (INTERFACES.md §1).
    """
    counts = {}
    for doc in docs:
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
    return {token: index for index, token in enumerate(sorted(counts))}


def _unigram_cumdist(counts: "np.ndarray") -> "np.ndarray":
    """Кумулятивная функция unigram-распределения негативов: ``count**0.75``.

    Возвращает массив длины ``n_words`` с ``cum[-1] == 1.0``. Вырожденный случай
    (все счётчики нулевые) -> равномерное распределение.
    """
    weights = np.power(np.asarray(counts, dtype=np.float64), 0.75)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones_like(weights)
        total = float(weights.sum())
    cumdist = np.cumsum(weights / total, dtype=np.float64)
    # Страховка от накопленной ошибки округления: последний элемент ровно 1.0,
    # иначе rng.random() == 1.0 - eps может уйти за правую границу.
    cumdist[-1] = 1.0
    return cumdist


def _sample_negatives(n_words: int, positive_index: int, negative: int,
                      rng: "np.random.RandomState", cumdist: "np.ndarray") -> "np.ndarray":
    """Индексы ``negative`` негативов из unigram-распределения (INTERFACES.md §1).

    Индекс: ``np.searchsorted(cumdist, rng.random())`` с клампом в
    ``[0, n_words - 1]``; негатив, совпавший с ``positive_index``, пересэмплируется.
    """
    count = max(0, int(negative))
    if count == 0 or n_words <= 1:
        return np.empty(0, dtype=np.intp)
    draws = rng.random(count) * float(cumdist[-1])
    indices = np.searchsorted(cumdist, draws)
    np.clip(indices, 0, n_words - 1, out=indices)
    # Пересэмплирование совпавших с положительным словом (их не больше count).
    attempts = 0
    while True:
        clashes = np.flatnonzero(indices == positive_index)
        if clashes.size == 0 or attempts >= 100:
            break
        draws = rng.random(clashes.size) * float(cumdist[-1])
        redrawn = np.searchsorted(cumdist, draws)
        np.clip(redrawn, 0, n_words - 1, out=redrawn)
        indices[clashes] = redrawn
        attempts += 1
    return indices


def _sgd_pair(inputs: "np.ndarray", input_index: int, outputs: "np.ndarray",
              output_index: int, label: int, lr: float, update_input: bool,
              update_output: bool = True) -> None:
    """Один шаг SGD по паре «вход × выход».

    ``label == 1`` — положительная пара; ``label == 0`` — негатив. При
    ``update_input=False`` входной вектор не обновляется (обновляется только
    выходной — так обновляются негативы в SGNS/PV-DBOW). При
    ``update_output=False`` не обновляется выходной вектор — так fold-in
    (:meth:`Doc2Vec.infer_vector`) обновляет ТОЛЬКО новый вектор документа,
    оставляя обученные векторы слов замороженными.
    """
    dot = float(np.dot(np.asarray(inputs[input_index], dtype=np.float64),
                       np.asarray(outputs[output_index], dtype=np.float64)))
    grad = float(lr) * (float(label) - _sigmoid(dot))
    if update_input:
        inputs[input_index] = (np.asarray(inputs[input_index], dtype=np.float32)
                               + np.float32(grad) * outputs[output_index])
    if update_output:
        outputs[output_index] = (np.asarray(outputs[output_index], dtype=np.float32)
                                 + np.float32(grad) * inputs[input_index])


# --------------------------------------------------------------------------- #
# Word2Vec
# --------------------------------------------------------------------------- #

class Word2Vec:
    """Skip-gram с negative sampling (SGNS) на `numpy` (INTERFACES.md §3).

    Параметры
    ---------
    dim:
        Размерность векторов (> 0).
    window:
        Полуширина окна контекста (>= 0): для центра ``i`` перебираются
        ``j ∈ [max(0, i-window), min(len(doc), i+window+1))``, ``j != i``.
    min_count:
        Минимальная частота токена (>= 1); редкие токены попадают в OOV.
    epochs:
        Число эпох обучения (> 0).
    negative:
        Число негативов на положительную пару (клампится в ``[1, n_words_-1]``;
        при ``n_words_ == 1`` негативы не сэмплируются).
    alpha, min_alpha:
        Начальный и минимальный learning rate линейного спада по эпохам.
    seed:
        Сид локального ``numpy.random.RandomState``.

    Атрибуты после :meth:`fit`
    --------------------------
    vocabulary_ : dict[str, int]
        Токен -> индекс, лексикографический порядок.
    vectors_ : numpy.ndarray
        Входная матрица слов, форма ``(n_words_, dim)``, ``float32``.
    counts_ : numpy.ndarray
        Частоты токенов в порядке словаря.
    dim, n_words_ : int
        Размерность и размер словаря.
    epochs_, negative_, alpha_, min_alpha_, min_count_, window_, seed_ :
        Зафиксированные параметры обучения (нужны для fold-in и диагностики).
    """

    def __init__(self, dim: int = 100, window: int = 5, min_count: int = 1,
                 epochs: int = 5, negative: int = 5, alpha: float = 0.025,
                 min_alpha: float = 0.0001, seed: int = 42):
        self.dim = _as_int(dim, "dim")
        self.window = _as_int(window, "window")
        self.min_count = _as_int(min_count, "min_count")
        self.epochs = _as_int(epochs, "epochs")
        self.negative = _as_int(negative, "negative")
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.seed = _as_int(seed, "seed")

        # Атрибуты, заполняемые fit (единый набор — чтобы `_require_fitted` и
        # использование атрибутов не падали с AttributeError).
        self.vocabulary_ = None
        self.vectors_ = None
        self.counts_ = None
        self.n_words_ = 0
        self.epochs_ = self.epochs
        self.negative_ = self.negative
        self.alpha_ = self.alpha
        self.min_alpha_ = self.min_alpha
        self.min_count_ = self.min_count
        self.window_ = self.window
        self.seed_ = self.seed
        # Внутренняя выходная (контекстная) матрица — не часть публичного
        # контракта: наружу отдаётся только входная матрица `vectors_`.
        self._context_vectors_ = None

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(self, tokenized_docs) -> "Word2Vec":
        """Обучает SGNS на ``tokenized_docs`` и возвращает ``self``.

        Документы приводятся к списку списков строк (стабильный порядок обхода),
        строится словарь в лексикографическом порядке, затем для каждой эпохи,
        каждого документа и каждого центрального слова обновляются положительная
        пара (вход центра + выход контекста) и ``negative`` негативов (обновляется
        только выходной вектор).

        Пустой/``None`` вход или пустой после ``min_count`` словарь -> ``ValueError``.
        """
        docs = self._prepare(docs=tokenized_docs)
        vocabulary = _sorted_vocabulary(docs)
        counts = self._counts(docs, vocabulary)
        kept = {token: index for token, index in vocabulary.items()
                if counts[index] >= self.min_count_}
        if not kept:
            raise ValueError(
                "словарь пуст после отсечения min_count={}".format(self.min_count_)
            )
        # Индексы перенумеровываются в том же (лексикографическом) порядке.
        vocabulary = {token: index for index, token in enumerate(sorted(kept))}
        counts = self._counts(docs, vocabulary)
        n_words = len(vocabulary)

        rng = np.random.RandomState(self.seed_)
        scale = 0.5 / float(self.dim)
        vectors = rng.uniform(-scale, scale, size=(n_words, self.dim)).astype(np.float32)
        context_vectors = rng.uniform(-scale, scale, size=(n_words, self.dim)).astype(np.float32)
        cumdist = _unigram_cumdist(counts)
        negative_max = max(1, n_words - 1)
        window = self.window_

        for epoch in range(self.epochs_):
            lr = max(self.min_alpha_,
                     self.alpha_ * (1.0 - float(epoch) / float(self.epochs_)))
            for doc in docs:
                length = len(doc)
                for i in range(length):
                    center = vocabulary.get(doc[i])
                    if center is None:
                        continue  # OOV (редкий токен) — пропускается
                    left = max(0, i - window)
                    right = min(length, i + window + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        context = vocabulary.get(doc[j])
                        if context is None:
                            continue
                        # Положительная пара: центр (вход) -> контекст (выход).
                        _sgd_pair(vectors, center, context_vectors, context,
                                  label=1, lr=lr, update_input=True)
                        negative = max(1, min(self.negative_, negative_max)) if n_words > 1 else 0
                        for index in _sample_negatives(n_words, context, negative, rng, cumdist):
                            # Негатив: обновляется ТОЛЬКО выходной вектор.
                            _sgd_pair(vectors, center, context_vectors, int(index),
                                      label=0, lr=lr, update_input=False)

        self.vocabulary_ = vocabulary
        self.counts_ = counts
        self.vectors_ = vectors
        self._context_vectors_ = context_vectors
        self.n_words_ = n_words
        return self

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def word_vector(self, word) -> "np.ndarray | None":
        """Входной вектор слова ``vectors_[idx]``; OOV -> ``None``."""
        vectors = self._require_fitted()
        index = self._index(word)
        if index is None:
            return None
        return np.array(vectors[index], dtype=np.float32, copy=True)

    def most_similar(self, word, top_n: int = 10) -> "list":
        """``list[(слово, косинус)]`` ближайших слов; OOV -> ``[]``.

        Строки ``vectors_`` нормируются один раз; само слово исключается из
        результата. Некорректный ``top_n`` трактуется как ``0`` -> ``[]``.
        """
        vectors = self._require_fitted()
        index = self._index(word)
        if index is None:
            return []
        limit = _coerce_top_n(top_n)
        if limit <= 0:
            return []
        order = self._similar_indices(vectors, index, limit)
        return [(self._token(int(other)), float(score)) for other, score in order]

    def doc_vector(self, tokens) -> "np.ndarray":
        """Mean-pooling входных векторов известных токенов, форма ``(dim,)``.

        OOV пропускаются; полностью OOV/пустой ввод -> нулевой вектор.
        """
        vectors = self._require_fitted()
        accumulator = np.zeros(self.dim, dtype=np.float64)
        known = 0
        for token in _tokens_of(tokens):
            index = self._index(token)
            if index is None:
                continue
            accumulator += np.asarray(vectors[index], dtype=np.float64)
            known += 1
        if known == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(accumulator / float(known), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _prepare(self, docs):
        """Валидирует параметры обучения и приводит документы к списку списков строк."""
        if self.dim <= 0:
            raise ValueError("dim должен быть > 0, получено {}".format(self.dim))
        if self.window_ < 0:
            raise ValueError("window должен быть >= 0, получено {}".format(self.window_))
        if self.epochs_ <= 0:
            raise ValueError("epochs должен быть > 0, получено {}".format(self.epochs_))
        if self.min_count_ < 1:
            raise ValueError("min_count должен быть >= 1, получено {}".format(self.min_count_))
        if self.negative_ < 0:
            raise ValueError("negative должен быть >= 0, получено {}".format(self.negative_))
        if docs is None:
            raise ValueError("tokenized_docs не может быть None: ожидается итерируемое списков токенов")
        try:
            materialized = [[_as_token(token) for token in doc] for doc in docs]
        except TypeError as exc:
            raise ValueError(
                "tokenized_docs должен быть итерируемым списков токенов, получено {!r}".format(docs)
            ) from exc
        if not materialized:
            raise ValueError("tokenized_docs пуст: нужен хотя бы один документ")
        return materialized

    def _counts(self, docs, vocabulary) -> "np.ndarray":
        """Частоты токенов в порядке индексов словаря (``int64``)."""
        counts = np.zeros(len(vocabulary), dtype=np.int64)
        for doc in docs:
            for token in doc:
                index = vocabulary.get(token)
                if index is not None:
                    counts[index] += 1
        return counts

    def _index(self, word):
        """Индекс слова в словаре; OOV/``None`` -> ``None``."""
        if self.vocabulary_ is None:
            return None
        try:
            key = _as_token(word)
        except ValueError:
            return None
        return self.vocabulary_.get(key)

    def _token(self, index: int) -> str:
        """Токен по индексу словаря (обратный индекс строится лениво)."""
        inverse = getattr(self, "_inverse_", None)
        if inverse is None or len(inverse) != len(self.vocabulary_):
            inverse = {value: key for key, value in self.vocabulary_.items()}
            self._inverse_ = inverse
        return inverse[index]

    def _similar_indices(self, vectors, index: int, limit: int) -> list:
        """``list[(индекс, косинус)]`` по убыванию, без самого слова ``index``."""
        normalized = _l2_normalized(vectors)
        scores = np.asarray(normalized @ normalized[index], dtype=np.float64)
        # Само слово (и любые нулевые строки) исключаются через -inf.
        scores[index] = -np.inf
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(other), float(scores[other])) for other in order
                if np.isfinite(scores[other])]

    def _require_fitted(self) -> "np.ndarray":
        if self.vectors_ is None:
            raise ValueError("модель не обучена: вызовите fit(tokenized_docs)")
        return self.vectors_

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "Word2Vec(dim={}, window={}, n_words_={})".format(
            self.dim, self.window, self.n_words_
        )


# --------------------------------------------------------------------------- #
# Doc2Vec
# --------------------------------------------------------------------------- #

class Doc2Vec:
    """PV-DBOW (`dm=0`) с negative sampling на `numpy` (INTERFACES.md §3).

    Параметры
    ---------
    dim, min_count, epochs, negative, alpha, min_alpha, seed:
        См. :class:`Word2Vec` (``window`` в PV-DBOW не используется).

    Атрибуты после :meth:`fit`
    --------------------------
    vocabulary_ : dict[str, int]
        Токен -> индекс, лексикографический порядок.
    doc_vectors_ : numpy.ndarray
        Входные векторы документов, форма ``(n_docs_, dim)``, ``float32``.
    word_vectors_ : numpy.ndarray
        Выходные векторы слов, форма ``(n_words_, dim)``, ``float32``.
    counts_, dim, n_docs_, n_words_ : numpy.ndarray | int
        Частоты токенов и размеры модели.
    """

    def __init__(self, dim: int = 100, min_count: int = 1, epochs: int = 10,
                 negative: int = 5, alpha: float = 0.025, min_alpha: float = 0.0001,
                 seed: int = 42):
        self.dim = _as_int(dim, "dim")
        self.min_count = _as_int(min_count, "min_count")
        self.epochs = _as_int(epochs, "epochs")
        self.negative = _as_int(negative, "negative")
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.seed = _as_int(seed, "seed")

        self.vocabulary_ = None
        self.doc_vectors_ = None
        self.word_vectors_ = None
        self.counts_ = None
        self.n_docs_ = 0
        self.n_words_ = 0
        self.epochs_ = self.epochs
        self.negative_ = self.negative
        self.alpha_ = self.alpha
        self.min_alpha_ = self.min_alpha
        self.min_count_ = self.min_count
        self.seed_ = self.seed

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #

    def fit(self, tokenized_docs) -> "Doc2Vec":
        """Обучает PV-DBOW: положительная пара (документ ``d`` -> слово).

        Для каждого документа ``d`` и каждого его слова обновляется вектор
        документа ``doc_vectors_[d]`` (вход) и выходной вектор слова
        ``word_vectors_[w]``; для каждого негатива обновляется ТОЛЬКО выходной
        вектор слова.

        Пустой/``None`` вход или пустой после ``min_count`` словарь -> ``ValueError``.
        """
        docs = self._prepare(docs=tokenized_docs)
        vocabulary = _sorted_vocabulary(docs)
        counts = self._counts(docs, vocabulary)
        kept = {token: index for token, index in vocabulary.items()
                if counts[index] >= self.min_count_}
        if not kept:
            raise ValueError(
                "словарь пуст после отсечения min_count={}".format(self.min_count_)
            )
        vocabulary = {token: index for index, token in enumerate(sorted(kept))}
        counts = self._counts(docs, vocabulary)
        n_words = len(vocabulary)
        n_docs = len(docs)

        rng = np.random.RandomState(self.seed_)
        scale = 0.5 / float(self.dim)
        doc_vectors = rng.uniform(-scale, scale, size=(n_docs, self.dim)).astype(np.float32)
        word_vectors = rng.uniform(-scale, scale, size=(n_words, self.dim)).astype(np.float32)
        cumdist = _unigram_cumdist(counts)
        negative_max = max(1, n_words - 1)

        for epoch in range(self.epochs_):
            lr = max(self.min_alpha_,
                     self.alpha_ * (1.0 - float(epoch) / float(self.epochs_)))
            for d, doc in enumerate(docs):
                for token in doc:
                    word = vocabulary.get(token)
                    if word is None:
                        continue  # OOV (редкий токен) — пропускается
                    _sgd_pair(doc_vectors, d, word_vectors, word,
                              label=1, lr=lr, update_input=True)
                    negative = max(1, min(self.negative_, negative_max)) if n_words > 1 else 0
                    for index in _sample_negatives(n_words, word, negative, rng, cumdist):
                        # Негатив: обновляется ТОЛЬКО выходной вектор слова.
                        _sgd_pair(doc_vectors, d, word_vectors, int(index),
                                  label=0, lr=lr, update_input=False)

        self.vocabulary_ = vocabulary
        self.counts_ = counts
        self.doc_vectors_ = doc_vectors
        self.word_vectors_ = word_vectors
        self.n_docs_ = n_docs
        self.n_words_ = n_words
        return self

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def doc_vector(self, index) -> "np.ndarray":
        """Вектор документа ``doc_vectors_[index]``.

        Индекс вне ``[0, n_docs_)`` -> ``IndexError`` (естественное поведение
        numpy; фасад защищает вызов).
        """
        vectors = self._require_fitted()
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise IndexError("index документа должен быть целым числом, получено {!r}".format(index))
        position = int(index)
        if not 0 <= position < self.n_docs_:
            raise IndexError(
                "индекс документа {} вне диапазона [0, {})".format(position, self.n_docs_)
            )
        return np.array(vectors[position], dtype=np.float32, copy=True)

    def infer_vector(self, tokens, epochs=None, alpha=None) -> "np.ndarray":
        """Fold-in нового документа: обучается ТОЛЬКО новый вектор документа.

        Выходные векторы слов ``word_vectors_`` заморожены. Стартовое значение —
        среднее выходных векторов известных токенов документа, нормированное к
        единичной норме (PV-DBOW сходится к центроиду векторов слов документа;
        случайная единичная инициализация непригодна: обученные ``word_vectors_``
        имеют норму ~0.04, логит насыщается и градиент почти нулевой). Затем
        выполняется ``epochs`` (по умолчанию ``self.epochs_``) проходов по
        известным токенам с learning rate ``alpha`` (по умолчанию ``self.alpha_``).
        Полностью OOV/пустой ввод -> нулевой вектор формы ``(dim,)``.
        """
        word_vectors = self._require_word_vectors()
        known = []
        for token in _tokens_of(tokens):
            index = self._index(token)
            if index is None:
                continue  # OOV — пропускается
            known.append(index)
        if not known:
            return np.zeros(self.dim, dtype=np.float32)

        if epochs is None:
            passes = self.epochs_
        else:
            passes = _as_int(epochs, "epochs")
        if passes <= 0:
            raise ValueError("epochs должен быть > 0, получено {}".format(passes))
        if alpha is None:
            lr = self.alpha_
        else:
            lr = float(alpha)

        n_words = self.n_words_
        rng = np.random.RandomState(self.seed_)
        vector = np.asarray(word_vectors[known], dtype=np.float64).mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        vector = np.asarray(vector, dtype=np.float32)
        held = np.asarray([vector], dtype=np.float32)

        cumdist = _unigram_cumdist(self.counts_)
        negative_max = max(1, n_words - 1)
        # Копия выходной матрицы: fold-in обязан оставить обученные векторы слов
        # неизменными, а `_sgd_pair` обновляет выходной вектор на месте.
        frozen = np.array(word_vectors, dtype=np.float32, copy=True)
        for _ in range(passes):
            for word in known:
                # Положительная пара и негативы обновляют ТОЛЬКО новый вектор
                # документа: `word_vectors_` (копия) заморожены.
                _sgd_pair(held, 0, frozen, word,
                          label=1, lr=lr, update_input=True, update_output=False)
                negative = max(1, min(self.negative_, negative_max)) if n_words > 1 else 0
                for index in _sample_negatives(n_words, word, negative, rng, cumdist):
                    _sgd_pair(held, 0, frozen, int(index),
                              label=0, lr=lr, update_input=True, update_output=False)
        return np.array(held[0], dtype=np.float32, copy=True)

    def word_vector(self, word) -> "np.ndarray | None":
        """Выходной вектор слова ``word_vectors_[idx]``; OOV -> ``None``."""
        word_vectors = self._require_word_vectors()
        index = self._index(word)
        if index is None:
            return None
        return np.array(word_vectors[index], dtype=np.float32, copy=True)

    def most_similar_words(self, word, top_n: int = 10) -> "list":
        """``list[(слово, косинус)]`` ближайших слов по ``word_vectors_``; OOV -> ``[]``."""
        word_vectors = self._require_word_vectors()
        index = self._index(word)
        if index is None:
            return []
        limit = _coerce_top_n(top_n)
        if limit <= 0:
            return []
        normalized = _l2_normalized(word_vectors)
        scores = np.asarray(normalized @ normalized[index], dtype=np.float64)
        scores[index] = -np.inf
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(self._token(int(other)), float(scores[other])) for other in order
                if np.isfinite(scores[other])]

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _prepare(self, docs):
        """Валидирует параметры обучения и приводит документы к списку списков строк."""
        if self.dim <= 0:
            raise ValueError("dim должен быть > 0, получено {}".format(self.dim))
        if self.epochs_ <= 0:
            raise ValueError("epochs должен быть > 0, получено {}".format(self.epochs_))
        if self.min_count_ < 1:
            raise ValueError("min_count должен быть >= 1, получено {}".format(self.min_count_))
        if self.negative_ < 0:
            raise ValueError("negative должен быть >= 0, получено {}".format(self.negative_))
        if docs is None:
            raise ValueError("tokenized_docs не может быть None: ожидается итерируемое списков токенов")
        try:
            materialized = [[_as_token(token) for token in doc] for doc in docs]
        except TypeError as exc:
            raise ValueError(
                "tokenized_docs должен быть итерируемым списков токенов, получено {!r}".format(docs)
            ) from exc
        if not materialized:
            raise ValueError("tokenized_docs пуст: нужен хотя бы один документ")
        return materialized

    def _counts(self, docs, vocabulary) -> "np.ndarray":
        """Частоты токенов в порядке индексов словаря (``int64``)."""
        counts = np.zeros(len(vocabulary), dtype=np.int64)
        for doc in docs:
            for token in doc:
                index = vocabulary.get(token)
                if index is not None:
                    counts[index] += 1
        return counts

    def _index(self, word):
        """Индекс слова в словаре; OOV/``None`` -> ``None``."""
        if self.vocabulary_ is None:
            return None
        try:
            key = _as_token(word)
        except ValueError:
            return None
        return self.vocabulary_.get(key)

    def _token(self, index: int) -> str:
        """Токен по индексу словаря (обратный индекс строится лениво)."""
        inverse = getattr(self, "_inverse_", None)
        if inverse is None or len(inverse) != len(self.vocabulary_):
            inverse = {value: key for key, value in self.vocabulary_.items()}
            self._inverse_ = inverse
        return inverse[index]

    def _require_fitted(self) -> "np.ndarray":
        if self.doc_vectors_ is None:
            raise ValueError("модель не обучена: вызовите fit(tokenized_docs)")
        return self.doc_vectors_

    def _require_word_vectors(self) -> "np.ndarray":
        """Выходная матрица слов ``word_vectors_`` (проверка факта обучения)."""
        if self.doc_vectors_ is None or self.word_vectors_ is None:
            raise ValueError("модель не обучена: вызовите fit(tokenized_docs)")
        return self.word_vectors_

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "Doc2Vec(dim={}, n_docs_={}, n_words_={})".format(
            self.dim, self.n_docs_, self.n_words_
        )


# --------------------------------------------------------------------------- #
# Приведение токенов
# --------------------------------------------------------------------------- #

def _as_token(token) -> str:
    """Токен -> ``str`` (единый ключ словаря). Нестроковый скаляр приводится.

    ``None`` недопустим как токен -> ``ValueError``.
    """
    if isinstance(token, str):
        return token
    if token is None:
        raise ValueError("токен не может быть None")
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    return str(token)


def _tokens_of(tokens) -> list:
    """Приводит аргумент-токены к списку строк; ``None`` -> пустой список."""
    if tokens is None:
        return []
    if isinstance(tokens, str):
        return [tokens]
    try:
        return [_as_token(token) for token in tokens]
    except TypeError:
        return [_as_token(tokens)]
