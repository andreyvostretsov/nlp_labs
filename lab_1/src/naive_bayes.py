"""Multinomial Naive Bayes, реализованный с нуля на стандартной библиотеке.

Модуль намеренно не использует numpy/sklearn/pandas: только ``math``, ``json``
и стандартные контейнеры.

Математика (Laplace / add-alpha сглаживание):

    P(token | class) = (count(token, class) + alpha) / (total_tokens(class) + alpha * vocab_size)
    P(class)         = docs(class) / docs_total
    log P(class | tokens) = log P(class) + sum_tokens log P(token | class)

Токен, которого нет в словаре модели (OOV), не даёт никакого вклада
(эквивалентно log 1 = 0), поэтому предсказание никогда не «ломается» на
незнакомом слове.
"""

import json
import math

MODEL_VERSION = 1

INCOMPATIBLE_MODEL_MESSAGE = "несовместимая версия модели, переобучите"

_REQUIRED_FIELDS = (
    "version",
    "alpha",
    "classes",
    "prior_log",
    "cond_log",
    "vocab_size",
    "label_order",
)

_NEG_INF = float("-inf")


class MultinomialNB:
    """Мультиномиальный наивный байесовский классификатор (чистый Python)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes = []
        self.label_order = []
        self.prior_log = {}
        self.cond_log = {}
        self.vocab_size = 0

    # -- обучение ----------------------------------------------------------

    def fit(self, documents: list, labels: list) -> None:
        """Обучить модель.

        ``documents`` — список списков токенов, ``labels`` — метки классов
        (по одной на документ). Пустые входные данные допустимы: модель
        остаётся «пустой», но не падает.
        """
        documents = list(documents) if documents else []
        labels = list(labels) if labels else []

        counts = {}          # class -> {token: count}
        totals = {}          # class -> общее число токенов
        doc_counts = {}      # class -> число документов
        label_order = []     # порядок первого появления классов
        vocab = set()

        for document, label in zip(documents, labels):
            label = str(label)
            if label not in counts:
                counts[label] = {}
                totals[label] = 0
                doc_counts[label] = 0
                label_order.append(label)
            doc_counts[label] += 1
            for token in document or []:
                token = str(token)
                counts[label][token] = counts[label].get(token, 0) + 1
                totals[label] += 1
                vocab.add(token)

        self.vocab_size = len(vocab)
        # sorted-but-stable список уникальных меток; label_order зеркалит classes
        self.classes = sorted(label_order)
        self.label_order = list(self.classes)

        n_documents = sum(doc_counts.values())
        self.prior_log = {}
        self.cond_log = {}

        for label in self.classes:
            if n_documents:
                self.prior_log[label] = math.log(doc_counts[label] / n_documents)
            else:
                self.prior_log[label] = _NEG_INF

        for label in self.classes:
            denominator = totals[label] + self.alpha * self.vocab_size
            entry = {}
            for token in vocab:
                numerator = counts[label].get(token, 0) + self.alpha
                if numerator > 0.0 and denominator > 0.0:
                    entry[token] = math.log(numerator / denominator)
                else:
                    entry[token] = _NEG_INF
            self.cond_log[label] = entry

    # -- предсказание ------------------------------------------------------

    def predict_log_proba(self, tokens: list) -> dict:
        """log P(class) + сумма log P(token|class); OOV-токены дают нулевой вклад."""
        result = {}
        for label in self.classes:
            total = float(self.prior_log.get(label, _NEG_INF))
            entry = self.cond_log.get(label, {})
            for token in tokens or []:
                total += entry.get(str(token), 0.0)
            result[label] = total
        return result

    def predict_proba(self, tokens: list) -> dict:
        """Нормированные вероятности классов (softmax по log-вероятностям)."""
        log_proba = self.predict_log_proba(tokens)
        n_classes = len(log_proba)
        if n_classes == 0:
            return {}

        finite = {c: v for c, v in log_proba.items() if math.isfinite(v)}
        if not finite:
            # вырожденный случай: все классы получили -inf -> равные доли
            return {c: 1.0 / n_classes for c in log_proba}

        best = max(finite.values())
        weights = {c: (math.exp(v - best) if math.isfinite(v) else 0.0) for c, v in log_proba.items()}
        total_weight = sum(weights.values())
        if total_weight <= 0.0 or not math.isfinite(total_weight):
            return {c: 1.0 / n_classes for c in log_proba}
        return {c: weight / total_weight for c, weight in weights.items()}

    def predict(self, tokens: list) -> str:
        """Метка класса с наибольшей log-вероятностью (при пустых токенах — наибольший prior)."""
        log_proba = self.predict_log_proba(tokens)
        if not log_proba:
            return ""
        return max(log_proba.items(), key=lambda item: item[1])[0]

    def score(self, documents: list, labels: list) -> float:
        """Доля правильных ответов (accuracy); пустой ``documents`` -> 0.0."""
        documents = list(documents) if documents else []
        if not documents:
            return 0.0
        labels = list(labels) if labels else []
        pairs = list(zip(documents, labels))
        if not pairs:
            return 0.0
        correct = sum(1 for document, label in pairs if self.predict(document) == str(label))
        return correct / len(pairs)

    # -- сериализация ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "alpha": float(self.alpha),
            "classes": list(self.classes),
            "prior_log": {label: float(value) for label, value in self.prior_log.items()},
            "cond_log": {
                label: {token: float(value) for token, value in entry.items()}
                for label, entry in self.cond_log.items()
            },
            "vocab_size": int(self.vocab_size),
            "label_order": list(self.label_order),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultinomialNB":
        if not isinstance(data, dict):
            raise ValueError(INCOMPATIBLE_MODEL_MESSAGE)
        for field in _REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                raise ValueError(INCOMPATIBLE_MODEL_MESSAGE)
        if data["version"] != MODEL_VERSION:
            raise ValueError(INCOMPATIBLE_MODEL_MESSAGE)

        model = cls(alpha=float(data["alpha"]))
        model.classes = [str(label) for label in data["classes"]]
        model.label_order = [str(label) for label in data["label_order"]] or list(model.classes)
        model.prior_log = {
            str(label): float(value) for label, value in dict(data["prior_log"]).items()
        }
        model.cond_log = {
            str(label): {str(token): float(value) for token, value in dict(entry).items()}
            for label, entry in dict(data["cond_log"]).items()
        }
        model.vocab_size = int(data["vocab_size"])
        return model


def save_model(nb: MultinomialNB, path: str) -> None:
    """Записать модель в JSON (utf-8) по указанному пути."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(nb.to_dict(), handle, ensure_ascii=False, indent=2)


def load_model(path: str) -> MultinomialNB:
    """Прочитать модель из JSON; при несовместимой версии/структуре — ValueError."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(INCOMPATIBLE_MODEL_MESSAGE) from exc
    return MultinomialNB.from_dict(data)
