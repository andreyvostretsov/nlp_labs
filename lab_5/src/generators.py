# -*- coding: utf-8 -*-
"""Генераторы текста на PyTorch: LSTM-языковая модель, sequence-to-sequence
(LSTM-энкодер → LSTM-декодер с вниманием) и шаблонный генератор
(INTERFACES.md §5).

Модуль не знает о семантическом анализаторе: он получает готовые
последовательности токенов, словарь и номера тем. Связывание с `lab_4`
(корпус, темы, векторы документов, поиск) выполняет фасад
:class:`lab_5.src.analyzer.GeneratingAnalyzer` — поэтому генерация остаётся
«только на основе» анализатора.

Специальные токены (единая нумерация для всех моделей):

* ``0`` — ``<pad>`` (паддинг);
* ``1`` — ``<bos>`` (начало последовательности);
* ``2`` — ``<eos>`` (конец последовательности);
* ``3`` — ``<unk>`` (слово вне словаря);
* слова корпуса — с ``4`` в лексикографическом порядке.

LSTM-языковая модель обусловлена темой: к эмбеддингу каждого токена
прибавляется (конкатенируется) эмбеддинг номера темы. Seq2seq-модель читает
исходную последовательность LSTM-энкодером и порождает целевую LSTM-декодером
с аддитивным вниманием Багданау.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PAD",
    "BOS",
    "EOS",
    "UNK",
    "SPECIAL_TOKENS",
    "build_vocabulary",
    "encode_tokens",
    "pad_token_sequences",
    "decode_ids",
    "format_tokens_as_text",
    "LSTMLanguageModel",
    "BahdanauAttention",
    "Seq2SeqLSTM",
    "TemplateGenerator",
]

#: Индексы специальных токенов.
PAD, BOS, EOS, UNK = 0, 1, 2, 3

#: Строковые имена специальных токенов (порядок = индексы 0..3).
SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

#: Первый индекс «настоящего» слова в словаре.
FIRST_WORD_INDEX = len(SPECIAL_TOKENS)


# --------------------------------------------------------------------------- #
# Словарь и последовательности
# --------------------------------------------------------------------------- #

def build_vocabulary(sequences: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Построить словарь ``{токен: индекс}`` по всем последовательностям.

    Слова сортируются лексикографически (воспроизводимость), индексы ``0..3``
    заняты специальными токенами.
    """
    words = set()
    for sequence in sequences:
        words.update(str(token) for token in sequence if str(token))
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for index, word in enumerate(sorted(words), start=FIRST_WORD_INDEX):
        vocabulary[word] = index
    return vocabulary


def encode_tokens(
    tokens: Sequence[str],
    vocabulary: Dict[str, int],
    *,
    add_bos: bool = False,
    add_eos: bool = False,
    max_len: Optional[int] = None,
    keep_unknown: bool = True,
) -> List[int]:
    """Закодировать токены индексами словаря.

    :param add_bos: добавить ``<bos>`` в начало.
    :param add_eos: добавить ``<eos>`` в конец.
    :param max_len: ограничить длину результата (спецтокены входят в лимит).
    :param keep_unknown: неизвестные токены -> ``<unk>`` (иначе отбрасываются).
    """
    encoded: List[int] = [BOS] if add_bos else []
    for token in tokens:
        index = vocabulary.get(str(token))
        if index is None:
            if keep_unknown:
                encoded.append(UNK)
            continue
        encoded.append(index)
    if add_eos:
        encoded.append(EOS)
    if max_len is not None and max_len > 0:
        encoded = encoded[:max_len]
        if add_eos and encoded and encoded[-1] != EOS:
            encoded[-1] = EOS
    return encoded


def pad_token_sequences(
    sequences: Sequence[Sequence[int]],
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Собрать батч: ``(LongTensor (n, L), длины (n,))``; ``0`` = PAD.

    Последовательности длиннее ``max_len`` обрезаются. Пустая последовательность
    даёт длину 0 (модели обязаны это переживать).
    """
    if not sequences:
        return (
            torch.zeros((0, 1), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths) if lengths else 0
    if max_len is not None and max_len > 0:
        width = min(width, int(max_len))
    width = max(width, 0)
    tensor = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        trimmed = list(sequence[:width])
        if trimmed:
            tensor[row, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)
    return tensor, torch.tensor(lengths, dtype=torch.long).clamp(max=width)


def decode_ids(ids: Sequence[int], vocabulary: Dict[str, int]) -> List[str]:
    """Индексы -> слова; специальные токены отбрасываются."""
    reverse = {index: token for token, index in vocabulary.items()}
    result: List[str] = []
    for index in ids:
        index = int(index)
        if index < FIRST_WORD_INDEX:
            continue
        token = reverse.get(index)
        if token is not None:
            result.append(token)
    return result


def format_tokens_as_text(tokens: Sequence[str], *, sentence: bool = True) -> str:
    """Склеить токены в читаемую строку: пробелы, заглавная буква, точка.

    Токенизатор `lab_1` удаляет пунктуацию, поэтому текст собирается из
    словоформ; завершающая точка добавляется, если её нет.
    """
    words = [str(token) for token in tokens if str(token)]
    if not words:
        return ""
    text = " ".join(words)
    if sentence:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?…":
            text += "."
    return text


# --------------------------------------------------------------------------- #
# LSTM-языковая модель
# --------------------------------------------------------------------------- #

class LSTMLanguageModel(nn.Module):
    """Тематически обусловленная LSTM-языковая модель (INTERFACES.md §5).

    Параметры
    ---------
    vocab_size:
        Число строк эмбеддинга включая специальные токены.
    embedding_dim:
        Размерность эмбеддинга слов.
    hidden_dim:
        Размерность скрытого состояния LSTM.
    num_layers:
        Число слоёв LSTM.
    dropout:
        Dropout между слоями LSTM и перед выходным слоем.
    n_topics:
        Число тем (номер ``0`` зарезервирован под «тема не задана»).
    topic_dim:
        Размерность эмбеддинга темы.
    padding_idx:
        Индекс паддинга (по умолчанию 0).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_topics: int = 1,
        topic_dim: int = 16,
        padding_idx: int = PAD,
    ):
        super().__init__()
        self.vocab_size = _as_int(vocab_size, "vocab_size")
        self.embedding_dim = _as_int(embedding_dim, "embedding_dim")
        self.hidden_dim = _as_int(hidden_dim, "hidden_dim")
        self.num_layers = _as_int(num_layers, "num_layers")
        self.n_topics = _as_int(n_topics, "n_topics")
        self.topic_dim = _as_int(topic_dim, "topic_dim")
        self.dropout_rate = float(dropout)
        self.padding_idx = _as_int(padding_idx, "padding_idx")

        if self.vocab_size <= FIRST_WORD_INDEX:
            raise ValueError(
                "vocab_size должен быть > {}, получено {}".format(
                    FIRST_WORD_INDEX, self.vocab_size
                )
            )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim должен быть > 0, получено {}".format(self.embedding_dim))
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim должен быть > 0, получено {}".format(self.hidden_dim))
        if self.num_layers <= 0:
            raise ValueError("num_layers должен быть > 0, получено {}".format(self.num_layers))
        if self.n_topics <= 0:
            raise ValueError("n_topics должен быть > 0, получено {}".format(self.n_topics))
        if self.topic_dim <= 0:
            raise ValueError("topic_dim должен быть > 0, получено {}".format(self.topic_dim))

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.topic_embedding = nn.Embedding(self.n_topics + 1, self.topic_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            self.embedding_dim + self.topic_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, tokens: torch.Tensor, topics: Optional[torch.Tensor] = None,
                hidden=None):
        """Логиты следующего токена для каждой позиции: ``(batch, L, vocab)``."""
        if topics is None:
            topics = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        embedded = self.embedding(tokens)
        topic_vector = self.topic_embedding(topics).unsqueeze(1)
        topic_vector = topic_vector.expand(-1, tokens.size(1), -1)
        output, hidden = self.lstm(torch.cat([embedded, topic_vector], dim=-1), hidden)
        return self.head(self.dropout(output)), hidden

    def step(self, token: torch.Tensor, topic: torch.Tensor, hidden):
        """Один шаг генерации: токен ``(batch,)`` -> логиты ``(batch, vocab)``."""
        token = token.view(-1, 1)
        topic = topic.view(-1)
        embedded = self.embedding(token)
        topic_vector = self.topic_embedding(topic).unsqueeze(1)
        output, hidden = self.lstm(torch.cat([embedded, topic_vector], dim=-1), hidden)
        return self.head(self.dropout(output[:, -1, :])), hidden


# --------------------------------------------------------------------------- #
# Sequence-to-sequence
# --------------------------------------------------------------------------- #

class BahdanauAttention(nn.Module):
    """Аддитивное внимание: ``score = vᵀ·tanh(W_enc·h_enc + W_dec·h_dec)``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = _as_int(hidden_dim, "hidden_dim")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim должен быть > 0, получено {}".format(self.hidden_dim))
        self.encoder_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.decoder_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.score = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вернуть контекст ``(batch, H)`` и веса внимания ``(batch, L)``."""
        projected = self.encoder_projection(encoder_outputs)
        projected = projected + self.decoder_projection(decoder_hidden).unsqueeze(1)
        scores = self.score(torch.tanh(projected)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Seq2SeqLSTM(nn.Module):
    """LSTM-энкодер → LSTM-декодер с вниманием (INTERFACES.md §5).

    Энкодер читает исходную последовательность и возвращает выходы всех позиций
    и последние ``(h, c)``. Декодер на каждом шаге получает эмбеддинг предыдущего
    токена и контекст внимания по выходам энкодера, обновляет ``LSTMCell`` и
    предсказывает следующий токен. Паддинг источника исключается из внимания
    маской по длинам.

    Параметры
    ---------
    vocab_size:
        Общий словарь источника и цели (включая специальные токены).
    embedding_dim:
        Размерность эмбеддинга.
    hidden_dim:
        Размерность скрытого состояния энкодера/декодера.
    num_layers:
        Число слоёв LSTM-энкодера.
    dropout:
        Dropout эмбеддингов и выхода декодера.
    n_topics:
        Число тем (номер ``0`` зарезервирован под «тема не задана»).
    topic_dim:
        Размерность эмбеддинга темы; тема конкатенируется с эмбеддингом токена
        и в энкодере, и в декодере.
    padding_idx:
        Индекс паддинга (0).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_topics: int = 1,
        topic_dim: int = 8,
        padding_idx: int = PAD,
    ):
        super().__init__()
        self.vocab_size = _as_int(vocab_size, "vocab_size")
        self.embedding_dim = _as_int(embedding_dim, "embedding_dim")
        self.hidden_dim = _as_int(hidden_dim, "hidden_dim")
        self.num_layers = _as_int(num_layers, "num_layers")
        self.n_topics = _as_int(n_topics, "n_topics")
        self.topic_dim = _as_int(topic_dim, "topic_dim")
        self.dropout_rate = float(dropout)
        self.padding_idx = _as_int(padding_idx, "padding_idx")

        if self.vocab_size <= FIRST_WORD_INDEX:
            raise ValueError(
                "vocab_size должен быть > {}, получено {}".format(
                    FIRST_WORD_INDEX, self.vocab_size
                )
            )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim должен быть > 0, получено {}".format(self.embedding_dim))
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim должен быть > 0, получено {}".format(self.hidden_dim))
        if self.num_layers <= 0:
            raise ValueError("num_layers должен быть > 0, получено {}".format(self.num_layers))
        if self.n_topics <= 0:
            raise ValueError("n_topics должен быть > 0, получено {}".format(self.n_topics))
        if self.topic_dim <= 0:
            raise ValueError("topic_dim должен быть > 0, получено {}".format(self.topic_dim))

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.topic_embedding = nn.Embedding(self.n_topics + 1, self.topic_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            self.embedding_dim + self.topic_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(self.hidden_dim)
        self.decoder = nn.LSTMCell(
            self.embedding_dim + self.topic_dim + self.hidden_dim, self.hidden_dim
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = nn.Linear(self.hidden_dim, self.vocab_size)

    # ------------------------------------------------------------------ #
    # Кодирование / декодирование
    # ------------------------------------------------------------------ #

    def _topic_vector(self, topics: Optional[torch.Tensor], batch: int,
                      width: int, device) -> torch.Tensor:
        """Эмбеддинг темы, размноженный по позициям: ``(batch, width, topic_dim)``."""
        if topics is None:
            topics = torch.zeros(batch, dtype=torch.long, device=device)
        topics = topics.to(device).view(-1)
        vector = self.topic_embedding(topics)
        return vector.unsqueeze(1).expand(batch, width, -1)

    def encode(self, source: torch.Tensor, lengths: torch.Tensor,
               topics: Optional[torch.Tensor] = None):
        """Закодировать источник: ``(выходы, h, c, маска)``."""
        lengths = lengths.to(torch.long).clamp(min=1)
        embedded = self.embedding(source)
        topic_vector = self._topic_vector(topics, source.size(0), source.size(1), source.device)
        embedded = torch.cat([embedded, topic_vector], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.encoder(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        mask = _sequence_mask(lengths, outputs.size(1), outputs.device)
        return outputs, hidden[-1], cell[-1], mask

    def forward(self, source: torch.Tensor, lengths: torch.Tensor,
                target_input: torch.Tensor,
                topics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher forcing: логиты ``(batch, T, vocab)`` по ``target_input``."""
        encoder_outputs, hidden, cell, mask = self.encode(source, lengths, topics)
        topic_vector = self._topic_vector(topics, source.size(0), 1, source.device).squeeze(1)
        logits = []
        for position in range(target_input.size(1)):
            context, _ = self.attention(hidden, encoder_outputs, mask)
            embedded = self.dropout(self.embedding(target_input[:, position]))
            hidden, cell = self.decoder(
                torch.cat([embedded, topic_vector, context], dim=-1), (hidden, cell)
            )
            logits.append(self.head(self.dropout(hidden)))
        if not logits:
            return torch.zeros(
                (source.size(0), 0, self.vocab_size),
                dtype=torch.float32,
                device=source.device,
            )
        return torch.stack(logits, dim=1)

    def decode_step(self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                    encoder_outputs: torch.Tensor, mask: torch.Tensor,
                    topic: Optional[torch.Tensor] = None):
        """Один шаг декодера: ``(логиты, h, c, веса внимания)``."""
        context, weights = self.attention(hidden, encoder_outputs, mask)
        embedded = self.dropout(self.embedding(token))
        topic_vector = self._topic_vector(topic, token.size(0), 1, token.device).squeeze(1)
        hidden, cell = self.decoder(
            torch.cat([embedded, topic_vector, context], dim=-1), (hidden, cell)
        )
        return self.head(self.dropout(hidden)), hidden, cell, weights


# --------------------------------------------------------------------------- #
# Шаблонный генератор
# --------------------------------------------------------------------------- #

class TemplateGenerator:
    """Генератор на шаблонах (допустимый по условию baseline, INTERFACES.md §5).

    Шаблоны заполняются **только результатами анализатора**: темой, ключевыми
    понятиями и наиболее близким фрагментом корпуса. Это даёт гарантированно
    связный и грамматически корректный текст, но без «свободной» генерации.
    """

    #: Шаблоны; ``{topic}``, ``{keywords}``, ``{evidence}``, ``{prompt}``, ``{source}``.
    TEMPLATES: Tuple[str, ...] = (
        "Тема «{topic}» раскрывается через понятия {keywords}. {evidence}",
        "В корпусе тема «{topic}» описывается словами {keywords}. {evidence}",
        "Судя по корпусу, ключевые понятия темы «{topic}» — {keywords}. {evidence}",
        "По запросу «{prompt}» ближе всего тема «{topic}»: {keywords}. {evidence}",
        "Ближайший документ по теме «{topic}» ({source}) сообщает: {evidence}",
    )

    def __init__(self, templates: Optional[Sequence[str]] = None):
        chosen = tuple(templates) if templates else self.TEMPLATES
        if not chosen:
            raise ValueError("нужен хотя бы один шаблон")
        self.templates = chosen

    @staticmethod
    def _join_keywords(keywords: Sequence[str], limit: int = 3) -> str:
        """Собрать «a, b и c» из ключевых понятий."""
        unique: List[str] = []
        for keyword in keywords:
            keyword = str(keyword).strip()
            if keyword and keyword not in unique:
                unique.append(keyword)
            if len(unique) >= limit:
                break
        if not unique:
            return "ключевые понятия корпуса"
        if len(unique) == 1:
            return unique[0]
        return ", ".join(unique[:-1]) + " и " + unique[-1]

    @staticmethod
    def _as_sentence(evidence: str) -> str:
        """Привести фрагмент к предложению с завершающей точкой."""
        text = " ".join(str(evidence or "").split())
        if not text:
            return ""
        if text[-1] not in ".!?…":
            text += "."
        return text

    def generate(
        self,
        prompt: str = "",
        topic: str = "",
        keywords: Sequence[str] = (),
        evidence: str = "",
        source: str = "",
        variant: int = 0,
    ) -> str:
        """Заполнить шаблон; пустые слоты получают нейтральные значения."""
        if not self.templates:
            return ""
        template = self.templates[int(variant) % len(self.templates)]
        prompt_text = " ".join(str(prompt or "").split())
        if len(prompt_text) > 80:
            prompt_text = prompt_text[:79].rstrip() + "…"
        if not prompt_text:
            prompt_text = "свободный запрос"
        evidence_text = self._as_sentence(evidence)
        if not evidence_text:
            evidence_text = "Корпус не содержит подходящего фрагмента, но тема остаётся «{}».".format(
                str(topic or "без темы")
            )
        text = template.format(
            topic=str(topic or "без темы"),
            keywords=self._join_keywords(keywords),
            evidence=evidence_text,
            prompt=prompt_text,
            source=str(source or "корпус"),
        )
        return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Внутреннее
# --------------------------------------------------------------------------- #

def _as_int(value, name: str) -> int:
    """Привести ``value`` к ``int``; неудачный ввод -> ``ValueError`` (по-русски)."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} должен быть целым числом, получено {!r}".format(name, value)
        ) from exc


def _sequence_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
    """Булева маска ``(batch, max_len)``: ``True`` для реальных позиций."""
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def mean_log_probability(logits: torch.Tensor, targets: torch.Tensor,
                         ignore_index: int = PAD) -> float:
    """Средний логарифм вероятности целевых токенов (для оценки «гладкости»).

    Используется фасадом как fluency-признак при ранжировании кандидатов.
    """
    if logits.numel() == 0 or targets.numel() == 0:
        return -math.inf
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    mask = targets.ne(ignore_index)
    if not bool(mask.any()):
        return -math.inf
    return float(gathered.masked_select(mask).mean().item())
