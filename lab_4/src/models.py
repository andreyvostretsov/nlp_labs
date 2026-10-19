# -*- coding: utf-8 -*-
"""Нейросети TextCNN и GRUEncoder на PyTorch (INTERFACES.md §3).

Модуль реализует два обучаемых энкодера документа:

* :class:`TextCNN` — CNN по классической схеме Kim (2014): параллельные одномерные
  свёртки нескольких размеров ядра по матрице эмбеддингов слов + ReLU + max-pool
  по времени (с маской по настоящим длинам документов) + конкатенация;
* :class:`GRUEncoder` — двунаправленный (опционально) GRU: ``pack_padded_sequence``
  корректно игнорирует паддинг, вектор документа — конкатенация последних скрытых
  состояний направлений.

Общая механика (INTERFACES.md §3):

* индекс **0 = PAD** (паддинг), реальные токены нумеруются с 1 в лексикографическом
  порядке словаря; ``nn.Embedding(vocab_size, embedding_dim, padding_idx=0)``;
* ``forward(tokens, lengths)`` возвращает логиты классов (для супервизии), а
  ``encode(tokens, lengths)`` — вектор документа (предпоследний слой), который и
  используется фасадом для семантического поиска;
* документ нулевой длины (все токены OOV) даёт нулевой вектор и конечные логиты.

Сериализация в этом модуле отсутствует: сохранение/загрузка весов — задача фасада
(INTERFACES.md §5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TextCNN", "GRUEncoder"]


def _as_int(value, name: str) -> int:
    """Приводит ``value`` к ``int``; неудачный ввод -> ``ValueError`` (по-русски)."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} должен быть целым числом, получено {!r}".format(name, value)
        ) from exc


# --------------------------------------------------------------------------- #
# TextCNN
# --------------------------------------------------------------------------- #

class TextCNN(nn.Module):
    """CNN по схеме Kim (2014): свёртки + max-over-time pooling (INTERFACES.md §3).

    Параметры
    ---------
    vocab_size:
        Общее число строк эмбеддинга **включая PAD** (``len(vocabulary) + 1``).
    embedding_dim:
        Размерность эмбеддинга слов (> 0).
    num_filters:
        Число фильтров на каждый размер ядра (> 0).
    kernel_sizes:
        Кортеж размеров ядер свёртки (непустой; каждый > 0).
    num_classes:
        Число классов классификации (>= 2).
    dropout:
        Вероятность дропаута перед классификатором.
    padding_idx:
        Индекс паддинга (по умолчанию 0).

    Атрибуты после ``__init__``
    ---------------------------
    embedding : nn.Embedding
        Матрица эмбеддингов (``vocab_size × embedding_dim``).
    convs : nn.ModuleList
        Параллельные ``nn.Conv1d`` по одному на размер ядра.
    fc : nn.Linear
        Классификатор ``(num_filters*len(kernel_sizes)) → num_classes``.
    """

    def __init__(self, vocab_size, embedding_dim=64, num_filters=100,
                 kernel_sizes=(3, 4, 5), num_classes=2, dropout=0.0, padding_idx=0):
        super().__init__()
        self.vocab_size = _as_int(vocab_size, "vocab_size")
        self.embedding_dim = _as_int(embedding_dim, "embedding_dim")
        self.num_filters = _as_int(num_filters, "num_filters")
        self.num_classes = _as_int(num_classes, "num_classes")
        self.dropout_rate = float(dropout)
        self.padding_idx = _as_int(padding_idx, "padding_idx")

        try:
            kernels = tuple(int(k) for k in kernel_sizes)
        except (TypeError, ValueError) as exc:
            raise ValueError("kernel_sizes должен быть итерируемым целых чисел") from exc

        if self.vocab_size <= 0:
            raise ValueError("vocab_size должен быть > 0, получено {}".format(self.vocab_size))
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim должен быть > 0, получено {}".format(self.embedding_dim))
        if self.num_filters <= 0:
            raise ValueError("num_filters должен быть > 0, получено {}".format(self.num_filters))
        if self.num_classes < 2:
            raise ValueError("num_classes должен быть >= 2, получено {}".format(self.num_classes))
        if not kernels or any(k <= 0 for k in kernels):
            raise ValueError("kernel_sizes должен быть непустым набором положительных целых")

        self.kernel_sizes = kernels
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim,
                                      padding_idx=self.padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.embedding_dim, self.num_filters, k) for k in self.kernel_sizes]
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), self.num_classes)

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def forward(self, tokens, lengths):
        """Логиты классов формы ``(batch, num_classes)`` (INTERFACES.md §3)."""
        features = self.encode(tokens, lengths)
        return self.fc(features)

    def encode(self, tokens, lengths):
        """Вектор документа = конкатенация max-pooled карт признаков (INTERFACES.md §3).

        ``tokens`` — LongTensor ``(batch, max_len)`` с ``0 = PAD``; ``lengths`` —
        ``(batch,)`` int64 настоящие длины документов. Возвращает ``(batch,
        num_filters*len(kernel_sizes))``.
        """
        embedded = self.embedding(tokens)             # (B, L, D)
        embedded = embedded.transpose(1, 2)           # (B, D, L)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(embedded))        # (B, F, L - k + 1)
            pooled.append(self._masked_max_pool1d(activated, lengths, conv.kernel_size[0]))
        features = torch.cat(pooled, dim=1)           # (B, F * len(kernels))
        return self.dropout(features)

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    @staticmethod
    def _masked_max_pool1d(x, lengths, kernel_size):
        """Max-pool по времени с маской: позиции за пределами длины документа дают 0.

        Позиция ``p`` карты признаков покрывает токены ``[p, p+k-1]``; она валидна,
        только если ``p + k <= length``. Невалидные позиции заменяются на ``-inf`` до
        max-pool; документ без валидных позиций (слишком короткий) даёт нулевой вектор.
        """
        batch, filters, out_len = x.shape
        device = x.device
        positions = torch.arange(out_len, device=device, dtype=torch.int64).unsqueeze(0)
        length_col = lengths.to(device).unsqueeze(1)          # (B, 1)
        # Позиция p окна свёртки покрывает токены [p, p+k-1]; она валидна, если
        # окно начинается раньше конца документа (p < length), то есть покрывает
        # хотя бы один настоящий токен. Позиции p >= length — целиком паддинг.
        mask = (positions < length_col).float()               # (B, out_len)
        mask = mask.unsqueeze(1).expand(batch, filters, out_len)
        masked = x.masked_fill(mask == 0.0, float("-inf"))
        pooled = masked.max(dim=2).values                     # (B, F)
        # Все позиции невалидны (max над -inf -> -inf): заменяем на нулевой вектор.
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


# --------------------------------------------------------------------------- #
# GRUEncoder
# --------------------------------------------------------------------------- #

class GRUEncoder(nn.Module):
    """RNN-энкодер на GRU (LSTM запрещён): последние скрытые состояния (INTERFACES.md §3).

    ``pack_padded_sequence`` корректно игнорирует паддинг; вектор документа —
    конкатенация последних скрытых состояний направлений (вперёд/назад).

    Параметры
    ---------
    vocab_size:
        Общее число строк эмбеддинга включая PAD (``len(vocabulary) + 1``).
    embedding_dim:
        Размерность эмбеддинга слов (> 0).
    hidden_dim:
        Размерность скрытого состояния GRU (> 0).
    num_layers:
        Число слоёв GRU (> 0).
    num_classes:
        Число классов классификации (>= 2).
    dropout:
        Вероятность дропаута (между слоями GRU при ``num_layers > 1`` и перед
        классификатором).
    bidirectional:
        Использовать двунаправленный GRU.
    padding_idx:
        Индекс паддинга (по умолчанию 0).
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_layers=1,
                 num_classes=2, dropout=0.0, bidirectional=True, padding_idx=0):
        super().__init__()
        self.vocab_size = _as_int(vocab_size, "vocab_size")
        self.embedding_dim = _as_int(embedding_dim, "embedding_dim")
        self.hidden_dim = _as_int(hidden_dim, "hidden_dim")
        self.num_layers = _as_int(num_layers, "num_layers")
        self.num_classes = _as_int(num_classes, "num_classes")
        self.dropout_rate = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.padding_idx = _as_int(padding_idx, "padding_idx")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size должен быть > 0, получено {}".format(self.vocab_size))
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim должен быть > 0, получено {}".format(self.embedding_dim))
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim должен быть > 0, получено {}".format(self.hidden_dim))
        if self.num_layers <= 0:
            raise ValueError("num_layers должен быть > 0, получено {}".format(self.num_layers))
        if self.num_classes < 2:
            raise ValueError("num_classes должен быть >= 2, получено {}".format(self.num_classes))

        self.num_directions = 2 if self.bidirectional else 1
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim,
                                      padding_idx=self.padding_idx)
        # dropout между слоями GRU имеет смысл только при num_layers > 1.
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.num_classes)

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def forward(self, tokens, lengths):
        """Логиты классов формы ``(batch, num_classes)`` (INTERFACES.md §3)."""
        features = self.encode(tokens, lengths)
        return self.fc(features)

    def encode(self, tokens, lengths):
        """Вектор документа = конкатенация последних скрытых состояний (INTERFACES.md §3).

        Возвращает ``(batch, hidden_dim * (2 if bidirectional else 1))``. Документ
        нулевой длины даёт нулевой вектор.
        """
        embedded = self.embedding(tokens)                     # (B, L, D)
        batch, max_len, _ = embedded.shape
        device = embedded.device
        lengths_cpu = lengths.detach().cpu()
        clamped = torch.clamp(lengths_cpu, min=1, max=max_len)

        # Сортируем по убыванию длины (требование pack_padded_sequence), затем
        # возвращаем h_n в исходный порядок батча.
        sorted_lengths, sorted_idx = clamped.sort(descending=True)
        sorted_embedded = embedded.index_select(0, sorted_idx.to(device))
        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_embedded, sorted_lengths, batch_first=True, enforce_sorted=True
        )
        _, h_n = self.gru(packed)                             # (num_layers*D, B, H)
        _, unsorted_idx = sorted_idx.sort()
        h_n = h_n.index_select(1, unsorted_idx.to(device))

        start = (self.num_layers - 1) * self.num_directions
        last_layer = h_n[start:start + self.num_directions]   # (D, B, H)
        features = last_layer.transpose(0, 1).contiguous()    # (B, D, H)
        features = features.view(batch, self.hidden_dim * self.num_directions)
        features = self.dropout(features)

        # Документы нулевой длины (все токены OOV) обнуляются.
        empty_mask = (lengths <= 0).to(device).float().unsqueeze(1)
        return features * (1.0 - empty_mask)

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return "GRUEncoder(embedding_dim={}, hidden_dim={}, num_layers={}, bidirectional={})".format(
            self.embedding_dim, self.hidden_dim, self.num_layers, self.bidirectional
        )
