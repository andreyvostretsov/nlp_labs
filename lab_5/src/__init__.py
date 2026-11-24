# -*- coding: utf-8 -*-
"""`lab_5` — генерация осмысленного текста на основе семантического анализатора
`lab_4` (LSTM + sequence-to-sequence, PyTorch).

Публичный API пакета (INTERFACES.md §8). Нормализация и разбиение на предложения
переиспользуются из `lab_1`; LSA-анализатор и загрузчики — из `lab_2`; Doc2Vec —
из `lab_3`; семантическое ядро CNN/GRU — из `lab_4`. Собственные LSTM-модель,
seq2seq-модель и шаблонный генератор реализованы в :mod:`lab_5.src.generators`,
фасад — в :mod:`lab_5.src.analyzer`.
"""
from .generators import (
    PAD,
    BOS,
    EOS,
    UNK,
    SPECIAL_TOKENS,
    LSTMLanguageModel,
    Seq2SeqLSTM,
    BahdanauAttention,
    TemplateGenerator,
    build_vocabulary,
    encode_tokens,
    decode_ids,
    format_tokens_as_text,
)
from .analyzer import Candidate, Generation, GeneratingAnalyzer
from .suite import AnalyzerView, SemanticSuite
from ._lab1 import (  # noqa: F401
    SentimentAnalyzer,
    sent_tokenize,
    canonical_tokens,
    load_corpus,
)
from ._lab2 import (  # noqa: F401
    SemanticAnalyzer,
    cosine_similarity,
    load_text_files,
    load_lines,
    load_reviews,
    doc_tokens,
)
from ._lab3 import DocVecAnalyzer  # noqa: F401
from ._lab4 import NeuralAnalyzer, Prediction, TextCNN, GRUEncoder  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "PAD",
    "BOS",
    "EOS",
    "UNK",
    "SPECIAL_TOKENS",
    "LSTMLanguageModel",
    "Seq2SeqLSTM",
    "BahdanauAttention",
    "TemplateGenerator",
    "build_vocabulary",
    "encode_tokens",
    "decode_ids",
    "format_tokens_as_text",
    "Candidate",
    "Generation",
    "GeneratingAnalyzer",
    "AnalyzerView",
    "SemanticSuite",
    "SentimentAnalyzer",
    "sent_tokenize",
    "canonical_tokens",
    "load_corpus",
    "SemanticAnalyzer",
    "cosine_similarity",
    "load_text_files",
    "load_lines",
    "load_reviews",
    "doc_tokens",
    "DocVecAnalyzer",
    "NeuralAnalyzer",
    "Prediction",
    "TextCNN",
    "GRUEncoder",
    "__version__",
]
