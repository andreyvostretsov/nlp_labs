# -*- coding: utf-8 -*-
"""Мосты lab_5 к lab_1–lab_4 (INTERFACES.md §1–§4)."""
from lab_5.src._lab1 import (  # noqa: F401
    STOPWORDS,
    canonical_tokens,
    load_corpus,
    sent_tokenize,
    tokenize,
)
from lab_5.src._lab2 import (  # noqa: F401
    Match,
    SemanticAnalyzer,
    cosine_similarity,
    doc_tokens,
    load_lines,
    load_reviews,
    load_text_files,
)
from lab_5.src._lab3 import DocVecAnalyzer
from lab_5.src._lab4 import GRUEncoder, NeuralAnalyzer, Prediction, TextCNN


def test_lab1_bridge():
    assert tokenize("Привет, мир!") == ["привет", "мир"]
    assert sent_tokenize("Первое. Второе!") == ["Первое.", "Второе!"]
    assert canonical_tokens("Отличные товары") == ["хороший", "товар"]
    assert isinstance(STOPWORDS, frozenset)
    assert callable(load_corpus)


def test_lab2_bridge():
    assert callable(load_text_files)
    assert callable(load_lines)
    assert callable(load_reviews)
    assert callable(doc_tokens)
    assert callable(cosine_similarity)
    assert SemanticAnalyzer is not None
    assert Match is not None


def test_lab3_bridge():
    assert DocVecAnalyzer.__name__ == "DocVecAnalyzer"


def test_lab4_bridge():
    assert NeuralAnalyzer.__name__ == "NeuralAnalyzer"
    assert Prediction is not None
    assert TextCNN.__name__ == "TextCNN"
    assert GRUEncoder.__name__ == "GRUEncoder"
