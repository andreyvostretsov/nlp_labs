# -*- coding: utf-8 -*-
"""Тесты сводки четырёх анализаторов SemanticSuite."""
from lab_5.src.suite import ANALYZER_NAMES, SemanticSuite


def test_demo_suite_loads_four_analyzers():
    suite = SemanticSuite(corpus="demo")
    available = suite.available
    for name in ("nb", "lsa", "doc2vec", "cnn", "rnn"):
        assert available[name], (name, suite.errors)


def test_analyze_returns_all_views():
    suite = SemanticSuite(corpus="demo")
    views = suite.analyze("Футбольный матч и тренировка команды", top_n=2)
    assert set(views) == set(ANALYZER_NAMES)
    for name in ANALYZER_NAMES:
        view = views[name]
        assert view.name == name
        assert view.title
    assert views["nb"].label in ("negative", "neutral", "positive")
    assert views["lsa"].top and views["lsa"].label
    assert views["cnn"].top and views["cnn"].label


def test_compare_and_agreement():
    suite = SemanticSuite(corpus="demo")
    queries = [
        "Футбольный матч и тренировка команды",
        "Новый ноутбук с мощным процессором",
        "Как приготовить суп из овощей",
    ]
    rows = suite.compare(queries)
    assert len(rows) == len(queries)
    for row in rows:
        assert set(row) == {"query", "nb", "lsa", "doc2vec", "cnn", "rnn"}
    agreement = suite.topic_agreement(queries)
    assert set(agreement) == {"lsa", "doc2vec", "cnn", "rnn"}
    assert agreement["lsa"] == 1.0


def test_custom_suite():
    suite = SemanticSuite(corpus="custom")
    assert all(suite.available[name] for name in ("lsa", "doc2vec", "cnn", "rnn"))
    views = suite.analyze("Новый альбом и живой концерт популярной группы", top_n=2)
    assert views["lsa"].label == "music"
    assert views["cnn"].label in {"music", "fashion", "edu", "auto"}


def test_invalid_corpus():
    try:
        SemanticSuite(corpus="unknown")
    except ValueError as exc:
        assert "corpus" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ожидался ValueError")
