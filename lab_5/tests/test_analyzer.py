# -*- coding: utf-8 -*-
"""Тесты фасада GeneratingAnalyzer: обучение, генерация, персистенция."""
import json
import shutil
from pathlib import Path

import pytest

from lab_5.src import GeneratingAnalyzer

TEXTS = [
    "Футбольный матч собрал полный стадион. Игрок забил решающий гол. Болельщики радовались победе.",
    "Тренер выбрал осторожную схему. Команда больше внимания уделяла обороне. Нападающий забил гол.",
    "Новый ноутбук получил быстрый процессор. Программа запускается мгновенно. Корпус остался тонким.",
    "Производитель показал новый смартфон. Экран стал ярче. Система работает без задержки.",
    "Врач осмотрел пациента. Боль усилилась к вечеру. Нужно принимать лекарство.",
    "Пациент жалуется на слабость. Осмотр показал простуду. Врач рекомендовал отдых.",
    "Туристы забронировали номер. Билеты купили заранее. Отель оказался рядом с морем.",
    "Поездка в горы планировалась долго. Маршрут выбрали опытные туристы. Погода была ясной.",
]
LABELS = ["sport", "sport", "tech", "tech", "health", "health", "travel", "travel"]
SOURCES = [
    "sport_01.txt", "sport_02.txt", "tech_01.txt", "tech_02.txt",
    "health_01.txt", "health_02.txt", "travel_01.txt", "travel_02.txt",
]

TINY = dict(
    semantic_epochs=4,
    epochs=5,
    seq2seq_epochs=5,
    embedding_dim=16,
    hidden_dim=24,
    num_layers=1,
    topic_dim=4,
    max_len=16,
    seq2seq_hidden_dim=16,
    seq2seq_num_layers=1,
    batch_size=2,
    semantic_embedding_dim=16,
    semantic_hidden_dim=16,
    semantic_num_filters=8,
    semantic_max_len=16,
    seed=42,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("gen_model")
    analyzer = GeneratingAnalyzer(str(model_dir), semantic_method="cnn")
    analyzer.fit(TEXTS, labels=LABELS, sources=SOURCES, **TINY)
    return analyzer


def test_fit_stats(trained):
    stats = trained.fit_stats()
    assert stats["n_docs"] == len(TEXTS)
    assert stats["n_classes"] == 4
    assert stats["semantic_method"] == "cnn"
    assert stats["topics"] == ["health", "sport", "tech", "travel"]
    assert stats["vocab_size"] > 4
    assert stats["next_pairs"] > 0
    assert trained.is_fitted
    assert trained.dim == 24
    assert trained.n_topics == 4
    assert trained.topics == ["health", "sport", "tech", "travel"]
    assert trained.params["semantic_method"] == "cnn"
    assert "lm_history" in trained.metrics


def test_generate_all_methods(trained):
    for method in ("template", "lstm", "seq2seq", "hybrid"):
        result = trained.generate("футбольный матч", topic="sport", method=method,
                                  max_words=8, min_words=3, seed=7)
        assert isinstance(result.text, str) and result.text
        assert result.topic == "sport"
        assert 0.0 <= result.score <= 1.0
        assert result.candidates
        for candidate in result.candidates:
            assert candidate.method in ("template", "lstm", "seq2seq")
            assert 0.0 <= candidate.score <= 1.0


def test_generate_is_deterministic_with_seed(trained):
    first = trained.generate("новый ноутбук", topic="tech", method="lstm",
                             max_words=8, seed=11)
    second = trained.generate("новый ноутбук", topic="tech", method="lstm",
                              max_words=8, seed=11)
    assert first.text == second.text


def test_generate_unknown_topic_and_method(trained):
    with pytest.raises(ValueError):
        trained.generate("текст", topic="unknown")
    with pytest.raises(ValueError):
        trained.generate("текст", method="unknown")


def test_generate_infers_topic(trained):
    result = trained.generate("футбольный матч и гол", method="template")
    assert result.topic in trained.topics


def test_summarize_and_empty(trained):
    summary = trained.summarize(TEXTS[0], max_words=6, min_words=2)
    assert isinstance(summary, str) and summary
    assert trained.summarize("") == ""


def test_continue_text_contains_prompt(trained):
    text = trained.continue_text("новый ноутбук", max_words=6, min_words=2, seed=3)
    assert "ноутбук" in text.lower()


def test_evaluate_keys(trained):
    report = trained.evaluate("Игрок забил гол на стадионе", prompt="футбольный матч",
                              topic="sport")
    assert set(report) >= {
        "score", "semantic", "topic_confidence", "fluency",
        "coherence", "uniqueness", "completeness", "predicted_topic", "topic",
    }
    assert 0.0 <= report["score"] <= 1.0


def test_template_uses_topic_evidence(trained):
    text = trained.template("футбольный матч", topic="sport")
    assert "sport" in text
    assert any(word in text.lower() for word in ("матч", "гол", "команд", "стадион"))


def test_save_load_roundtrip(trained, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(trained.model_dir, target)
    loaded = GeneratingAnalyzer(str(target), semantic_method="cnn")
    assert loaded.is_fitted and loaded.load_error is None
    assert loaded.topics == trained.topics
    assert loaded.n_docs == trained.n_docs
    first = trained.generate("футбольный матч", topic="sport", method="hybrid",
                             max_words=8, min_words=3, seed=5).text
    second = loaded.generate("футбольный матч", topic="sport", method="hybrid",
                             max_words=8, min_words=3, seed=5).text
    assert first == second
    assert loaded.summarize(TEXTS[0], max_words=6, min_words=2)


def test_missing_model_is_unfitted(tmp_path):
    analyzer = GeneratingAnalyzer(str(tmp_path / "none"), semantic_method="cnn")
    assert not analyzer.is_fitted
    assert analyzer.load_error is None
    for call in (
        lambda: analyzer.generate("текст"),
        lambda: analyzer.summarize("текст"),
        lambda: analyzer.continue_text("текст"),
        lambda: analyzer.template("текст"),
        lambda: analyzer.evaluate("текст"),
    ):
        with pytest.raises(RuntimeError):
            call()


def test_incompatible_version_reports_load_error(trained, tmp_path):
    target = tmp_path / "broken"
    shutil.copytree(trained.model_dir, target)
    meta_path = target / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["version"] = 99
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    analyzer = GeneratingAnalyzer(str(target), semantic_method="cnn")
    assert not analyzer.is_fitted
    assert GeneratingAnalyzer.INCOMPATIBLE_MARKER in (analyzer.load_error or "")


def test_fit_validation_errors(tmp_path):
    analyzer = GeneratingAnalyzer(str(tmp_path / "a"), semantic_method="cnn")
    with pytest.raises(ValueError):
        analyzer.fit([])
    with pytest.raises(ValueError):
        analyzer.fit(["один текст", "другой текст"], labels=["one", "one"])
    with pytest.raises(ValueError):
        analyzer.fit(["текст"], labels=["one", "two"])


def test_fit_without_next_pairs(tmp_path):
    texts = [
        "Футбольный матч собрал стадион.",
        "Тренер выбрал осторожную схему.",
        "Новый ноутбук получил процессор.",
        "Программа запускается мгновенно.",
    ]
    labels = ["sport", "sport", "tech", "tech"]
    analyzer = GeneratingAnalyzer(str(tmp_path / "one_sentence"), semantic_method="cnn")
    stats = analyzer.fit(texts, labels=labels, **TINY)
    assert stats["next_pairs"] == 0
    assert analyzer.is_fitted
    assert analyzer.generate("футбольный матч", topic="sport", method="seq2seq",
                             max_words=6, min_words=2, seed=1).text
    assert analyzer.summarize(texts[0], max_words=6, min_words=2)


def test_rnn_semantic_core(tmp_path):
    analyzer = GeneratingAnalyzer(str(tmp_path / "rnn"), semantic_method="rnn")
    stats = analyzer.fit(TEXTS, labels=LABELS, **TINY)
    assert stats["semantic_method"] == "rnn"
    assert analyzer.generate("футбольный матч", topic="sport", method="template").text
