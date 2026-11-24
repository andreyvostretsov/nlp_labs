# -*- coding: utf-8 -*-
"""Тесты генераторов: словарь, LSTM-языковая модель, seq2seq, шаблоны."""
import torch

from lab_5.src.generators import (
    BOS,
    EOS,
    FIRST_WORD_INDEX,
    PAD,
    SPECIAL_TOKENS,
    UNK,
    BahdanauAttention,
    LSTMLanguageModel,
    Seq2SeqLSTM,
    TemplateGenerator,
    build_vocabulary,
    decode_ids,
    encode_tokens,
    format_tokens_as_text,
    pad_token_sequences,
)


def test_vocabulary_starts_with_special_tokens():
    vocabulary = build_vocabulary([["б", "а"], ["в"]])
    assert tuple(vocabulary[token] for token in SPECIAL_TOKENS) == (PAD, BOS, EOS, UNK)
    assert vocabulary["а"] == FIRST_WORD_INDEX
    assert vocabulary["б"] == FIRST_WORD_INDEX + 1
    assert vocabulary["в"] == FIRST_WORD_INDEX + 2


def test_encode_decode_roundtrip_and_unknown():
    vocabulary = build_vocabulary([["матч", "гол"]])
    encoded = encode_tokens(["матч", "неизвестно", "гол"], vocabulary,
                            add_bos=True, add_eos=True)
    assert encoded[0] == BOS and encoded[-1] == EOS
    assert UNK in encoded
    assert decode_ids(encoded, vocabulary) == ["матч", "гол"]


def test_encode_max_len_keeps_eos():
    vocabulary = build_vocabulary([["a", "b", "c"]])
    encoded = encode_tokens(["a", "b", "c"], vocabulary,
                            add_bos=True, add_eos=True, max_len=3)
    assert len(encoded) == 3
    assert encoded[-1] == EOS


def test_pad_token_sequences():
    tensor, lengths = pad_token_sequences([[1, 2, 3], [4]])
    assert tensor.shape == (2, 3)
    assert tensor.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert lengths.tolist() == [3, 1]
    empty, empty_lengths = pad_token_sequences([])
    assert empty.shape[0] == 0 and empty_lengths.shape[0] == 0


def test_format_tokens_as_text():
    assert format_tokens_as_text(["новый", "ноутбук"]) == "Новый ноутбук."
    assert format_tokens_as_text(["готово!"]) == "Готово!"
    assert format_tokens_as_text([]) == ""


def test_lstm_language_model_shapes_and_step():
    vocabulary = build_vocabulary([["a", "b", "c"]])
    model = LSTMLanguageModel(len(vocabulary), embedding_dim=8, hidden_dim=12,
                              num_layers=2, n_topics=2, topic_dim=4)
    tokens = torch.tensor([[BOS, 4, 5, EOS], [BOS, 6, EOS, PAD]])
    topics = torch.tensor([1, 2])
    logits, hidden = model(tokens, topics)
    assert logits.shape == (2, 4, len(vocabulary))
    assert torch.isfinite(logits).all()
    single_logits, single_hidden = model(tokens[:1], topics[:1])
    step_logits, _ = model.step(torch.tensor([4]), torch.tensor([2]), single_hidden)
    assert step_logits.shape == (1, len(vocabulary))


def test_seq2seq_forward_and_decode_step():
    vocabulary = build_vocabulary([["a", "b", "c"]])
    model = Seq2SeqLSTM(len(vocabulary), embedding_dim=8, hidden_dim=12,
                        n_topics=3, topic_dim=4)
    source = torch.tensor([[4, 5, EOS], [6, EOS, PAD]])
    lengths = torch.tensor([3, 2])
    target_input = torch.tensor([[BOS, 4, 5], [BOS, 6, EOS]])
    topics = torch.tensor([1, 2])
    logits = model(source, lengths, target_input, topics)
    assert logits.shape == (2, 3, len(vocabulary))
    assert torch.isfinite(logits).all()

    outputs, hidden, cell, mask = model.encode(source, lengths, topics)
    assert outputs.shape == (2, 3, 12)
    assert hidden.shape == (2, 12)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    step_logits, _, _, weights = model.decode_step(
        torch.tensor([BOS, BOS]), hidden, cell, outputs, mask, topics
    )
    assert step_logits.shape == (2, len(vocabulary))
    assert weights.shape == (2, 3)


def test_attention_masks_padding():
    attention = BahdanauAttention(8)
    encoder_outputs = torch.randn(1, 4, 8)
    hidden = torch.randn(1, 8)
    mask = torch.tensor([[True, True, False, False]])
    context, weights = attention(hidden, encoder_outputs, mask)
    assert context.shape == (1, 8)
    assert float(weights[0, 2].item()) < 1e-6
    assert float(weights[0, 3].item()) < 1e-6


def test_template_generator_variants_and_empty_evidence():
    generator = TemplateGenerator()
    text = generator.generate(prompt="футбол", topic="sport",
                              keywords=["матч", "гол"], evidence="Игрок забил гол",
                              source="sport_01.txt", variant=0)
    assert "sport" in text and "матч" in text and "Игрок забил гол." in text
    fallback = generator.generate(topic="sport", variant=1)
    assert "Корпус не содержит" in fallback or "sport" in fallback
    assert generator.generate(variant=99)
