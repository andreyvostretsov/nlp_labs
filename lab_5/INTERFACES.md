# INTERFACES.md — замороженный контракт интерфейсов проекта `lab_5`

Единственный источник истины для сигнатур, форматов данных и схемы модели.
Все модули обязаны следовать ему дословно.

## Язык и окружение
- Python 3.9+.
- Зависимости (`lab_5/requirements.txt`): `torch>=2.0` (CPU-сборка достаточна),
  `numpy`, `scipy` (косинусная близость из `lab_2`), `pymorphy3`,
  `pymorphy3-dicts-ru`, `nltk`, `pytest`, `jupyter`, `nbconvert`, `ipykernel`.
  `matplotlib` необязателен (только визуализация в ноутбуке, импорт в `try/except`).
  **`gensim`/`sklearn` НЕ используются.**
- Генерация строится **только на основе** семантического анализатора `lab_4`:
  корпус, темы, словарь и представления документов берутся из него. Внешние
  предобученные языковые модели и сетевые API не используются.
- Обязательные классы нейросетей: **LSTM** (языковая модель и seq2seq) и
  **sequence-to-sequence** (LSTM-энкодер → LSTM-декодер с вниманием).
  Шаблонный генератор разрешён условием и используется как baseline и кандидат
  гибридного ранжирования.
- Нормализация и разбиение на предложения **переиспользуются из `lab_1`** через
  мост `lab_5/src/_lab1.py`; LSA и загрузчики — из `lab_2`; Doc2Vec — из `lab_3`;
  CNN/GRU-анализатор — из `lab_4`.

## Глобальные ограничения
- Код `lab_1` … `lab_4` НЕ модифицируется; доступ только через мосты.
- Обучение на CPU, воспроизводимость: фиксируются `torch.manual_seed(seed)` и
  `numpy.random.seed(seed)`, порядок документов/предложений стабилен; два `fit`
  с одинаковым `seed` дают одинаковые веса и тексты (при одинаковом сэмплировании).
- Пустой/мусорный ввод не приводит к исключениям, кроме документированных
  `ValueError` (неизвестная тема/режим, пустой корпус, < 2 тем).

---

## 1. `lab_5/src/_lab1.py` — мост к `lab_1`
Добавляет корень проекта в `sys.path` и реэкспортирует: `tokenize`,
`sent_tokenize`, `normalize_text`, `normalize_token`, `canonical_tokens`,
`NormalizedToken`, `STOPWORDS`, `load_corpus`, `SentimentAnalyzer`,
`SentimentResult`.

## 2. `lab_5/src/_lab2.py` — мост к `lab_2`
```python
from lab_2.src.analyzer import SemanticAnalyzer, Match
from lab_2.src.lsa import cosine_similarity
from lab_2.src.corpus import load_text_files, load_lines, load_reviews, doc_tokens
```

## 3. `lab_5/src/_lab3.py` — мост к `lab_3`
```python
from lab_3.src.analyzer import DocVecAnalyzer
```

## 4. `lab_5/src/_lab4.py` — мост к `lab_4`
```python
from lab_4.src.analyzer import NeuralAnalyzer, Prediction, Match
from lab_4.src.models import TextCNN, GRUEncoder
```

---

## 5. `lab_5/src/generators.py` — LSTM, seq2seq и шаблоны (PyTorch)

### 5.1 Специальные токены и словарь
```python
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
FIRST_WORD_INDEX = 4

def build_vocabulary(sequences) -> dict[str, int]:
    """Словарь {токен: индекс}; слова сортируются лексикографически с индекса 4."""

def encode_tokens(tokens, vocabulary, *, add_bos=False, add_eos=False,
                  max_len=None, keep_unknown=True) -> list[int]: ...

def pad_token_sequences(sequences, max_len=None) -> tuple[Tensor, Tensor]:
    """(LongTensor (n, L), длины (n,)); 0 = PAD; пустой вход -> (0,1)/(0,)."""

def decode_ids(ids, vocabulary) -> list[str]:
    """Индексы -> слова без специальных токенов."""

def format_tokens_as_text(tokens, *, sentence=True) -> str:
    """Склеить словоформы, поставить заглавную букву и завершающую точку."""
```

### 5.2 LSTM-языковая модель
```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                 num_layers=2, dropout=0.1, n_topics=1, topic_dim=16,
                 padding_idx=0): ...
    def forward(self, tokens, topics=None, hidden=None):
        """logits (batch, L, vocab); тема конкатенируется с эмбеддингом токена."""
    def step(self, token, topic, hidden):
        """Один шаг: logits (batch, vocab), новое hidden."""
```
Тема — номер от `1` до `n_topics`; `0` = тема не задана. Эмбеддинг темы
конкатенируется с эмбеддингом слова на каждом шаге.

### 5.3 Sequence-to-sequence
```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim): ...
    def forward(self, decoder_hidden, encoder_outputs, mask):
        """context (batch, H), weights (batch, L); паддинг замаскирован."""

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=96,
                 num_layers=1, dropout=0.1, n_topics=1, topic_dim=8,
                 padding_idx=0): ...
    def encode(self, source, lengths, topics=None):
        """(выходы, h, c, маска); тема конкатенируется с эмбеддингами источника."""
    def forward(self, source, lengths, target_input, topics=None):
        """Teacher forcing: logits (batch, T, vocab)."""
    def decode_step(self, token, hidden, cell, encoder_outputs, mask, topic=None):
        """(logits, h, c, веса внимания); тема подаётся и в декодер."""
```
Энкодер — `nn.LSTM`; декодер — `nn.LSTMCell`; вход декодера —
`[эмбеддинг слова; эмбеддинг темы; контекст внимания]`. Паддинг источника
исключается маской. Документ нулевой длины в `encode` не допускается (длины
клампятся к 1).

### 5.4 Шаблонный генератор
```python
class TemplateGenerator:
    TEMPLATES: tuple[str, ...]  # 5 шаблонов
    def __init__(self, templates=None): ...
    def generate(self, prompt="", topic="", keywords=(), evidence="",
                 source="", variant=0) -> str: ...
```
Слоты: `{topic}`, `{keywords}`, `{evidence}`, `{prompt}`, `{source}`. Пустой
`evidence` заменяется нейтральной фразой, пустой `prompt` — «свободный запрос».
Шаблоны заполняются **только** результатами анализатора.

---

## 6. `lab_5/src/analyzer.py` — фасад `GeneratingAnalyzer`

### 6.1 Результаты
```python
@dataclass
class Candidate:
    text: str
    method: str          # "template" | "lstm" | "seq2seq"
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
    text: str
    method: str          # запрошенный режим
    topic: str
    score: float
    prompt: str = ""
    candidates: list[Candidate] = []
```

### 6.2 Фасад
```python
class GeneratingAnalyzer:
    MODEL_VERSION = 1
    NOT_FITTED_MESSAGE = "модель не обучена — вызовите fit(...) или выполните ноутбук"
    INCOMPATIBLE_MARKER = "несовместимая версия модели, переобучите"
    GEN_METHODS = ("template", "lstm", "seq2seq", "hybrid")

    def __init__(self, model_dir=None, semantic_method="cnn"): ...
    def fit(self, texts, *, labels=None, sources=None,
            semantic_method=None,
            semantic_epochs=40, semantic_batch_size=8, semantic_max_len=48,
            semantic_embedding_dim=64, semantic_hidden_dim=64,
            semantic_num_filters=64,
            embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.1,
            topic_dim=16, max_len=32,
            seq2seq_hidden_dim=96, seq2seq_num_layers=1,
            epochs=40, seq2seq_epochs=60, batch_size=8,
            learning_rate=1e-3, seq2seq_learning_rate=1e-3,
            max_src_len=48, max_tgt_len=24,
            mode="raw", seed=42) -> dict
    def generate(self, prompt="", *, topic=None, method="hybrid",
                 max_words=24, temperature=0.8, top_k=20,
                 num_candidates=3, no_repeat_ngram_size=2,
                 constrain=True, min_words=4, seed=42) -> Generation
    def summarize(self, text, *, max_words=32, temperature=0.0,
                  min_words=4, seed=42) -> str
    def continue_text(self, prompt, *, max_words=24, temperature=0.8,
                      top_k=20, no_repeat_ngram_size=2, min_words=4,
                      seed=42) -> str
    def template(self, prompt="", *, topic=None, variant=0) -> str
    def evaluate(self, text, *, prompt="", topic=None) -> dict
    def save(self, model_dir=None) -> str
    @classmethod
    def load(cls, model_dir) -> "GeneratingAnalyzer"
    # свойства: is_fitted, semantic_analyzer, n_docs, n_features, dim,
    #           n_topics, topics, params, metrics
    # метод: fit_stats() -> dict
```

- `semantic_method` ∈ `{"cnn", "rnn"}`; иначе `ValueError`.
- `model_dir` по умолчанию `<пакет>/models/gen_cnn_demo` или
  `<пакет>/models/gen_rnn_demo`. Если каталог содержит модель — загрузить; если
  каталога нет — **необученный** экземпляр; если модель несовместима —
  необученный экземпляр с текстом ошибки в `load_error`.
- `generate`/`summarize`/`continue_text`/`template`/`evaluate` до `fit` →
  `RuntimeError` с `NOT_FITTED_MESSAGE`.
- `method` вне `GEN_METHODS` → `ValueError`; неизвестная `topic` → `ValueError`.
- `generate` с пустым `prompt` определяет тему по явному `topic`, иначе по самой
  частой теме корпуса; `generate` с непустым `prompt` — по классификатору
  анализатора, при ошибке — по самой частой теме.
- `summarize("")` → `""`; seq2seq-модель реферирования отсутствует →
  `RuntimeError`.
- `fit`: `texts` пусто/`None` → `ValueError`; менее 2 тем → `ValueError`;
  пустой словарь → `ValueError`. `labels`/`sources` должны совпадать по длине
  с `texts` (иначе `ValueError`).
- `fit` обучает: (1) семантическое ядро `lab_4` и сохраняет его в `semantic/`;
  (2) LSTM-языковую модель на **отдельных предложениях** корпуса;
  (3) две seq2seq-модели: `lead` (документ → первое предложение) и `next`
  (предложение → следующее предложение). Возвращает `fit_stats()`.

### 6.3 Декодирование и ранжирование
- Сэмплирование: `temperature` (`0` — greedy), `top_k`, штраф за повтор
  n-грамм (`no_repeat_ngram_size`), минимальная длина `min_words` (EOS
  запрещён, пока фраза не завершена и не закончилась на «плохом» слове).
- `constrain=True` — n-граммные ограничения корпуса: следующий токен выбирается
  только из переходов, встречавшихся в предложениях корпуса (сначала триграмма,
  затем биграмма); если ограничения отсекают все слова, они игнорируются.
- Гибридный режим собирает кандидатов `template` (2 варианта), `lstm`
  (`num_candidates`) и `seq2seq` (`num_candidates`) и сортирует по
  `score = 0.28*semantic + 0.22*topic_confidence + 0.12*fluency +
  0.20*coherence + 0.10*uniqueness + 0.08*completeness`, где
  `semantic` — косинус к запросу/центроиду темы (векторы `lab_4`),
  `topic_confidence` — вероятность темы по классификатору `lab_4`,
  `fluency` — `exp(средний log P)` LSTM,
  `coherence` — доля биграмм текста, встречающихся в корпусе,
  `uniqueness` — доля уникальных слов,
  `completeness` — штраф за обрыв на предлоге/союзе или длину < 4.
- Возвращается лучший кандидат и все кандидаты в `Generation.candidates`.

---

## 7. `lab_5/src/suite.py` — сводка четырёх анализаторов
```python
@dataclass
class AnalyzerView:
    name: str
    title: str
    label: str | None = None
    probabilities: dict[str, float] = {}
    top: list[tuple[str, float]] = []
    note: str = ""

class SemanticSuite:
    def __init__(self, project_root=None, corpus="demo"): ...
    @property
    def available(self) -> dict[str, bool]: ...
    @property
    def errors(self) -> dict[str, str]: ...
    def analyze(self, query, top_n=3) -> dict[str, AnalyzerView]: ...
    def compare(self, queries, top_n=3) -> list[dict]: ...
    def topic_agreement(self, queries) -> dict[str, float]: ...
```
`corpus` ∈ `{"demo", "custom"}`; иначе `ValueError`. Загружаются:
`lab_1/src/models/sentiment_model.json`, `lab_2/src/models/lsa_<corpus>`,
`lab_3/src/models/d2v_<corpus>`, `lab_4/src/models/cnn_<corpus>` и
`lab_4/src/models/rnn_<corpus>`. Ошибки загрузки не бросаются, а попадают в
`errors`. `analyze` возвращает записи для всех имён
`("nb", "lsa", "doc2vec", "cnn", "rnn")`; недоступные — с пояснением в `note`.
`topic_agreement` сравнивает темы с LSA (опорный анализатор).

---

## 8. `lab_5/src/__init__.py` — публичный API
```python
from .generators import (PAD, BOS, EOS, UNK, SPECIAL_TOKENS,
                         LSTMLanguageModel, Seq2SeqLSTM, BahdanauAttention,
                         TemplateGenerator, build_vocabulary, encode_tokens,
                         decode_ids, format_tokens_as_text)
from .analyzer import Candidate, Generation, GeneratingAnalyzer
from .suite import AnalyzerView, SemanticSuite
from ._lab1 import SentimentAnalyzer, sent_tokenize, canonical_tokens, load_corpus
from ._lab2 import (SemanticAnalyzer, cosine_similarity, load_text_files,
                    load_lines, load_reviews, doc_tokens)
from ._lab3 import DocVecAnalyzer
from ._lab4 import NeuralAnalyzer, Prediction, TextCNN, GRUEncoder
__version__ = "0.1.0"
```

---

## 9. Формат модели (каталог = 3 файла + `semantic/`, `version == 1`)

`meta.json`:
```json
{
  "version": 1,
  "semantic_method": "cnn",
  "n_docs": 42,
  "n_topics": 6,
  "topics": ["finance", "food", "health", "sport", "tech", "travel"],
  "vocabulary": ["<pad>", "<bos>", "<eos>", "<unk>", "термин1", "..."],
  "params": {
    "semantic_method": "cnn", "semantic_epochs": 40,
    "semantic_batch_size": 8, "semantic_max_len": 48,
    "semantic_embedding_dim": 64, "semantic_hidden_dim": 64,
    "semantic_num_filters": 64,
    "embedding_dim": 64, "hidden_dim": 96, "num_layers": 2,
    "dropout": 0.1, "topic_dim": 16, "max_len": 24,
    "seq2seq_hidden_dim": 64, "seq2seq_num_layers": 1,
    "epochs": 80, "seq2seq_epochs": 80, "batch_size": 8,
    "learning_rate": 0.001, "seq2seq_learning_rate": 0.001,
    "max_src_len": 48, "max_tgt_len": 24, "mode": "raw", "seed": 42
  },
  "metrics": {
    "semantic_accuracy": 1.0, "lm_loss": 0.858,
    "lm_perplexity": 2.36, "lm_token_accuracy": 0.903,
    "lead_pairs": 42, "next_pairs": 84,
    "lead_loss": 1.303, "next_loss": 0.848,
    "lead_token_accuracy": 0.72, "next_token_accuracy": 0.81,
    "lm_history": [6.8, 5.9, "..."], "lead_history": [6.9, 5.7, "..."],
    "next_history": [6.9, 5.6, "..."], "seconds": 51.5
  }
}
```
`weights.pt` — `torch.save({"lm": state_dict, "lead": state_dict|None,
"next": state_dict|None})`.
`corpus.json` — `{"texts": [...], "sources": [...], "labels": [...]}`.
`semantic/` — каталог модели `lab_4` (`meta.json`, `weights.pt`, `docs.json`).

`GeneratingAnalyzer.load` проверяет `version == 1`, `semantic_method`,
порядок специальных токенов, `n_topics == len(topics)`,
`n_docs == len(texts) == len(sources) == len(labels)`, наличие и обученность
семантического ядра, наличие состояний `lm`; при нарушении — `ValueError` с
`INCOMPATIBLE_MARKER`. Конструктор ловит это и кладёт текст в `load_error`.

---

## 10. Формат пользовательских данных
Идентичен `lab_2` §7: `load_text_files`/`load_lines` — UTF-8, один документ
на файл/строку; `load_reviews` — корпус `lab_1`. Для обучения нужны минимум
две темы: явные `labels` или имена файлов вида `<тема>_<номер>.txt`.

## 11. Тестовый bootstrap
`lab_5/tests/conftest.py` добавляет корень проекта в `sys.path`; тесты
импортируют как `from lab_5.src.<module> import ...`.
