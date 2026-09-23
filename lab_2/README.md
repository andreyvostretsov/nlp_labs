# Лабораторная работа 2: семантический анализатор (Latent Semantic Analysis, LSA)

Прототип латентно-семантического анализа естественного языка, построенный на
результатах **Практической работы №1** (`../lab_1`). Анализатор строит
**TF-IDF**-векторы документов, вычисляет **векторы тем** усечённым сингулярным
разложением (SVD) и отвечает на запросы семантического поиска и близости
документов. Нормализация (токенизация, **стемминг** Snowball, лемматизация
pymorphy3, синонимия, стоп-слова) **переиспользуется из `lab_1`** и не дублируется.

> **Основной файл лабораторной работы — [`notebooks/lab_2.ipynb`](notebooks/lab_2.ipynb).**
> Он отражает весь ход решения: данные, нормализацию из `lab_1`, TF-IDF, SVD и
> темы, семантический поиск, близость документов, влияние стемминга и примеры на
> пользовательских данных. Для воспроизводимости тем перед каждым `fit` в ноутбуке
> вызывается `np.random.seed(42)` (стартовый вектор ARPACK-решателя SVD случаен).

## Что умеет анализатор

1. **TF-IDF**-представление корпуса (разреженная матрица `scipy.sparse`, формулы
   `tf = 1 + ln(count)` и `idf = ln((1+N)/(1+df)) + 1`, L2-нормировка строк);
2. **векторы тем** — усечённое SVD `X ≈ U·S·Vᵀ`, документы проецируются как
   `X·Vᵀ`; термины темы — по убыванию модуля нагрузки;
3. **семантический поиск** — запрос проецируется (fold-in) и сравнивается по
   косинусной близости со всеми документами;
4. **близость документов** — ближайшие соседи документа в латентном пространстве;
5. **интерпретация тем** (`topics`, `topic_weights`).

## Установка

```bash
# из корня проекта (где лежат lab_1/ и lab_2/)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r lab_2/requirements.txt
```

Зависимости (`lab_2/requirements.txt`): `numpy`, `scipy` (линейная алгебра: TF-IDF
и SVD), `matplotlib` (необязательно, только визуализация в ноутбуке), а также
зависимости `lab_1` — `pymorphy3`, `pymorphy3-dicts-ru`, `nltk`, `pytest`,
`jupyter`/`nbconvert`/`ipykernel`.

> Python 3.9+ (для Python 3.9 `scipy` ставится версии `<1.14`).

## Запуск лабораторной работы (ноутбук)

```bash
jupyter notebook lab_2/notebooks/lab_2.ipynb
```

Ноутбук полностью воспроизводим: `Kernel → Restart & Run All` обучает все модели
заново (демо-корпус, пользовательский корпус и подвыборку RuReviews) за
1–2 минуты. Альтернативно — выполнить без GUI:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --inplace lab_2/notebooks/lab_2.ipynb
```

## Использование как библиотеки

```python
import glob
from pathlib import Path
from lab_2.src import load_text_files, SemanticAnalyzer

# 1) обучение на демо-корпусе (или на любом своём наборе .txt-файлов)
analyzer = SemanticAnalyzer("lab_2/src/models/lsa_demo")
paths = sorted(glob.glob("lab_2/data/sample_docs/*.txt"))
stats = analyzer.fit(load_text_files(paths), k=10, sources=[Path(p).name for p in paths])
# stats: {'n_docs': 42, 'n_features': 666, 'k': 10, 'sparsity': 0.965,
#         'explained_variance_ratio': [...], 'seconds': 0.003}

# 2) семантический поиск
for m in analyzer.query("Новый ноутбук с мощным процессором", top_n=3):
    print(m.score, m.source, m.text[:60])
# 0.969 tech_01.txt  Новый ноутбук получил быстрый процессор и экран...
# 0.916 tech_02.txt  Процессор нового поколения содержит больше ядер...

# 3) близость документов и темы
analyzer.similar(0, top_n=5, exclude_self=True)   # соседи документа 0
analyzer.topics(n_terms=8)                        # топ-термины каждой темы
analyzer.topic_weights("Футбольный матч")         # {тема: вес, ...}
```

После обучения модель сохраняется в каталог из трёх файлов (`meta.json`,
`svd.npz`, `docs.json`), поэтому в следующий раз она загружается мгновенно:

```python
from lab_2.src import SemanticAnalyzer
analyzer = SemanticAnalyzer("lab_2/src/models/lsa_demo")   # уже обучена
analyzer.query("Как приготовить суп из овощей")
```

## Примеры работы

### Семантический поиск по демо-корпусу (42 документа, 6 тем)

| Запрос (своими словами) | Лучший документ | Близость |
| --- | --- | --- |
| «Футбольный матч и тренировка команды» | `sport_06.txt` | 0.974 |
| «Как приготовить вкусный суп из свежих овощей» | `food_01.txt` | 0.894 |
| «Новый ноутбук с мощным процессором» | `tech_01.txt` | 0.969 |

Запросы не совпадают с текстами дословно — LSA находит документы по смыслу.

### Векторы тем (k=10, примеры топ-терминов)

- тема 0: `матч, команда, тренер, игрок, игра` — спорт;
- тема 4: `программа, ноутбук, файл, система` — техника;
- тема 5: `поездка, отель, билет, турист, гора` — путешествия;
- тема 6: `доход, акция, платёж, банк` — финансы;
- тема 9: `добавлять, подавать, духовка, вкус` — кулинария.

### Пользовательские данные

Достаточно положить свои документы (по файлу на документ) в каталог и обучить:

```python
import glob
from pathlib import Path
from lab_2.src import load_text_files, SemanticAnalyzer

paths = sorted(glob.glob("lab_2/data/custom_corpus/*.txt"))
analyzer = SemanticAnalyzer("lab_2/src/models/lsa_custom")
analyzer.fit(load_text_files(paths), k=8, sources=[Path(p).name for p in paths])

for m in analyzer.query("Двигатель и коробка передач автомобиля", top_n=2):
    print(m.score, m.source)   # 0.995 auto_01.txt; 0.934 auto_02.txt
```

В репозитории есть готовый «пользовательский» корпус `lab_2/data/custom_corpus/`
(20 документов по 4 темам: музыка, автомобили, образование, мода) — его можно
заменить своими файлами. Также корпус можно задать списком строк:

```python
analyzer = SemanticAnalyzer("lab_2/src/models/my_model")   # отдельный каталог
analyzer.fit(["Текст первого документа.", "Текст второго документа."], k=4)
```

> `fit` сохраняет модель в каталог, переданный конструктору, — указывайте
> отдельный каталог (не `lsa_demo`), чтобы не перезаписать поставляемую модель.
> Для двух документов эффективное число тем равно `k = 1` (ограничение ранга).

### Масштаб на отзывах RuReviews (данные lab_1)

Тот же конвейер без разметки применяется к отзывам `../lab_1/data/rureviews.csv`
(см. ноутбук, раздел 13): `load_reviews(..., limit=2000)` → `fit(k=40, min_df=5)`.
Темами становятся аспекты отзывов (доставка, размер, качество, ткань и т.п.).

## Источник данных и переиспользование lab_1

- **Демо-корпус** `data/sample_docs/` — 42 коротких русских документа по 6 темам
  (сгенерированы в рамках работы; сценарий `data/_gen_task4.py`).
- **Пользовательский корпус** `data/custom_corpus/` — 20 документов по 4 темам.
- **RuReviews** ([Smetanin & Komarov, 2019](https://ieeexplore.ieee.org/document/8807792),
  Apache 2.0) переиспользуется из `lab_1` — тексты загружаются через
  `lab_2.src.corpus.load_reviews`, который вызывает загрузчик `lab_1`
  (`lab_1.src.data.load_corpus`).

Нормализация (`doc_tokens(text, "canonical")` — лемма + синонимы без стоп-слов)
делегируется в `lab_1` через мост `lab_2/src/_lab1.py`. На демо-корпусе размер
словаря по режимам: `raw` (только нижний регистр) — 923, `lemma` — 675,
`canonical` (лемма + синонимы) — 666, `stem` — 660; нормализация сводит
словоформы («футболист»/«футболисты» → «футболист»).

## Структура проекта

```
lab_2/
  src/
    _lab1.py        # мост к lab_1 (нормализация, загрузчик отзывов)
    corpus.py       # загрузка корпусов, doc_tokens (режимы canonical/stem/lemma/raw)
    vectorizer.py   # TfidfVectorizer (scipy.sparse)
    lsa.py          # LsaModel (усечённый SVD), cosine_similarity
    analyzer.py     # SemanticAnalyzer — сквозной фасад
    models/         # обученные модели (meta.json + svd.npz + docs.json)
  notebooks/lab_2.ipynb  # основной файл лабораторной работы
  data/
    sample_docs/    # демо-корпус (42 документа, 6 тем)
    custom_corpus/  # пользовательский корпус (20 документов, 4 темы)
  tests/            # pytest-тесты
  INTERFACES.md     # контракт интерфейсов (сигнатуры, схема модели)
  requirements.txt  # зависимости
```

## Тесты

```bash
cd /data/Projects/NLP && .venv/bin/python -m pytest lab_2/tests -q
# 228 passed
```

## Формат модели

Каталог модели (например `src/models/lsa_demo/`) состоит из трёх файлов:

- `meta.json` — `version`, `k`, `n_docs`, параметры векторизатора, `vocabulary`,
  `idf`;
- `svd.npz` — `vt` (k×m), `s` (k), `doc_vectors` (n×k);
- `docs.json` — `texts`, `sources` (тексты документов для сниппетов).

`SemanticAnalyzer.load` проверяет `version == 1` и согласованность длин; при
несовпадении — `ValueError` («несовместимая версия модели, переобучите»).

## Сборка архива проекта (опционально)

```bash
zip -r lab2_lsa.zip lab_2
```
