# Уровень решения проблемы обработки и генерации естественного языка, ч.4

Расширение семантического анализатора **Лабораторной работы 3** (Doc2Vec) новым
методом сравнения набора документов на схожесть — **нейросетевыми представлениями**.
Реализованы две модели на **PyTorch**: **CNN (TextCNN)** — одномерные свёртки
нескольких размеров ядра + max-over-time pooling (Kim, 2014), и **RNN
(двунаправленный GRU)** — последовательный энкодер с конкатенацией последних
скрытых состояний направлений. Обе модели обучаются **с супервизией** на
классификацию тем корпуса; вектор документа — предпоследний (пулинговый/скрытый)
слой. Сверху построен фасад `NeuralAnalyzer` (обучение, семантический поиск,
близость документов, классификация темы, векторы слов), а сам анализ сравнивается
с LSA-анализатором `SemanticAnalyzer` (`lab_2`) и Doc2Vec-анализатором
`DocVecAnalyzer` (`lab_3`) на одних и тех же запросах.

> **Основной файл лабораторной работы — [`notebooks/lab_4.ipynb`](notebooks/lab_4.ipynb).**
> Он отражает весь ход решения: данные, нормализацию из `lab_1`, TextCNN, GRU,
> семантический поиск, близость документов, классификацию темы, сравнение с LSA и
> Doc2Vec, пользовательский корпус и масштабный пример на отзывах RuReviews. Для
> воспроизводимости все источники случайности фиксируются `seed` внутри `fit`
> (по умолчанию `seed=42`), а обучение ведётся на CPU — два `fit` с одинаковым
> `seed` дают одинаковые веса и векторы.

## Что умеет анализатор

1. **CNN (TextCNN)** — эмбеддинг слов → параллельные свёртки с ядрами `(3, 4, 5)`
   → ReLU → max-pool по времени с маской по длине документа → конкатенация;
2. **RNN (двунаправленный GRU)** — `pack_padded_sequence` игнорирует паддинг;
   вектор документа — конкатенация последних скрытых состояний вперёд/назад;
3. **семантический поиск** — запрос сворачивается тем же энкодером (`encode`) и
   сравнивается по косинусной близости со всеми документами;
4. **близость документов** — ближайшие соседи документа в пространстве представлений;
5. **классификация темы** (`predict`) — вероятности по всем классам (следствие супервизии);
6. **словесная семантика** (`most_similar`, `word_vector`) — по обученной матрице
   эмбеддингов; для OOV возвращается `[]`/`None`, а не исключение;
7. **сравнение с LSA и Doc2Vec** — тот же набор запросов прогоняется через
   `SemanticAnalyzer` (`lab_2`) и `DocVecAnalyzer` (`lab_3`) без переобучения.

## Установка (сборка и развёртывание)

```bash
# из корня проекта (где лежат lab_1/, lab_2/, lab_3/, lab_4/)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch для CPU (компактный wheel, без CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости (numpy/scipy + зависимости lab_1 + pytest/jupyter):
pip install -r lab_4/requirements.txt
```

Зависимости (`lab_4/requirements.txt`): `torch>=2.0` (CPU-сборка достаточна),
`numpy`, `scipy` (косинусная близость из `lab_2`), `matplotlib` (необязательно,
только визуализация в ноутбуке), а также зависимости `lab_1` — `pymorphy3`,
`pymorphy3-dicts-ru`, `nltk`, `pytest`, `jupyter`/`nbconvert`/`ipykernel`.

## Запуск лабораторной работы (ноутбук)

```bash
jupyter notebook lab_4/notebooks/lab_4.ipynb
```

Ноутбук полностью воспроизводим: `Kernel → Restart & Run All` обучает все модели
заново (демо-корпус, пользовательский корпус и подвыборку RuReviews) за
1–2 минуты. Альтернативно — выполнить без GUI:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --inplace lab_4/notebooks/lab_4.ipynb
```

## Использование как библиотеки

```python
import glob
from pathlib import Path
from lab_4.src import NeuralAnalyzer, load_text_files

# 1) обучение CNN на демо-корпусе (или на любом своём наборе .txt-файлов)
analyzer = NeuralAnalyzer("lab_4/src/models/cnn_demo", method="cnn")
paths = sorted(glob.glob("lab_2/data/sample_docs/*.txt"))
texts = load_text_files(paths)
stats = analyzer.fit(texts, sources=[Path(p).name for p in paths],
                     embedding_dim=64, num_filters=64, kernel_sizes=(3, 4, 5),
                     epochs=40, batch_size=8, max_len=48, seed=42)
# stats: {'n_docs': 42, 'n_features': 666, 'n_classes': 6, 'dim': 192,
#         'method': 'cnn', 'epochs': 40, 'accuracy': 1.0, 'seconds': 2.1}

# 2) семантический поиск «своими словами»
for m in analyzer.query("Новый ноутбук с мощным процессором и большим экраном", top_n=2):
    print(round(m.score, 3), m.source)
# 0.81 tech_01.txt
# 0.769 tech_05.txt

# 3) классификация темы
prediction = analyzer.predict("Футбольный матч и тренировка команды")
print(prediction.label, prediction.probabilities)
# sport {'finance': 0.03, 'food': 0.07, 'health': 0.05, 'sport': 0.72, ...}

# 4) близость документов
analyzer.similar(0, top_n=6, exclude_self=True)   # соседи finance_01.txt
```

После обучения модель сохраняется в каталог из трёх файлов (`meta.json`,
`weights.pt`, `docs.json`), поэтому в следующий раз она загружается мгновенно:

```python
from lab_4.src import NeuralAnalyzer
analyzer = NeuralAnalyzer("lab_4/src/models/cnn_demo")   # уже обучена
analyzer.query("Футбольный матч и тренировка команды на стадионе")
```

Второй метод того же фасада — RNN: `NeuralAnalyzer(path, method="rnn")` обучает
двунаправленный GRU, а вектор документа — конкатенация последних скрытых
состояний направлений:

```python
rnn = NeuralAnalyzer("lab_4/src/models/rnn_demo", method="rnn")
rnn.fit(texts, sources=[Path(p).name for p in paths],
        embedding_dim=64, hidden_dim=64, num_layers=1, bidirectional=True,
        epochs=40, batch_size=8, max_len=48, seed=42)
rnn.query("Новый ноутбук с мощным процессором и большим экраном", top_n=1)
# 0.561 tech_02.txt
```

## Три семантических анализатора: примеры работы

Ниже — примеры трёх полученных анализаторов на **одних и тех же данных**
(демо-корпус `lab_2/data/sample_docs/`, 42 документа по 6 темам).

### 1. LSA (`lab_2`, класс `SemanticAnalyzer`)

```python
from lab_4.src import SemanticAnalyzer   # реэкспорт из lab_2 через мост _lab2.py

lsa = SemanticAnalyzer("lab_2/src/models/lsa_demo")   # n_docs=42, k=10
for m in lsa.query("Футбольный матч и тренировка команды", top_n=2):
    print(round(m.score, 3), m.source)
# 0.974 sport_06.txt
# 0.969 sport_02.txt
```

### 2. Doc2Vec (`lab_3`, класс `DocVecAnalyzer`)

```python
from lab_3.src import DocVecAnalyzer

d2v = DocVecAnalyzer("lab_3/src/models/d2v_demo")   # n_docs=42, dim=100
for m in d2v.query("Футбольный матч и тренировка команды", top_n=2):
    print(round(m.score, 3), m.source)
# 0.728 sport_02.txt
# 0.566 sport_06.txt
```

### 3. CNN/RNN (`lab_4`, класс `NeuralAnalyzer`)

```python
from lab_4.src import NeuralAnalyzer

cnn = NeuralAnalyzer("lab_4/src/models/cnn_demo", method="cnn")   # dim=192
rnn = NeuralAnalyzer("lab_4/src/models/rnn_demo", method="rnn")   # dim=128
for m in cnn.query("Футбольный матч и тренировка команды на стадионе", top_n=2):
    print(round(m.score, 3), m.source)
# 0.847 sport_07.txt
# 0.838 sport_06.txt

for m in rnn.query("Футбольный матч и тренировка команды на стадионе", top_n=2):
    print(round(m.score, 3), m.source)
# 0.843 sport_01.txt
# 0.79 sport_05.txt
```

| Запрос (своими словами) | CNN top-1 | GRU top-1 |
| --- | --- | --- |
| «Футбольный матч и тренировка команды на стадионе» | `sport_07.txt` +0.847 | `sport_01.txt` +0.843 |
| «Новый ноутбук с мощным процессором и большим экраном» | `tech_01.txt` +0.810 | `tech_02.txt` +0.561 |
| «Как приготовить вкусный суп из свежих овощей» | `travel_05.txt` +0.781 | `tech_04.txt` +0.380 |

Запросы не совпадают с текстами дословно — анализатор находит документы по смыслу.
Запрос про «суп/овощи» даёт тематически неверный top-1 у CNN и GRU — это **честный
результат**, показывающий ограничение на маленьком корпусе (42 коротких документа,
666 терминов): обучаемых параметров много, а данных мало (то же самое ограничение
демонстрировал Doc2Vec в `lab_3`). При этом классификатор `predict` на этом запросе
даёт верную тему `food` у CNN.

#### Близость документов и словесная семантика

`similar(0, top_n=6, exclude_self=True)` — документ `finance_01.txt` (CNN):

```
[+0.952] finance_05.txt   [+0.949] finance_03.txt   [+0.947] finance_04.txt
[+0.945] finance_02.txt   [+0.945] finance_07.txt   [+0.941] finance_06.txt
```

Все 6 ближайших — своя тема `finance`: документы группируются по смыслу без
подсказок. `most_similar` по обученной матрице эмбеддингов:

```
матч -> стойка(+0.32), лекарство(+0.31), хоккеист(+0.26), гулять(+0.28)
футбол -> OOV, [] (метод не падает)
```

### 4. Сравнение трёх анализаторов на 6 общих запросах

Оба анализатора получают одни и те же запросы; тема документа выводится из его
метки (префикс имени файла до `_`).

| Запрос | LSA | Doc2Vec | CNN | GRU |
| --- | --- | --- | --- | --- |
| Игроки вышли на поле и боролись за победу в важном турнире | sport | sport | sport | sport |
| Мама положила в кастрюлю овощи и поставила на плиту готовиться обед | food | sport | travel | food |
| Программист обновил систему и ускорил работу устройства | tech | tech | tech | tech |
| Туристы забронировали номер и купили билеты на поезд заранее | travel | travel | travel | tech |
| Пациент жалуется на боль и слабость, нужен осмотр врача | health | health | health | health |
| Он вложил сбережения, чтобы получить прибыль через год | finance | finance | tech | health |

- **top-1 agreement: LSA 6/6, Doc2Vec 5/6, CNN 4/6, GRU 4/6.**

На малом демо-корпусе LSA остаётся самым «уверенным» (косинусы 0.89–0.98 и
объяснимые темы), Doc2Vec и CNN/GRU дают близкий результат по темам, а расходится
в основном короткий запрос про «овощи/суп» и перефразированный запрос про
финансы. CNN/GRU, в отличие от LSA/Doc2Vec, дополнительно дают **классификацию
темы** — прямое следствие супервизии.

### 5. Пользовательские данные

Достаточно положить свои документы (по файлу на документ) в каталог и обучить —
**все три анализатора** работают с одним и тем же форматом данных:

```python
import glob
from pathlib import Path
from lab_4.src import NeuralAnalyzer, load_text_files

paths = sorted(glob.glob("lab_2/data/custom_corpus/*.txt"))   # 20 документов, 4 темы
texts = load_text_files(paths)
names = [Path(p).name for p in paths]

cnn_custom = NeuralAnalyzer("lab_4/src/models/cnn_custom", method="cnn")
cnn_custom.fit(texts, sources=names, embedding_dim=64, num_filters=64,
               kernel_sizes=(3, 4, 5), epochs=40, batch_size=8, max_len=48, seed=42)

for q in ["Новый альбом и живой концерт популярной группы",
          "Двигатель и коробка передач автомобиля"]:
    print([(round(m.score, 3), m.source) for m in cnn_custom.query(q, top_n=2)])
# [(0.765, 'music_04.txt'), (0.763, 'music_03.txt')]
# [(0.659, 'auto_01.txt'), (0.658, 'auto_05.txt')]
```

В репозитории есть готовый «пользовательский» корпус `lab_2/data/custom_corpus/`
(20 документов по 4 темам: музыка, автомобили, образование, мода) — его можно
заменить своими файлами. Также корпус можно задать списком строк:

```python
analyzer = NeuralAnalyzer("lab_4/src/models/my_model", method="cnn")
analyzer.fit(["Текст первого документа.", "Текст второго документа."],
             labels=["тема_a", "тема_b"], embedding_dim=32, epochs=20)
```

> `fit` сохраняет модель в каталог, переданный конструктору, — указывайте
> отдельный каталог (не `cnn_demo`), чтобы не перезаписать поставляемую модель.
> Если каталога нет, конструктор возвращает **необученный** экземпляр без
> исключения; вызов `query`/`similar`/`most_similar`/`word_vector`/`doc_vector`
> до `fit` поднимает `RuntimeError` с сообщением «модель не обучена — вызовите
> `fit(...)` или выполните ноутбук». Для обучения нужны **минимум 2 класса**:
> темы берутся из `labels`, либо выводятся из префиксов `sources` (до первого
> `_`/`.`).

### 6. Масштаб на отзывах RuReviews (данные `lab_1`)

Тот же конвейер применяется к сбалансированной подвыборке из **1998 отзывов**
(по ~666 на класс negative/neutral/positive), метки — тональность из `lab_1`:

```python
from collections import defaultdict
from lab_1.src.data import load_corpus
from lab_4.src import NeuralAnalyzer

rows = load_corpus("lab_1/data/rureviews.csv")
by_class = defaultdict(list)
for label, text in rows:
    by_class[label].append(text)
per = 2000 // len(by_class)
reviews, labels = [], []
for label, docs in by_class.items():
    reviews.extend(docs[:per]); labels.extend([label] * per)

analyzer = NeuralAnalyzer("lab_4/src/models/cnn_rureviews", method="cnn")
analyzer.fit(reviews, labels=labels, embedding_dim=64, num_filters=64,
             kernel_sizes=(3, 4, 5), epochs=10, batch_size=32, max_len=48, seed=42)
# n_docs=1998, n_classes=3, accuracy=0.986
```

Запрос «Куртка оказалась мала, пришлось возвращать, но доставка быстрая», top-2
(CNN) — тематика отзывов про одежду/доставку:

```
[+0.869] doc_652  еслиб я знала что будет такое отличие я бы не заказывала. уж…
[+0.869] doc_749  Куртка шла долго. Устала ждать. В итоге пришла с браком. При…
```

Тематиками становятся аспекты отзывов (посадка/размер, доставка, качество) —
разметка тональности нужна лишь как обучающий сигнал. Эти модели
(`lab_4/src/models/cnn_rureviews`, `rnn_rureviews`) воспроизводятся ноутбуком и
не хранятся в git.

### 7. Визуализация

[`lab_4/data/cnn_2d.png`](data/cnn_2d.png) — scatter 42 документов демо-корпуса по
первым двум главным компонентам матрицы `doc_vectors_` CNN (усечённое SVD
центрированной матрицы), раскраска по темам и подписи файлов; **объяснённая
дисперсия первых двух компонент — 35.3 %**. Картинка встроена в ноутбук как вывод
`image/png`.

## Сравнительное описание трёх анализаторов

Три представления — **LSA** (`lab_2`), **Doc2Vec** (`lab_3`) и **CNN/RNN** (`lab_4`)
— устроены принципиально по-разному:

| Критерий | LSA (`lab_2`) | Doc2Vec (`lab_3`) | CNN (TextCNN, `lab_4`) | RNN (GRU, `lab_4`) |
| --- | --- | --- | --- | --- |
| Представление документа | TF-IDF + усечённый SVD (латентные темы) | собственный вектор документа (PV-DBOW) | max-pooled карты свёрток | конкатенация скрытых состояний |
| Математический метод | матричное разложение (SVD) | shallow-сеть, SGD, negative sampling | свёрточная сеть, SGD | рекуррентная сеть, SGD |
| Супервизия | нет | нет | да (тема документа) | да (тема документа) |
| Размерность | `k` тем (10) | `dim` (100) | `num_filters × len(kernels)` (192) | `2 × hidden_dim` (128) |
| Интерпретируемость | высокая: тема = список терминов | средняя: ближайшие слова/документы | средняя: ближайшие слова | средняя: ближайшие слова |
| Скорость обучения | быстрая, детерминированная (0.003 c на демо) | медленнее (2.45 c на демо) | 2.1 c на демо | 5.6 c на демо |
| Что даёт | объяснимые темы, точные косинусы на малых корпусах | семантику слов и документов | классификацию темы + порядок слов | классификацию темы + порядок слов |
| Новый документ | fold-in проекцией | fold-in (`infer_vector`) | `encode` (свёртка + max-pool) | `encode` (GRU) |
| Ограничение | не учитывает порядок слов | не учитывает порядок слов | фиксированное окно свёртки | обучение медленнее |

**Ни одна из трёх моделей не использует CNN/RNN/LSTM, кроме самой lab_4, где
CNN/RNN и есть предмет работы.** LSA — чистая линейная алгебра: TF-IDF-матрица
раскладывается усечённым сингулярным разложением `X ≈ U·S·Vᵀ`, документы
проецируются как `X·Vᵀ`. Doc2Vec — PV-DBOW: однослойная модель, предсказывающая
слова документа по его вектору с negative sampling. TextCNN — свёрточная сеть:
свёртки с ядрами `(3, 4, 5)` захватывают n-граммы слов, max-pool даёт
фиксированный вектор документа. GRU — рекуррентная сеть (LSTM **не
используется**), последовательно обрабатывающая слова и сжимающая документ в
скрытое состояние; обе lab_4-модели обучаются **с супервизией** сквозным
градиентным спуском (Adam) на кросс-энтропии. Никаких sequence-to-sequence
архитектур нет.

Выбор метода — компромисс: **LSA** даёт объяснимые темы и точные косинусы на
малых корпусах без обучения; **Doc2Vec** — семантику слов и документов без
разметки; **CNN/GRU** — обучаемое сквозное представление, учитывающее порядок
слов и дополнительно дающее классификатор темы. На практике на малом демо-корпусе
(42 документа) LSA сильнее (top-1 agreement 6/6), Doc2Vec и CNN/GRU близки
(5/6 и 4/6 соответственно); на подвыборке из 2000 отзывов CNN/GRU дают точность
классификации тональности 0.986/0.975.

## Источник данных и переиспользование lab_1, lab_2 и lab_3

- **Демо-корпус** `../lab_2/data/sample_docs/` — 42 коротких русских документа по
  6 темам (finance 7, food 7, health 7, sport 7, tech 7, travel 7).
- **Пользовательский корпус** `../lab_2/data/custom_corpus/` — 20 документов по
  4 темам (auto 5, edu 5, fashion 5, music 5).
- **RuReviews** ([Smetanin & Komarov, 2019](https://ieeexplore.ieee.org/document/8807792),
  Apache 2.0) переиспользуется из `lab_1` — тексты и метки загружаются через
  `lab_4.src._lab1.load_corpus`.

Код работы целиком лежит в каталоге **`./lab_4`** и **не дублирует** предыдущие
лабораторные:

- нормализация (токенизация, стемминг Snowball, лемматизация pymorphy3, синонимия,
  стоп-слова) делегируется в `lab_1` через мост `lab_4/src/_lab1.py`;
- LSA-анализатор `SemanticAnalyzer`, косинусная близость и загрузчики корпусов
  (`load_text_files`, `load_lines`, `load_reviews`, `doc_tokens`) делегируются в
  `lab_2` через мост `lab_4/src/_lab2.py`;
- Doc2Vec-анализатор `DocVecAnalyzer` (`lab_3`) импортируется напрямую для
  сравнительного описания (модель загружается без переобучения);
- мосты — тонкие реэкспорты, **код `lab_1`, `lab_2`, `lab_3` не модифицируется**.

Токенизация для моделей — `doc_tokens(text, "canonical")` (лемма + синонимы без
стоп-слов), то есть **те же канонические токены**, что и у LSA/Doc2Vec в `lab_2`/
`lab_3`: различие результатов объясняется моделью, а не разной предобработкой. На
демо-корпусе размер словаря в канонических токенах — **666**; доступен и режим
`mode="raw"` (только нижний регистр).

## Структура проекта

```
lab_4/
  src/
    _lab1.py        # мост к lab_1 (нормализация, загрузчик отзывов)
    _lab2.py        # мост к lab_2 (LSA, Match, cosine_similarity, загрузчики, doc_tokens)
    models.py       # TextCNN и GRUEncoder (nn.Module)
    analyzer.py     # NeuralAnalyzer — сквозной фасад
    models/         # обученные модели (meta.json + weights.pt + docs.json)
  notebooks/lab_4.ipynb  # основной файл лабораторной работы
  data/
    cnn_2d.png      # 2D-проекция векторов документов CNN
  requirements.txt  # зависимости
```

## Тесты

```bash
cd /data/Projects/NLP && .venv/bin/python -m pytest lab_4/tests -q
# 30 passed
```

## Формат модели

Каталог модели (например `src/models/cnn_demo/`) состоит из **трёх файлов**:

- `meta.json` — `version`, `method` (`cnn`/`rnn`), `n_docs`, `n_classes`,
  `doc_vector_dim`, `vocabulary` (список терминов в лексикографическом порядке),
  `label_order` и `params` (`embedding_dim`, `hidden_dim`, `num_filters`,
  `kernel_sizes`, `num_layers`, `bidirectional`, `dropout`, `max_len`, `mode`,
  `seed`, `epochs`, `learning_rate`, `batch_size`, `accuracy`);
- `weights.pt` — `torch.save({"state_dict": module.state_dict()})`;
- `docs.json` — `{"texts": [...], "sources": [...], "labels": [...]}`.

`NeuralAnalyzer.load` (и конструктор) проверяет `version == 1` и согласованность
длин (`len(vocabulary) + 1 == embedding.weight.shape[0]`, `n_docs == len(texts) ==
len(labels) == len(sources)`); при несовпадении — `ValueError` с текстом
«несовместимая версия модели, переобучите». Конструктор ловит это `ValueError` и
возвращает **необученный** экземпляр, положив текст ошибки в атрибут `load_error`.
После успешного `load` внутренний модуль восстанавливается целиком, а векторы
документов **пересчитываются** одним `encode`-проходом (без повторного обучения),
поэтому `query`, `similar`, `predict`, `most_similar`, `word_vector` и `doc_vector`
работают без повторного обучения.

```python
import json, shutil, tempfile
from lab_4.src import NeuralAnalyzer

d = tempfile.mkdtemp()                                  # копия модели с чужим version
for f in ("meta.json", "weights.pt", "docs.json"):
    shutil.copy(f"lab_4/src/models/cnn_demo/{f}", d)
meta = json.load(open(f"{d}/meta.json", encoding="utf-8"))
meta["version"] = 2
json.dump(meta, open(f"{d}/meta.json", "w", encoding="utf-8"), ensure_ascii=False)

broken = NeuralAnalyzer(d)
print(broken.load_error)
# 'несовместимая версия модели, переобучите: version=2 в .../meta.json, ожидается 1'
```

Если каталога модели нет вовсе, исключения тоже не возникает: возвращается
необученный анализатор (`load_error is None`), а любое обращение к модели до
`fit`/`load` поднимает `RuntimeError` с сообщением «модель не обучена — вызовите
`fit(...)` или выполните ноутбук».

