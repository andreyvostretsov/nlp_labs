# Лабораторная работа 1:  уровень решения проблемы обработки и генерации естественного языка ч.1

Учебный проект по обработке естественного языка (NLP). Реализует полный конвейер:

1. **Токенизация** произвольного документа или набора документов;
2. **Нормализация** словаря — нижний регистр, стемминг (Snowball, русский),
   лемматизация (pymorphy3), синонимия (каноническая форма), удаление стоп-слов;
3. **Анализатор тональности** — мультиномиальный наивный байесовский классификатор
   (3 класса: `negative` / `neutral` / `positive`), реализованный с нуля на
   стандартной библиотеке Python (без numpy / scikit-learn).

> **Основной файл лабораторной работы — [`notebooks/lab_1.ipynb`](notebooks/lab_1.ipynb).**
> Он отражает весь ход решения: описание данных, откуда они загружены и из чего
> состоят, токенизация, нормализация, словарь, живое обучение модели на RuReviews,
> оценка точности, предсказания и пример на пользовательских данных.

## Установка

```bash
# из корня проекта
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Зависимости (`requirements.txt`): `pymorphy3`, `pymorphy3-dicts-ru`, `nltk`,
`pytest`, а также `jupyter`/`nbconvert`/`ipykernel` для запуска ноутбука.

## Запуск лабораторной работы (ноутбук)

```bash
jupyter notebook notebooks/lab_1.ipynb
```


## Использование как библиотеки

```python
from src import tokenize, canonical_tokens, build_vocabulary
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()  # загружает lab_1/models/sentiment_model.json
result = analyzer.analyze("Отличный товар, очень доволен, рекомендую!")
print(result.label, result.probabilities)
# positive {'negative': 0.0363, 'neutral': 0.0087, 'positive': 0.9551}

# обучение на своём корпусе CSV (label,text; все три класса обязательны)
stats = analyzer.train("my_reviews.csv", "lab_1/models/my_model.json")
print(stats)  # {'accuracy': 0.8333, 'n_train': 18, 'n_test': 6}
```

Скачать полный датасет RuReviews (~21 МБ) из Python:

```python
from src.data import download_rureviews

download_rureviews("data/rureviews.csv")
```

## Пайплайн нормализации

Каждый токен проходит: нижний регистр → стемминг (`SnowballStemmer('russian')`) →
лемматизация (`pymorphy3.MorphAnalyzer`) → приведение синонимов к канонической
лемме (`resources/synonyms_ru.json`). Стоп-слова (`resources/stopwords_ru.txt`)
помечаются и отбрасываются при формировании признаков. Признаками классификатора
служат канонические леммы без стоп-слов; перед обучением редкие слова (встречающиеся
менее чем в `min_df` документах, по умолчанию 3) отбрасываются. Порог адаптивен —
`max(1, min(min_df, n_train // 10))`, поэтому малые корпуса не обнуляются.

## Источник данных и лицензия

Предобученная модель обучена на датасете **RuReviews**
([Smetanin & Komarov, 2019](https://ieeexplore.ieee.org/document/8807792)) —
отзывы о товарах, 3 сбалансированных класса (≈30 000 на класс), лицензия
**Apache 2.0**. Зеркало: <https://github.com/sismetanin/rureviews>. В исходном
файле нейтральный класс записан с опечаткой `neautral` — загрузчик приводит его
к каноническому `neutral`.

Полный файл (`data/rureviews.csv`) скачивается функцией `download_rureviews(...)`;
в репозиторий вложена компактная выборка `lab_1/resources/train_data.csv`
(~1000 отзывов) для офлайн-воспроизводимости.

## Структура проекта

```
lab_1/
  tokenizer.py        # токенизация и разбиение на предложения
  normalization.py    # стемминг, лемматизация, синонимия, стоп-слова
  vocabulary.py       # нормализованный словарь набора документов
  naive_bayes.py      # мультиномиальный наивный Байес (с нуля)
  sentiment.py        # SentimentAnalyzer — сквозной конвейер
  data.py             # загрузка датасета/корпуса, выборки, разбиение
  resources/          # stopwords_ru.txt, synonyms_ru.json, train_data.csv
  models/             # sentiment_model.json (предобученная модель)
notebooks/lab_1.ipynb  # основной файл лабораторной работы
data/sample_docs/     # демо-документы разной тональности
data/rureviews.csv    # полный датасет (скачивается: download_rureviews)
tests/                # pytest-тесты
INTERFACES.md         # контракт интерфейсов (сигнатуры, схема модели)
requirements.txt      # зависимости
```

## Тесты

```bash
python -m pytest tests/ -q
```

## Формат модели

`lab_1/models/sentiment_model.json` — JSON со схемой `version: 1`:

```json
{
  "version": 1,
  "alpha": 1.0,
  "classes": ["negative", "neutral", "positive"],
  "prior_log": {"negative": -1.098, "neutral": -1.098, "positive": -1.098},
  "cond_log": {"negative": {"отличный": -2.1}, "neutral": {}, "positive": {}},
  "vocab_size": 3257,
  "label_order": ["negative", "neutral", "positive"]
}
```

## Сборка архива проекта (опционально)

```bash
zip -r lab1_sentiment.zip lab_1 notebooks data/sample_docs tests README.md INTERFACES.md requirements.txt
```
