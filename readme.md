# Прогнозирование энергопотребления (линейная регрессия)

Небольшой проект на Python для обучения простой линейной регрессии, предсказания энергопотребления здания и веб-интерфейса на Streamlit для визуализации и быстрого ввода данных.

Файлы в репозитории:

- `train.py` — скрипт для обучения модели на `train_energy_data.csv` и сохранения обученной модели в `energy_model.pkl`.
- `predict_energy.py` — скрипт для интерактивного предсказания в терминале (загружает `energy_model.pkl`).
- `app.py` — Streamlit-приложение, которое загружает `energy_model.pkl`, строит график "Actual vs Predicted" и позволяет вводить новые примеры для предсказания.
- `train_energy_data.csv`, `test_energy_data.csv` — примеры данных (CSV). В репозитории также может быть `test_energy_data.csv` и `train_energy_data.csv`.
- `requirements.txt` — список зависимостей Python.

Что делает проект

1. Обучает простую модель (Pipeline: предобработка + LinearRegression) в `train.py`.
2. Сохраняет модель в `energy_model.pkl`.
3. `predict_energy.py` позволяет вводить параметры здания в терминале и получает предсказание энергопотребления.
4. `app.py` — Streamlit-приложение для визуализации и быстрых прогнозов через веб-интерфейс.

Установка

Рекомендуется создать виртуальное окружение и установить зависимости из `requirements.txt`.

```bash
# macOS / zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если в `requirements.txt` нет необходимых пакетов, минимальный набор включает:

- pandas
- scikit-learn
- joblib
- matplotlib
- streamlit

Запуск обучения (train.py)

Скрипт `train.py` читает `train_energy_data.csv`, обучает модель и сохраняет её в `energy_model.pkl`.

```bash
python train.py
```

После успешного выполнения в папке появится `energy_model.pkl`.

Запуск локального терминального предсказания (predict_energy.py)

Скрипт `predict_energy.py` загружает `energy_model.pkl` и запрашивает у пользователя значения признаков. Просто запустите:

```bash
python predict_energy.py
```

Следуйте подсказкам для ввода значений (например: Building Type, Square Footage, Number of Occupants, Appliances Used, Average Temperature, Day of Week). Скрипт выведет предсказанное энергопотребление.

Запуск Streamlit-приложения (app.py)

Streamlit-приложение отображает график фактических значений против предсказанных по `test_energy_data.csv` и предоставляет форму для ввода новых данных.

```bash
streamlit run app.py
```

Ожидается, что `energy_model.pkl` и `test_energy_data.csv` находятся в той же директории, что и `app.py`.

Формат данных

Ожидаемые названия столбцов в CSV (на основе кода):

- Building Type (категориальная)
- Square Footage (числовая)
- Number of Occupants (числовая)
- Appliances Used (числовая)
- Average Temperature (числовая)
- Day of Week (категориальная)
- Energy Consumption (целевой столбец)

Примечания и возможные улучшения

- Добавить валидацию входных данных и unit-тесты.
- Добавить Dockerfile для удобного развёртывания.
- Добавить скрипт/Makefile для автоматизации установки окружения и запуска.
- При большом наборе данных рекомендуется использовать кросс-валидацию и масштабирование признаков.
