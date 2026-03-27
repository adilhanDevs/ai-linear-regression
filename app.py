import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Title
st.title('Energy Consumption Prediction')
st.write('This app loads a trained linear regression model and displays a scatter plot of actual vs predicted energy consumption.')

# Load model
model = joblib.load('energy_model.pkl')

# Load test data
test_df = pd.read_csv('test_energy_data.csv')

target = 'Energy Consumption'
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# Predict on test data
y_pred = model.predict(X_test)

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual Energy Consumption')
ax.set_ylabel('Predicted Energy Consumption')
ax.set_title('Actual vs Predicted')
ax.grid(True)

# Display plot in Streamlit
st.pyplot(fig)

# Original results table
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Покажем пример 10 записей из исходного датасета (все поля) — это полезно, чтобы пользователи видели формат данных
st.subheader('Dataset')
st.write(test_df.head(10))



# -----------------------------
# USER INPUT SECTION (at bottom)
# -----------------------------
st.subheader('Enter New Data for Prediction')

# Подготовим поля ввода: для числовых колонок - number_input, для категориальных - selectbox или text_input
user_input = {}

for col in X_test.columns:
    col_series = X_test[col]
    # Числовая колонка
    if pd.api.types.is_numeric_dtype(col_series):
        non_na = col_series.dropna()
        # определим, является ли колонка целочисленной (int dtype или все значения без дробной части)
        is_int_like = pd.api.types.is_integer_dtype(col_series) or (
            (not non_na.empty) and ((non_na % 1 == 0).all())
        )

        if is_int_like:
            default = int(non_na.iloc[0]) if not non_na.empty else 0
            user_input[col] = st.number_input(f'Enter {col}', value=default, step=1)
        else:
            default = float(non_na.iloc[0]) if not non_na.empty else 0.0
            user_input[col] = st.number_input(f'Enter {col}', value=default, format='%.4f')
    else:
        # Категориальная / текстовая колонка: если небольшое число уникальных значений — показываем selectbox
        unique_vals = col_series.dropna().unique().tolist()
        unique_vals_str = [str(v) for v in unique_vals]
        if 0 < len(unique_vals_str) <= 50:
            # используем selectbox с первым элементом по умолчанию
            user_input[col] = st.selectbox(f'Enter {col}', options=unique_vals_str, index=0)
        else:
            # много уникальных значений или пусто — даём текстовое поле
            user_input[col] = st.text_input(f'Enter {col}', value=unique_vals_str[0] if unique_vals_str else '')

# Predict button
if st.button('Predict'):
    # Собираем входной DataFrame и приводим типы числовых колонок
    input_df = pd.DataFrame([user_input])

    for col in X_test.columns:
        if pd.api.types.is_numeric_dtype(X_test[col]):
            # преобразуем вход в числовой тип
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Предсказание
    try:
        user_prediction = model.predict(input_df)[0]
        st.success(f'Predicted Energy Consumption: {user_prediction:.4f}')

        # Добавим новую строку в таблицу результатов (Actual нет для пользовательского ввода)
        new_row = {'Actual': None, 'Predicted': user_prediction}
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        st.error(f'Ошибка при предсказании: {e}')
