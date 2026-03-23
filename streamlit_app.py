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

st.subheader('Sample Predictions')
st.write(results.head())

# -----------------------------
# USER INPUT SECTION (at bottom)
# -----------------------------
st.subheader('Enter New Data for Prediction')

# Колдонуучу киргизе турган input талаалар
user_input = {}

for col in X_test.columns:
    user_input[col] = st.number_input(
        f'Enter {col}',
        value=float(X_test[col].iloc[0]) if pd.api.types.is_numeric_dtype(X_test[col]) else 0.0
    )

# Predict button
if st.button('Predict'):
    input_df = pd.DataFrame([user_input])
    user_prediction = model.predict(input_df)[0]

    st.success(f'Predicted Energy Consumption: {user_prediction:.4f}')

    # Жаңы prediction'ды results таблицасына кошобуз
    new_row = {'Actual': None, 'Predicted': user_prediction}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    st.subheader('Last 10 Results')
    st.write(results.tail(10))
else:
    st.subheader('Last 10 Results')
    st.write(results.tail(10))
