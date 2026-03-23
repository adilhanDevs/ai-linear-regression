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

# Predict
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

# Optionally show a table of first few predictions
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
st.subheader('Sample Predictions')
st.write(results.head())
