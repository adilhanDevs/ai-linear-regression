import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page settings
# ----------------------------
st.title("Energy Consumption Prediction")
st.write(
    "Бул тиркеме колдонуучу киргизген маалымат боюнча "
    "energy consumption маанисин божомолдойт."
)

# ----------------------------
# Load model
# ----------------------------
model = joblib.load("energy_model.pkl")

# ----------------------------
# Feature names
# Model кандай feature'лерди күтөрүн автоматтык алабыз
# ----------------------------
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    st.error(
        "Модель feature_names_in_ маалыматын сактабаптыр. "
        "Feature аттарын кол менен жазыш керек."
    )
    st.stop()

# ----------------------------
# Session state for history
# ----------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------
# User input form
# ----------------------------
st.subheader("Маалыматтарды киргизиңиз")

user_input = {}

with st.form("prediction_form"):
    for feature in feature_names:
        user_input[feature] = st.number_input(
            label=feature,
            value=0.0,
            step=0.1,
            format="%.4f"
        )

    predict_button = st.form_submit_button("Predict")

# ----------------------------
# Predict
# ----------------------------
if predict_button:
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Energy Consumption: {prediction:.4f}")

    # History'ге сактайбыз
    record = user_input.copy()
    record["Predicted Energy Consumption"] = prediction

    st.session_state.prediction_history.append(record)

# ----------------------------
# Show last 10 results
# ----------------------------
if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.subheader("Акыркы 10 prediction")
    st.write(history_df.tail(10).reset_index(drop=True))
