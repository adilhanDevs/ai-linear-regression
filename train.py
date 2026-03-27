import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# 1) Data
train_df = pd.read_csv("train_energy_data.csv")
test_df = pd.read_csv("test_energy_data.csv")

# 2) Features / target
target = "Energy Consumption"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# 3) Column types
categorical_features = ["Building Type", "Day of Week"]
numeric_features = [
    "Square Footage",
    "Number of Occupants",
    "Appliances Used",
    "Average Temperature"
]

# 4) Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5) Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6) Train
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "energy_model.pkl")
print("Model saved as energy_model.pkl")

# 7) Predict
y_pred = model.predict(X_test)

# 8) Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("R2 score:", round(r2, 4))



# 9) Compare actual vs predicted
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(5))

# 10) Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()