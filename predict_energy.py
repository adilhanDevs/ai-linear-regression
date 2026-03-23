import pandas as pd
import joblib

# Load saved model
model = joblib.load("energy_model.pkl")

# User input
building_type = input("Building Type (Residential/Commercial/Industrial): ")
square_footage = float(input("Square Footage: "))
num_occupants = int(input("Number of Occupants: "))
appliances_used = int(input("Appliances Used: "))
avg_temp = float(input("Average Temperature: "))
day_of_week = input("Day of Week: ")

# Create dataframe from input
new_data = pd.DataFrame([{
    "Building Type": building_type,
    "Square Footage": square_footage,
    "Number of Occupants": num_occupants,
    "Appliances Used": appliances_used,
    "Average Temperature": avg_temp,
    "Day of Week": day_of_week
}])

# Predict
prediction = model.predict(new_data)

print("Predicted Energy Consumption:", round(prediction[0], 2))
