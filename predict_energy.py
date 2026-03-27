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
import pandas as pd
import joblib

try:
    # Load saved model
    model = joblib.load("energy_model.pkl")
except FileNotFoundError:
    print("Модель файлы табылган жок. 'energy_model.pkl' файлынын бар экенин текшериңиз.")
    exit(1)
except Exception as e:
    print(f"Моделди жүктөөдө ката кетти: {e}")
    exit(1)

# User input with error handling
try:
    building_type = input("Building Type (Residential/Commercial/Industrial): ")
    
    # Sanalik maalymatty aluu jana teksheruu
    try:
        square_footage = float(input("Square Footage: "))
    except ValueError:
        print("Ката: Square Footage сан менен көрсөтүлүшү керек. Мисалы: 1500.5")
        exit(1)
    
    try:
        num_occupants = int(input("Number of Occupants: "))
    except ValueError:
        print("Ката: Number of Occupants бүтүн сан менен көрсөтүлүшү керек. Мисалы: 4")
        exit(1)
    
    try:
        appliances_used = int(input("Appliances Used: "))
    except ValueError:
        print("Ката: Appliances Used бүтүн сан менен көрсөтүлүшү керек. Мисалы: 5")
        exit(1)
    
    try:
        avg_temp = float(input("Average Temperature: "))
    except ValueError:
        print("Ката: Average Temperature сан менен көрсөтүлүшү керек. Мисалы: 22.5")
        exit(1)
    
    day_of_week = input("Day of Week: ")
    
    # Sanalardy teksheruu (teris san bolbosun)
    if square_footage <= 0:
        print("Ката: Square Footage 0дон чоң болушу керек.")
        exit(1)
    
    if num_occupants < 0:
        print("Ката: Number of Occupants терс сан болбошу керек.")
        exit(1)
    
    if appliances_used < 0:
        print("Ката: Appliances Used терс сан болбошу керек.")
        exit(1)
    
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

except KeyboardInterrupt:
    print("\nПрограмма колдонуучу тарабынан токтотулду.")
    exit(0)
except Exception as e:
    print(f"Күтүлбөгөн ката кетти: {e}")
    exit(1)
