import pandas as pd
import joblib

# Load trained model
model_path = "model/recommendation_model.pkl"
model = joblib.load(model_path)

print("Trained model loaded successfully!")


# Sample user information
user_data = pd.DataFrame([{
    "Age": 19,
    "Gender": "Female",
    "Weight_kg": 52,
    "Height_cm": 159,
    "BMI": 20.57,
    "Disease_Type": "Healthy",
    "Severity": "Moderate",
    "Physical_Activity_Level": "Moderate",
    "Weekly_Exercise_Hours": 5.0,
    "Daily_Caloric_Intake": 2200,
    "Cholesterol_mg/dL": 170,
    "Blood_Pressure_mmHg": 120,
    "Glucose_mg/dL": 90,
    "Dietary_Restrictions": "None",
    "Allergies": "None",
    "Preferred_Cuisine": "Indian",
    "Adherence_to_Diet_Plan": 0.8,
    "Dietary_Nutrient_Imbalance_Score": 1.0
}])


# Make prediction
prediction = model.predict(user_data)

print("\n==============================")
print("PERSONALIZED RECOMMENDATION")
print("==============================")

print("Predicted Diet Recommendation:", prediction[0])
