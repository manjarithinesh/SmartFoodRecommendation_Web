import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. LOAD DATASET
# ==============================

file_path = "dataset/diet_recommendations_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==============================
# 2. HANDLE MISSING VALUES
# ==============================

categorical_columns = [
    "Disease_Type",
    "Dietary_Restrictions",
    "Allergies"
]

for column in categorical_columns:
    df[column] = df[column].fillna("None")


# ==============================
# 3. SELECT FEATURES
# ==============================

features = [
    "Age",
    "Gender",
    "Weight_kg",
    "Height_cm",
    "BMI",
    "Disease_Type",
    "Severity",
    "Physical_Activity_Level",
    "Weekly_Exercise_Hours",
    "Daily_Caloric_Intake",
    "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg",
    "Glucose_mg/dL",
    "Dietary_Restrictions",
    "Allergies",
    "Preferred_Cuisine",
    "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score"
]

target = "Diet_Recommendation"

X = df[features]
y = df[target]


# ==============================
# 4. IDENTIFY COLUMN TYPES
# ==============================

categorical_features = [
    "Gender",
    "Disease_Type",
    "Severity",
    "Physical_Activity_Level",
    "Dietary_Restrictions",
    "Allergies",
    "Preferred_Cuisine",
    "Adherence_to_Diet_Plan"
]

numeric_features = [
    "Age",
    "Weight_kg",
    "Height_cm",
    "BMI",
    "Weekly_Exercise_Hours",
    "Daily_Caloric_Intake",
    "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg",
    "Glucose_mg/dL",
    "Dietary_Nutrient_Imbalance_Score"
]


# ==============================
# 5. PREPROCESSING
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ==============================
# 6. CREATE RANDOM FOREST MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==============================
# 7. CREATE PIPELINE
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# 8. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==============================
# 9. TRAIN MODEL
# ==============================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed.")


# ==============================
# 10. TEST MODEL
# ==============================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 11. SAVE MODEL
# ==============================

model_path = "model/recommendation_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully!")
print("Saved file:", model_path)
