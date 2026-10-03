from flask import Flask, render_template, request
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from ml.recommendation_engine import generate_recommendation


app = Flask(__name__)


# ==========================================================
# LOAD TRAINED RANDOM FOREST MODEL
# ==========================================================

model = joblib.load("model/recommendation_model.pkl")


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

DATASET_PATH = "dataset/diet_recommendations_dataset.csv"

FEATURES = [
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

TARGET = "Diet_Recommendation"


CATEGORICAL_FEATURES = [
    "Gender",
    "Disease_Type",
    "Severity",
    "Physical_Activity_Level",
    "Dietary_Restrictions",
    "Allergies",
    "Preferred_Cuisine"
]


NUMERIC_FEATURES = [
    "Age",
    "Weight_kg",
    "Height_cm",
    "BMI",
    "Weekly_Exercise_Hours",
    "Daily_Caloric_Intake",
    "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg",
    "Glucose_mg/dL",
    "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score"
]


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# PERSONALIZED PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ------------------------------
    # USER INPUT
    # ------------------------------

    name = request.form["name"]

    age = int(request.form["age"])

    gender = request.form["gender"]

    weight = float(request.form["weight"])

    height = float(request.form["height"])

    disease_type = request.form["disease_type"]

    severity = request.form["severity"]

    activity = request.form["activity"]

    exercise_hours = float(
        request.form["exercise_hours"]
    )

    calories = int(
        request.form["calories"]
    )

    cholesterol = float(
        request.form["cholesterol"]
    )

    blood_pressure = int(
        request.form["blood_pressure"]
    )

    glucose = float(
        request.form["glucose"]
    )

    nutrient_score = float(
        request.form["nutrient_score"]
    )

    adherence = float(
        request.form["adherence"]
    )

    restrictions = request.form["restrictions"]

    allergies = request.form["allergies"]

    cuisine = request.form["cuisine"]


    # ------------------------------
    # BMI
    # ------------------------------

    height_m = height / 100

    bmi = weight / (height_m * height_m)

    bmi = round(bmi, 2)


    if bmi < 18.5:

        bmi_category = "Underweight"

    elif bmi < 25:

        bmi_category = "Healthy Weight"

    elif bmi < 30:

        bmi_category = "Overweight"

    else:

        bmi_category = "Obesity"


    # ------------------------------
    # CREATE USER DATA
    # ------------------------------

    user_data = pd.DataFrame([{

        "Age": age,

        "Gender": gender,

        "Weight_kg": weight,

        "Height_cm": height,

        "BMI": bmi,

        "Disease_Type": disease_type,

        "Severity": severity,

        "Physical_Activity_Level": activity,

        "Weekly_Exercise_Hours": exercise_hours,

        "Daily_Caloric_Intake": calories,

        "Cholesterol_mg/dL": cholesterol,

        "Blood_Pressure_mmHg": blood_pressure,

        "Glucose_mg/dL": glucose,

        "Dietary_Restrictions": restrictions,

        "Allergies": allergies,

        "Preferred_Cuisine": cuisine,

        "Adherence_to_Diet_Plan": adherence,

        "Dietary_Nutrient_Imbalance_Score": nutrient_score

    }])


    # ------------------------------
    # ML PREDICTION
    # ------------------------------

    prediction = model.predict(user_data)

    recommendation = prediction[0]


    # ------------------------------
    # MODEL PROBABILITIES
    # ------------------------------

    probabilities = model.predict_proba(user_data)[0]

    classes = model.classes_


    prediction_probabilities = []


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        prediction_probabilities.append({

            "name": class_name,

            "probability": round(
                probability * 100,
                2
            )

        })


    confidence = round(
        max(probabilities) * 100,
        2
    )


    # ------------------------------
    # FOOD RECOMMENDATION
    # ------------------------------

    food_result = generate_recommendation(

        recommendation,

        disease_type,

        cuisine,

        restrictions,

        allergies

    )


    description = food_result["description"]

    recommended_foods = food_result["recommended_foods"]

    foods_to_limit = food_result["foods_to_limit"]


    # ------------------------------
    # RESULT PAGE
    # ------------------------------

    return render_template(

        "result.html",

        name=name,

        age=age,

        gender=gender,

        weight=weight,

        height=height,

        bmi=bmi,

        bmi_category=bmi_category,

        disease_type=disease_type,

        severity=severity,

        activity=activity,

        exercise_hours=exercise_hours,

        calories=calories,

        cholesterol=cholesterol,

        blood_pressure=blood_pressure,

        glucose=glucose,

        nutrient_score=nutrient_score,

        adherence=adherence,

        restrictions=restrictions,

        allergies=allergies,

        cuisine=cuisine,

        recommendation=recommendation,

        confidence=confidence,

        prediction_probabilities=prediction_probabilities,

        description=description,

        recommended_foods=recommended_foods,

        foods_to_limit=foods_to_limit

    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

@app.route("/model-comparison")
def model_comparison():

    # ------------------------------
    # LOAD DATASET
    # ------------------------------

    df = pd.read_csv(DATASET_PATH)


    # ------------------------------
    # HANDLE MISSING VALUES
    # ------------------------------

    categorical_columns = [
        "Disease_Type",
        "Dietary_Restrictions",
        "Allergies"
    ]


    for column in categorical_columns:

        df[column] = df[column].fillna("None")


    # ------------------------------
    # SELECT FEATURES
    # ------------------------------

    X = df[FEATURES]

    y = df[TARGET]


    # ------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    # ------------------------------
    # PREPROCESSING
    # ------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                CATEGORICAL_FEATURES
            ),

            (
                "numeric",

                StandardScaler(),

                NUMERIC_FEATURES
            )

        ]

    )


    # ======================================================
    # RANDOM FOREST
    # ======================================================

    random_forest = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",

                RandomForestClassifier(

                    n_estimators=200,

                    random_state=42

                )

            )

        ]

    )


    # ======================================================
    # DECISION TREE
    # ======================================================

    decision_tree = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                DecisionTreeClassifier(

                    random_state=42

                )

            )

        ]

    )


    # ======================================================
    # KNN
    # ======================================================

    knn = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                KNeighborsClassifier(

                    n_neighbors=5

                )

            )

        ]

    )


    # ======================================================
    # LOGISTIC REGRESSION
    # ======================================================

    logistic_regression = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                LogisticRegression(

                    max_iter=2000

                )

            )

        ]

    )


    models = {

        "Random Forest":
            random_forest,

        "Decision Tree":
            decision_tree,

        "KNN":
            knn,

        "Logistic Regression":
            logistic_regression

    }


    # ------------------------------
    # TRAIN AND TEST MODELS
    # ------------------------------

    results = []


    for model_name, ml_model in models.items():

        ml_model.fit(
            X_train,
            y_train
        )


        predictions = ml_model.predict(
            X_test
        )


        accuracy = accuracy_score(

            y_test,

            predictions

        )


        results.append({

            "name": model_name,

            "accuracy": round(
                accuracy * 100,
                2
            )

        })


    # ------------------------------
    # SORT BY ACCURACY
    # ------------------------------

    results = sorted(

        results,

        key=lambda x: x["accuracy"],

        reverse=True

    )


    # ------------------------------
    # DISPLAY COMPARISON PAGE
    # ------------------------------

    return render_template(

        "model_comparison.html",

        results=results,

        total_records=len(df),

        training_records=len(X_train),

        testing_records=len(X_test)

    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(debug=True)