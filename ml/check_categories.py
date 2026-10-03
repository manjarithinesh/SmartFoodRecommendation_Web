import pandas as pd

file_path = "dataset/diet_recommendations_dataset.csv"

df = pd.read_csv(file_path)

columns = [
    "Diet_Recommendation",
    "Disease_Type",
    "Severity",
    "Physical_Activity_Level",
    "Preferred_Cuisine",
    "Dietary_Restrictions",
    "Allergies"
]

for column in columns:
    print("\n================================")
    print(column)
    print("================================")

    values = df[column].dropna().unique()

    for value in sorted(values.astype(str)):
        print("-", value)
