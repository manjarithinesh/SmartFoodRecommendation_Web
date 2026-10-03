import pandas as pd

# Load the Kaggle dataset
file_path = "dataset/diet_recommendations_dataset.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Fill missing categorical values
categorical_columns = [
    "Disease_Type",
    "Dietary_Restrictions",
    "Allergies"
]

for column in categorical_columns:
    df[column] = df[column].fillna("None")

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

print("\nDataset preprocessing completed successfully.")
