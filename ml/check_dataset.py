import pandas as pd

# Load dataset
file_path = "dataset/diet_recommendations_dataset.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========\n")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
