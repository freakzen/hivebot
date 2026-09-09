import pandas as pd

file_path = "data/twcs.csv"

df = pd.read_csv(file_path, nrows=10000)

print("Shape of sample:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)