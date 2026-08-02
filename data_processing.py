import pandas as pd


# Load dataset
data = pd.read_csv("illu.csv")


# Convert date column
data["date"] = pd.to_datetime(data["date"])


# Extract useful information from date
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month


# Remove original date column
data = data.drop("date", axis=1)


# Save processed data
data.to_csv("processed_data.csv", index=False)


print("Feature engineering completed")
print(data.head())
print(data.columns)