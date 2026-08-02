import pandas as pd

data = pd.read_csv("illu.csv")

# Check dataset information
print(data.info())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Statistical summary
print("\nStatistics:")
print(data.describe())
