import pandas as pd

data = pd.read_csv("illu.csv")

correlation = data.corr(numeric_only=True)

print(correlation["price"].sort_values(ascending=False))