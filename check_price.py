import pandas as pd

data = pd.read_csv("illu.csv")

print(data[data["price"] == 0])