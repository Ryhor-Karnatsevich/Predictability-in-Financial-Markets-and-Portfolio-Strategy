import pandas as pd

df = pd.read_csv(r"../../Data/all_stocks.csv")
df['Date'] = pd.to_datetime(df["Date"])
#print(df.head())
filter = df[df["Date"].dt.year >= 2019].reset_index(drop=True)
filter.to_csv(r"Data_prep\test.csv")
