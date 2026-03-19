import pandas as pd

df = pd.read_csv(r"../../Data/all_stocks.csv")

dft = df.groupby("Ticker").agg(
    close = ('Close','mean')
).reset_index()

dft['close'] = round(dft['close'],2)

print(dft.head())