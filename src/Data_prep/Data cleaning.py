import pandas as pd

df = pd.read_csv(r"../../Data/test_data.csv")

dft = df.groupby("Ticker").agg(
    close = ('Close','mean'),
    open = ('Open','mean')
).reset_index()

dft['close'] = round(dft['close'],2)
dft['open'] = round(dft['open'],2)

print(dft.head())