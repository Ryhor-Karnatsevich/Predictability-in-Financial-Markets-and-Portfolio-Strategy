import pandas as pd
import numpy as np

is_test = False
if is_test:
    path = r"../../Data/Test Data/test_data_cleaned.csv"
    output_path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_cleaned.csv"
    output_path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

grouped = df.groupby("Ticker")
### Returns
df["Returns"] = grouped["Close"].pct_change()

### Log Returns
df["Log_Returns"] = np.log(df["Close"]) - np.log(grouped["Close"].shift(1))

### Standard Deviation
N = 10
df["Volatility"] = grouped["Returns"].rolling(window=N).std().reset_index(0,drop=True)

### SMA's
df["SMA_10"] = grouped["Close"].rolling(window=10).mean().reset_index(0, drop=True)
df["SMA_50"] = grouped["Close"].rolling(window=50).mean().reset_index(0, drop=True)

### Momentum 10 days
df["Momentum"] = grouped["Close"].pct_change(periods = 10)

### Volume change
df["Volume change"] = grouped["Volume"].pct_change()

### Target
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
df["Target"] = df.groupby("Ticker")["Returns"].shift(-1)

df = df.dropna()

df.to_csv(output_path, index=False)
