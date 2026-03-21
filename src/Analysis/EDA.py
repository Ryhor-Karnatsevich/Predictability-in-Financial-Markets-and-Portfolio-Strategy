import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

is_test = True
if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
    output_path = r"../../Data/Test Data/test_eda.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"
    output_path = r"../../Data/Main Data/all_stocks_eda.csv"

df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])


all_tickers = df['Ticker'].unique()
#### Graphics
def graphics(n):
    random_tickers = np.random.choice(all_tickers, n)
    for ticker in random_tickers:
        df[df['Ticker'] == ticker].plot(x = 'Date',y = "Close", title = ticker)
        plt.show()

graphics(0)

### Histograms
def histogram(n):
    random_tickers = np.random.choice(all_tickers, n)
    for ticker in random_tickers:
        data = df[df['Ticker'] == ticker]
        data['Returns'].hist(bins=100, figsize=(10, 5))
        plt.title(f"Returns Distribution: {ticker}")
        plt.show()

histogram(0)

### Volatility Clustering
def clustering(n):
    random_tickers = np.random.choice(all_tickers, n)
    for ticker in random_tickers:
        data = df[df['Ticker'] == ticker]
        data.plot(x='Date', y='Volatility', figsize=(10, 5), title=f"Volatility over time: {ticker}")
        plt.show()

clustering(10)

###
