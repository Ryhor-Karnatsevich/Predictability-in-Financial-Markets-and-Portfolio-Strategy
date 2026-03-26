import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


is_test = False

if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])
#Filter
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)
# Seed Setup
np.random.seed(52)


### SETUP
split_date = "2019-01-01"
use_random = False
n = 2

if use_random:
    tickers = np.random.choice(df["Ticker"].unique(), size=n, replace=False)
else:
    tickers = ["AAPL", "NVDA"]


### GARCH Model
def garch_run(df,ticker,split_date,verbose=True):
    data = df[df['Ticker'] == ticker].copy()
    data.set_index("Date", inplace=True)

#-----------------------------------------------------------
    returns = data["Returns"].replace([np.inf, -np.inf], np.nan).dropna() * 100

    train = returns[returns.index < split_date]
    test = returns[returns.index >= split_date]

    print(f"Ticker: {ticker}")
    print("Train size:", len(train))
    print("Test size:", len(test))
    # Additional Length filtrating
    if len(train) < 500:
        print(f"{ticker} has less than 500 stocks")
        return None
#-----------------------------------------------------------
    model  = arch_model(
        train,
        vol='GARCH',
        p=1,
        q=1,
        dist = "t"
    )
    result = model.fit(disp=0)

    model_full = arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
    # Метод .fix() подставляет в эту модель параметры, которые ты нашел на train
    test_res = model_full.fix(result.params)
    test_sigma = test_res.conditional_volatility[test.index]
    mae = np.mean(np.abs(np.abs(test) - test_sigma))

    # Parameters
    alpha = result.params.get("alpha[1]", np.nan)
    beta = result.params.get("beta[1]", np.nan)



# Forecast
    forecast = result.forecast(horizon=1)
    future_variance = forecast.variance.iloc[-1].values[0]

    if verbose:
        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(test), color='gray', alpha=0.3, label='Realized Vol')
        plt.plot(test_sigma, color='red', label='GARCH Forecast')
        plt.title(f"Volatility Test for {ticker}")
        plt.legend()
        plt.show()

    return {
        "ticker": ticker,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "tomorrow_variance": future_variance,
        "test_mae": mae,
        "train size": len(train),
        "test size": len(test)
    }

results = []
for ticker in tickers:
    result = garch_run(df,ticker, split_date)
    if result is not None:
        results.append(result)

print(pd.DataFrame(results))
