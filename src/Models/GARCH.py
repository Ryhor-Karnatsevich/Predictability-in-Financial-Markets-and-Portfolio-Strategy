import pandas as pd
import numpy as np
from arch import arch_model

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
def garch_run(df,ticker,split_date,verbose=False):
    data = df[df['Ticker'] == ticker].copy()
    data.set_index("Date", inplace=True)

#-----------------------------------------------------------
    returns = data["Returns"].dropna() * 100

    train = returns[returns.index < split_date]
    test = returns[returns.index >= split_date]

    print(f"Ticker: {ticker}")
    print("Train size:", len(train))
    print("Test size:", len(test))
    # Additional Length filtrating
    if len(train) < 750:
        return None
#-----------------------------------------------------------
    model  = arch_model(
        train,
        mean='constant',
        vol='GARCH',
        p=1,
        q=1,
        dist = "t"
    )
    result = model.fit(disp=0)

    if verbose:
        print(result.summary())
    # Parameters
    alpha = result.params.get("alpha[1]", np.nan)
    beta = result.params.get("beta[1]", np.nan)

# Forecast
    forecast = result.forecast(horizon=1)
    print(forecast.variance)

    return {
        "ticker": ticker,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "train size": len(train),
        "test size": len(test)
    }

results = []
for ticker in tickers:
    result = garch_run(df,ticker, split_date, verbose = False)
    if result is not None:
        results.append(result)

print(results)
