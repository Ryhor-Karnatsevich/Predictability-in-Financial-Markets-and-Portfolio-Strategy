import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.4f}'.format


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
# configurations for grid
CONFIGS = {
    "DEFAULT": {"type": "EGARCH", "p": 2, "q": 1}
}

### SETUP
MODE = "FINAL"
split_date = "2019-01-01"
use_random = True
n = 10

if use_random:
    tickers = np.random.choice(df["Ticker"].unique(), size=n, replace=False)
else:
    tickers = ["AAPL", "NVDA"]


### GARCH Model
def garch_run(df,ticker,split_date,type,p,q,verbose=True):
#-----------------------------------------------------------
    data = df[df['Ticker'] == ticker].copy()
    data.set_index("Date", inplace=True)
    returns = data["Returns"].replace([np.inf, -np.inf], np.nan).dropna() * 100

    train = returns[returns.index < split_date]
    test = returns[returns.index >= split_date]
    # Additional Length filtrating
    if len(train) < 500:
        print(f"{ticker} has less than 500 stocks")
        return None
#-----------------------------------------------------------
# Model creating
    model = arch_model(train, vol=type, p=p, q=q, dist='t')
    result = model.fit(disp=0)

    if result.convergence_flag != 0:
         print(f"{ticker} {type}({p},{q}) does not match (flag={result.convergence_flag})")
         return None

    model_full = arch_model(returns, vol=type, p=p, q=q, dist='t')
    test_res = model_full.fix(result.params)

# Metrics
    test_sigma = test_res.conditional_volatility[test.index] # prediction line
    #mae
    mae = mean_absolute_error(np.abs(test), test_sigma)
    # relative mae for comparison
    mean_abs_return = np.mean(np.abs(test))
    relative_mae = mae / mean_abs_return


    alpha_sum = 0
    beta_sum = 0

    for i in range(1, p + 1):
        alpha_sum += result.params.get(f"alpha[{i}]", 0)

    for i in range(1, q + 1):
        beta_sum += result.params.get(f"beta[{i}]", 0)

    alpha = result.params.get("alpha[1]", np.nan)
    beta = result.params.get("beta[1]", np.nan)
    if type == "APARCH":
        delta = result.params.get("delta", np.nan)
    else:
        delta = np.nan


    if type == "EGARCH":
        persistence = beta_sum
    else:
        persistence = alpha_sum + beta_sum


    if persistence > 1.00001 and type != "EGARCH":
        print(f"Warning: {ticker} {type}({p},{q}) has persistence {persistence} >= 1")

# Forecast 1 day in the future variance
    final_forecast = test_res.forecast(horizon=1)
    future_variance = final_forecast.variance.iloc[-1].values[0]

    if verbose:
        plt.figure(figsize=(10, 4))
        plt.plot(np.abs(test), color='gray', alpha=0.3, label='Realized Vol')
        plt.plot(test_sigma, color='red', label=f'{type} Forecast')
        plt.title(f"{ticker} | {type}({p},{q})")
        plt.legend()
        plt.show()

    return {
        "ticker": ticker,
        "Model": type,
        "p": p,
        "q": q,
        "AIC": result.aic,
        "alpha": alpha,
        "beta": beta,
        "converged": result.convergence_flag == 0,
        "delta": delta,
        "persistence": persistence,
        "tomorrow_variance": future_variance,
        "MAE": mae,
        "Relative MAE": relative_mae,
        "train size": len(train),
        "test size": len(test)
    }

results = []

if MODE == "GRID":
    for ticker in tickers:
        for v in ["GARCH", "EGARCH", "APARCH"]:
            for p, q in [ (1,1),(2,1),(1,2)]:
                res = garch_run(df, ticker, split_date, type=v, p=p, q=q, verbose=False)
                if res: results.append(res)
    results_df = pd.DataFrame(results).sort_values(["ticker", "AIC"])

else: # MODE == "FINAL"
    for ticker in tickers:
        c = CONFIGS.get(ticker, CONFIGS["DEFAULT"])
        res = garch_run(df, ticker, split_date, type=c["type"], p=c["p"], q=c["q"], verbose=True)
        if res: results.append(res)
    results_df = pd.DataFrame(results)

print(results_df)