import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

is_test = False
if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
    output_path = r"../../Data/Test Data/test_eda.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"
    output_path = r"../../Data/Main Data/all_stocks_eda.csv"

df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(["Ticker", "Date"])
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)


df["Return_lag1"] = df.groupby("Ticker")["Returns"].shift(1)
df["Volume_lag1"] = df.groupby("Ticker")["Volume change"].shift(1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Return_lag1", "Volume_lag1", "Target"])

# Ordinary Least Squares
# Can I predict Return ?
# Return_t = β1 * Return_(t-1) + β2 * Volume_change + ε
X = df[["Return_lag1", "Volume_lag1"]]
y = df["Target"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

# RA model
# Return_t = α + β * Return_(t-1) + ε


























