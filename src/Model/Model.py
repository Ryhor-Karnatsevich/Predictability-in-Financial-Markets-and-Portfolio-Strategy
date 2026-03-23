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

# 1. Создаем лаги для новых признаков (из тех, что ты уже посчитал в feature engineering)
df["Vol_lag1"] = df.groupby("Ticker")["Volatility"].shift(1)
df["Mom_lag1"] = df.groupby("Ticker")["Momentum"].shift(1)

# 2. Очистка (теперь проверяем все 4 колонки)
df = df.replace([np.inf, -np.inf], np.nan)
cols_to_clean = ["Return_lag1", "Volume_lag1", "Vol_lag1", "Mom_lag1", "Target"]
df = df.dropna(subset=cols_to_clean)

# 3. СТАНДАРТИЗАЦИЯ (Z-score) — ОЧЕНЬ ВАЖНО
# Волатильность может быть 0.01, а Моментум 0.20. Модель может запутаться.
# Мы делаем так, чтобы среднее было 0, а отклонение 1.
features = ["Return_lag1", "Volume_lag1", "Vol_lag1", "Mom_lag1"]
df[features] = (df[features] - df[features].mean()) / df[features].std()

# 4. Собираем модель
X = df[features]
y = df["Target"]

X = sm.add_constant(X) # Добавляем константу (бета-ноль)

model = sm.OLS(y, X).fit()
print(model.summary())

# RA model
# Return_t = α + β * Return_(t-1) + ε


























