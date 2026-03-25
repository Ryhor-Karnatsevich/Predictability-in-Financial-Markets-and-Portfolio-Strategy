import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

is_test = False

if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])

ticker = 'AAPL'
data = df[df['Ticker'] == ticker].copy()
data = data.sort_values("Date")
data = data.set_index("Date")

data = data.asfreq("B")
data = data.ffill()

# TRAIN / TEST SPLIT
train = data[data.index < "2019-01-01"]["Returns"]
test = data[data.index >= "2019-01-01"]["Returns"]
print("Train size:", len(train))
print("Test size:", len(test))

model  = ARIMA(train,  order=(1,0,0))
result = model.fit()

print(result.summary())

# Forecast
forecast = result.forecast(steps=len(test))

# METRICS
mse = mean_squared_error(test, forecast)

direction = np.mean(np.sign(forecast) == np.sign(test))

print("MSE:", mse)
print("Direction:", direction)

