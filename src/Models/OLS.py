import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

is_test = False

if is_test:
    path = r"../../Data/Test Data/test_analysis.csv"
else:
    path = r"../../Data/Main Data/all_stocks_analysis.csv"

df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"])
# Filter new stocks
min_obs = 500
df = df.groupby("Ticker").filter(lambda x: len(x) >= min_obs)


df["Return_lag1"] = df.groupby("Ticker")["Returns"].shift(1)
df["Volume_lag1"] = df.groupby("Ticker")["Volume change"].shift(1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Return_lag1", "Volume_lag1", "Returns"])


# TRAIN / TEST SPLIT
train = df[df["Date"] < "2019-01-01"]
test  = df[df["Date"] >= "2019-01-01"]


print("Train size:", len(train))
print("Test size:", len(test))



# OLS Model Function
def run_model(train, test, features, target="Returns", name="Model"):
    # TRAIN
    X_train = train[features]
    y_train = train[target]
    X_train = sm.add_constant(X_train)

    model = sm.OLS(y_train, X_train).fit(cov_type='HC3')

    print(f"\n{name}")
    print(model.summary())

    # TEST
    X_test = test[features]
    X_test = sm.add_constant(X_test)

    preds = model.predict(X_test)
    preds = np.where(np.abs(preds) < 0.0001, 0, preds)

    # METRICS
    accuracy = np.mean(np.sign(preds) == np.sign(test[target]))
    mse = mean_squared_error(test[target], preds)

    print("Directional accuracy:", accuracy)
    print("MSE:", mse)

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "mse": mse
    }

# Return(t) = β0 + β1 * Return(t-1) + ε
model_1 = run_model(
    train,
    test,
    features=["Return_lag1"],
    name="AR(1) Model"
)

# Return(t) = β0 + β1 * Volume + ε
model_2 = run_model(
    train,
    test,
    features=["Volume_lag1"],
    name="Volume Model"
)

# Return(t) = β0 + β1 * Return(t-1) + β2 * Volume + ε
model_3 = run_model(
    train,
    test,
    features=["Return_lag1","Volume_lag1"],
    name="Return & Volume Model"
)























