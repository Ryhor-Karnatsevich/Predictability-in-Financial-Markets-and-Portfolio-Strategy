import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import pickle

# Window settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import trained EGARCH(2,1)
with open("../../Data/Results/portfolio.pkl", "rb") as f:
    results = pickle.load(f)

# Extract tickers
ticker_list = list(set(r["summary"]["Ticker"] for r in results))
print(f"Tickers found: {len(ticker_list)}")
print(f"-" * 37)

# =====================================================
# Fast csv import and filtration to add "Close" column
path = r"../../Data/Main Data/all_stocks_analysis.csv"
df = (
    pl.scan_csv(path)
    .select(["Date", "Ticker", "Close"])
    .filter(
        (pl.col("Ticker").is_in(ticker_list)) &
        (pl.col("Date") > "2004-06-01")
    )
    .collect()
)
df = df.to_pandas()
prices_pivot = df.pivot(index="Date", columns="Ticker", values="Close")
prices_pivot.index = pd.to_datetime(prices_pivot.index)

path_2 = r"../../Data/dataset/symbols_valid_meta.csv"
companies_df = pd.read_csv(path_2)
companies_df = companies_df[["Symbol", "Security Name"]]
companies_df = companies_df[companies_df["Symbol"].isin(ticker_list)]


# =====================================================

# Setup for correct summary table printing
def print_styled_table(df, title,length = 160):
    print("\n" + "=" * length)
    print(title.center(length))
    print("=" * length)

    # Format columns
    df_print = df.copy()
    df_print = df_print.rename(columns={
        "Total Return": "Total Return",
        "Sharpe": "Sharpe Ratio",
        "Max_Drawdown": "Max Drawdown",
        "Annual_Vol": "Volatility",
        "Hit_Ratio": "Hit Ratio",
        "Turnover": "Turnover",
        "CVaR": "CVaR",
        "TRR": "Tail Risk Reduction"
    })
    percent_cols = ["Total Return", "Max Drawdown", "Volatility", "CVaR"]
    ratio_cols = ["Sharpe Ratio", "Hit Ratio", "Tail Risk Reduction"]
    other_cols = ["Turnover"]

    for col in percent_cols:
        if col in df_print.columns:
            df_print[col] = df_print[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "-")

    for col in ratio_cols:
        if col in df_print.columns:
            df_print[col] = df_print[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    for col in other_cols:
        if col in df_print.columns:
            df_print[col] = df_print[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    ordered_cols = [
        "Total Return",
        "Sharpe Ratio",
        "Volatility",
        "Max Drawdown",
        "CVaR",
        "Tail Risk Reduction",
        "Hit Ratio",
        "Turnover"
    ]

    df_print = df_print[[col for col in ordered_cols if col in df_print.columns]]
    df_print = df_print.fillna("-")
    df_print = df_print.sort_values(by=["Sharpe Ratio"], ascending=False)

    output = df_print.to_string(justify='center', col_space=14)
    print(output)
    print("-" * length)


#  ========================================================================================================================================================
#  ========================================================================================================================================================
#  Strategies Function  ( ENGINE of the project )
def strategies_backtest(results, rebalance=0.05, vol_discount=1):
    all_metrics = []
    diagnostic_equities = []

    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100
        volatility = r["series"]["volatility"] / 100

        # [1] "Buy & Hold" strategy
        fixed_fee = 0.0005
        weight_fixed = pd.Series(1, index=returns.index)
        return_fixed = weight_fixed * returns
        return_fixed.iloc[0] -= fixed_fee
        return_fixed.iloc[-1] -= fixed_fee

        # [2] "TVS with transaction and margin costs"
        target_0 = 0.02
        margin_annual_rate_0 = 0.05
        margin_daily_rate_0 = margin_annual_rate_0 / 252
        comm_rate_0 = 0.0005
        position_basic = (target_0 / volatility).clip(0, 2)
        return_raw_basic = position_basic * returns
        trades_basic = position_basic.diff().abs().fillna(0)
        turnover_basic = trades_basic.mean() * 252
        commissions_0 = trades_basic * comm_rate_0
        leverage_used_0 = (position_basic - 1).clip(lower=0)
        margin_costs_0 = leverage_used_0 * margin_daily_rate_0
        return_net_basic = return_raw_basic - commissions_0 - margin_costs_0

        # ==============================================================================================================
        # [3] "Target Volatility Scaling Advanced"
        leverage = 2
        comm_rate = 0.0005
        margin_daily_rate = 0.05 / 252
        volatility = volatility.clip(lower=0.012)

        volatility_target = volatility.rolling(100).median().fillna(0.02)

        raw_position = (volatility_target / (volatility * vol_discount)).clip(0, leverage)

        position = raw_position.copy()
        for i in range(1, len(position)):
            if abs(raw_position.iloc[i] - position.iloc[i - 1]) < rebalance:
                position.iloc[i] = position.iloc[i - 1]

        temp_returns = position * returns
        equity_temp = (1 + (temp_returns)).cumprod()
        drawdown_temp = equity_temp / equity_temp.cummax() - 1
        risk_multiplier = np.where(drawdown_temp < -0.1, 0.5, 1.0)
        risk_multiplier = pd.Series(risk_multiplier, index=position.index).shift(1).fillna(1.0)
        position = position * risk_multiplier

        return_raw = position * returns
        trades = position.diff().abs().fillna(0)
        commissions = trades * comm_rate
        turnover = trades.mean() * 252

        leverage_used = (position - 1).clip(lower=0)
        margin_costs = leverage_used * margin_daily_rate

        return_net = return_raw - commissions - margin_costs
        # ==============================================================================================================

        strategies = {
            "Buy & Hold": (return_fixed, weight_fixed, 0, return_fixed),
            "TVS with transaction and margin costs": (return_net_basic, position_basic, turnover_basic, return_fixed),
            "Target Volatility Scaling (TVS)": (return_net, position, turnover, return_fixed)
        }

        for name, (strategy_return, weight, turnover, return_fixed) in strategies.items():
            equity = (1 + strategy_return).cumprod()
            metrics = calculate_metrics(strategy_return, equity, turnover, return_fixed)

            diagnostic_equities.append({
                "Ticker": ticker,
                "Strategy": name,
                "Returns": strategy_return,
                "Dates": strategy_return.index
            })

            if metrics:
                all_metrics.append({
                    "Ticker": ticker,
                    "Strategy": name,
                    **metrics
                })
    return pd.DataFrame(all_metrics), diagnostic_equities


# Function to calculate metrics
def calculate_metrics(returns_series, equity_series, turnover, return_fixed):
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) == 0:
        return None

    total_return = equity_series.iloc[-1] - 1
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(252)
    max_dd = (equity_series / equity_series.cummax() - 1).min()
    annual_vol = returns_series.std() * np.sqrt(252)
    hit_ratio = (returns_series > 0).mean()

    percentile = 0.05
    var = np.percentile(returns_series, percentile * 100)
    cvar = returns_series[returns_series <= var].mean()

    if return_fixed is not None:
        bh_var = np.percentile(return_fixed, 5)
        bh_cvar = return_fixed[return_fixed <= bh_var].mean()
        if np.isnan(bh_cvar) or bh_cvar == 0:
            tail_risk_reduction = np.nan
        else:
            tail_risk_reduction = (cvar - bh_cvar) / abs(bh_cvar)
    else:
        tail_risk_reduction = np.nan

    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Annual_Vol": annual_vol,
        "Hit_Ratio": hit_ratio,
        "Turnover": turnover,
        "CVaR": cvar,
        "TRR": tail_risk_reduction
    }


# ======================================================================================================================
# EXECUTION
# ======================================================================================================================
print("Running backtest for single period...")
results_df, equities = strategies_backtest(results)

# ======================================================================================================================
# PORTFOLIO CONSTRUCTION (EQUAL WEIGHT)
# ======================================================================================================================
returns_list = []
for e in equities:
    temp = pd.DataFrame({
        "Date": e["Dates"],
        "Strategy": e["Strategy"],
        "Ticker": e["Ticker"],
        "Ret": e["Returns"].values
    })
    returns_list.append(temp)

returns_all = pd.concat(returns_list).reset_index(drop=True)

# Create B&H portfolio for TRR calculation
bh_strategy_data = returns_all[returns_all["Strategy"] == "Buy & Hold"]
bh_pivot = bh_strategy_data.pivot_table(index="Date", columns="Ticker", values="Ret")
bh_portfolio_ret = bh_pivot.mean(axis=1)
bh_portfolio_ret = bh_portfolio_ret.replace([np.inf, -np.inf], np.nan).dropna()

portfolio_returns = {}
portfolio_metrics = []

for strategy in returns_all["Strategy"].unique():
    strat_data = returns_all[returns_all["Strategy"] == strategy]
    pivot = strat_data.pivot_table(
        index="Date",
        columns="Ticker",
        values="Ret"
    )

    portfolio_ret = pivot.mean(axis=1)
    portfolio_returns[strategy] = portfolio_ret

    n_stocks = pivot.shape[1]
    portfolio_turnover = results_df[results_df["Strategy"] == strategy]["Turnover"].mean()

    portfolio_ret = portfolio_ret.replace([np.inf, -np.inf], np.nan).dropna()
    equity = (1 + portfolio_ret.fillna(0)).cumprod()

    if strategy == "Buy & Hold":
        metrics = calculate_metrics(
            portfolio_ret,
            equity,
            turnover=portfolio_turnover,
            return_fixed=None
        )
    else:
        metrics = calculate_metrics(
            portfolio_ret,
            equity,
            turnover=portfolio_turnover,
            return_fixed=bh_portfolio_ret
        )

    portfolio_metrics.append({
        "Strategy": strategy,
        **metrics
    })

portfolio_df = pd.DataFrame(portfolio_metrics).set_index("Strategy")
print_styled_table(portfolio_df, f"PORTFOLIO PERFORMANCE (2015.01-2020.04)")


# ======================================================================================================================
# INDIVIDUAL TICKER PERFORMANCE TABLE
# ======================================================================================================================
tvs_results = results_df[results_df["Strategy"] == "Target Volatility Scaling (TVS)"].copy()
tvs_results = tvs_results.drop(columns=["Strategy","CVaR"])
tvs_results = tvs_results.set_index("Ticker")

print_styled_table(tvs_results, "INDIVIDUAL TICKER PERFORMANCE (TVS STRATEGY)", 122)





# ======================================================================================================================
# VISUALISATION
# ======================================================================================================================
SHOW_PLOTS = True

if SHOW_PLOTS:
    # ======================================================================================================================
    # Portfolio Equity Curve
    initial_capital = 100

    plt.figure(figsize=(24, 12))
    for strategy, ret in portfolio_returns.items():
        equity = initial_capital * (1 + ret.fillna(0)).cumprod()
        if strategy == "Target Volatility Scaling (TVS)":
            plt.plot(equity, label=strategy, linewidth=2)
        elif strategy == "Buy & Hold":
            plt.plot(equity, label=strategy, linewidth=1.5, alpha=0.8)
        else:
            plt.plot(equity, label=strategy, linewidth=1.5, alpha=0.6)

    plt.title("Cumulative Portfolio Performance",fontsize=34)
    plt.ylabel("Portfolio Value (Indexed to 100)",fontsize=26)
    plt.xlabel("Date",fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.show()
    # ======================================================================================================================

    # ======================================================================================================================
    # Realized Volatility vs DYNAMIC Target
    plt.figure(figsize=(24, 12))

    window = 60
    strategy_tvs = "Target Volatility Scaling (TVS)"
    strategy_bh = "Buy & Hold"

    realized_vol_tvs = portfolio_returns[strategy_tvs].rolling(window).std() * np.sqrt(252)
    realized_vol_bh = portfolio_returns[strategy_bh].rolling(window).std() * np.sqrt(252)

    plt.plot(realized_vol_bh, label=f"{strategy_bh} (Unmanaged)", color='C0', alpha=0.7, linewidth=2)
    plt.plot(realized_vol_tvs, label=f"{strategy_tvs} (Managed)", color='C2', linewidth=2)

    plt.title("Realized Volatility: TVS vs Buy & Hold", fontsize=34)
    plt.ylabel("Annualized Volatility", fontsize=26)
    plt.xlabel("Date", fontsize=26)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', fontsize=24)
    plt.grid(True, alpha=0.2)
    plt.show()
    # ==================================================================================================================



    # ======================================================================================================================
    # Drawdown Curve
    plt.figure(figsize=(44, 12))
    for strategy, ret in portfolio_returns.items():
        equity = (1 + ret.fillna(0)).cumprod()
        drawdown = (equity / equity.cummax() - 1) * 100
        if strategy == "Target Volatility Scaling (TVS)":
            plt.plot(drawdown, label=strategy, linewidth=2)
        elif strategy == "Buy & Hold":
            plt.plot(drawdown, label=strategy, linewidth=1.5, alpha=0.8)
        else:
            plt.plot(drawdown, label=strategy, linewidth=1.5, alpha=0.5)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=1)
    plt.title("Portfolio Drawdowns", fontsize=34)
    plt.ylabel("Drawdown (%)", fontsize=26)
    plt.xlabel("Date", fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.fill_between(drawdown.index, 0, drawdown, color="C2", alpha=0.2)
    plt.show()
    # ======================================================================================================================



    # ======================================================================================================================
    # Drawdown Curve (Specific Period)
    plt.figure(figsize=(24, 12))

    start_date = '2018-01-01'
    end_date = '2019-06-01'
    start_date = start_date[:7].replace('-', '.')
    end_date = end_date[:7].replace('-', '.')

    for strategy, ret in portfolio_returns.items():
        # Filter for specific period
        ret_filtered = ret[(ret.index >= start_date) & (ret.index <= end_date)]
        if len(ret_filtered) > 0:
            equity = (1 + ret_filtered.fillna(0)).cumprod()
            drawdown = (equity / equity.cummax() - 1) * 100
            if strategy == "Target Volatility Scaling (TVS)":
                plt.plot(drawdown, label=strategy, linewidth=2.5)
            elif strategy == "Buy & Hold":
                plt.plot(drawdown, label=strategy, linewidth=2, alpha=0.8)
            else:
                plt.plot(drawdown, label=strategy, linewidth=1.5, alpha=0.5)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=1)
    plt.title(f"Drawdowns During Selected Period ({start_date} to {end_date})", fontsize=34)
    plt.ylabel("Drawdown (%)", fontsize=26)
    plt.xlabel("Date", fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.fill_between(drawdown.index, 0, drawdown, color="C2", alpha=0.2)
    plt.show()
    # ======================================================================================================================


    # ======================================================================================================================
    # Rolling Sharpe (252-day window) with colored fill
    plt.figure(figsize=(24, 12))
    window = 252

    rolling_sharpes = {}
    for strategy, ret in portfolio_returns.items():
        rolling_sharpe = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8) * np.sqrt(252)
        rolling_sharpes[strategy] = rolling_sharpe

    if "Target Volatility Scaling (TVS)" in rolling_sharpes and "Buy & Hold" in rolling_sharpes:
        difference = rolling_sharpes["Target Volatility Scaling (TVS)"] - rolling_sharpes["Buy & Hold"]

        # Линия
        plt.plot(difference, label="TVS - Buy & Hold", linewidth=2, color='black')

        # Заливка: зеленый где > 0, красный где < 0
        plt.fill_between(difference.index, 0, difference,
                         where=(difference >= 0),
                         color='C2', alpha=0.3, interpolate=True)
        plt.fill_between(difference.index, 0, difference,
                         where=(difference < 0),
                         color='C3', alpha=0.3, interpolate=True)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.title(f"Rolling Sharpe Ratio Difference (TVS minus Buy & Hold)", fontsize=34)
    plt.ylabel("Sharpe Ratio Difference", fontsize=26)
    plt.xlabel("Date", fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.show()
    # ==================================================================================================================


