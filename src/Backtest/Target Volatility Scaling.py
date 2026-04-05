import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Window settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import trained EGARCH(2,1)
with open("../../Data/Results/garch_results.pkl", "rb") as f:
    results = pickle.load(f)

# Extract unique periods
periods = sorted(set(r["summary"]["Period"] for r in results))
p = len(periods)
print(f"Periods found: {p}")
print(f"-"*30)
# For 3 PERIODS ANALYSIS
target_periods = ['2007-06-01','2015-06-01','2018-06-01']
# Extract tickers
ticker_list = list(set(r["summary"]["Ticker"] for r in results))

# Setup for correct summary table printing
def print_styled_table(df, title):
    print("\n" + "=" * 170)
    print(title.center(170))
    print("=" * 170)
    output = df.to_string(justify='center', col_space=10)
    print(output)
    print("-" * 170)


#  ========================================================================================================================================================
#  ========================================================================================================================================================
#  Strategies Function  ( ENGINE of the project )
def strategies_backtest(results, rebalance=0.05, vol_discount=1):

    all_metrics = []
    diagnostic_positions = [] # for sensitivity plot
    diagnostic_equities = [] #  equity curves

    for r in results:
        ticker = r["summary"]["Ticker"]
        returns = r["series"]["returns"] / 100
        volatility = r["series"]["volatility"] / 100
        period = r["summary"]["Period"]

        # cutting test to 2 years
        start_dt = pd.to_datetime(period)
        end_dt = start_dt + pd.DateOffset(years=2)
        returns = returns[returns.index <= end_dt]
        volatility = volatility[volatility.index <= end_dt]

        # [1] "Buy & Hold" strategy
        fixed_fee = 0.0005
        weight_fixed = pd.Series(1, index=returns.index)
        return_fixed = weight_fixed * returns
        return_fixed.iloc[0] -= fixed_fee
        return_fixed.iloc[-1] -= fixed_fee


        # [2] "TVS with transaction and margin costs"
        # Costs Setup
        target_0 = 0.02
        margin_annual_rate_0 = 0.05  # 5% annually for leverage
        margin_daily_rate_0 = margin_annual_rate_0 / 252
        comm_rate_0 = 0.0005  # 0.05% - transaction cost
        # Raw return
        position_basic = (target_0 / volatility).clip(0, 2)
        return_raw_basic = position_basic * returns
        # Transaction Costs and turnover
        trades_basic = position_basic.diff().abs().fillna(0)
        turnover_basic = trades_basic.mean() * 252
        commissions_0 = trades_basic * comm_rate_0
        # Margin Costs
        leverage_used_0 = (position_basic - 1).clip(lower=0)
        margin_costs_0 = leverage_used_0 * margin_daily_rate_0
        # Final Return
        return_net_basic = return_raw_basic - commissions_0 - margin_costs_0


        # ==============================================================================================================
        # [3] Target Volatility Scaling Advanced
        # 1  Setup: Leverage, Costs and Prices
        leverage = 2
        # vol_discount = 1  # value to offset predicted volatility, if it is overrated
        # rebalance = 0.05  # value for rebalancing threshold
        comm_rate = 0.0005
        margin_daily_rate = 0.05 / 252  # annual rate / trading days
        volatility = volatility.clip(lower=0.012)  # limit flour for volatility

        # 2  Symmetric Target
        volatility_target = volatility.rolling(100).median().fillna(0.02)

        # 3  Volatility Risk Premium Offset
        raw_position = (volatility_target / (volatility * vol_discount)).clip(0, leverage)

        # 4  Rebalancing Threshold (Cutting unnecessary turnover & slippage)
        position = raw_position.copy()
        for i in range(1, len(position)):
            # Only update position if the change is greater than 5%
            if abs(raw_position.iloc[i] - position.iloc[i - 1]) < rebalance:
                position.iloc[i] = position.iloc[i - 1]

        diagnostic_positions.extend(position.values) # for sensitivity plot

        # 5  Drawdown Protection (Circuit Breaker)
        # Calculating equity and drawdown to adjust position risk
        temp_returns = position * returns
        equity_temp = (1 + (temp_returns)).cumprod()
        drawdown_temp = equity_temp / equity_temp.cummax() - 1
        risk_multiplier = np.where(drawdown_temp < -0.1, 0.5, 1.0)
        risk_multiplier = pd.Series(risk_multiplier, index=position.index).shift(1).fillna(1.0)
        position = position * risk_multiplier

        # 6  Returns & Execution Costs
        return_raw = position * returns
        trades = position.diff().abs().fillna(0)
        commissions = trades * comm_rate
        turnover = trades.mean() * 252

        # 7 Margin Costs (Cost of Leverage)
        leverage_used = (position - 1).clip(lower=0)
        margin_costs = leverage_used * margin_daily_rate

        # 8 Final Net Return for Optimized Strategy
        return_net = return_raw - commissions - margin_costs
        # ==============================================================================================================

        # Dictionary with return and weight for final table
        strategies = {
            "Buy & Hold": (return_fixed, weight_fixed,0,None, None),
            "TVS with transaction and margin costs": (return_net_basic, position_basic,turnover_basic,target_0,return_fixed),
            "Target Volatility Scaling (TVS)": (return_net, position,turnover,volatility_target,return_fixed)
        }

        # Another subiteration to add values to final table and to plot dictionary
        for name, (strategy_return, weight,turnover,target,return_fixed) in strategies.items():
            equity = (1 + strategy_return).cumprod()  # creates equity curve (line of returns of all previous time)
            metrics = calculate_metrics(strategy_return, equity, turnover, target, return_fixed)  # Use function that below

            diagnostic_equities.append({
                "Ticker": ticker,
                "Period": period,
                "Strategy": name,
                "Returns": strategy_return,
                "Dates": strategy_return.index
            })

            if metrics:
                # Adding data to list for final table
                all_metrics.append({
                    "Ticker": ticker,
                    "Period": period,
                    "Strategy": name,
                    **metrics # from function "calculate_metrics"
                })

    return pd.DataFrame(all_metrics), diagnostic_positions, diagnostic_equities


# Function to calculate metrics
# ----------------------------------------------------------------------------------------------------------------------------------
def calculate_metrics(returns_series, equity_series, turnover,target,return_fixed ):
    # Data cleaning
    returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_series) == 0:
        return None

    # Metrics
    total_return = equity_series.iloc[-1] - 1  # Return for the period
    sharpe = (returns_series.mean() / (returns_series.std() + 1e-8)) * np.sqrt(252)  # Sharpe
    max_dd = (equity_series / equity_series.cummax() - 1).min()  # Max Drawdown
    annual_vol = returns_series.std() * np.sqrt(252)  # Annual Volatility
    hit_ratio = (returns_series > 0).mean()  # Percentage of Days when strategy return > 0

    # CVAR
    percentile = 0.05
    var = np.percentile(returns_series, percentile * 100)
    cvar = returns_series[returns_series <= var].mean()

    # Volatility Target Deviation
    if target is not None:
        realized_vol = returns_series.rolling(20).std().rolling(20).mean() # use also mean to smoothen it(Double smoothing)
        if isinstance(target, pd.Series):
            target_aligned = target.reindex(returns_series.index)
        else:
            target_aligned = pd.Series(target, index=returns_series.index)
        vol_target_dev = np.nanmean(np.abs(realized_vol - target_aligned))
    else:
        vol_target_dev = np.nan

    # Tail Risk Reduction
    if return_fixed is not None:
        bh_var = np.percentile(return_fixed, 5)
        bh_cvar = return_fixed[return_fixed <= bh_var].mean()
        if np.isnan(bh_cvar) or bh_cvar == 0:
            tail_risk_reduction = np.nan
        else:
            tail_risk_reduction = (cvar - bh_cvar) / abs(bh_cvar)
    else:
        tail_risk_reduction = np.nan


    # That dictionary appends to ticker dictionary in all_metrics list
    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Annual_Vol": annual_vol,
        "Hit_Ratio": hit_ratio,
        "Turnover": turnover,
        "CVaR": cvar,
        "RMSE_Vol": vol_target_dev,
        "TRR": tail_risk_reduction
    }
#  ========================================================================================================================================================
#  ========================================================================================================================================================



# ======================================================================================================================
# EXECUTION Setup
all_results = [] # Storage for ENGINE calculations for every period without sensitivity
all_positions = [] # Storage for position curves for each ticker and each period (leverage plot)
all_equities = [] # for equity curve

# sensitivity setup
rebalance_values = [0.02, 0.03, 0.04, 0.05]
vol_discount_values = [0.6, 0.8, 1.0]
sensitivity_results = [] # Results from ENGINE but with sensitivity loop

# Main Loop
for period in periods:
    print(f"Calculating Period: {period}")
    print(f"-"*30)

    filtered_results = [r for r in results if r["summary"]["Period"] == period]

    results_df, period_positions,equities = strategies_backtest(filtered_results)
    all_results.append(results_df)
    all_positions.extend(period_positions) # for leverage plot
    all_equities.extend(equities)    # for equity curve

    # Sensitivity loop
    for reb in rebalance_values:
        for vd in vol_discount_values:

            tmp_df, _ , _ = strategies_backtest(
                filtered_results,
                rebalance=reb,
                vol_discount=vd
            )
            # for data integrity safety
            if not tmp_df.empty:
                tmp_df["Period"] = period # exceptionally for safety
                tmp_df["rebalance"] = reb
                tmp_df["vol_discount"] = vd
                sensitivity_results.append(tmp_df)

final_df = pd.concat(all_results)                # make table out of raw data
sensitivity_df = pd.concat(sensitivity_results)
# ======================================================================================================================



# Creating "3 PERIODS ANALYSIS"
# ======================================================================================================================
regime_df = final_df[final_df["Period"].isin(target_periods)].copy()
regime_df["Periods"] = regime_df["Period"].map({
    target_periods[0]: "2007-2009",
    target_periods[1]: "2015-2017",
    target_periods[2]: "2018-2020"
}).fillna("Other")

regime_summary = regime_df.groupby(["Periods", "Strategy"]).agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max_Drawdown": "mean",
    "Annual_Vol": "mean",
    "Hit_Ratio": "mean",
    "Turnover": "mean",
    "CVaR": "mean",
    "RMSE_Vol": "mean",
    "TRR": "mean"
}).sort_values(by=["Periods", "Sharpe"],ascending=[True, False])

# Adding Outperformance column
pivot_regime = regime_df.pivot_table(
    index=['Periods', 'Ticker'],
    columns='Strategy',
    values='Sharpe'
)
win_rates_regime = {}
for (period_name, strategy), _ in regime_summary.iterrows():
    if strategy != "Buy & Hold":
        current_period_data = pivot_regime.xs(period_name, level='Periods')
        wins = (current_period_data[strategy] > current_period_data["Buy & Hold"]).sum()
        total = len(current_period_data)
        win_rates_regime[(period_name, strategy)] = f"{wins}/{total} ({wins / total:.1%})"

regime_summary["Outperf. (vs B&H)"] = [
    win_rates_regime.get(index, "-") for index in regime_summary.index
]

# Periods printing setup
regime_print = regime_summary.copy()
cols_2_decimal = ["Turnover", "Max_Drawdown", "Annual_Vol"]
cols_4_decimal = ["Sharpe", "Total Return","Hit_Ratio", "CVaR", "RMSE_Vol", "TRR"]
for col in cols_2_decimal:
    if col in regime_print.columns:
        regime_print[col] = regime_print[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
for col in cols_4_decimal:
    if col in regime_print.columns:
        regime_print[col] = regime_print[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
regime_print = regime_print.fillna("-")

# Printing
print_styled_table(regime_print, "3 PERIODS ANALYSIS")
# ======================================================================================================================


# Creating "AVERAGE PERFORMANCE ACROSS SELECTED PERIODS"
# ======================================================================================================================
summary = final_df.groupby("Strategy").agg({
    "Total Return": "mean",
    "Sharpe": "mean",
    "Max_Drawdown": "mean",
    "Annual_Vol": "mean",
    "Hit_Ratio": "mean",
    "Turnover": "mean",
    "CVaR": "mean",
    "RMSE_Vol": "mean",
    "TRR": "mean"
}).sort_values("Sharpe", ascending=False)

# Average table printing setup
summary_print = summary.copy()
for col in cols_2_decimal:
    if col in summary_print.columns:
        summary_print[col] = summary_print[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
for col in cols_4_decimal:
    if col in summary_print.columns:
        summary_print[col] = summary_print[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
summary_print = summary_print.fillna("-")

# Printing
print_styled_table(summary_print, "AVERAGE PERFORMANCE ACROSS SELECTED PERIODS")
# ======================================================================================================================





### ==================================================================================================================
### VISUALISATION
### ==================================================================================================================


# ====================================================================================================================
# ROBUSTNESS ANALYSIS (4 PLOTS)
# Sharpe, Max Drawdown, Return, Equity Curve
sharpe_dyn = final_df.groupby(["Period", "Strategy"])["Sharpe"].mean().unstack()
dd_dyn = final_df.groupby(["Period", "Strategy"])["Max_Drawdown"].mean().unstack()
ret_dyn = final_df.groupby(["Period", "Strategy"])["Total Return"].mean().unstack()

# insert year for axes
year_labels = [(pd.to_datetime(p) + pd.DateOffset(years=2)).strftime('%Y') for p in sharpe_dyn.index]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("GLOBAL STRATEGY ROBUSTNESS ANALYSIS (2007-2020)", fontsize=20, fontweight='bold')

# 1. Sharpe Ratio
sharpe_dyn.reset_index(drop=True).plot(ax=axes[0, 0], marker='o', markersize=5, linewidth=2)
axes[0, 0].set_title("Sharpe Ratio Dynamics", fontsize=14)

# 2. Max Drawdown
dd_dyn.reset_index(drop=True).plot(ax=axes[0, 1], marker='s', markersize=5, linewidth=2)
axes[0, 1].set_title("Max Drawdown per Period", fontsize=14)

# 3. Total Return
ret_dyn.reset_index(drop=True).plot(ax=axes[1, 0], marker='d', markersize=5, linewidth=2)
axes[1, 0].set_title("Total Return per Period", fontsize=14)

# 4. Equity Curve
returns_list = []
for e in all_equities:
    temp_series = pd.DataFrame({
        "Date": e["Dates"],
        "Strategy": e["Strategy"],
        "Ret": e["Returns"].values
    })
    returns_list.append(temp_series)

returns_all = pd.concat(returns_list).reset_index(drop=True)
avg_daily_ret = returns_all.groupby(["Date", "Strategy"])["Ret"].mean().unstack()
equity_final = (1 + avg_daily_ret.fillna(0)).cumprod()
equity_smooth = equity_final.resample("ME").last().rolling(window=3).mean().bfill()
equity_smooth.plot(ax=axes[1, 1], linewidth=2)
axes[1, 1].set_title("Average Equity Curve", fontsize=14)
axes[1, 1].set_ylabel("Portfolio Value")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=8)
years = pd.date_range(
    start=equity_smooth.index.min(),
    end=equity_smooth.index.max(),
    freq="YS"
)
axes[1, 1].set_xticks(years)
axes[1, 1].set_xticklabels([d.year for d in years], rotation=0)
axes[1, 1].set_xlabel("Year")
from matplotlib.ticker import ScalarFormatter
axes[1, 1].set_yscale('log')
for axis in [axes[1, 1].yaxis]:
    axis.set_major_formatter(ScalarFormatter())
axes[1, 1].set_yticks([1, 1.2, 1.44, 1.73, 2.07, 2.49, 3, 3.6, 4.32, 5.18, 6.22, 7])


# Customization for 3 plots
for i in range(3):
    ax = axes.flat[i]
    clean_labels = []
    prev = None
    for y in year_labels:
        if y != prev:
            clean_labels.append(y)
            prev = y
        else:
            clean_labels.append("")

    ax.set_xticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=0)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ====================================================================================================================


# ====================================================================================================================
# MECHANICS ANALYSIS
# Leverage Distribution (Main Strategy Only)
plt.figure(figsize=(10, 6))
plt.hist(all_positions, bins=50, color='teal', alpha=0.7, edgecolor='white')
plt.axvline(x=1.0, color='red', linestyle='--', label='No Leverage (100% Weight)')
plt.title("Leverage Usage Distribution (Default Parameters)", fontsize=14)
plt.xlabel("Position Size (0.0 to 2.0)")
plt.ylabel("Frequency (Days)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
# ====================================================================================================================


# ====================================================================================================================
# SENSITIVITY ANALYSIS (4 HEATMAPS)
# Sharpe, CVaR, Return, Std Sharpe
tvs_sens = sensitivity_df[sensitivity_df["Strategy"] == "Target Volatility Scaling (TVS)"]

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("SENSITIVITY ANALYSIS: RISK & PERFORMANCE HEATMAPS", fontsize=22, fontweight='bold')

metrics = [
    ("Sharpe", "Mean Sharpe Ratio (Efficiency)", "YlGnBu"),
    ("CVaR", "Mean CVaR (Tail Risk)", "Reds_r"),
    ("Total Return", "Mean Total Return (Performance)", "Greens"),
    ("Sharpe", "Std Dev of Sharpe (Stability)", "Purples")
]

for i, (metric, title, cmap) in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    if "Std Dev" in title:
        pivot_data = tvs_sens.groupby(["rebalance", "vol_discount"])[metric].std().unstack()
    else:
        pivot_data = tvs_sens.groupby(["rebalance", "vol_discount"])[metric].mean().unstack()

    fmt = ".2%" if metric == "Total Return" else ".3f"

    sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={'label': metric})

    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Rebalance Threshold")
    ax.set_xlabel("Volatility Discount")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ====================================================================================================================