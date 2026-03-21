# Stock Market Data Analysis (2000–2021)

## Project Overview

This project focuses on the statistical and econometric analysis of stock market data.
The goal is to examine return behavior, volatility patterns, and relationships between trading activity and price dynamics.

---

## Dataset

Source: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

- Historical stock data (OHLCV)
- Multiple tickers
- Time period: 2000–2021

---

## Data Cleaning

The dataset was preprocessed to ensure data quality:

### Date
- Converted to datetime format

### Missing Values
- Removed 649 rows with missing values

### Duplicates
- All duplicate rows were removed

### Logical Consistency Checks
- Removed invalid rows where:
  - High < Low
  - High < Close
  - Low > Close
  - Volume < 0

### Invalid Prices
- Records with zero or invalid prices (e.g., Open = 0) were treated as missing and removed
- Approximately 1 million rows removed to ensure data integrity

---

## Feature Engineering

The following variables were created:

- Returns (daily percentage change)
- Log Returns
- Volatility (rolling standard deviation)
- Moving averages (SMA 10, SMA 50)
- Momentum (10-day return)
- Volume change

A target variable was defined as the next-day return to support predictive modeling.

---

## Next Steps

- Exploratory Data Analysis (EDA)
- Econometric modeling (OLS, AR models)
- Interpretation of statistical relationships