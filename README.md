# Multi-Agent Time Series Forecasting

## Project Overview
Time series data—whether tracking daily financial markets, climate changes, or monthly sales—often contains a complex mixture of signals. A single dataset might simultaneously exhibit long-term macro trends, rigid seasonality, short-term momentum shifts, and unpredictable volatility. Relying on a single machine learning model to capture all these dynamics usually leads to underfitting or severe overfitting. 

This project solves that problem by building a **multi-agent forecasting system**. Instead of a single monolithic algorithm, the system relies on specialized "agents." Each agent utilizes a different statistical, machine learning, or deep learning model to target a specific component of the time series. By isolating features like trend, momentum, and volatility, the ensemble can generate highly optimized, robust forecasts capable of driving actionable quantitative trading signals.

## Methodology & Architecture

### 1. Foundation
The foundation of the forecasting pipeline relies heavily on data transformation using NumPy and Pandas. To ensure the models learned genuine patterns rather than relying on future data, strict $t-1$ shifts were enforced across all datasets to absolutely prevent data leakage. 
* **Exploratory Analysis:** Applied ACF (Autocorrelation) and PACF (Partial Autocorrelation) alongside multiplicative seasonal decomposition to identify significant temporal structures (e.g., confirming a lag-12 correlation of 0.989 in airline data).
* **Financial Indicators:** Engineered specialized features for S&P 500 (SPY) and VIX data, including Relative Strength Index (RSI), MACD, and Bollinger Band widths.
* **Temporal Features:** Generated custom lag features (1, 2, 3, 6, 12) and rolling statistics to capture serial dependence.

### 2. The Multi-Agent Framework
An object-oriented `BaseAgent` interface was developed in Python to standardize the training and prediction pipeline across entirely different model families.
* **Trend Agent:** Utilizes Linear Regression paired with Fourier seasonality terms to capture long-run drift and annual cycles.
* **Momentum Agent:** An XGBoost regressor trained specifically on short-term lagged returns and momentum indicators to react to immediate market swings.
* **Volatility Agent:** A secondary XGBoost model configured to recognize and forecast based on changing market regimes and VIX fluctuations.
* **Sequence Agent:** A Deep Learning LSTM (Long Short-Term Memory) network built with PyTorch, evaluating raw sliding windows of time-series sequences with early stopping protocols.

### 3. Hybrid Multi-Step Modeling
Before applying the framework to financial data, the architecture was heavily benchmarked on classic datasets (Melbourne climate, Iris, and Air Passengers). A recursive multi-step forecast was built to blend models: using a linear model for trend/Fourier seasonality, and passing the residuals to an XGBoost model. 

## Key Outcomes
The multi-stage pipeline demonstrated significant predictive improvements across diverse datasets:
* **S&P 500 Forecasting:** When tested on unseen 2019 market data, all four specialized agents consistently achieved a sub-0.6% Mean Absolute Error (MAE), proving accurate enough for live quantitative trading environments.
* **Climate Data Benchmarking (3,650 rows):** The Random Forest implementation achieved a test MAE of 1.69, outperforming the Linear Regression baseline (MAE 2.17) by 22%. Furthermore, XGBoost reduced the error to an MAE of 1.77 compared to a standard Decision Tree (MAE 2.61), showcasing extreme robustness on noisy data.
* **Airline Passenger Volume:** The hybrid linear-residual model successfully forecasted a full 12-month horizon with an MAE of 33.7 and an RMSE of 41.7 passengers.

## Repository Structure
* **`features.py`**: Handles automated historical data fetching via `yfinance` and executes all strict feature engineering (lags, rolling stats, RSI, MACD).
* **`agents.py`**: Contains the core `BaseAgent` class and the implementation of the four specialized forecasters (Trend, Momentum, Volatility, Sequence).
* **`main.py`**: The primary execution script that orchestrates data preparation, train/test splitting, agent initialization, and performance evaluation.
* **`air_passengers_hybrid.py`**: A standalone script demonstrating the recursive multi-step hybrid forecasting pipeline.

## Tech Stack
* **Languages:** Python
* **Data Processing & Stats:** Pandas, NumPy, Statsmodels
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deep Learning:** PyTorch
* **Data Sourcing:** yfinance

## Setup & Usage

**1. Install required dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost statsmodels torch yfinance matplotlib
