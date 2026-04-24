# Apple Stock Price Forecasting
### Time Series Analysis using ARIMA · Prophet · XGBoost · LSTM

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Included-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project is a complete end-to-end time series forecasting pipeline built on Apple Inc. (AAPL) daily stock data from January 2020 to January 2026. The goal is to predict the next day's closing price using historical price data, comparing four different modelling approaches — a classical statistical model, a decomposition-based model, a tree-based ML model, and a deep learning model.

The project is deployed as an interactive Streamlit web application where users can explore the data, compare model results, and generate forecasts.

---

## App Preview

![App Screenshot](./Screenshot.png)

---

## Live Demo

🔗 [Open the Streamlit App](https://apple-stock-price-forecasting.streamlit.app/)

---

## Project Structure

```
stock_forecasting/
├── app.py                  ← Streamlit web application (4 pages)
├── notebook.ipynb          ← Complete analysis notebook (6 phases)
├── lstm_model.h5           ← Pre-trained LSTM model
├── scaler.pkl              ← Fitted MinMaxScaler
├── aapl_data.csv           ← Cleaned AAPL dataset
├── requirements.txt        ← Python dependencies
├── runtime.txt             ← Python version for Streamlit Cloud
└── README.md               ← This file
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Yahoo Finance via `yfinance` |
| Ticker | AAPL (Apple Inc.) |
| Period | January 2020 – January 2026 |
| Rows | 1,508 trading days |
| Features | Close, High, Low, Open, Volume |
| Missing values | None |

> Stock data excludes weekends and public holidays — ~252 trading days per year.

---

## Methodology — 6 Phase Pipeline

### Phase 1 · Data Collection & Inspection
- Downloaded 6 years of daily AAPL data using `yfinance`
- Flattened MultiIndex columns, verified shape and data types
- Confirmed zero missing values across all 5 columns
- Date range: 1,508 trading days (Jan 2020 → Jan 2026)

### Phase 2 · Exploratory Data Analysis (EDA)
- Plotted closing price with 30-day and 90-day rolling moving averages
- Correlation heatmap revealed Open/High/Low/Close are ~0.99 correlated — confirming only `Close` is needed for modelling
- Daily returns computed using `pct_change()` — roughly bell-shaped with fat tails
- 30-day rolling volatility revealed the COVID-19 spike in March 2020
- Volume analysis identified earnings-related trading surges

### Phase 3 · Decomposition & Stationarity Testing
Multiplicative seasonal decomposition (period = 252 trading days):

| Component | Strength | Interpretation |
|-----------|----------|----------------|
| Trend | **1.00** | Extremely strong upward trend |
| Seasonality | **0.26** | Weak — no meaningful yearly cycle |

Augmented Dickey-Fuller (ADF) test:

| Series | p-value | Result |
|--------|---------|--------|
| Raw closing price | ~1.0000 | Non-stationary ✗ |
| First difference | 0.0000 | Stationary ✓ |

### Phase 4 · ARIMA Modelling
- ACF and PACF plots showed classic random walk pattern — no significant lags outside confidence bands
- Manual model: `ARIMA(1,1,0)` — AR term statistically insignificant (p = 0.438)
- `auto_arima` selected `ARIMA(0,1,0)` — a pure random walk with drift
- Every attempt to add complexity increased AIC — confirms no linear pattern exists
- Establishes **baseline RMSE = 24.71**

### Phase 5a · Prophet Modelling
- Facebook Prophet with weekly and yearly seasonality enabled
- Original model (`changepoint_prior_scale=0.05`) overshot during mid-2025 correction
- Tuned model (`changepoint_prior_scale=0.3`) improved RMSE: 28.69 → 27.34
- Component plots confirmed strong trend and weak seasonality

### Phase 5b · XGBoost Modelling
- XGBoost cannot natively understand time sequences — requires manual feature engineering
- Created 19 features from closing price alone:
  - **Lag features:** lag_1 through lag_21 (7 features)
  - **Rolling statistics:** MA_7, MA_21, MA_50, rolling_std_7, rolling_std_21
  - **Momentum:** momentum_5, momentum_10, pct_change_1, pct_change_5
  - **Calendar:** day_of_week, month, quarter
- Anti-overfit settings: `max_depth=2`, `min_child_weight=10`, L1/L2 regularisation
- Result: train RMSE = 0.54 vs test RMSE = 18.69 — overfitting detected despite regularisation
- Feature importance: `lag_1` (31.9%) + `lag_2` (21.9%) dominated — confirms random walk behaviour

### Phase 5c · LSTM Modelling
Architecture: 2-layer stacked LSTM with Dropout regularisation

```
Input  → LSTM(64, return_sequences=True) → Dropout(0.2)
       → LSTM(32)                         → Dropout(0.2)
       → Dense(1)

Total trainable parameters: 29,345
```

- Data preparation: MinMaxScaler → 60-day sliding window → 3D reshape `(samples, 60, 1)`
- Training: EarlyStopping (patience=10) triggered at epoch 20 — no overfitting
- 80/20 train-test split: 1,158 training samples, 290 test samples

### Phase 6 · Findings & Conclusion
See results table below and the final notebook section for full write-up.

---

## Results

| Model | RMSE | MAE | MAPE | vs ARIMA |
|-------|------|-----|------|----------|
| ARIMA(0,1,0) | 24.71 | 20.17 | — | baseline |
| Prophet (original) | 28.69 | 22.56 | 10.35% | -16.1% |
| Prophet (tuned) | 27.34 | 21.14 | 9.64% | -10.7% |
| XGBoost | 18.69 | 11.43 | 4.42% | +24.3% |
| **LSTM** | **8.96** | **6.88** | **3.01%** | **+63.7%** |

**Best model: LSTM** — predicting Apple's closing price within an average of $6.88 (~3%) on a stock trading around $220–270.

---

## Key Findings

**1. Apple stock follows a random walk**
Auto-ARIMA confirmed `ARIMA(0,1,0)` — a pure random walk — is the best linear model. No AR or MA term improved performance, validating the Efficient Market Hypothesis for this dataset.

**2. Strong trend, weak seasonality**
Decomposition revealed trend strength of 1.00 and seasonal strength of 0.26. Apple's price is driven by long-term momentum, not yearly cycles.

**3. XGBoost overfit despite regularisation**
With only ~1,500 rows and univariate features, XGBoost memorised training data (train RMSE = 0.54) but generalised poorly (test RMSE = 18.69). Feature importance confirmed it learned a random walk — dominated by lag_1 and lag_2. Needs more data and richer features (RSI, MACD, volume).

**4. LSTM significantly outperformed all models**
The 60-day memory window allowed LSTM to learn momentum and trend direction that ARIMA, Prophet, and XGBoost could not capture. RMSE improved by 63.7% over baseline.

**5. Prophet forecast in the right direction but overshot**
Prophet correctly identified the upward trend but was penalised when the actual price corrected in mid-2025 while Prophet continued forecasting upward.

---

## Limitations

- **Univariate model** — only closing price used; news, earnings, macro events not considered
- **One-step-ahead prediction** — uses real historical data per step, not true multi-step recursive forecast
- **XGBoost data constraint** — tree-based models need significantly more rows and richer features
- **No external signals** — sentiment, interest rates, sector data excluded

---

## Future Improvements

- Add technical indicators (RSI, MACD, Bollinger Bands) → multivariate LSTM and XGBoost
- Incorporate trading volume as additional input feature
- Sentiment analysis from financial news headlines
- Experiment with GRU or Transformer architecture
- Walk-forward validation for more realistic evaluation
- MLflow experiment tracking for systematic model comparison

---

## Streamlit App Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Key metrics, price history chart, model comparison table, ADF results |
| **EDA** | Price & MA plots, volume, daily returns, volatility, correlation heatmap, decomposition |
| **Model Comparison** | RMSE/MAE bar charts for all 4 models, insight cards, performance table |
| **Predict** | Slider to choose forecast horizon → LSTM generates day-by-day price predictions |

---

## Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/himanshu-shekhar2327/Apple-stock-price-forecasting
cd Apple-stock-price-forecasting

# Create and activate environment
conda create -n stock_app python=3.11
conda activate stock_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Requirements

```
streamlit
yfinance
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
prophet
statsmodels
joblib
plotly
pmdarima
xgboost
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Stock data fetching |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `statsmodels` | ARIMA, ADF test, decomposition |
| `pmdarima` | Auto-ARIMA parameter search |
| `prophet` | Facebook Prophet model |
| `xgboost` | XGBoost tree-based model |
| `tensorflow` / `keras` | LSTM deep learning model |
| `scikit-learn` | Scaling, evaluation metrics |
| `streamlit` | Web application deployment |

---

## Author

**Himanshu Shekhar**
- GitHub: [@himanshu-shekhar2327](https://github.com/himanshu-shekhar2327)

---

## Disclaimer

This project is for **educational purposes only**. Stock price predictions made by this model should not be used for actual investment decisions. Past price patterns do not guarantee future performance.
