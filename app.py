# ════════════════════════════════════════════════════════
#  AAPL Stock Price Forecasting App
#   Foundation + Sidebar + Dashboard
# ════════════════════════════════════════════════════════

import os
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AAPL Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load saved artifacts (cached so they load only once) ──
@st.cache_resource
def load_artifacts():
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('aapl_data.csv', index_col='Date', parse_dates=True)
    return df

model, scaler = load_artifacts()
df            = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 AAPL Forecaster")
    st.caption("Apple Inc. stock price prediction")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Dashboard", "EDA", "Model Comparison", "Predict"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("About this app")
    st.info(
        "Built using LSTM deep learning model trained on "
        "Apple stock data (2020–2026). "
        "Models compared: ARIMA · Prophet · LSTM"
    )


    # ── Helper: matplotlib style ───────────────────────────────
def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor']   = 'white'


# ════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("📈 Apple Stock Price Forecasting App")
    # st.caption("Time Series Analysis Project")

    st.divider()

    
    st.title("Dashboard")
    st.caption("Apple Inc. (AAPL) · Jan 2020 – Jan 2026 · LSTM forecasting")
    st.divider()

    # ── Row 1: Key metric cards ────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    latest_price  = df['Close'].iloc[-1]
    prev_price    = df['Close'].iloc[-2]
    price_change  = latest_price - prev_price
    price_pct     = (price_change / prev_price) * 100

    col1.metric(
        label="Latest close price",
        value=f"${latest_price:.2f}",
        delta=f"{price_pct:+.2f}% vs prev day"
    )
    col2.metric(
        label="LSTM RMSE",
        value="8.96",
        delta="63.7% better than ARIMA",
        delta_color="normal"
    )
    col3.metric(
        label="LSTM MAPE",
        value="3.01%",
        delta="avg prediction error",
        delta_color="off"
    )
    col4.metric(
        label="Total trading days",
        value=f"{len(df):,}",
        delta="Jan 2020 → Jan 2026",
        delta_color="off"
    )

    st.divider()

    # ── Row 2: Main price chart ────────────────────────
    st.subheader("Closing price history")

    set_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df.index, df['Close'],
            color='steelblue', linewidth=1.2, label='Close price')

    # Add 90-day MA overlay
    ma90 = df['Close'].rolling(90).mean()
    ax.plot(df.index, ma90,
            color='coral', linewidth=1.5,
            linestyle='--', label='90-day MA', alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Row 3: Two columns ─────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Model comparison — RMSE")

        results = pd.DataFrame({
            'Model'  : ['ARIMA(0,1,0)', 'Prophet (tuned)', 'LSTM'],
            'RMSE'   : [24.71, 27.34, 8.96],
            'MAE'    : [20.17, 21.14, 6.88],
            'MAPE'   : ['—', '9.64%', '3.01%']
        })

        # Highlight LSTM row
        def highlight_lstm(row):
            if row['Model'] == 'LSTM':
                return ['background-color: #e8f5e9'] * len(row)
            return [''] * len(row)

        st.dataframe(
            results.style.apply(highlight_lstm, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # Bar chart of RMSE
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        colors = ['#ef9a9a', '#ef9a9a', '#66bb6a']
        bars = ax2.barh(
            results['Model'], results['RMSE'],
            color=colors, edgecolor='white', height=0.5
        )
        ax2.set_xlabel('RMSE (lower is better)')
        ax2.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
        ax2.set_xlim(0, 35)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_right:
        st.subheader("Data characteristics")

        # Decomposition strength
        st.metric("Trend strength",    "1.00",  "very strong upward trend")
        st.metric("Seasonal strength", "0.26",  "weak — no yearly cycle",
                  delta_color="off")

        st.divider()

        # Price range stats
        st.caption("Price statistics (full dataset)")
        stats = pd.DataFrame({
            'Metric': ['Min price', 'Max price', 'Mean price', 'Std deviation'],
            'Value' : [
                f"${df['Close'].min():.2f}",
                f"${df['Close'].max():.2f}",
                f"${df['Close'].mean():.2f}",
                f"${df['Close'].std():.2f}"
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

        st.divider()
        st.caption("ADF stationarity test")
        adf_df = pd.DataFrame({
            'Series'   : ['Raw close price', 'First difference'],
            'p-value'  : ['~1.0000', '0.0000'],
            'Result'   : ['Non-stationary ✗', 'Stationary ✓']
        })
        st.dataframe(adf_df, use_container_width=True, hide_index=True)



# ════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ════════════════════════════════════════════════════════
elif page == "EDA":

    st.title("Exploratory Data Analysis")
    st.caption("Visual exploration of Apple stock data before modelling")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Price & moving averages",
        "Volume & returns",
        "Volatility",
        "Decomposition"
    ])

    # ── Tab 1: Price & MA ──────────────────────────────
    with tab1:
        st.subheader("Closing price with moving averages")
        st.caption("Rolling averages smooth out daily noise to reveal the underlying trend.")

        ma30 = df['Close'].rolling(30).mean()
        ma90 = df['Close'].rolling(90).mean()

        set_style()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['Close'], color='steelblue',
                linewidth=0.8, alpha=0.6, label='Close price')
        ax.plot(df.index, ma30, color='coral',
                linewidth=1.5, label='30-day MA')
        ax.plot(df.index, ma90, color='seagreen',
                linewidth=1.5, label='90-day MA')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "The 30-day MA reacts quickly to price changes. "
            "The 90-day MA shows the broader trend direction. "
            "When price stays above the 90-day MA, it signals a sustained uptrend."
        )

    # ── Tab 2: Volume & Returns ────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price & volume")
            st.caption("Volume spikes often coincide with earnings or major news events.")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                            sharex=True)
            ax1.plot(df.index, df['Close'],
                     color='steelblue', linewidth=1)
            ax1.set_ylabel('Price (USD)')
            ax1.set_title('Closing price')

            ax2.bar(df.index, df['Volume'],
                    color='steelblue', alpha=0.5, width=1)
            ax2.set_ylabel('Volume')
            ax2.set_title('Trading volume')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Daily returns distribution")
            st.caption("Returns should be roughly bell-shaped around zero.")

            daily_returns = df['Close'].pct_change() * 100

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
            ax1.plot(df.index, daily_returns,
                     color='coral', linewidth=0.7)
            ax1.axhline(0, color='black',
                        linewidth=0.8, linestyle='--')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Daily returns over time')

            ax2.hist(daily_returns.dropna(), bins=60,
                     color='steelblue', edgecolor='white')
            ax2.set_xlabel('Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Return distribution')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Stats
            st.caption("Return statistics")
            ret_stats = pd.DataFrame({
                'Metric': ['Mean daily return', 'Std deviation',
                           'Best day', 'Worst day'],
                'Value': [
                    f"{daily_returns.mean():.3f}%",
                    f"{daily_returns.std():.3f}%",
                    f"{daily_returns.max():.2f}%",
                    f"{daily_returns.min():.2f}%"
                ]
            })
            st.dataframe(ret_stats, use_container_width=True,
                         hide_index=True)

    # ── Tab 3: Volatility ──────────────────────────────
    with tab3:
        st.subheader("30-day rolling volatility")
        st.caption(
            "Volatility = rolling standard deviation of daily returns. "
            "High volatility means harder-to-forecast periods."
        )

        daily_returns = df['Close'].pct_change() * 100
        vol30 = daily_returns.rolling(30).std()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, vol30, color='coral', linewidth=1)
        ax.fill_between(df.index, vol30, alpha=0.2, color='coral')
        ax.set_ylabel('Std of daily return (%)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "Notice the large volatility spike in early 2020 (COVID-19 crash). "
            "These high-volatility regimes are where all models struggle most — "
            "unpredictable external events cannot be learned from price history alone."
        )

        # Correlation heatmap
        st.subheader("Feature correlation heatmap")
        st.caption("Shows how closely related Open, High, Low, Close, and Volume are.")

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(4, 3))
        corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f',
                    cmap='Blues', ax=ax, square=False,
                    annot_kws={"size": 8},
                    cbar_kws={'shrink': 0.6})
        ax.set_title('OHLCV correlation',fontsize=8)
        plt.tight_layout(pad=0.5)
        # st.pyplot(fig)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        st.info(
            "Open, High, Low, Close are ~0.99 correlated — "
            "they carry almost identical information. "
            "This is why we use only Close price for modelling."
        )

    # ── Tab 4: Decomposition ───────────────────────────
    with tab4:
        st.subheader("Seasonal decomposition")
        st.caption(
            "Multiplicative decomposition splits the series into "
            "Trend × Seasonality × Residual."
        )

        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp = seasonal_decompose(
            df['Close'], model='multiplicative', period=252
        )

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        components = [
            (decomp.observed,  'Observed',  'steelblue'),
            (decomp.trend,     'Trend',     'coral'),
            (decomp.seasonal,  'Seasonal',  'seagreen'),
            (decomp.resid,     'Residual',  'purple'),
        ]
        for ax, (data, label, color) in zip(axes, components):
            ax.plot(data, color=color, linewidth=0.8)
            ax.set_ylabel(label, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Strength scores
        col1, col2 = st.columns(2)
        trend_strength = 1 - (
            decomp.resid.var() /
            (decomp.trend + decomp.resid).var()
        )
        seasonal_strength = 1 - (
            decomp.resid.var() /
            (decomp.seasonal + decomp.resid).var()
        )
        col1.metric("Trend strength",    f"{trend_strength:.4f}",
                    "Very strong — price driven by trend")
        col2.metric("Seasonal strength", f"{seasonal_strength:.4f}",
                    "Weak — no strong yearly cycle",
                    delta_color="off")
        

# ════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════
elif page == "Model Comparison":

    st.title("Model Comparison")
    st.caption("ARIMA vs Prophet vs LSTM on Apple stock (test period: Oct 2024 – Jan 2026)")
    st.divider()

    # ── Metrics table ──────────────────────────────────
    st.subheader("Performance metrics")

    results = pd.DataFrame({
        'Model'   : ['ARIMA(0,1,0)', 'Prophet (original)',
                     'Prophet (tuned)', 'LSTM'],
        'RMSE'    : [24.71, 28.69, 27.34, 8.96],
        'MAE'     : [20.17, 22.56, 21.14, 6.88],
        'MAPE'    : ['—', '10.35%', '9.64%', '3.01%'],
        'vs ARIMA': ['baseline', '-16.1%', '-10.7%', '+63.7%']
    })

    def highlight_best(row):
        if row['Model'] == 'LSTM':
            return ['background-color: #e8f5e9; font-weight: bold'] * len(row)
        if row['Model'] == 'ARIMA(0,1,0)':
            return ['background-color: #e3f2fd'] * len(row)
        return [''] * len(row)

    st.dataframe(
        results.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ── Bar charts side by side ────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['ARIMA', 'Prophet\n(orig)', 'Prophet\n(tuned)', 'LSTM']
        rmse   = [24.71, 28.69, 27.34, 8.96]
        colors = ['#90caf9', '#ef9a9a', '#ef9a9a', '#66bb6a']
        bars   = ax.bar(models, rmse, color=colors,
                        edgecolor='white', width=0.5)
        ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_ylim(0, 35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("MAE comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        mae    = [20.17, 22.56, 21.14, 6.88]
        bars   = ax.bar(models, mae, color=colors,
                        edgecolor='white', width=0.5)
        ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
        ax.set_ylabel('MAE (lower is better)')
        ax.set_ylim(0, 30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Insight boxes ──────────────────────────────────
    st.subheader("Key insights")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**ARIMA** confirmed the random walk hypothesis. "
            "Best auto-ARIMA model was (0,1,0) — adding any "
            "AR or MA terms made it worse."
        )
    with c2:
        st.warning(
            "**Prophet** forecast in the right direction "
            "(upward trend) but overshot during the mid-2025 "
            "correction, inflating RMSE above ARIMA."
        )
    with c3:
        st.success(
            "**LSTM** outperformed both by 63.7%. "
            "Its 60-day memory window captured momentum "
            "and trend direction that linear models missed."
        )



# ════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT
# ════════════════════════════════════════════════════════
elif page == "Predict":

    st.title("Predict future prices")
    st.caption("Uses the pre-trained LSTM model to forecast AAPL closing price")
    st.divider()

    WINDOW = 60

    # ── Controls ───────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_days = st.slider(
            "Number of days to forecast", 7, 60, 30
        )
        st.caption(
            "Note: longer forecasts are less accurate because "
            "each prediction feeds into the next."
        )
        run_btn = st.button("Run forecast", type="primary",
                            use_container_width=True)

    with col2:
        st.info(
            f"The model will look at the last **{WINDOW} days** "
            f"of real price data, then predict the next "
            f"**{forecast_days} days** step by step. "
            "Each predicted price is used as input for the next prediction."
        )

    st.divider()

    if run_btn:
        with st.spinner("Generating forecast..."):

            # Get last WINDOW days of real data
            last_data = df['Close'].values[-WINDOW:]

            # Scale
            last_scaled = scaler.transform(
                last_data.reshape(-1, 1)
            )

            # Generate predictions step by step
            predictions = []
            current_seq = last_scaled.copy().tolist()

            for _ in range(forecast_days):
                X_input = np.array(current_seq[-WINDOW:])
                X_input = X_input.reshape(1, WINDOW, 1)

                pred_scaled = model.predict(X_input, verbose=0)
                pred_price  = scaler.inverse_transform(
                    pred_scaled
                )[0][0]
                predictions.append(pred_price)

                current_seq.append(pred_scaled[0].tolist())

            # Build dates for forecast
            last_date    = df.index[-1]
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days
            )

            forecast_df = pd.DataFrame({
                'Date'            : future_dates,
                'Predicted Price' : predictions
            }).set_index('Date')

        st.success("Forecast complete!")

        # ── Metric summary ─────────────────────────────
        first_pred = predictions[0]
        last_pred  = predictions[-1]
        total_chg  = ((last_pred - first_pred) / first_pred) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Last known price",
                  f"${df['Close'].iloc[-1]:.2f}")
        c2.metric("Day 1 forecast",
                  f"${first_pred:.2f}",
                  f"{((first_pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100:+.2f}%")
        c3.metric(f"Day {forecast_days} forecast",
                  f"${last_pred:.2f}",
                  f"{total_chg:+.2f}% over forecast period")

        st.divider()

        # ── Forecast plot ──────────────────────────────
        st.subheader("Forecast chart")

        # Show last 90 days of history + forecast
        history_days  = 90
        history       = df['Close'].iloc[-history_days:]

        set_style()
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(history.index, history.values,
                color='steelblue', linewidth=1.2,
                label='Historical price')

        ax.plot(forecast_df.index,
                forecast_df['Predicted Price'],
                color='seagreen', linewidth=2,
                linestyle='--', label='LSTM forecast',
                marker='o', markersize=3)

        # Vertical line at forecast start
        ax.axvline(x=df.index[-1], color='gray',
                   linewidth=1, linestyle=':', alpha=0.7)
        ax.text(df.index[-1], ax.get_ylim()[0],
                ' forecast start', fontsize=9,
                color='gray', va='bottom')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Forecast table ─────────────────────────────
        st.subheader("Predicted prices — day by day")

        forecast_df['Change vs prev'] = (
            forecast_df['Predicted Price'].pct_change() * 100
        ).round(2).astype(str) + '%'

        forecast_df['Predicted Price'] = forecast_df[
            'Predicted Price'
        ].round(2).apply(lambda x: f"${x:.2f}")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        st.caption(
            "Disclaimer: This forecast is for educational purposes only. "
            "Stock prices are inherently unpredictable and this model "
            "does not account for news, earnings, or market events."
        )
