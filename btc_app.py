# model = load_model("notebooks/btc_lstm_model.keras")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import plotly.graph_objects as go

# ----- Page Setup -----
st.set_page_config(page_title="Bitcoin LSTM Predictor", layout="wide")
st.title("📈 Bitcoin Price Predictor (LSTM)")
st.markdown("This app predicts the **next-day closing price** of Bitcoin using LSTM neural networks.")

# ----- Load Model -----
model = load_model("notebooks/btc_lstm_model.keras") 

# ----- Load Data -----
@st.cache_data
def load_data():
    df = yf.download("BTC-USD", start="2020-01-01", end=str(date.today()))
    return df[['Close']].dropna()

df = load_data()

# ----- Preprocessing -----
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

sequence_length = 60
X = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
X = np.array(X).reshape((X.__len__(), sequence_length, 1))

# ----- Predictions -----
predicted_scaled = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_scaled)
actual_prices = df.iloc[sequence_length:]

# ----- RMSE -----
rmse = np.sqrt(mean_squared_error(actual_prices['Close'], predicted_prices))
avg_price = float(actual_prices['Close'].mean())

rmse_pct = (rmse / avg_price) * 100

# ----- Latest Prediction -----
last_60 = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))
next_day_pred_scaled = model.predict(last_60)
next_day_price = scaler.inverse_transform(next_day_pred_scaled)[0][0]

# ----- Metrics -----
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"${rmse:,.2f}")
col2.metric("Avg Price", f"${avg_price:,.2f}")
col3.metric("Error %", f"{rmse_pct:.2f}%")

# ----- Chart -----
import matplotlib.pyplot as plt

# Create chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices.index, actual_prices['Close'], label="Actual")
ax.plot(actual_prices.index, predicted_prices, label="Predicted", color='orange')
ax.set_title("BTC Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)

# Show in Streamlit
st.pyplot(fig)

# ----- Tomorrow's Prediction -----
st.subheader("🔮 Tomorrow’s BTC Price Prediction")
st.success(f"Predicted closing price for **{date.today() + timedelta(days=1)}** is: **${next_day_price:,.2f}**")

# ----- Explanation -----
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This LSTM model uses the last 60 days of Bitcoin prices to predict the next day’s price.
    
    - Data: Yahoo Finance via `yfinance`
    - Model: LSTM (TensorFlow/Keras)
    - Visuals: Streamlit + matplotlib
    """)
