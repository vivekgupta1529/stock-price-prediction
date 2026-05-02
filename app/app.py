import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import plotly.graph_objects as go
from yahooquery import Ticker

st.set_page_config(page_title="Pro Stock Predictor", layout="wide")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Sidebar
st.sidebar.title("⚙️ Settings")
stock = st.sidebar.text_input("Stock Symbol", "TCS.NS")
period = st.sidebar.selectbox("Time Period", ["6mo", "1y", "2y"])
show_ema = st.sidebar.checkbox("Show EMA (20)", True)

st.title("📈 Pro Stock Price Prediction Dashboard")

# ---------------- DATA FETCH ----------------
try:
    ticker = Ticker(stock)
    data = ticker.history(period=period)

    if data.empty:
        st.error("❌ No data found")
        st.stop()

    # 🔥 MultiIndex fix
    if isinstance(data.index, pd.MultiIndex):
        data = data.xs(stock)

    data = data.reset_index()

    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data.set_index('Date', inplace=True)

except Exception as e:
    st.error("❌ Data fetch error")
    st.write(e)
    st.stop()

# ---------------- INDICATORS ----------------
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()
data["EMA20"] = data["Close"].ewm(span=20).mean()

delta = data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# ---------------- CHART ----------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="MA200"))

if show_ema:
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA20"))

st.plotly_chart(fig, use_container_width=True)

# RSI
st.subheader("RSI")
st.line_chart(data["RSI"])

# ---------------- SIGNAL ----------------
st.subheader("Signal")

rsi = data["RSI"].iloc[-1]

if data["MA50"].iloc[-1] > data["MA200"].iloc[-1] and rsi < 70:
    st.success("🟢 BUY")
elif data["MA50"].iloc[-1] < data["MA200"].iloc[-1] and rsi > 30:
    st.error("🔴 SELL")
else:
    st.warning("🟡 HOLD")

# ---------------- PREDICTION ----------------
last_price = data[["Close"]].iloc[-1:]
prediction = model.predict(last_price)

st.metric("Prediction", f"₹ {prediction[0]:.2f}")

# ---------------- DATA ----------------
st.subheader("Recent Data")
st.dataframe(data.tail())