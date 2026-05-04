import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import requests
import plotly.graph_objects as go

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Vivek's Trading Dashboard", layout="wide")

# ------------------ TITLE ------------------
st.markdown("<h2>📈 Smart Trading Dashboard</h2>", unsafe_allow_html=True)
st.markdown("### 👨‍💻 Developed by Vivek Gupta")

# ------------------ SEARCH (TOP - NOT SIDEBAR) ------------------
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input("🔍 Search Stock (e.g. SUZLON, TCS, INFY)", "")

with col2:
    search_clicked = st.button("Search")

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## ⚙ Settings")

timeframe = st.sidebar.selectbox("⏱ Timeframe", [
    "1m", "5m", "15m", "30m",
    "1h", "2h", "4h",
    "1d", "7d", "1mo", "3mo", "6mo", "1y"
])

show_ema = st.sidebar.checkbox("Show EMA (20)", True)

# ------------------ SMART SEARCH ------------------
def smart_search(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        data = res.json()

        quotes = data.get("quotes", [])

        for q in quotes:
            if q.get("exchange") == "NSI":
                return q.get("symbol")

        if quotes:
            return quotes[0].get("symbol")

        return None
    except:
        return None

# ------------------ TIMEFRAME LOGIC ------------------
def get_interval_period(tf):
    mapping = {
        "1m": ("1m", "5d"),
        "5m": ("5m", "5d"),
        "15m": ("15m", "1mo"),
        "30m": ("30m", "1mo"),
        "1h": ("1h", "3mo"),
        "2h": ("1h", "3mo"),
        "4h": ("1h", "6mo"),
        "1d": ("1d", "1y"),
        "7d": ("1d", "5y"),
        "1mo": ("1mo", "10y"),
        "3mo": ("1mo", "10y"),
        "6mo": ("1mo", "10y"),
        "1y": ("1mo", "max"),
    }
    return mapping.get(tf, ("1d", "1y"))

# ------------------ LOAD DATA ------------------
def load_data(symbol, timeframe):
    try:
        interval, period = get_interval_period(timeframe)

        ticker = Ticker(symbol)
        df = ticker.history(interval=interval, period=period)

        if df is None or df.empty:
            return None

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol)

        df = df.reset_index()
        return df
    except:
        return None

# ------------------ INDICATORS ------------------
def add_indicators(df):
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

# ------------------ SIGNAL ------------------
def get_signal(df):
    latest = df.iloc[-1]
    rsi = latest["RSI"]

    if latest["EMA20"] > latest["MA50"] and rsi < 70:
        return "🟢 BUY"
    elif latest["EMA20"] < latest["MA50"] and rsi > 30:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"

# ------------------ MAIN ------------------
if not user_input:
    st.info("👆 Enter stock name above (e.g. SUZLON, TCS, INFY)")

elif search_clicked:
    symbol = smart_search(user_input)

    if symbol is None:
        st.error("❌ Stock not found")
    else:
        st.success(f"🔍 Found: {symbol}")

        data = load_data(symbol, timeframe)

        if data is None:
            st.error("❌ No data available")
        else:
            data = add_indicators(data)

            price = data["close"].iloc[-1]
            st.subheader(f"💰 Price: ₹ {round(price, 2)}")

            signal = get_signal(data)
            st.subheader("📊 Signal")
            st.markdown(f"### {signal}")

            # ------------------ PRO GRAPH ------------------
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=data["date"],
                y=data["close"],
                name="Price",
                line=dict(width=3)
            ))

            fig.add_trace(go.Scatter(
                x=data["date"],
                y=data["MA50"],
                name="MA50"
            ))

            if show_ema:
                fig.add_trace(go.Scatter(
                    x=data["date"],
                    y=data["EMA20"],
                    name="EMA20",
                    line=dict(dash="dot")
                ))

            fig.update_layout(
                height=550,
                template="plotly_dark",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # RSI
            st.subheader("📉 RSI")
            st.line_chart(data["RSI"])

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Made with ❤️ by **Vivek Gupta**")