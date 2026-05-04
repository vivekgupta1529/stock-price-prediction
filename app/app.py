import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import requests

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Vivek's Trading Dashboard", layout="wide")

# ------------------ TITLE ------------------
st.title("📈 Smart Trading Dashboard")
st.markdown("### 👨‍💻 Developed by Vivek Gupta")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙ Settings")

user_input = st.sidebar.text_input("🔍 Search Stock", "")
search_clicked = st.sidebar.button("🔍 Search")

timeframe = st.sidebar.selectbox("⏱ Timeframe", [
    "1m", "5m", "15m", "30m",
    "1h", "2h", "4h",
    "1d", "7d", "1mo", "3mo", "6mo", "1y"
])

show_ema = st.sidebar.checkbox("Show EMA (20)", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Vivek Gupta")
st.sidebar.markdown("🚀 Data Science Project")

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
    if tf == "1m":
        return "1m", "5d"
    elif tf == "5m":
        return "5m", "5d"
    elif tf == "15m":
        return "15m", "1mo"
    elif tf == "30m":
        return "30m", "1mo"
    elif tf == "1h":
        return "1h", "3mo"
    elif tf == "2h":
        return "1h", "3mo"
    elif tf == "4h":
        return "1h", "6mo"
    elif tf == "1d":
        return "1d", "1y"
    elif tf == "7d":
        return "1d", "5y"
    elif tf == "1mo":
        return "1mo", "10y"
    elif tf == "3mo":
        return "1mo", "10y"
    elif tf == "6mo":
        return "1mo", "10y"
    elif tf == "1y":
        return "1mo", "max"

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
    st.info("👆 Enter a stock name in sidebar and click Search (e.g. SUZLON, SAIL, TCS)")

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

            chart_data = data[["close", "MA50", "MA200"]].dropna()

            if show_ema:
                chart_data["EMA20"] = data["EMA20"]

            st.line_chart(chart_data)

            st.subheader("📉 RSI")
            st.line_chart(data["RSI"])

            with st.expander("📄 Data"):
                st.write(data.tail())

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Made with ❤️ by **Vivek Gupta**")