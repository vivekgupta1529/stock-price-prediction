import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Vivek's Trading Dashboard", layout="wide")

# ============================================================
#  SESSION STATE
# ============================================================
if "symbol"       not in st.session_state: st.session_state.symbol       = None
if "trading_mode" not in st.session_state: st.session_state.trading_mode = False

# ============================================================
#  TRADING MODE — dark terminal CSS injected when active
# ============================================================
if st.session_state.trading_mode:
    st.markdown("""
    <style>
    /* Full dark terminal theme */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    section[data-testid="stSidebar"] {
        background-color: #090e1a !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] { background-color: #0d1526 !important; border-right: 1px solid #1e3a5f !important; }
    .stTextInput > div > div > input {
        background: #0d1526 !important; color: #7dd3fc !important;
        border: 1px solid #1e3a5f !important; font-family: 'Courier New', monospace !important;
    }
    .stButton > button {
        background: #0f3460 !important; color: #7dd3fc !important;
        border: 1px solid #1e6fbf !important; font-family: 'Courier New', monospace !important;
    }
    .stButton > button:hover { background: #1e6fbf !important; color: #ffffff !important; }
    label, .stRadio label, p, span, div { color: #94a3b8 !important; }
    .stRadio > div > label > div { color: #7dd3fc !important; }
    [data-testid="metric-container"] {
        background: #0d1526 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
#  SMART SEARCH
# ============================================================
def smart_search(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        quotes = data.get("quotes", [])
        for q in quotes:
            if q.get("exchange") == "NSI":
                return q.get("symbol")
        if quotes:
            return quotes[0].get("symbol")
        return None
    except Exception as e:
        st.warning(f"Search error: {e}")
        return None

# ============================================================
#  TIMEFRAME LOGIC
# ============================================================
def get_interval_period(tf):
    mapping = {
        "1m":  ("1m",  "5d"),
        "5m":  ("5m",  "5d"),
        "15m": ("15m", "1mo"),
        "30m": ("30m", "1mo"),
        "1h":  ("1h",  "3mo"),
        "2h":  ("1h",  "3mo"),
        "4h":  ("1h",  "6mo"),
        "1d":  ("1d",  "1y"),
        "7d":  ("1d",  "5y"),
        "1mo": ("1mo", "10y"),
        "3mo": ("1mo", "10y"),
        "6mo": ("1mo", "10y"),
        "1y":  ("1mo", "max"),
    }
    return mapping.get(tf, ("1d", "1y"))

# ============================================================
#  LOAD DATA
# ============================================================
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
    except Exception as e:
        st.warning(f"Data load error: {e}")
        return None

# ============================================================
#  INDICATORS
# ============================================================
def add_indicators(df):
    df["MA50"]  = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    df["BB_mid"]   = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # RSI
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df

# ============================================================
#  CANDLESTICK PATTERN DETECTION
# ============================================================
def detect_patterns(df):
    """
    Detects 12 professional patterns on the last 5 candles.
    Returns list of dicts: {name, type, emoji, description, candle_index}
    """
    patterns = []
    n = len(df)
    if n < 3:
        return patterns

    def body(i):     return abs(df["close"].iloc[i] - df["open"].iloc[i])
    def upper_wick(i): return df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])
    def lower_wick(i): return min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]
    def is_bull(i):  return df["close"].iloc[i] > df["open"].iloc[i]
    def is_bear(i):  return df["close"].iloc[i] < df["open"].iloc[i]
    def candle_range(i): return df["high"].iloc[i] - df["low"].iloc[i]

    # Check last 5 candles
    check_range = range(max(2, n - 5), n)

    for i in check_range:
        o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        b = body(i)
        uw = upper_wick(i)
        lw = lower_wick(i)
        cr = candle_range(i)
        if cr == 0: continue

        # ── SINGLE CANDLE PATTERNS ──────────────────────────

        # Doji — open ≈ close, indecision
        if b <= 0.05 * cr and cr > 0:
            patterns.append({
                "name": "Doji",
                "type": "neutral",
                "emoji": "➖",
                "strength": "Medium",
                "description": "Open ≈ Close. Market indecision — neither buyers nor sellers in control. Watch for breakout confirmation.",
                "idx": i
            })

        # Hammer — small body at top, long lower wick (bullish reversal)
        elif lw >= 2 * b and uw <= 0.3 * b and is_bear(i) or (lw >= 2 * b and uw <= b and is_bull(i)):
            if i > 0 and df["close"].iloc[i-1] < df["open"].iloc[i-1]:  # after downtrend
                patterns.append({
                    "name": "Hammer",
                    "type": "bullish",
                    "emoji": "🔨",
                    "strength": "Strong",
                    "description": "Long lower wick after downtrend. Buyers pushed price back up strongly. Potential bullish reversal.",
                    "idx": i
                })

        # Hanging Man — same shape as hammer but after uptrend (bearish)
        elif lw >= 2 * b and uw <= b:
            if i > 0 and df["close"].iloc[i-1] > df["open"].iloc[i-1]:  # after uptrend
                patterns.append({
                    "name": "Hanging Man",
                    "type": "bearish",
                    "emoji": "🪢",
                    "strength": "Medium",
                    "description": "Looks like a hammer but appears after uptrend. Warning: sellers are starting to push back.",
                    "idx": i
                })

        # Shooting Star — small body at bottom, long upper wick (bearish)
        elif uw >= 2 * b and lw <= 0.3 * b:
            if i > 0 and df["close"].iloc[i-1] > df["open"].iloc[i-1]:
                patterns.append({
                    "name": "Shooting Star",
                    "type": "bearish",
                    "emoji": "💫",
                    "strength": "Strong",
                    "description": "Long upper wick after uptrend. Buyers tried to push higher but sellers took control. Bearish reversal signal.",
                    "idx": i
                })

        # Inverted Hammer — small body, long upper wick, after downtrend (bullish)
        elif uw >= 2 * b and lw <= 0.3 * b:
            if i > 0 and df["close"].iloc[i-1] < df["open"].iloc[i-1]:
                patterns.append({
                    "name": "Inverted Hammer",
                    "type": "bullish",
                    "emoji": "🔁",
                    "strength": "Medium",
                    "description": "Long upper wick after downtrend. Buyers tried to push up — watch for follow-through confirmation.",
                    "idx": i
                })

        # Spinning Top — small body, long equal wicks
        elif b <= 0.3 * cr and uw >= 0.25 * cr and lw >= 0.25 * cr:
            patterns.append({
                "name": "Spinning Top",
                "type": "neutral",
                "emoji": "🌀",
                "strength": "Weak",
                "description": "Small body with long wicks on both sides. Market is very undecided. Wait for the next candle direction.",
                "idx": i
            })

        # Marubozu Bull — full body, no wicks (very strong buyers)
        elif b >= 0.95 * cr and is_bull(i):
            patterns.append({
                "name": "Bullish Marubozu",
                "type": "bullish",
                "emoji": "💚",
                "strength": "Very Strong",
                "description": "Full green candle with no wicks. Extremely strong buying pressure — bulls fully in control all session.",
                "idx": i
            })

        # Marubozu Bear — full body, no wicks (very strong sellers)
        elif b >= 0.95 * cr and is_bear(i):
            patterns.append({
                "name": "Bearish Marubozu",
                "type": "bearish",
                "emoji": "🔴",
                "strength": "Very Strong",
                "description": "Full red candle with no wicks. Extremely strong selling pressure — bears fully in control all session.",
                "idx": i
            })

        # ── TWO CANDLE PATTERNS ─────────────────────────────

        if i >= 1:
            prev_b = body(i-1)

            # Bullish Engulfing
            if (is_bear(i-1) and is_bull(i) and
                    df["open"].iloc[i]  < df["close"].iloc[i-1] and
                    df["close"].iloc[i] > df["open"].iloc[i-1]):
                patterns.append({
                    "name": "Bullish Engulfing",
                    "type": "bullish",
                    "emoji": "🟢",
                    "strength": "Very Strong",
                    "description": "Green candle fully swallows the previous red candle. Strong bullish reversal — buyers took complete control.",
                    "idx": i
                })

            # Bearish Engulfing
            elif (is_bull(i-1) and is_bear(i) and
                    df["open"].iloc[i]  > df["close"].iloc[i-1] and
                    df["close"].iloc[i] < df["open"].iloc[i-1]):
                patterns.append({
                    "name": "Bearish Engulfing",
                    "type": "bearish",
                    "emoji": "🔴",
                    "strength": "Very Strong",
                    "description": "Red candle fully swallows the previous green candle. Strong bearish reversal — sellers took complete control.",
                    "idx": i
                })

            # Bullish Harami — small candle inside big bearish candle
            elif (is_bear(i-1) and is_bull(i) and
                    df["open"].iloc[i]  > df["close"].iloc[i-1] and
                    df["close"].iloc[i] < df["open"].iloc[i-1] and
                    b < prev_b * 0.6):
                patterns.append({
                    "name": "Bullish Harami",
                    "type": "bullish",
                    "emoji": "🤱",
                    "strength": "Medium",
                    "description": "Small green candle inside big red candle. Selling momentum is slowing down — potential reversal forming.",
                    "idx": i
                })

            # Bearish Harami
            elif (is_bull(i-1) and is_bear(i) and
                    df["open"].iloc[i]  < df["close"].iloc[i-1] and
                    df["close"].iloc[i] > df["open"].iloc[i-1] and
                    b < prev_b * 0.6):
                patterns.append({
                    "name": "Bearish Harami",
                    "type": "bearish",
                    "emoji": "🤱",
                    "strength": "Medium",
                    "description": "Small red candle inside big green candle. Buying momentum is slowing down — watch for trend reversal.",
                    "idx": i
                })

        # ── THREE CANDLE PATTERNS ───────────────────────────

        if i >= 2:
            # Morning Star (bullish reversal at bottom)
            if (is_bear(i-2) and body(i-1) <= 0.3 * candle_range(i-1) and is_bull(i) and
                    df["close"].iloc[i] > (df["open"].iloc[i-2] + df["close"].iloc[i-2]) / 2):
                patterns.append({
                    "name": "Morning Star",
                    "type": "bullish",
                    "emoji": "🌅",
                    "strength": "Very Strong",
                    "description": "3-candle pattern: big red → small indecision → big green. One of the strongest bullish reversals.",
                    "idx": i
                })

            # Evening Star (bearish reversal at top)
            elif (is_bull(i-2) and body(i-1) <= 0.3 * candle_range(i-1) and is_bear(i) and
                    df["close"].iloc[i] < (df["open"].iloc[i-2] + df["close"].iloc[i-2]) / 2):
                patterns.append({
                    "name": "Evening Star",
                    "type": "bearish",
                    "emoji": "🌇",
                    "strength": "Very Strong",
                    "description": "3-candle pattern: big green → small indecision → big red. One of the strongest bearish reversals.",
                    "idx": i
                })

            # Three White Soldiers
            elif (is_bull(i-2) and is_bull(i-1) and is_bull(i) and
                    df["close"].iloc[i]   > df["close"].iloc[i-1] and
                    df["close"].iloc[i-1] > df["close"].iloc[i-2] and
                    body(i) > 0 and body(i-1) > 0 and body(i-2) > 0):
                patterns.append({
                    "name": "Three White Soldiers",
                    "type": "bullish",
                    "emoji": "⚔️",
                    "strength": "Very Strong",
                    "description": "3 consecutive rising green candles. Very strong bullish momentum — bulls in complete control.",
                    "idx": i
                })

            # Three Black Crows
            elif (is_bear(i-2) and is_bear(i-1) and is_bear(i) and
                    df["close"].iloc[i]   < df["close"].iloc[i-1] and
                    df["close"].iloc[i-1] < df["close"].iloc[i-2]):
                patterns.append({
                    "name": "Three Black Crows",
                    "type": "bearish",
                    "emoji": "🐦‍⬛",
                    "strength": "Very Strong",
                    "description": "3 consecutive falling red candles. Very strong bearish momentum — sellers in complete control.",
                    "idx": i
                })

    # Deduplicate: one entry per pattern name only (keep the latest occurrence)
    seen = {}
    for p in patterns:
        seen[p["name"]] = p   # later candle overwrites earlier same-name pattern
    return list(seen.values())

# ============================================================
#  SIGNAL
# ============================================================
def get_signal(df):
    if df["RSI"].isna().all():
        return "🟡 Not enough data — try a longer timeframe"
    latest = df.iloc[-1]
    rsi    = latest["RSI"]
    if pd.isna(rsi):
        return "🟡 Hold — Not enough RSI data"
    if latest["EMA20"] > latest["MA50"] and rsi < 70:
        return "🟢 Strong Buy — Trend is bullish and RSI has room to grow"
    elif latest["EMA20"] < latest["MA50"] and rsi > 30:
        return "🔴 Sell — Trend is bearish, consider exiting your position"
    else:
        return "🟡 Hold — No clear trend right now, wait for a stronger signal"

# ============================================================
#  PRO CHART
# ============================================================
def build_pro_chart(data, show_ema, show_ma50, show_ma200, show_bb, show_macd, patterns, trading_mode):
    rows    = 3 if not show_macd else 4
    heights = [0.55, 0.15, 0.15, 0.15] if show_macd else [0.60, 0.20, 0.20]
    titles  = ("", "Volume", "RSI (14)", "MACD") if show_macd else ("", "Volume", "RSI (14)")

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
        subplot_titles=titles
    )

    x = data["date"]

    # ── Bollinger Bands (behind candles) ────────────────────
    if show_bb:
        fig.add_trace(go.Scatter(
            x=x, y=data["BB_upper"], name="BB Upper",
            line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=data["BB_lower"], name="BB Lower",
            line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.05)",
            showlegend=False), row=1, col=1)

    # ── Candlestick ─────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=data["open"], high=data["high"],
        low=data["low"],   close=data["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── Pattern annotations on chart ────────────────────────
    for p in patterns:
        idx = p["idx"]
        if idx >= len(data): continue
        y_pos = data["high"].iloc[idx] * 1.005 if p["type"] == "bearish" else data["low"].iloc[idx] * 0.995
        ay    = -30 if p["type"] == "bullish" else 30
        color = "#26a69a" if p["type"] == "bullish" else ("#ef5350" if p["type"] == "bearish" else "#f59e0b")
        fig.add_annotation(
            x=data["date"].iloc[idx],
            y=y_pos,
            text=f"{p['emoji']} {p['name']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=color,
            ay=ay,
            font=dict(size=10, color=color),
            bgcolor="rgba(9,14,26,0.85)",
            bordercolor=color,
            borderwidth=1,
            row=1, col=1
        )

    # ── Moving averages ─────────────────────────────────────
    if show_ma50:
        fig.add_trace(go.Scatter(x=x, y=data["MA50"],  name="MA 50",
                                 line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
    if show_ma200:
        fig.add_trace(go.Scatter(x=x, y=data["MA200"], name="MA 200",
                                 line=dict(color="#a78bfa", width=1.2)), row=1, col=1)
    if show_ema:
        fig.add_trace(go.Scatter(x=x, y=data["EMA20"], name="EMA 20",
                                 line=dict(color="#38bdf8", width=1.2, dash="dot")), row=1, col=1)

    # ── Volume ──────────────────────────────────────────────
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(data["close"], data["open"])]
    fig.add_trace(go.Bar(x=x, y=data["volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.7, showlegend=False), row=2, col=1)

    # ── RSI ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=x, y=data["RSI"], name="RSI",
                             line=dict(color="#f472b6", width=1.5)), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, row=3, col=1, fillcolor="rgba(239,83,80,0.08)",  line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=3, col=1, fillcolor="rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, row=3, col=1, line=dict(color="#ef5350", width=0.8, dash="dash"))
    fig.add_hline(y=30, row=3, col=1, line=dict(color="#26a69a", width=0.8, dash="dash"))
    fig.add_hline(y=50, row=3, col=1, line=dict(color="#64748b", width=0.5, dash="dot"))

    # ── MACD (optional 4th panel) ────────────────────────────
    if show_macd:
        fig.add_trace(go.Scatter(x=x, y=data["MACD"],        name="MACD",
                                 line=dict(color="#818cf8", width=1.3)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=data["MACD_signal"], name="Signal",
                                 line=dict(color="#fb923c", width=1.3)), row=4, col=1)
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in data["MACD_hist"]]
        fig.add_trace(go.Bar(x=x, y=data["MACD_hist"], name="Histogram",
                             marker_color=hist_colors, opacity=0.6, showlegend=False), row=4, col=1)

    # ── Layout ───────────────────────────────────────────────
    bg   = "#090e1a" if trading_mode else "rgba(0,0,0,0)"
    grid = "rgba(30,58,95,0.4)" if trading_mode else "rgba(100,116,139,0.15)"

    fig.update_layout(
        height=750 if show_macd else 700,
        template="plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor",
                   spikecolor="#38bdf8", spikethickness=1, spikedash="dot",
                   showgrid=True, gridcolor=grid),
        yaxis=dict(showspikes=True, spikecolor="#38bdf8", spikethickness=1, spikedash="dot",
                   showgrid=True, gridcolor=grid),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        font=dict(family="Courier New" if trading_mode else "sans-serif", color="#7dd3fc" if trading_mode else "#e2e8f0"),
    )
    fig.update_yaxes(range=[0, 100],         row=3, col=1)
    fig.update_yaxes(rangemode="nonnegative", row=2, col=1)

    return fig

# ============================================================
#  HEADER ROW — title + trading mode toggle
# ============================================================
h1, h2 = st.columns([5, 1])
with h1:
    if st.session_state.trading_mode:
        st.markdown("""
        <div style="font-family:'Courier New',monospace;font-size:1.4rem;
                    color:#7dd3fc;letter-spacing:0.15em;padding:4px 0;">
            ⚡ VIVEK'S TRADING TERMINAL
        </div>
        <div style="font-family:'Courier New',monospace;font-size:0.75rem;
                    color:#334155;letter-spacing:0.2em;">
            PROFESSIONAL MODE • LIVE ANALYSIS
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='margin:0'>📈 Smart Trading Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0;color:#64748b;font-size:0.85rem;'>👨‍💻 Developed by Vivek Gupta</p>", unsafe_allow_html=True)

with h2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    mode_label = "🌙 Exit Trading Mode" if st.session_state.trading_mode else "⚡ Trading Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.trading_mode = not st.session_state.trading_mode
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ============================================================
#  SEARCH ROW
# ============================================================
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("🔍 Search Stock (e.g. SUZLON, TCS, INFY)", "",
                               placeholder="Enter NSE symbol or company name...")
with col2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    search_clicked = st.button("🔎 Search", use_container_width=True)

if search_clicked and user_input:
    found = smart_search(user_input)
    if found:
        st.session_state.symbol = found
    else:
        st.error("❌ Stock not found. Try full name or NSE ticker (e.g. SUZLON.NS)")

# ============================================================
#  TIMEFRAME ROW
# ============================================================
tf_options = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "7d", "1mo", "3mo", "6mo", "1y"]
timeframe = st.radio("Timeframe", options=tf_options, index=tf_options.index("1d"),
                     horizontal=True, label_visibility="collapsed")

# ============================================================
#  SIDEBAR — advanced options
# ============================================================
st.sidebar.markdown("## ⚙️ Overlays")
show_ema  = st.sidebar.checkbox("EMA 20",  True)
show_ma50 = st.sidebar.checkbox("MA 50",   True)
show_ma200= st.sidebar.checkbox("MA 200",  False)
show_bb   = st.sidebar.checkbox("Bollinger Bands", False)

st.sidebar.markdown("## 📊 Extra Panels")
show_macd = st.sidebar.checkbox("MACD Panel", False)

st.sidebar.markdown("## 🕯️ Pattern Detection")
show_patterns = st.sidebar.checkbox("Show Candlestick Patterns", True)

# ============================================================
#  MAIN RENDER
# ============================================================
if not st.session_state.symbol:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Search for a stock above and click Search to begin")
else:
    symbol = st.session_state.symbol
    data   = load_data(symbol, timeframe)

    if data is None:
        st.error("❌ No data available for this symbol / timeframe combination.")
    else:
        data     = add_indicators(data)
        patterns = detect_patterns(data) if show_patterns else []

        # ── Metrics ──────────────────────────────────────────
        price   = data["close"].iloc[-1]
        change  = data["close"].iloc[-1] - data["close"].iloc[-2]
        pct     = (change / data["close"].iloc[-2]) * 100
        high      = data["high"].max()
        low       = data["low"].min()
        high_date = data.loc[data["high"].idxmax(), "date"]
        low_date  = data.loc[data["low"].idxmin(),  "date"]
        high_date_str = pd.to_datetime(high_date).strftime("%d %b %Y")
        low_date_str  = pd.to_datetime(low_date).strftime("%d %b %Y")
        vol_cur   = data["volume"].iloc[-1]
        vol_week  = data["volume"].tail(5).mean()    # ~1 week (5 trading days)
        vol_month = data["volume"].tail(20).mean()   # ~1 month (20 trading days)
        signal  = get_signal(data)
        rsi_val = data["RSI"].iloc[-1]
        atr_val = data["ATR"].iloc[-1]

        delta_color = "#26a69a" if change >= 0 else "#ef5350"
        delta_arrow = "▲" if change >= 0 else "▼"
        rsi_str = f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "N/A"
        atr_str = f"{atr_val:.2f}" if not pd.isna(atr_val) else "N/A"
        def vol_spike(cur, avg, label):
            if avg == 0 or pd.isna(avg): return f"{label}: N/A"
            pct = cur / avg * 100
            if pct >= 200:   tag = "🔥 Spike"
            elif pct >= 130: tag = "⬆ High"
            elif pct >= 70:  tag = "➡ Normal"
            else:            tag = "⬇ Low"
            return f"{label}: {pct:.0f}% {tag}"
        week_str  = vol_spike(vol_cur, vol_week,  "1W")
        month_str = vol_spike(vol_cur, vol_month, "1M")
        rsi_color = "#ef5350" if not pd.isna(rsi_val) and rsi_val > 70 else ("#26a69a" if not pd.isna(rsi_val) and rsi_val < 30 else "#e2e8f0")
        rsi_label = "Overbought" if not pd.isna(rsi_val) and rsi_val > 70 else ("Oversold" if not pd.isna(rsi_val) and rsi_val < 30 else "Neutral")

        # Smart volume formatter — handles index (small) and stock (large) volumes
        def fmt_volume(v):
            if pd.isna(v) or v == 0:
                return "N/A", "No volume data"
            elif v >= 1e7:
                return f"{v/1e7:.2f} Cr", "Crores"
            elif v >= 1e5:
                return f"{v/1e5:.2f} L", "Lakhs"
            elif v >= 1e3:
                return f"{v/1e3:.2f} K", "Thousands"
            else:
                return f"{int(v):,}", "Units"
        vol_display, vol_unit = fmt_volume(vol_cur)

        st.markdown(f"""
        <style>
        .mvg-row {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin: 10px 0 14px;
        }}
        .mvg-card {{
            background: #0d1526;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 12px 14px;
            min-width: 0;
        }}
        .mvg-label {{
            font-size: 11px;
            color: #64748b;
            letter-spacing: 0.04em;
            margin-bottom: 5px;
        }}
        .mvg-value {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #e2e8f0;
            word-break: break-all;
        }}
        .mvg-sub {{
            font-size: 11px;
            margin-top: 4px;
            color: #64748b;
        }}
        </style>
        <div class="mvg-row">
          <div class="mvg-card">
            <div class="mvg-label">💰 LTP (₹)</div>
            <div class="mvg-value">₹{price:,.2f}</div>
            <div class="mvg-sub" style="color:{delta_color};">{delta_arrow} {abs(change):.2f} ({pct:+.2f}%)</div>
          </div>
          <div class="mvg-card">
            <div class="mvg-label">📈 Period High</div>
            <div class="mvg-value">₹{high:,.2f}</div>
            <div class="mvg-sub">{high_date_str}</div>
          </div>
          <div class="mvg-card">
            <div class="mvg-label">📉 Period Low</div>
            <div class="mvg-value">₹{low:,.2f}</div>
            <div class="mvg-sub">{low_date_str}</div>
          </div>
          <div class="mvg-card">
            <div class="mvg-label">📐 RSI (14)</div>
            <div class="mvg-value" style="color:{rsi_color};">{rsi_str}</div>
            <div class="mvg-sub">{rsi_label}</div>
          </div>
          <div class="mvg-card">
            <div class="mvg-label">📏 ATR (14)</div>
            <div class="mvg-value">{atr_str}</div>
            <div class="mvg-sub">Volatility</div>
          </div>
          <div class="mvg-card">
            <div class="mvg-label">📦 Volume</div>
            <div class="mvg-value">{vol_display}</div>
            <div class="mvg-sub" style="line-height:1.7;">{vol_unit}<br>{week_str}<br>{month_str}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Signal banner ─────────────────────────────────────
        sig_lower = signal.lower()
        if "buy" in sig_lower:
            bg = "linear-gradient(135deg,#064e3b,#065f46)"; border = "#10b981"; emoji = "🚀"
        elif "sell" in sig_lower:
            bg = "linear-gradient(135deg,#7f1d1d,#991b1b)"; border = "#ef4444"; emoji = "🔻"
        else:
            bg = "linear-gradient(135deg,#1c1917,#292524)";  border = "#f59e0b"; emoji = "⏳"

        st.markdown(f"""
        <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                    padding:16px 24px;margin:10px 0 6px;display:flex;align-items:center;gap:16px;">
            <span style="font-size:2.2rem;line-height:1;">{emoji}</span>
            <div>
                <div style="font-size:11px;color:#94a3b8;letter-spacing:0.09em;
                            text-transform:uppercase;margin-bottom:4px;">📊 Trading Signal</div>
                <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">{signal}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Pattern cards ─────────────────────────────────────
        if show_patterns and patterns:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown("#### 🕯️ Detected Candlestick Patterns")
            cols = st.columns(min(len(patterns), 4))
            for i, p in enumerate(patterns[-4:]):  # show latest 4
                col_i = i % len(cols)
                color = "#26a69a" if p["type"] == "bullish" else ("#ef5350" if p["type"] == "bearish" else "#f59e0b")
                strength_map = {"Very Strong": "🔥🔥🔥", "Strong": "🔥🔥", "Medium": "🔥", "Weak": "💧"}
                strength_icon = strength_map.get(p["strength"], "🔥")
                cols[col_i].markdown(f"""
                <div style="border:1px solid {color};border-radius:10px;padding:12px 14px;
                            background:rgba(0,0,0,0.3);height:100%;">
                    <div style="font-size:1.4rem;margin-bottom:4px;">{p['emoji']}</div>
                    <div style="font-weight:700;font-size:0.9rem;color:{color};">{p['name']}</div>
                    <div style="font-size:0.75rem;color:#64748b;margin:2px 0 6px;">
                        {strength_icon} {p['strength']}
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.4;">{p['description']}</div>
                </div>""", unsafe_allow_html=True)

        elif show_patterns:
            st.info("🕯️ No strong candlestick patterns detected in the last 5 candles.")

        st.divider()

        # ── Chart ─────────────────────────────────────────────
        fig = build_pro_chart(data, show_ema, show_ma50, show_ma200,
                              show_bb, show_macd, patterns,
                              st.session_state.trading_mode)
        st.plotly_chart(fig, use_container_width=True)

        # ── Pattern legend (trading mode) ─────────────────────
        if st.session_state.trading_mode and show_patterns and patterns:
            st.markdown("---")
            st.markdown("**📖 Pattern Log (Trading Mode)**")
            for p in patterns:
                color = "🟢" if p["type"]=="bullish" else ("🔴" if p["type"]=="bearish" else "🟡")
                candle_time = data["date"].iloc[p["idx"]]
                st.markdown(f"{color} **{p['emoji']} {p['name']}** — {p['strength']} — `{candle_time}` — {p['description']}")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("---")
st.markdown("Made with ❤️ by **Vivek Gupta** &nbsp;|&nbsp; Powered by Yahoo Finance & Plotly")
