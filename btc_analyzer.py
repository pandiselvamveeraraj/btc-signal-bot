"""
BTC Options Analyzer — Visual Email Edition
Sends a beautiful HTML email with inline SVG charts, gauge meters,
signal bars, candlestick pattern icons — easy to read at a glance.
"""

import requests
import pandas as pd
import ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import math

# ── CONFIG ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TO_EMAIL   = os.environ["TO_EMAIL"]

SYMBOL   = "BTC-USDT"
INTERVAL = "15min"
LIMIT    = 200


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_ohlcv():
    url = "https://api.kucoin.com/api/v1/market/candles"
    r = requests.get(url, params={"symbol": SYMBOL, "type": INTERVAL}, timeout=10)
    r.raise_for_status()
    raw = list(reversed(r.json()["data"]))[-LIMIT:]
    df = pd.DataFrame(raw, columns=["open_time","open","close","high","low","volume","turnover"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    return df.reset_index(drop=True)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema9"]   = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema20"]  = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df["rsi"]    = ta.momentum.RSIIndicator(c, 14).rsi()
    stoch = ta.momentum.StochRSIIndicator(c, 14, 3, 3)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["atr"]     = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    df["vwap"]      = (c * v).rolling(24).sum() / v.rolling(24).sum()
    df["mom10"]     = ta.momentum.ROCIndicator(c, 10).roc()
    return df


# ── SUPPORT / RESISTANCE ──────────────────────────────────────────────────────
def find_sr_levels(df, lookback=80, tolerance=0.005):
    highs = df["high"].tail(lookback)
    lows  = df["low"].tail(lookback)
    levels = []
    for i in range(2, len(highs)-2):
        h = highs.iloc[i]
        if h > highs.iloc[i-1] and h > highs.iloc[i-2] and h > highs.iloc[i+1] and h > highs.iloc[i+2]:
            levels.append(h)
        lo = lows.iloc[i]
        if lo < lows.iloc[i-1] and lo < lows.iloc[i-2] and lo < lows.iloc[i+1] and lo < lows.iloc[i+2]:
            levels.append(lo)
    levels = sorted(set(levels))
    clustered = []
    for lv in levels:
        if not clustered or abs(lv - clustered[-1]) / clustered[-1] > tolerance:
            clustered.append(lv)
    return clustered


# ── CANDLESTICK PATTERNS ──────────────────────────────────────────────────────
def detect_patterns(df):
    patterns = []
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    i = len(c) - 1
    body       = abs(c[i] - o[i])
    candle_rng = h[i] - l[i]
    upper_wick = h[i] - max(c[i], o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    prev_body  = abs(c[i-1] - o[i-1])
    if candle_rng > 0 and body / candle_rng < 0.1:
        patterns.append(("🕯️", "Doji", "Indecision — reversal possible", "neutral"))
    if lower_wick > 2 * body and upper_wick < body * 0.5 and c[i-1] < o[i-1]:
        patterns.append(("🔨", "Hammer", "Bullish reversal signal", "call"))
    if upper_wick > 2 * body and lower_wick < body * 0.5 and c[i-1] > o[i-1]:
        patterns.append(("⭐", "Shooting Star", "Bearish reversal signal", "put"))
    if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1] and body > prev_body:
        patterns.append(("📗", "Bullish Engulfing", "Strong reversal upward", "call"))
    if c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1] and body > prev_body:
        patterns.append(("📕", "Bearish Engulfing", "Strong reversal downward", "put"))
    if candle_rng > 0 and c[i] > o[i] and body / candle_rng > 0.85:
        patterns.append(("💚", "Bullish Marubozu", "Strong buying pressure", "call"))
    if candle_rng > 0 and c[i] < o[i] and body / candle_rng > 0.85:
        patterns.append(("🔻", "Bearish Marubozu", "Strong selling pressure", "put"))
    if i >= 2:
        if c[i-2] < o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2:
            patterns.append(("🌅", "Morning Star", "Bullish 3-candle reversal", "call"))
        if c[i-2] > o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2:
            patterns.append(("🌆", "Evening Star", "Bearish 3-candle reversal", "put"))
    return patterns


# ── SCORING ───────────────────────────────────────────────────────────────────
def analyze(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]
    rsi   = last["rsi"]
    sk    = last["stoch_k"]
    sd    = last["stoch_d"]
    mh    = last["macd_hist"]
    mh_p  = prev["macd_hist"]
    ml    = last["macd"]
    ms    = last["macd_signal"]

    call_score = put_score = 0
    reasons = []

    # EMA Stack
    if last["ema9"] > last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 3; reasons.append(("📈","Full EMA bullish (9>20>50>200)","call","Trend"))
    elif last["ema9"] < last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 3;  reasons.append(("📉","Full EMA bearish (9<20<50<200)","put","Trend"))
    elif last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 2; reasons.append(("📈","EMA 20/50/200 bullish stack","call","Trend"))
    elif last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 2;  reasons.append(("📉","EMA 20/50/200 bearish stack","put","Trend"))
    else:
        reasons.append(("⚠️","EMAs mixed — no clear trend","neutral","Trend"))

    if price > last["ema200"]:
        call_score += 1; reasons.append(("✅",f"Above EMA200 ${last['ema200']:,.0f} — macro bull","call","Trend"))
    else:
        put_score += 1;  reasons.append(("❌",f"Below EMA200 ${last['ema200']:,.0f} — macro bear","put","Trend"))

    adx_val = last["adx"]
    if adx_val > 25:
        if last["adx_pos"] > last["adx_neg"]:
            call_score += 2; reasons.append(("💪",f"ADX {adx_val:.0f} strong BULLISH trend","call","Trend"))
        else:
            put_score += 2;  reasons.append(("💪",f"ADX {adx_val:.0f} strong BEARISH trend","put","Trend"))
    elif adx_val < 20:
        reasons.append(("😴",f"ADX {adx_val:.0f} — choppy, no strong trend","neutral","Trend"))
    else:
        reasons.append(("〰️",f"ADX {adx_val:.0f} — trend developing","neutral","Trend"))

    if rsi < 30:
        call_score += 3; reasons.append(("🟢",f"RSI {rsi:.1f} — strongly oversold, high bounce prob","call","Momentum"))
    elif rsi < 40:
        call_score += 1; reasons.append(("🟢",f"RSI {rsi:.1f} — oversold zone","call","Momentum"))
    elif rsi > 70:
        put_score += 3;  reasons.append(("🔴",f"RSI {rsi:.1f} — strongly overbought","put","Momentum"))
    elif rsi > 60:
        put_score += 1;  reasons.append(("🔴",f"RSI {rsi:.1f} — overbought zone","put","Momentum"))
    else:
        reasons.append(("⚪",f"RSI {rsi:.1f} — neutral","neutral","Momentum"))

    if sk < 20 and sd < 20:
        call_score += 2; reasons.append(("🟢",f"Stoch RSI oversold K:{sk:.0f} D:{sd:.0f}","call","Momentum"))
    elif sk > 80 and sd > 80:
        put_score += 2;  reasons.append(("🔴",f"Stoch RSI overbought K:{sk:.0f} D:{sd:.0f}","put","Momentum"))
    elif sk > sd and sk < 50:
        call_score += 1; reasons.append(("🟡",f"Stoch RSI bullish cross low zone K:{sk:.0f}","call","Momentum"))
    elif sk < sd and sk > 50:
        put_score += 1;  reasons.append(("🟡",f"Stoch RSI bearish cross high zone K:{sk:.0f}","put","Momentum"))

    if ml > ms and ml > 0 and mh > mh_p:
        call_score += 3; reasons.append(("📈",f"MACD above signal+positive+rising {mh:+.1f}","call","MACD"))
    elif ml > ms and mh > mh_p:
        call_score += 2; reasons.append(("📈",f"MACD bullish cross + rising {mh:+.1f}","call","MACD"))
    elif ml > ms:
        call_score += 1; reasons.append(("📈",f"MACD above signal {mh:+.1f}","call","MACD"))
    elif ml < ms and ml < 0 and mh < mh_p:
        put_score += 3;  reasons.append(("📉",f"MACD below signal+negative+falling {mh:+.1f}","put","MACD"))
    elif ml < ms and mh < mh_p:
        put_score += 2;  reasons.append(("📉",f"MACD bearish cross + falling {mh:+.1f}","put","MACD"))
    elif ml < ms:
        put_score += 1;  reasons.append(("📉",f"MACD below signal {mh:+.1f}","put","MACD"))

    bb_pct = last["bb_pct"]
    if price <= last["bb_lower"]:
        call_score += 2; reasons.append(("🟢","Price at lower BB — mean reversion CALL","call","Volatility"))
    elif bb_pct < 0.2:
        call_score += 1; reasons.append(("🟢",f"Price near lower BB ({bb_pct:.2f})","call","Volatility"))
    elif price >= last["bb_upper"]:
        put_score += 2;  reasons.append(("🔴","Price at upper BB — mean reversion PUT","put","Volatility"))
    elif bb_pct > 0.8:
        put_score += 1;  reasons.append(("🔴",f"Price near upper BB ({bb_pct:.2f})","put","Volatility"))

    bb_width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
    if bb_width < 2.0:
        reasons.append(("💥",f"BB Squeeze! Big move incoming (width {bb_width:.1f}%)","neutral","Volatility"))

    if pd.notna(last["vwap"]):
        if price > last["vwap"] * 1.005:
            call_score += 1; reasons.append(("✅",f"Above VWAP ${last['vwap']:,.0f} — bullish","call","VWAP"))
        elif price < last["vwap"] * 0.995:
            put_score += 1;  reasons.append(("❌",f"Below VWAP ${last['vwap']:,.0f} — bearish","put","VWAP"))

    vol_ratio = last["vol_ratio"]
    is_green  = last["close"] > last["open"]
    if vol_ratio > 2.0:
        if is_green:
            call_score += 2; reasons.append(("📊",f"High vol green candle {vol_ratio:.1f}x — strong buy","call","Volume"))
        else:
            put_score += 2;  reasons.append(("📊",f"High vol red candle {vol_ratio:.1f}x — strong sell","put","Volume"))
    elif vol_ratio > 1.5:
        if is_green:
            call_score += 1; reasons.append(("📊",f"Above avg vol green {vol_ratio:.1f}x","call","Volume"))
        else:
            put_score += 1;  reasons.append(("📊",f"Above avg vol red {vol_ratio:.1f}x","put","Volume"))
    elif vol_ratio < 0.6:
        reasons.append(("⚠️",f"Low volume {vol_ratio:.1f}x — weak conviction","neutral","Volume"))

    mom = last["mom10"]
    if mom > 3:
        call_score += 1; reasons.append(("🚀",f"Strong upward momentum ROC:{mom:+.1f}%","call","Momentum"))
    elif mom < -3:
        put_score += 1;  reasons.append(("💣",f"Strong downward momentum ROC:{mom:+.1f}%","put","Momentum"))

    patterns = detect_patterns(df)
    for icon, name, desc, side in patterns:
        if side == "call":
            call_score += 1; reasons.append((icon, f"{name}: {desc}", "call", "Pattern"))
        elif side == "put":
            put_score += 1;  reasons.append((icon, f"{name}: {desc}", "put", "Pattern"))
        else:
            reasons.append((icon, f"{name}: {desc}", "neutral", "Pattern"))

    sr_levels = find_sr_levels(df, lookback=80)
    near_support    = [lv for lv in sr_levels if 0 < (price - lv) / price < 0.015]
    near_resistance = [lv for lv in sr_levels if 0 < (lv - price) / price < 0.015]
    if near_support:
        call_score += 1; reasons.append(("🛡️",f"Near support ${near_support[-1]:,.0f} — bounce zone","call","S/R"))
    if near_resistance:
        put_score += 1;  reasons.append(("🧱",f"Near resistance ${near_resistance[0]:,.0f} — rejection","put","S/R"))

    atr_pct  = (last["atr"] / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(last["atr"] * 1.5 / 100) * 100

    # Price history for mini chart (last 30 candles)
    chart_closes = df["close"].tail(30).tolist()
    chart_highs  = df["high"].tail(30).tolist()
    chart_lows   = df["low"].tail(30).tolist()
    chart_opens  = df["open"].tail(30).tolist()

    return {
        "call_score": call_score, "put_score": put_score, "reasons": reasons,
        "price": price, "rsi": rsi, "stoch_k": sk, "stoch_d": sd,
        "adx": adx_val, "atr_pct": atr_pct, "atr": last["atr"],
        "macd_hist": mh, "macd": ml, "macd_signal": ms,
        "bb_pct": bb_pct, "bb_upper": last["bb_upper"], "bb_lower": last["bb_lower"],
        "vol_ratio": vol_ratio, "vwap": last["vwap"], "mom10": mom,
        "ema20": last["ema20"], "ema50": last["ema50"], "ema200": last["ema200"],
        "ema9": last["ema9"],
        "atm": atm, "otm_call": atm + atr_step, "otm_put": atm - atr_step,
        "sr_levels": sr_levels, "patterns": patterns,
        "chart_closes": chart_closes, "chart_highs": chart_highs,
        "chart_lows": chart_lows, "chart_opens": chart_opens,
        "bb_width": (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100,
    }


# ── SVG GAUGE ─────────────────────────────────────────────────────────────────
def svg_gauge(value, min_val, max_val, label, low_color="#e24b4a", high_color="#1d9e75", mid_color="#f59e0b"):
    pct = max(0, min(1, (value - min_val) / (max_val - min_val)))
    angle = -140 + pct * 280
    rad   = math.radians(angle)
    nx = 50 + 36 * math.cos(math.radians(angle - 90))
    ny = 54 + 36 * math.sin(math.radians(angle - 90))

    if pct < 0.35:
        color = low_color
    elif pct > 0.65:
        color = high_color
    else:
        color = mid_color

    val_str = f"{value:.1f}"

    return f"""
<svg width="110" height="80" viewBox="0 0 110 80" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g{label}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{low_color}"/>
      <stop offset="50%" style="stop-color:{mid_color}"/>
      <stop offset="100%" style="stop-color:{high_color}"/>
    </linearGradient>
  </defs>
  <!-- Track -->
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="#e8e7e3" stroke-width="7" stroke-linecap="round"/>
  <!-- Fill -->
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="url(#g{label})" stroke-width="7"
        stroke-linecap="round" stroke-dasharray="226" stroke-dashoffset="{226 - pct*226:.1f}"/>
  <!-- Needle -->
  <line x1="55" y1="54" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#333" stroke-width="2" stroke-linecap="round"/>
  <circle cx="55" cy="54" r="4" fill="#333"/>
  <!-- Value -->
  <text x="55" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{val_str}</text>
  <text x="55" y="12" text-anchor="middle" font-size="9" fill="#999">{label}</text>
</svg>"""


# ── SVG MINI CANDLESTICK CHART ────────────────────────────────────────────────
def svg_candle_chart(closes, highs, lows, opens, width=560, height=120):
    n = len(closes)
    if n == 0:
        return ""
    mn = min(lows)
    mx = max(highs)
    rng = mx - mn or 1
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 20
    cw = (width - pad_l - pad_r) / n
    bar_w = max(2, cw * 0.6)

    def fy(v):
        return pad_t + (1 - (v - mn) / rng) * (height - pad_t - pad_b)

    candles = ""
    for i in range(n):
        x   = pad_l + i * cw + cw / 2
        bull = closes[i] >= opens[i]
        col  = "#1d9e75" if bull else "#e24b4a"
        o_y  = fy(opens[i])
        c_y  = fy(closes[i])
        h_y  = fy(highs[i])
        l_y  = fy(lows[i])
        body_y = min(o_y, c_y)
        body_h = max(abs(o_y - c_y), 1)
        candles += f'<line x1="{x:.1f}" y1="{h_y:.1f}" x2="{x:.1f}" y2="{l_y:.1f}" stroke="{col}" stroke-width="1"/>'
        candles += f'<rect x="{x - bar_w/2:.1f}" y="{body_y:.1f}" width="{bar_w:.1f}" height="{body_h:.1f}" fill="{col}"/>'

    # Y axis labels
    labels = ""
    for i in range(3):
        v   = mn + rng * i / 2
        y   = fy(v)
        labels += f'<text x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="8" fill="#aaa">${v:,.0f}</text>'
        labels += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#f0eeea" stroke-width="0.5"/>'

    # Last price line
    last_y = fy(closes[-1])
    labels += f'<line x1="{pad_l}" y1="{last_y:.1f}" x2="{width - pad_r}" y2="{last_y:.1f}" stroke="#f59e0b" stroke-width="0.8" stroke-dasharray="3,2"/>'

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"
     style="background:#fafaf8;border-radius:8px;">
  {labels}
  {candles}
  <text x="{width - pad_r}" y="{last_y - 3:.1f}" text-anchor="end" font-size="8" fill="#f59e0b">${closes[-1]:,.0f}</text>
</svg>"""


# ── SVG SIGNAL BAR ────────────────────────────────────────────────────────────
def svg_signal_bar(call_score, put_score):
    total = call_score + put_score or 1
    call_pct = call_score / total * 100
    put_pct  = put_score  / total * 100
    return f"""
<svg width="520" height="36" viewBox="0 0 520 36" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="8" width="520" height="20" rx="10" fill="#e8e7e3"/>
  <rect x="0" y="8" width="{call_pct * 5.2:.1f}" height="20" rx="10" fill="#1d9e75"/>
  <rect x="{520 - put_pct * 5.2:.1f}" y="8" width="{put_pct * 5.2:.1f}" height="20" rx="10" fill="#e24b4a"/>
  <text x="8" y="23" font-size="10" font-weight="700" fill="white">CALL {call_score}</text>
  <text x="512" y="23" text-anchor="end" font-size="10" font-weight="700" fill="white">PUT {put_score}</text>
</svg>"""


# ── BUILD EMAIL ───────────────────────────────────────────────────────────────
def build_email(d):
    cs = d["call_score"]
    ps = d["put_score"]
    total = cs + ps or 1
    gap   = abs(cs - ps)
    pct   = round(max(cs, ps) / total * 100)
    strength = "Very Strong 🔥" if gap >= 8 else "Strong 💪" if gap >= 5 else "Moderate ⚡" if gap >= 3 else "Weak ⚠️"

    if cs > ps:
        verdict     = "BUY CALL"
        verdict_emoji = "🟢"
        v_color     = "#1d9e75"
        v_bg        = "#eaf3de"
        v_border    = "#a3d977"
        action_tip  = f"Enter a CALL option near ATM <strong>${d['atm']:,}</strong>. Target OTM at <strong>${d['otm_call']:,}</strong>."
    elif ps > cs:
        verdict     = "BUY PUT"
        verdict_emoji = "🔴"
        v_color     = "#a32d2d"
        v_bg        = "#fcebeb"
        v_border    = "#f7a3a3"
        action_tip  = f"Enter a PUT option near ATM <strong>${d['atm']:,}</strong>. Target OTM at <strong>${d['otm_put']:,}</strong>."
    else:
        verdict     = "NO CLEAR SIGNAL"
        verdict_emoji = "🟡"
        v_color     = "#854f0b"
        v_bg        = "#faeeda"
        v_border    = "#f5c97e"
        action_tip  = "Wait for a stronger signal before entering any position."

    # Build candle chart SVG
    chart_svg = svg_candle_chart(
        d["chart_closes"], d["chart_highs"], d["chart_lows"], d["chart_opens"]
    )

    # Build signal bar SVG
    bar_svg = svg_signal_bar(cs, ps)

    # Gauges
    rsi_gauge   = svg_gauge(d["rsi"],   0,   100, "RSI", "#e24b4a", "#1d9e75", "#f59e0b")
    stoch_gauge = svg_gauge(d["stoch_k"], 0, 100, "Stoch RSI", "#e24b4a", "#1d9e75", "#f59e0b")
    adx_gauge   = svg_gauge(d["adx"],   0,   60,  "ADX", "#aaa",  "#6366f1", "#6366f1")
    mom_gauge   = svg_gauge(d["mom10"], -10,  10,  "Momentum", "#e24b4a", "#1d9e75", "#f59e0b")

    # Group reasons by category
    cats = ["Trend","Momentum","MACD","Volatility","VWAP","Volume","S/R","Pattern"]
    grouped = {c: [] for c in cats}
    for icon, text, side, cat in d["reasons"]:
        grouped.get(cat, grouped["Trend"]).append((icon, text, side))

    reason_html = ""
    for cat in cats:
        items = grouped[cat]
        if not items:
            continue
        reason_html += f"""
        <div style="margin-bottom:12px;">
          <div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;
               margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #f0eeea;">{cat}</div>"""
        for icon, text, side in items:
            bg    = "#f0faf5" if side=="call" else "#fdf2f2" if side=="put" else "#f8f7f5"
            color = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888"
            badge = "CALL" if side=="call" else "PUT" if side=="put" else ""
            badge_html = f'<span style="font-size:9px;background:{color};color:white;padding:1px 5px;border-radius:3px;margin-left:6px;">{badge}</span>' if badge else ""
            reason_html += f"""
          <div style="display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
               background:{bg};border-radius:6px;margin-bottom:4px;">
            <span style="font-size:14px;flex-shrink:0;">{icon}</span>
            <span style="font-size:12px;color:{color};line-height:1.4;">{text}{badge_html}</span>
          </div>"""
        reason_html += "</div>"

    # S/R levels table
    sr_html = ""
    if d["sr_levels"]:
        price = d["price"]
        close_levels = sorted(d["sr_levels"], key=lambda x: abs(x - price))[:8]
        close_levels = sorted(close_levels)
        for lv in close_levels:
            diff = ((lv - price) / price) * 100
            typ  = "Resistance 🧱" if lv > price else "Support 🛡️"
            col  = "#a32d2d" if lv > price else "#1d9e75"
            sr_html += f"""
          <tr>
            <td style="padding:6px 10px;font-size:12px;font-weight:600;color:{col};">${lv:,.0f}</td>
            <td style="padding:6px 10px;font-size:12px;color:{col};">{typ}</td>
            <td style="padding:6px 10px;font-size:12px;color:#888;">{diff:+.2f}%</td>
          </tr>"""

    # Pattern section
    pattern_html = ""
    if d["patterns"]:
        for icon, name, desc, side in d["patterns"]:
            col = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888"
            bg  = "#f0faf5" if side=="call" else "#fdf2f2" if side=="put" else "#f8f7f5"
            pattern_html += f"""
          <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
               background:{bg};border-radius:8px;margin-bottom:6px;">
            <span style="font-size:22px;">{icon}</span>
            <div>
              <div style="font-size:13px;font-weight:700;color:{col};">{name}</div>
              <div style="font-size:11px;color:#888;">{desc}</div>
            </div>
          </div>"""
    else:
        pattern_html = '<div style="font-size:12px;color:#aaa;padding:8px;">No major pattern detected on this candle.</div>'

    # ADX label
    adx_label = "Strong Trend" if d["adx"] > 25 else "Weak/Choppy" if d["adx"] < 20 else "Developing"
    premium_note = (
        "🔥 Premiums EXPENSIVE — only enter on very strong signal" if d["atr_pct"] > 3
        else "😴 Premiums CHEAP — good time to buy options" if d["atr_pct"] < 1
        else "📊 Premiums FAIR — normal market conditions"
    )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0efeb;font-family:-apple-system,Arial,sans-serif;">
<div style="max-width:600px;margin:24px auto;background:#ffffff;border-radius:16px;
     overflow:hidden;border:1px solid #e0dfd8;box-shadow:0 4px 24px rgba(0,0,0,.08);">

  <!-- ═══ HEADER ═══ -->
  <div style="background:linear-gradient(135deg,#0f0f0f 0%,#1a1a2e 100%);
       padding:20px 24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <div style="color:#f59e0b;font-size:12px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;">BTC Options Signal</div>
        <div style="color:#fff;font-size:28px;font-weight:800;margin:4px 0;">₿ BTC/USDT</div>
        <div style="color:#666;font-size:11px;">{now} · KuCoin · 15M · 12 Indicators</div>
      </div>
      <div style="text-align:right;">
        <div style="color:#fff;font-size:30px;font-weight:800;">${d['price']:,.0f}</div>
        <div style="color:#888;font-size:11px;">Live Price</div>
      </div>
    </div>
  </div>

  <!-- ═══ VERDICT BANNER ═══ -->
  <div style="background:{v_bg};border-left:5px solid {v_border};padding:20px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td>
          <div style="font-size:11px;color:#888;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:8px;">Today's Signal</div>
          <div style="font-size:32px;font-weight:900;color:{v_color};
               letter-spacing:-1px;">{verdict_emoji} {verdict}</div>
          <div style="font-size:14px;font-weight:600;color:{v_color};margin-top:4px;">
               {strength} Signal &nbsp;·&nbsp; {pct}% Confidence</div>
          <div style="font-size:12px;color:#666;margin-top:8px;
               background:rgba(255,255,255,0.6);padding:8px 10px;border-radius:6px;">
               💡 {action_tip}</div>
        </td>
        <td width="100" style="text-align:center;vertical-align:middle;">
          <div style="font-size:48px;line-height:1;">{verdict_emoji}</div>
          <div style="font-size:11px;color:#888;margin-top:4px;">Expiry: Today (Intraday)</div>
        </td>
      </tr>
    </table>
    <!-- Signal bar -->
    <div style="margin-top:12px;">
      {bar_svg}
    </div>
  </div>

  <!-- ═══ PRICE CHART ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:8px;">📈 Last 30 Candles (1H)</div>
    {chart_svg}
    <div style="display:flex;gap:16px;margin-top:8px;">
      <span style="font-size:10px;color:#1d9e75;">█ Bullish candle</span>
      <span style="font-size:10px;color:#e24b4a;">█ Bearish candle</span>
      <span style="font-size:10px;color:#f59e0b;">── Current price</span>
    </div>
  </div>

  <!-- ═══ GAUGE METERS ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:8px;">🎯 Indicator Gauges</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="25%" style="text-align:center;">{rsi_gauge}</td>
        <td width="25%" style="text-align:center;">{stoch_gauge}</td>
        <td width="25%" style="text-align:center;">{adx_gauge}</td>
        <td width="25%" style="text-align:center;">{mom_gauge}</td>
      </tr>
    </table>
    <div style="display:flex;gap:8px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;
           padding:2px 8px;border-radius:4px;">RSI &lt;35 = Oversold → CALL</span>
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;
           padding:2px 8px;border-radius:4px;">RSI &gt;65 = Overbought → PUT</span>
      <span style="font-size:10px;background:#eef2ff;color:#6366f1;
           padding:2px 8px;border-radius:4px;">ADX &gt;25 = Strong trend</span>
    </div>
  </div>

  <!-- ═══ KEY METRICS TABLE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">📊 Key Metrics</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;width:30%;">EMA 9 / 20 / 50 / 200</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
          ${d['ema9']:,.0f} / ${d['ema20']:,.0f} / ${d['ema50']:,.0f} / ${d['ema200']:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">VWAP (24h)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['price']>d['vwap'] else '#a32d2d'};">
            ${d['vwap']:,.0f} — Price is {'above ↑' if d['price']>d['vwap'] else 'below ↓'} VWAP</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">Bollinger Bands</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            Lower: ${d['bb_lower']:,.0f} &nbsp;|&nbsp; Upper: ${d['bb_upper']:,.0f}
            &nbsp;|&nbsp; BB%: {d['bb_pct']:.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">Volume Ratio</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['vol_ratio']>1.5 else '#888'};">
            {d['vol_ratio']:.2f}x average {'🔥 High volume spike!' if d['vol_ratio']>2 else '✅ Above average' if d['vol_ratio']>1.5 else '⚠️ Low volume' if d['vol_ratio']<0.6 else ''}</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">ATR / Premium</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            {d['atr_pct']:.2f}% &nbsp;·&nbsp; {premium_note}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">ADX Trend Strength</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#6366f1' if d['adx']>25 else '#888'};">
            {d['adx']:.1f} — {adx_label} (+DI: {d['adx']:,.0f})</td>
      </tr>
    </table>
  </div>

  <!-- ═══ CANDLESTICK PATTERNS ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🕯️ Candlestick Patterns</div>
    {pattern_html}
  </div>

  <!-- ═══ FULL SIGNAL BREAKDOWN ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🔍 Full Signal Breakdown</div>
    {reason_html}
  </div>

  <!-- ═══ SUPPORT / RESISTANCE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🗺️ Key Support &amp; Resistance Levels</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Price</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Type</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Distance</th>
      </tr>
      {sr_html}
    </table>
  </div>

  <!-- ═══ STRIKE PRICE GUIDE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:12px;">🎯 Strike Price Guide (ATR-based)</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="33%" style="padding:0 4px 0 0;">
          <div style="background:#fcebeb;border:1px solid #f7a3a3;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#a32d2d;margin-bottom:6px;font-weight:600;">
              OTM PUT STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#a32d2d;">${d['otm_put']:,}</div>
            <div style="font-size:10px;color:#c07070;margin-top:4px;">
              Cheaper · Needs bigger move</div>
          </div>
        </td>
        <td width="34%" style="padding:0 4px;">
          <div style="background:#fff;border:2px solid #d3d1c7;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#888;margin-bottom:6px;font-weight:600;">
              ATM STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#333;">${d['atm']:,}</div>
            <div style="font-size:10px;color:#aaa;margin-top:4px;">
              Safer · Higher premium</div>
          </div>
        </td>
        <td width="33%" style="padding:0 0 0 4px;">
          <div style="background:#eaf3de;border:1px solid #a3d977;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#3b6d11;margin-bottom:6px;font-weight:600;">
              OTM CALL STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#3b6d11;">${d['otm_call']:,}</div>
            <div style="font-size:10px;color:#6a9b3a;margin-top:4px;">
              Cheaper · Needs bigger move</div>
          </div>
        </td>
      </tr>
    </table>
    <div style="margin-top:10px;padding:8px 12px;background:#fff8e7;border-radius:6px;
         font-size:11px;color:#854f0b;">
      💡 <strong>ATM</strong> = safer, more expensive. <strong>OTM</strong> = cheaper, needs larger price move to profit. Always buy on your signal direction only.
    </div>
  </div>

  <!-- ═══ FOOTER ═══ -->
  <div style="padding:16px 24px;background:#f8f7f5;">
    <div style="font-size:10px;color:#bbb;text-align:center;line-height:1.8;">
      Automated analysis using 12 indicators (EMA · RSI · Stoch RSI · MACD · BB · ADX · VWAP · Volume · ATR · Momentum · S/R · Candlestick Patterns)<br>
      <strong style="color:#e24b4a;">Not financial advice.</strong> Options can expire worthless. Always manage risk. Never invest more than you can afford to lose.
    </div>
  </div>

</div>
</body></html>"""

    return html, f"{verdict_emoji} {verdict}"


# ── SEND EMAIL ────────────────────────────────────────────────────────────────
def send_email(html_body, verdict_text, price):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"BTC Signal: {verdict_text} | ${price:,.0f} | {datetime.utcnow().strftime('%H:%M UTC')}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
    print(f"✅ Email sent: {verdict_text} | ${price:,.0f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching BTC data from KuCoin...")
    df = fetch_ohlcv()
    print(f"Got {len(df)} candles. Price: ${df.iloc[-1]['close']:,.0f}")
    df = add_indicators(df)
    result = analyze(df)
    print(f"Call: {result['call_score']} | Put: {result['put_score']}")
    html, verdict = build_email(result)
    send_email(html, verdict, result["price"])

if __name__ == "__main__":
    main()





"""
BTC Options Analyzer — Visual Email Edition
Sends a beautiful HTML email with inline SVG charts, gauge meters,
signal bars, candlestick pattern icons — easy to read at a glance.
"""

import requests
import pandas as pd
import ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import math

# ── CONFIG ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TO_EMAIL   = os.environ["TO_EMAIL"]

SYMBOL   = "BTC-USDT"
INTERVAL = "1hour"
LIMIT    = 300


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_ohlcv():
    url = "https://api.kucoin.com/api/v1/market/candles"
    r = requests.get(url, params={"symbol": SYMBOL, "type": INTERVAL}, timeout=10)
    r.raise_for_status()
    raw = list(reversed(r.json()["data"]))[-LIMIT:]
    df = pd.DataFrame(raw, columns=["open_time","open","close","high","low","volume","turnover"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    return df.reset_index(drop=True)


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema9"]   = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema20"]  = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df["rsi"]    = ta.momentum.RSIIndicator(c, 14).rsi()
    stoch = ta.momentum.StochRSIIndicator(c, 14, 3, 3)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["atr"]     = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    df["vwap"]      = (c * v).rolling(24).sum() / v.rolling(24).sum()
    df["mom10"]     = ta.momentum.ROCIndicator(c, 10).roc()
    return df


# ── SUPPORT / RESISTANCE ──────────────────────────────────────────────────────
def find_sr_levels(df, lookback=80, tolerance=0.005):
    highs = df["high"].tail(lookback)
    lows  = df["low"].tail(lookback)
    levels = []
    for i in range(2, len(highs)-2):
        h = highs.iloc[i]
        if h > highs.iloc[i-1] and h > highs.iloc[i-2] and h > highs.iloc[i+1] and h > highs.iloc[i+2]:
            levels.append(h)
        lo = lows.iloc[i]
        if lo < lows.iloc[i-1] and lo < lows.iloc[i-2] and lo < lows.iloc[i+1] and lo < lows.iloc[i+2]:
            levels.append(lo)
    levels = sorted(set(levels))
    clustered = []
    for lv in levels:
        if not clustered or abs(lv - clustered[-1]) / clustered[-1] > tolerance:
            clustered.append(lv)
    return clustered


# ── CANDLESTICK PATTERNS ──────────────────────────────────────────────────────
def detect_patterns(df):
    patterns = []
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    i = len(c) - 1
    body       = abs(c[i] - o[i])
    candle_rng = h[i] - l[i]
    upper_wick = h[i] - max(c[i], o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    prev_body  = abs(c[i-1] - o[i-1])
    if candle_rng > 0 and body / candle_rng < 0.1:
        patterns.append(("🕯️", "Doji", "Indecision — reversal possible", "neutral"))
    if lower_wick > 2 * body and upper_wick < body * 0.5 and c[i-1] < o[i-1]:
        patterns.append(("🔨", "Hammer", "Bullish reversal signal", "call"))
    if upper_wick > 2 * body and lower_wick < body * 0.5 and c[i-1] > o[i-1]:
        patterns.append(("⭐", "Shooting Star", "Bearish reversal signal", "put"))
    if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1] and body > prev_body:
        patterns.append(("📗", "Bullish Engulfing", "Strong reversal upward", "call"))
    if c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1] and body > prev_body:
        patterns.append(("📕", "Bearish Engulfing", "Strong reversal downward", "put"))
    if candle_rng > 0 and c[i] > o[i] and body / candle_rng > 0.85:
        patterns.append(("💚", "Bullish Marubozu", "Strong buying pressure", "call"))
    if candle_rng > 0 and c[i] < o[i] and body / candle_rng > 0.85:
        patterns.append(("🔻", "Bearish Marubozu", "Strong selling pressure", "put"))
    if i >= 2:
        if c[i-2] < o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2:
            patterns.append(("🌅", "Morning Star", "Bullish 3-candle reversal", "call"))
        if c[i-2] > o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2:
            patterns.append(("🌆", "Evening Star", "Bearish 3-candle reversal", "put"))
    return patterns


# ── SCORING ───────────────────────────────────────────────────────────────────
def analyze(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]
    rsi   = last["rsi"]
    sk    = last["stoch_k"]
    sd    = last["stoch_d"]
    mh    = last["macd_hist"]
    mh_p  = prev["macd_hist"]
    ml    = last["macd"]
    ms    = last["macd_signal"]

    call_score = put_score = 0
    reasons = []

    # EMA Stack
    if last["ema9"] > last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 3; reasons.append(("📈","Full EMA bullish (9>20>50>200)","call","Trend"))
    elif last["ema9"] < last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 3;  reasons.append(("📉","Full EMA bearish (9<20<50<200)","put","Trend"))
    elif last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 2; reasons.append(("📈","EMA 20/50/200 bullish stack","call","Trend"))
    elif last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 2;  reasons.append(("📉","EMA 20/50/200 bearish stack","put","Trend"))
    else:
        reasons.append(("⚠️","EMAs mixed — no clear trend","neutral","Trend"))

    if price > last["ema200"]:
        call_score += 1; reasons.append(("✅",f"Above EMA200 ${last['ema200']:,.0f} — macro bull","call","Trend"))
    else:
        put_score += 1;  reasons.append(("❌",f"Below EMA200 ${last['ema200']:,.0f} — macro bear","put","Trend"))

    adx_val = last["adx"]
    if adx_val > 25:
        if last["adx_pos"] > last["adx_neg"]:
            call_score += 2; reasons.append(("💪",f"ADX {adx_val:.0f} strong BULLISH trend","call","Trend"))
        else:
            put_score += 2;  reasons.append(("💪",f"ADX {adx_val:.0f} strong BEARISH trend","put","Trend"))
    elif adx_val < 20:
        reasons.append(("😴",f"ADX {adx_val:.0f} — choppy, no strong trend","neutral","Trend"))
    else:
        reasons.append(("〰️",f"ADX {adx_val:.0f} — trend developing","neutral","Trend"))

    if rsi < 30:
        call_score += 3; reasons.append(("🟢",f"RSI {rsi:.1f} — strongly oversold, high bounce prob","call","Momentum"))
    elif rsi < 40:
        call_score += 1; reasons.append(("🟢",f"RSI {rsi:.1f} — oversold zone","call","Momentum"))
    elif rsi > 70:
        put_score += 3;  reasons.append(("🔴",f"RSI {rsi:.1f} — strongly overbought","put","Momentum"))
    elif rsi > 60:
        put_score += 1;  reasons.append(("🔴",f"RSI {rsi:.1f} — overbought zone","put","Momentum"))
    else:
        reasons.append(("⚪",f"RSI {rsi:.1f} — neutral","neutral","Momentum"))

    if sk < 20 and sd < 20:
        call_score += 2; reasons.append(("🟢",f"Stoch RSI oversold K:{sk:.0f} D:{sd:.0f}","call","Momentum"))
    elif sk > 80 and sd > 80:
        put_score += 2;  reasons.append(("🔴",f"Stoch RSI overbought K:{sk:.0f} D:{sd:.0f}","put","Momentum"))
    elif sk > sd and sk < 50:
        call_score += 1; reasons.append(("🟡",f"Stoch RSI bullish cross low zone K:{sk:.0f}","call","Momentum"))
    elif sk < sd and sk > 50:
        put_score += 1;  reasons.append(("🟡",f"Stoch RSI bearish cross high zone K:{sk:.0f}","put","Momentum"))

    if ml > ms and ml > 0 and mh > mh_p:
        call_score += 3; reasons.append(("📈",f"MACD above signal+positive+rising {mh:+.1f}","call","MACD"))
    elif ml > ms and mh > mh_p:
        call_score += 2; reasons.append(("📈",f"MACD bullish cross + rising {mh:+.1f}","call","MACD"))
    elif ml > ms:
        call_score += 1; reasons.append(("📈",f"MACD above signal {mh:+.1f}","call","MACD"))
    elif ml < ms and ml < 0 and mh < mh_p:
        put_score += 3;  reasons.append(("📉",f"MACD below signal+negative+falling {mh:+.1f}","put","MACD"))
    elif ml < ms and mh < mh_p:
        put_score += 2;  reasons.append(("📉",f"MACD bearish cross + falling {mh:+.1f}","put","MACD"))
    elif ml < ms:
        put_score += 1;  reasons.append(("📉",f"MACD below signal {mh:+.1f}","put","MACD"))

    bb_pct = last["bb_pct"]
    if price <= last["bb_lower"]:
        call_score += 2; reasons.append(("🟢","Price at lower BB — mean reversion CALL","call","Volatility"))
    elif bb_pct < 0.2:
        call_score += 1; reasons.append(("🟢",f"Price near lower BB ({bb_pct:.2f})","call","Volatility"))
    elif price >= last["bb_upper"]:
        put_score += 2;  reasons.append(("🔴","Price at upper BB — mean reversion PUT","put","Volatility"))
    elif bb_pct > 0.8:
        put_score += 1;  reasons.append(("🔴",f"Price near upper BB ({bb_pct:.2f})","put","Volatility"))

    bb_width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
    if bb_width < 2.0:
        reasons.append(("💥",f"BB Squeeze! Big move incoming (width {bb_width:.1f}%)","neutral","Volatility"))

    if pd.notna(last["vwap"]):
        if price > last["vwap"] * 1.005:
            call_score += 1; reasons.append(("✅",f"Above VWAP ${last['vwap']:,.0f} — bullish","call","VWAP"))
        elif price < last["vwap"] * 0.995:
            put_score += 1;  reasons.append(("❌",f"Below VWAP ${last['vwap']:,.0f} — bearish","put","VWAP"))

    vol_ratio = last["vol_ratio"]
    is_green  = last["close"] > last["open"]
    if vol_ratio > 2.0:
        if is_green:
            call_score += 2; reasons.append(("📊",f"High vol green candle {vol_ratio:.1f}x — strong buy","call","Volume"))
        else:
            put_score += 2;  reasons.append(("📊",f"High vol red candle {vol_ratio:.1f}x — strong sell","put","Volume"))
    elif vol_ratio > 1.5:
        if is_green:
            call_score += 1; reasons.append(("📊",f"Above avg vol green {vol_ratio:.1f}x","call","Volume"))
        else:
            put_score += 1;  reasons.append(("📊",f"Above avg vol red {vol_ratio:.1f}x","put","Volume"))
    elif vol_ratio < 0.6:
        reasons.append(("⚠️",f"Low volume {vol_ratio:.1f}x — weak conviction","neutral","Volume"))

    mom = last["mom10"]
    if mom > 3:
        call_score += 1; reasons.append(("🚀",f"Strong upward momentum ROC:{mom:+.1f}%","call","Momentum"))
    elif mom < -3:
        put_score += 1;  reasons.append(("💣",f"Strong downward momentum ROC:{mom:+.1f}%","put","Momentum"))

    patterns = detect_patterns(df)
    for icon, name, desc, side in patterns:
        if side == "call":
            call_score += 1; reasons.append((icon, f"{name}: {desc}", "call", "Pattern"))
        elif side == "put":
            put_score += 1;  reasons.append((icon, f"{name}: {desc}", "put", "Pattern"))
        else:
            reasons.append((icon, f"{name}: {desc}", "neutral", "Pattern"))

    sr_levels = find_sr_levels(df, lookback=80)
    near_support    = [lv for lv in sr_levels if 0 < (price - lv) / price < 0.015]
    near_resistance = [lv for lv in sr_levels if 0 < (lv - price) / price < 0.015]
    if near_support:
        call_score += 1; reasons.append(("🛡️",f"Near support ${near_support[-1]:,.0f} — bounce zone","call","S/R"))
    if near_resistance:
        put_score += 1;  reasons.append(("🧱",f"Near resistance ${near_resistance[0]:,.0f} — rejection","put","S/R"))

    atr_pct  = (last["atr"] / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(last["atr"] * 1.5 / 100) * 100

    # Price history for mini chart (last 30 candles)
    chart_closes = df["close"].tail(30).tolist()
    chart_highs  = df["high"].tail(30).tolist()
    chart_lows   = df["low"].tail(30).tolist()
    chart_opens  = df["open"].tail(30).tolist()

    return {
        "call_score": call_score, "put_score": put_score, "reasons": reasons,
        "price": price, "rsi": rsi, "stoch_k": sk, "stoch_d": sd,
        "adx": adx_val, "atr_pct": atr_pct, "atr": last["atr"],
        "macd_hist": mh, "macd": ml, "macd_signal": ms,
        "bb_pct": bb_pct, "bb_upper": last["bb_upper"], "bb_lower": last["bb_lower"],
        "vol_ratio": vol_ratio, "vwap": last["vwap"], "mom10": mom,
        "ema20": last["ema20"], "ema50": last["ema50"], "ema200": last["ema200"],
        "ema9": last["ema9"],
        "atm": atm, "otm_call": atm + atr_step, "otm_put": atm - atr_step,
        "sr_levels": sr_levels, "patterns": patterns,
        "chart_closes": chart_closes, "chart_highs": chart_highs,
        "chart_lows": chart_lows, "chart_opens": chart_opens,
        "bb_width": (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100,
    }


# ── SVG GAUGE ─────────────────────────────────────────────────────────────────
def svg_gauge(value, min_val, max_val, label, low_color="#e24b4a", high_color="#1d9e75", mid_color="#f59e0b"):
    pct = max(0, min(1, (value - min_val) / (max_val - min_val)))
    angle = -140 + pct * 280
    rad   = math.radians(angle)
    nx = 50 + 36 * math.cos(math.radians(angle - 90))
    ny = 54 + 36 * math.sin(math.radians(angle - 90))

    if pct < 0.35:
        color = low_color
    elif pct > 0.65:
        color = high_color
    else:
        color = mid_color

    val_str = f"{value:.1f}"

    return f"""
<svg width="110" height="80" viewBox="0 0 110 80" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g{label}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{low_color}"/>
      <stop offset="50%" style="stop-color:{mid_color}"/>
      <stop offset="100%" style="stop-color:{high_color}"/>
    </linearGradient>
  </defs>
  <!-- Track -->
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="#e8e7e3" stroke-width="7" stroke-linecap="round"/>
  <!-- Fill -->
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="url(#g{label})" stroke-width="7"
        stroke-linecap="round" stroke-dasharray="226" stroke-dashoffset="{226 - pct*226:.1f}"/>
  <!-- Needle -->
  <line x1="55" y1="54" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#333" stroke-width="2" stroke-linecap="round"/>
  <circle cx="55" cy="54" r="4" fill="#333"/>
  <!-- Value -->
  <text x="55" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{val_str}</text>
  <text x="55" y="12" text-anchor="middle" font-size="9" fill="#999">{label}</text>
</svg>"""


# ── SVG MINI CANDLESTICK CHART ────────────────────────────────────────────────
def svg_candle_chart(closes, highs, lows, opens, width=560, height=120):
    n = len(closes)
    if n == 0:
        return ""
    mn = min(lows)
    mx = max(highs)
    rng = mx - mn or 1
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 20
    cw = (width - pad_l - pad_r) / n
    bar_w = max(2, cw * 0.6)

    def fy(v):
        return pad_t + (1 - (v - mn) / rng) * (height - pad_t - pad_b)

    candles = ""
    for i in range(n):
        x   = pad_l + i * cw + cw / 2
        bull = closes[i] >= opens[i]
        col  = "#1d9e75" if bull else "#e24b4a"
        o_y  = fy(opens[i])
        c_y  = fy(closes[i])
        h_y  = fy(highs[i])
        l_y  = fy(lows[i])
        body_y = min(o_y, c_y)
        body_h = max(abs(o_y - c_y), 1)
        candles += f'<line x1="{x:.1f}" y1="{h_y:.1f}" x2="{x:.1f}" y2="{l_y:.1f}" stroke="{col}" stroke-width="1"/>'
        candles += f'<rect x="{x - bar_w/2:.1f}" y="{body_y:.1f}" width="{bar_w:.1f}" height="{body_h:.1f}" fill="{col}"/>'

    # Y axis labels
    labels = ""
    for i in range(3):
        v   = mn + rng * i / 2
        y   = fy(v)
        labels += f'<text x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="8" fill="#aaa">${v:,.0f}</text>'
        labels += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#f0eeea" stroke-width="0.5"/>'

    # Last price line
    last_y = fy(closes[-1])
    labels += f'<line x1="{pad_l}" y1="{last_y:.1f}" x2="{width - pad_r}" y2="{last_y:.1f}" stroke="#f59e0b" stroke-width="0.8" stroke-dasharray="3,2"/>'

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"
     style="background:#fafaf8;border-radius:8px;">
  {labels}
  {candles}
  <text x="{width - pad_r}" y="{last_y - 3:.1f}" text-anchor="end" font-size="8" fill="#f59e0b">${closes[-1]:,.0f}</text>
</svg>"""


# ── SVG SIGNAL BAR ────────────────────────────────────────────────────────────
def svg_signal_bar(call_score, put_score):
    total = call_score + put_score or 1
    call_pct = call_score / total * 100
    put_pct  = put_score  / total * 100
    return f"""
<svg width="520" height="36" viewBox="0 0 520 36" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="8" width="520" height="20" rx="10" fill="#e8e7e3"/>
  <rect x="0" y="8" width="{call_pct * 5.2:.1f}" height="20" rx="10" fill="#1d9e75"/>
  <rect x="{520 - put_pct * 5.2:.1f}" y="8" width="{put_pct * 5.2:.1f}" height="20" rx="10" fill="#e24b4a"/>
  <text x="8" y="23" font-size="10" font-weight="700" fill="white">CALL {call_score}</text>
  <text x="512" y="23" text-anchor="end" font-size="10" font-weight="700" fill="white">PUT {put_score}</text>
</svg>"""


# ── BUILD EMAIL ───────────────────────────────────────────────────────────────
def build_email(d):
    cs = d["call_score"]
    ps = d["put_score"]
    total = cs + ps or 1
    gap   = abs(cs - ps)
    pct   = round(max(cs, ps) / total * 100)
    strength = "Very Strong 🔥" if gap >= 8 else "Strong 💪" if gap >= 5 else "Moderate ⚡" if gap >= 3 else "Weak ⚠️"

    if cs > ps:
        verdict     = "BUY CALL"
        verdict_emoji = "🟢"
        v_color     = "#1d9e75"
        v_bg        = "#eaf3de"
        v_border    = "#a3d977"
        action_tip  = f"Enter a CALL option near ATM <strong>${d['atm']:,}</strong>. Target OTM at <strong>${d['otm_call']:,}</strong>."
    elif ps > cs:
        verdict     = "BUY PUT"
        verdict_emoji = "🔴"
        v_color     = "#a32d2d"
        v_bg        = "#fcebeb"
        v_border    = "#f7a3a3"
        action_tip  = f"Enter a PUT option near ATM <strong>${d['atm']:,}</strong>. Target OTM at <strong>${d['otm_put']:,}</strong>."
    else:
        verdict     = "NO CLEAR SIGNAL"
        verdict_emoji = "🟡"
        v_color     = "#854f0b"
        v_bg        = "#faeeda"
        v_border    = "#f5c97e"
        action_tip  = "Wait for a stronger signal before entering any position."

    # Build candle chart SVG
    chart_svg = svg_candle_chart(
        d["chart_closes"], d["chart_highs"], d["chart_lows"], d["chart_opens"]
    )

    # Build signal bar SVG
    bar_svg = svg_signal_bar(cs, ps)

    # Gauges
    rsi_gauge   = svg_gauge(d["rsi"],   0,   100, "RSI", "#e24b4a", "#1d9e75", "#f59e0b")
    stoch_gauge = svg_gauge(d["stoch_k"], 0, 100, "Stoch RSI", "#e24b4a", "#1d9e75", "#f59e0b")
    adx_gauge   = svg_gauge(d["adx"],   0,   60,  "ADX", "#aaa",  "#6366f1", "#6366f1")
    mom_gauge   = svg_gauge(d["mom10"], -10,  10,  "Momentum", "#e24b4a", "#1d9e75", "#f59e0b")

    # Group reasons by category
    cats = ["Trend","Momentum","MACD","Volatility","VWAP","Volume","S/R","Pattern"]
    grouped = {c: [] for c in cats}
    for icon, text, side, cat in d["reasons"]:
        grouped.get(cat, grouped["Trend"]).append((icon, text, side))

    reason_html = ""
    for cat in cats:
        items = grouped[cat]
        if not items:
            continue
        reason_html += f"""
        <div style="margin-bottom:12px;">
          <div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;
               margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #f0eeea;">{cat}</div>"""
        for icon, text, side in items:
            bg    = "#f0faf5" if side=="call" else "#fdf2f2" if side=="put" else "#f8f7f5"
            color = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888"
            badge = "CALL" if side=="call" else "PUT" if side=="put" else ""
            badge_html = f'<span style="font-size:9px;background:{color};color:white;padding:1px 5px;border-radius:3px;margin-left:6px;">{badge}</span>' if badge else ""
            reason_html += f"""
          <div style="display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
               background:{bg};border-radius:6px;margin-bottom:4px;">
            <span style="font-size:14px;flex-shrink:0;">{icon}</span>
            <span style="font-size:12px;color:{color};line-height:1.4;">{text}{badge_html}</span>
          </div>"""
        reason_html += "</div>"

    # S/R levels table
    sr_html = ""
    if d["sr_levels"]:
        price = d["price"]
        close_levels = sorted(d["sr_levels"], key=lambda x: abs(x - price))[:8]
        close_levels = sorted(close_levels)
        for lv in close_levels:
            diff = ((lv - price) / price) * 100
            typ  = "Resistance 🧱" if lv > price else "Support 🛡️"
            col  = "#a32d2d" if lv > price else "#1d9e75"
            sr_html += f"""
          <tr>
            <td style="padding:6px 10px;font-size:12px;font-weight:600;color:{col};">${lv:,.0f}</td>
            <td style="padding:6px 10px;font-size:12px;color:{col};">{typ}</td>
            <td style="padding:6px 10px;font-size:12px;color:#888;">{diff:+.2f}%</td>
          </tr>"""

    # Pattern section
    pattern_html = ""
    if d["patterns"]:
        for icon, name, desc, side in d["patterns"]:
            col = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888"
            bg  = "#f0faf5" if side=="call" else "#fdf2f2" if side=="put" else "#f8f7f5"
            pattern_html += f"""
          <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
               background:{bg};border-radius:8px;margin-bottom:6px;">
            <span style="font-size:22px;">{icon}</span>
            <div>
              <div style="font-size:13px;font-weight:700;color:{col};">{name}</div>
              <div style="font-size:11px;color:#888;">{desc}</div>
            </div>
          </div>"""
    else:
        pattern_html = '<div style="font-size:12px;color:#aaa;padding:8px;">No major pattern detected on this candle.</div>'

    # ADX label
    adx_label = "Strong Trend" if d["adx"] > 25 else "Weak/Choppy" if d["adx"] < 20 else "Developing"
    premium_note = (
        "🔥 Premiums EXPENSIVE — only enter on very strong signal" if d["atr_pct"] > 3
        else "😴 Premiums CHEAP — good time to buy options" if d["atr_pct"] < 1
        else "📊 Premiums FAIR — normal market conditions"
    )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0efeb;font-family:-apple-system,Arial,sans-serif;">
<div style="max-width:600px;margin:24px auto;background:#ffffff;border-radius:16px;
     overflow:hidden;border:1px solid #e0dfd8;box-shadow:0 4px 24px rgba(0,0,0,.08);">

  <!-- ═══ HEADER ═══ -->
  <div style="background:linear-gradient(135deg,#0f0f0f 0%,#1a1a2e 100%);
       padding:20px 24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <div style="color:#f59e0b;font-size:12px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;">BTC Options Signal</div>
        <div style="color:#fff;font-size:28px;font-weight:800;margin:4px 0;">₿ BTC/USDT</div>
        <div style="color:#666;font-size:11px;">{now} · KuCoin · 1H · 12 Indicators</div>
      </div>
      <div style="text-align:right;">
        <div style="color:#fff;font-size:30px;font-weight:800;">${d['price']:,.0f}</div>
        <div style="color:#888;font-size:11px;">Live Price</div>
      </div>
    </div>
  </div>

  <!-- ═══ VERDICT BANNER ═══ -->
  <div style="background:{v_bg};border-left:5px solid {v_border};padding:20px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td>
          <div style="font-size:11px;color:#888;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:8px;">Today's Signal</div>
          <div style="font-size:32px;font-weight:900;color:{v_color};
               letter-spacing:-1px;">{verdict_emoji} {verdict}</div>
          <div style="font-size:14px;font-weight:600;color:{v_color};margin-top:4px;">
               {strength} Signal &nbsp;·&nbsp; {pct}% Confidence</div>
          <div style="font-size:12px;color:#666;margin-top:8px;
               background:rgba(255,255,255,0.6);padding:8px 10px;border-radius:6px;">
               💡 {action_tip}</div>
        </td>
        <td width="100" style="text-align:center;vertical-align:middle;">
          <div style="font-size:48px;line-height:1;">{verdict_emoji}</div>
          <div style="font-size:11px;color:#888;margin-top:4px;">Expiry: 1–3 days</div>
        </td>
      </tr>
    </table>
    <!-- Signal bar -->
    <div style="margin-top:12px;">
      {bar_svg}
    </div>
  </div>

  <!-- ═══ PRICE CHART ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:8px;">📈 Last 30 Candles (1H)</div>
    {chart_svg}
    <div style="display:flex;gap:16px;margin-top:8px;">
      <span style="font-size:10px;color:#1d9e75;">█ Bullish candle</span>
      <span style="font-size:10px;color:#e24b4a;">█ Bearish candle</span>
      <span style="font-size:10px;color:#f59e0b;">── Current price</span>
    </div>
  </div>

  <!-- ═══ GAUGE METERS ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:8px;">🎯 Indicator Gauges</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="25%" style="text-align:center;">{rsi_gauge}</td>
        <td width="25%" style="text-align:center;">{stoch_gauge}</td>
        <td width="25%" style="text-align:center;">{adx_gauge}</td>
        <td width="25%" style="text-align:center;">{mom_gauge}</td>
      </tr>
    </table>
    <div style="display:flex;gap:8px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;
           padding:2px 8px;border-radius:4px;">RSI &lt;35 = Oversold → CALL</span>
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;
           padding:2px 8px;border-radius:4px;">RSI &gt;65 = Overbought → PUT</span>
      <span style="font-size:10px;background:#eef2ff;color:#6366f1;
           padding:2px 8px;border-radius:4px;">ADX &gt;25 = Strong trend</span>
    </div>
  </div>

  <!-- ═══ KEY METRICS TABLE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">📊 Key Metrics</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;width:30%;">EMA 9 / 20 / 50 / 200</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
          ${d['ema9']:,.0f} / ${d['ema20']:,.0f} / ${d['ema50']:,.0f} / ${d['ema200']:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">VWAP (24h)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['price']>d['vwap'] else '#a32d2d'};">
            ${d['vwap']:,.0f} — Price is {'above ↑' if d['price']>d['vwap'] else 'below ↓'} VWAP</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">Bollinger Bands</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            Lower: ${d['bb_lower']:,.0f} &nbsp;|&nbsp; Upper: ${d['bb_upper']:,.0f}
            &nbsp;|&nbsp; BB%: {d['bb_pct']:.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">Volume Ratio</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['vol_ratio']>1.5 else '#888'};">
            {d['vol_ratio']:.2f}x average {'🔥 High volume spike!' if d['vol_ratio']>2 else '✅ Above average' if d['vol_ratio']>1.5 else '⚠️ Low volume' if d['vol_ratio']<0.6 else ''}</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">ATR / Premium</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            {d['atr_pct']:.2f}% &nbsp;·&nbsp; {premium_note}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">ADX Trend Strength</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#6366f1' if d['adx']>25 else '#888'};">
            {d['adx']:.1f} — {adx_label} (+DI: {d['adx']:,.0f})</td>
      </tr>
    </table>
  </div>

  <!-- ═══ CANDLESTICK PATTERNS ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🕯️ Candlestick Patterns</div>
    {pattern_html}
  </div>

  <!-- ═══ FULL SIGNAL BREAKDOWN ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🔍 Full Signal Breakdown</div>
    {reason_html}
  </div>

  <!-- ═══ SUPPORT / RESISTANCE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:10px;">🗺️ Key Support &amp; Resistance Levels</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Price</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Type</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Distance</th>
      </tr>
      {sr_html}
    </table>
  </div>

  <!-- ═══ STRIKE PRICE GUIDE ═══ -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;
         margin-bottom:12px;">🎯 Strike Price Guide (ATR-based)</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="33%" style="padding:0 4px 0 0;">
          <div style="background:#fcebeb;border:1px solid #f7a3a3;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#a32d2d;margin-bottom:6px;font-weight:600;">
              OTM PUT STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#a32d2d;">${d['otm_put']:,}</div>
            <div style="font-size:10px;color:#c07070;margin-top:4px;">
              Cheaper · Needs bigger move</div>
          </div>
        </td>
        <td width="34%" style="padding:0 4px;">
          <div style="background:#fff;border:2px solid #d3d1c7;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#888;margin-bottom:6px;font-weight:600;">
              ATM STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#333;">${d['atm']:,}</div>
            <div style="font-size:10px;color:#aaa;margin-top:4px;">
              Safer · Higher premium</div>
          </div>
        </td>
        <td width="33%" style="padding:0 0 0 4px;">
          <div style="background:#eaf3de;border:1px solid #a3d977;border-radius:10px;
               padding:14px;text-align:center;">
            <div style="font-size:10px;color:#3b6d11;margin-bottom:6px;font-weight:600;">
              OTM CALL STRIKE</div>
            <div style="font-size:20px;font-weight:800;color:#3b6d11;">${d['otm_call']:,}</div>
            <div style="font-size:10px;color:#6a9b3a;margin-top:4px;">
              Cheaper · Needs bigger move</div>
          </div>
        </td>
      </tr>
    </table>
    <div style="margin-top:10px;padding:8px 12px;background:#fff8e7;border-radius:6px;
         font-size:11px;color:#854f0b;">
      💡 <strong>ATM</strong> = safer, more expensive. <strong>OTM</strong> = cheaper, needs larger price move to profit. Always buy on your signal direction only.
    </div>
  </div>

  <!-- ═══ FOOTER ═══ -->
  <div style="padding:16px 24px;background:#f8f7f5;">
    <div style="font-size:10px;color:#bbb;text-align:center;line-height:1.8;">
      Automated analysis using 12 indicators (EMA · RSI · Stoch RSI · MACD · BB · ADX · VWAP · Volume · ATR · Momentum · S/R · Candlestick Patterns)<br>
      <strong style="color:#e24b4a;">Not financial advice.</strong> Options can expire worthless. Always manage risk. Never invest more than you can afford to lose.
    </div>
  </div>

</div>
</body></html>"""

    return html, f"{verdict_emoji} {verdict}"


# ── SEND EMAIL ────────────────────────────────────────────────────────────────
def send_email(html_body, verdict_text, price):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"BTC Signal: {verdict_text} | ${price:,.0f} | {datetime.utcnow().strftime('%H:%M UTC')}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
    print(f"✅ Email sent: {verdict_text} | ${price:,.0f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching BTC data from KuCoin...")
    df = fetch_ohlcv()
    print(f"Got {len(df)} candles. Price: ${df.iloc[-1]['close']:,.0f}")
    df = add_indicators(df)
    result = analyze(df)
    print(f"Call: {result['call_score']} | Put: {result['put_score']}")
    html, verdict = build_email(result)
    send_email(html, verdict, result["price"])

if __name__ == "__main__":
    main()
