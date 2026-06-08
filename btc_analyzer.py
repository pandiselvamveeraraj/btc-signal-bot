"""
BTC Options Analyzer — Advanced Edition
Indicators: EMA, RSI, MACD, Bollinger Bands, ADX, VWAP, Volume,
            Support/Resistance, Candlestick Patterns, ATR
"""

import requests
import pandas as pd
import ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


# ── CONFIG ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TO_EMAIL   = os.environ["TO_EMAIL"]

SYMBOL   = "BTC-USDT"
INTERVAL = "1hour"
LIMIT    = 300   # more candles = better S/R + ADX accuracy


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

    # Trend EMAs
    df["ema9"]   = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema20"]  = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(c, 14).rsi()

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(c, 14, 3, 3)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100

    # MACD
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()   # 0=lower, 1=upper

    # ADX — trend strength
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]    = adx.adx()
    df["adx_pos"] = adx.adx_pos()   # +DI
    df["adx_neg"] = adx.adx_neg()   # -DI

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()

    # Volume MA
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]  # >1.5 = volume spike

    # VWAP (rolling daily proxy — 24 candles for 1h)
    df["vwap"] = (c * v).rolling(24).sum() / v.rolling(24).sum()

    # Momentum
    df["mom10"] = ta.momentum.ROCIndicator(c, 10).roc()

    return df


# ── SUPPORT / RESISTANCE ──────────────────────────────────────────────────────
def find_sr_levels(df, lookback=60, tolerance=0.005):
    """Find key support/resistance from recent pivot highs/lows."""
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

    # Cluster close levels
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

    body      = abs(c[i] - o[i])
    candle_rng = h[i] - l[i]
    upper_wick = h[i] - max(c[i], o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    prev_body  = abs(c[i-1] - o[i-1])

    # Doji
    if candle_rng > 0 and body / candle_rng < 0.1:
        patterns.append(("🕯️", "Doji — indecision, reversal possible", "neutral"))

    # Hammer (bullish)
    if (lower_wick > 2 * body and upper_wick < body * 0.5
            and c[i-1] < o[i-1]):
        patterns.append(("🔨", "Hammer — bullish reversal signal", "call"))

    # Shooting Star (bearish)
    if (upper_wick > 2 * body and lower_wick < body * 0.5
            and c[i-1] > o[i-1]):
        patterns.append(("⭐", "Shooting Star — bearish reversal signal", "put"))

    # Bullish Engulfing
    if (c[i] > o[i] and c[i-1] < o[i-1]
            and c[i] > o[i-1] and o[i] < c[i-1]
            and body > prev_body):
        patterns.append(("🟢", "Bullish Engulfing — strong reversal up", "call"))

    # Bearish Engulfing
    if (c[i] < o[i] and c[i-1] > o[i-1]
            and c[i] < o[i-1] and o[i] > c[i-1]
            and body > prev_body):
        patterns.append(("🔴", "Bearish Engulfing — strong reversal down", "put"))

    # Bullish Marubozu
    if c[i] > o[i] and body / candle_rng > 0.85:
        patterns.append(("💚", "Bullish Marubozu — strong buying pressure", "call"))

    # Bearish Marubozu
    if c[i] < o[i] and body / candle_rng > 0.85:
        patterns.append(("🔻", "Bearish Marubozu — strong selling pressure", "put"))

    # Morning Star (3-candle bullish)
    if i >= 2:
        if (c[i-2] < o[i-2] and
            abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and
            c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2):
            patterns.append(("🌅", "Morning Star — bullish reversal (3-candle)", "call"))

    # Evening Star (3-candle bearish)
    if i >= 2:
        if (c[i-2] > o[i-2] and
            abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3 and
            c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2):
            patterns.append(("🌆", "Evening Star — bearish reversal (3-candle)", "put"))

    return patterns


# ── MAIN SCORING ENGINE ───────────────────────────────────────────────────────
def analyze(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]

    call_score = 0
    put_score  = 0
    reasons    = []

    # ── 1. EMA TREND STRUCTURE (max 3 pts) ───────────────────────────────────
    if last["ema9"] > last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 3
        reasons.append(("📈", "Full EMA bullish stack (9>20>50>200)", "call", "Trend"))
    elif last["ema9"] < last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 3
        reasons.append(("📉", "Full EMA bearish stack (9<20<50<200)", "put", "Trend"))
    elif last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 2
        reasons.append(("📈", "EMA20/50/200 bullish stack", "call", "Trend"))
    elif last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 2
        reasons.append(("📉", "EMA20/50/200 bearish stack", "put", "Trend"))
    else:
        reasons.append(("⚠️", "EMAs mixed — no clear trend", "neutral", "Trend"))

    # ── 2. PRICE vs EMA200 ────────────────────────────────────────────────────
    if price > last["ema200"]:
        call_score += 1
        reasons.append(("✅", f"Price above EMA200 (${last['ema200']:,.0f}) — macro bullish", "call", "Trend"))
    else:
        put_score += 1
        reasons.append(("✅", f"Price below EMA200 (${last['ema200']:,.0f}) — macro bearish", "put", "Trend"))

    # ── 3. ADX TREND STRENGTH ─────────────────────────────────────────────────
    adx_val = last["adx"]
    if adx_val > 25:
        if last["adx_pos"] > last["adx_neg"]:
            call_score += 2
            reasons.append(("💪", f"ADX {adx_val:.0f} — strong BULLISH trend confirmed", "call", "Trend"))
        else:
            put_score += 2
            reasons.append(("💪", f"ADX {adx_val:.0f} — strong BEARISH trend confirmed", "put", "Trend"))
    elif adx_val < 20:
        reasons.append(("😴", f"ADX {adx_val:.0f} — weak trend, choppy market", "neutral", "Trend"))
    else:
        reasons.append(("〰️", f"ADX {adx_val:.0f} — trend developing", "neutral", "Trend"))

    # ── 4. RSI ────────────────────────────────────────────────────────────────
    rsi = last["rsi"]
    if rsi < 30:
        call_score += 3
        reasons.append(("🟢", f"RSI strongly oversold ({rsi:.1f}) — high bounce probability", "call", "Momentum"))
    elif rsi < 40:
        call_score += 1
        reasons.append(("🟢", f"RSI oversold zone ({rsi:.1f}) — bounce likely", "call", "Momentum"))
    elif rsi > 70:
        put_score += 3
        reasons.append(("🔴", f"RSI strongly overbought ({rsi:.1f}) — pullback likely", "put", "Momentum"))
    elif rsi > 60:
        put_score += 1
        reasons.append(("🔴", f"RSI overbought zone ({rsi:.1f}) — watch for reversal", "put", "Momentum"))
    elif 48 < rsi < 55:
        call_score += 1
        reasons.append(("⚪", f"RSI neutral-bullish ({rsi:.1f})", "call", "Momentum"))
    else:
        reasons.append(("⚪", f"RSI neutral ({rsi:.1f})", "neutral", "Momentum"))

    # ── 5. STOCHASTIC RSI ─────────────────────────────────────────────────────
    sk, sd = last["stoch_k"], last["stoch_d"]
    if sk < 20 and sd < 20:
        call_score += 2
        reasons.append(("🟢", f"Stoch RSI oversold (K:{sk:.0f} D:{sd:.0f}) — buy signal", "call", "Momentum"))
    elif sk > 80 and sd > 80:
        put_score += 2
        reasons.append(("🔴", f"Stoch RSI overbought (K:{sk:.0f} D:{sd:.0f}) — sell signal", "put", "Momentum"))
    elif sk > sd and sk < 50:
        call_score += 1
        reasons.append(("🟡", f"Stoch RSI bullish cross in lower zone (K:{sk:.0f})", "call", "Momentum"))
    elif sk < sd and sk > 50:
        put_score += 1
        reasons.append(("🟡", f"Stoch RSI bearish cross in upper zone (K:{sk:.0f})", "put", "Momentum"))

    # ── 6. MACD ───────────────────────────────────────────────────────────────
    mh   = last["macd_hist"]
    mh_p = prev["macd_hist"]
    ml   = last["macd"]
    ms   = last["macd_signal"]

    if ml > ms and ml > 0 and mh > mh_p:
        call_score += 3
        reasons.append(("📈", f"MACD above signal + positive + rising (hist:{mh:+.1f})", "call", "MACD"))
    elif ml > ms and mh > mh_p:
        call_score += 2
        reasons.append(("📈", f"MACD bullish cross + rising (hist:{mh:+.1f})", "call", "MACD"))
    elif ml > ms:
        call_score += 1
        reasons.append(("📈", f"MACD above signal line (hist:{mh:+.1f})", "call", "MACD"))
    elif ml < ms and ml < 0 and mh < mh_p:
        put_score += 3
        reasons.append(("📉", f"MACD below signal + negative + falling (hist:{mh:+.1f})", "put", "MACD"))
    elif ml < ms and mh < mh_p:
        put_score += 2
        reasons.append(("📉", f"MACD bearish cross + falling (hist:{mh:+.1f})", "put", "MACD"))
    elif ml < ms:
        put_score += 1
        reasons.append(("📉", f"MACD below signal line (hist:{mh:+.1f})", "put", "MACD"))

    # ── 7. BOLLINGER BANDS ────────────────────────────────────────────────────
    bb_pct = last["bb_pct"]
    if price <= last["bb_lower"]:
        call_score += 2
        reasons.append(("🟢", f"Price AT lower BB — mean reversion setup → CALL", "call", "Volatility"))
    elif bb_pct < 0.2:
        call_score += 1
        reasons.append(("🟢", f"Price near lower BB (BB%:{bb_pct:.2f}) — oversold zone", "call", "Volatility"))
    elif price >= last["bb_upper"]:
        put_score += 2
        reasons.append(("🔴", f"Price AT upper BB — mean reversion setup → PUT", "put", "Volatility"))
    elif bb_pct > 0.8:
        put_score += 1
        reasons.append(("🔴", f"Price near upper BB (BB%:{bb_pct:.2f}) — overbought zone", "put", "Volatility"))

    # BB squeeze (low volatility = big move coming)
    bb_width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
    if bb_width < 2.0:
        reasons.append(("💥", f"BB Squeeze detected (width:{bb_width:.1f}%) — big move imminent!", "neutral", "Volatility"))

    # ── 8. VWAP ───────────────────────────────────────────────────────────────
    if pd.notna(last["vwap"]):
        if price > last["vwap"] * 1.005:
            call_score += 1
            reasons.append(("✅", f"Price above VWAP (${last['vwap']:,.0f}) — bullish intraday", "call", "VWAP"))
        elif price < last["vwap"] * 0.995:
            put_score += 1
            reasons.append(("✅", f"Price below VWAP (${last['vwap']:,.0f}) — bearish intraday", "put", "VWAP"))

    # ── 9. VOLUME ─────────────────────────────────────────────────────────────
    vol_ratio = last["vol_ratio"]
    is_green  = last["close"] > last["open"]
    if vol_ratio > 2.0:
        if is_green:
            call_score += 2
            reasons.append(("📊", f"High volume green candle (vol {vol_ratio:.1f}x avg) — strong buying", "call", "Volume"))
        else:
            put_score += 2
            reasons.append(("📊", f"High volume red candle (vol {vol_ratio:.1f}x avg) — strong selling", "put", "Volume"))
    elif vol_ratio > 1.5:
        if is_green:
            call_score += 1
            reasons.append(("📊", f"Above-avg volume green candle ({vol_ratio:.1f}x)", "call", "Volume"))
        else:
            put_score += 1
            reasons.append(("📊", f"Above-avg volume red candle ({vol_ratio:.1f}x)", "put", "Volume"))
    elif vol_ratio < 0.6:
        reasons.append(("⚠️", f"Very low volume ({vol_ratio:.1f}x avg) — weak conviction", "neutral", "Volume"))

    # ── 10. MOMENTUM (ROC) ────────────────────────────────────────────────────
    mom = last["mom10"]
    if mom > 3:
        call_score += 1
        reasons.append(("🚀", f"Strong upward momentum (ROC:{mom:+.1f}%)", "call", "Momentum"))
    elif mom < -3:
        put_score += 1
        reasons.append(("💣", f"Strong downward momentum (ROC:{mom:+.1f}%)", "put", "Momentum"))

    # ── 11. CANDLESTICK PATTERNS ──────────────────────────────────────────────
    patterns = detect_patterns(df)
    for icon, text, side in patterns:
        if side == "call":
            call_score += 1
            reasons.append((icon, text, "call", "Pattern"))
        elif side == "put":
            put_score += 1
            reasons.append((icon, text, "put", "Pattern"))
        else:
            reasons.append((icon, text, "neutral", "Pattern"))

    # ── 12. SUPPORT / RESISTANCE ──────────────────────────────────────────────
    sr_levels = find_sr_levels(df, lookback=80)
    near_support    = [lv for lv in sr_levels if 0 < (price - lv) / price < 0.015]
    near_resistance = [lv for lv in sr_levels if 0 < (lv - price) / price < 0.015]

    if near_support:
        call_score += 1
        reasons.append(("🛡️", f"Near key support: ${near_support[-1]:,.0f} — bounce zone", "call", "S/R"))
    if near_resistance:
        put_score += 1
        reasons.append(("🧱", f"Near key resistance: ${near_resistance[0]:,.0f} — rejection zone", "put", "S/R"))

    # ── COMPILE RESULT ────────────────────────────────────────────────────────
    atr_pct  = (last["atr"] / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(last["atr"] * 1.5 / 100) * 100

    return {
        "call_score": call_score,
        "put_score": put_score,
        "reasons": reasons,
        "price": price,
        "rsi": rsi,
        "stoch_k": sk,
        "stoch_d": sd,
        "adx": adx_val,
        "atr_pct": atr_pct,
        "atr": last["atr"],
        "macd_hist": mh,
        "bb_pct": bb_pct,
        "vol_ratio": vol_ratio,
        "vwap": last["vwap"],
        "mom10": mom,
        "ema20": last["ema20"],
        "ema50": last["ema50"],
        "ema200": last["ema200"],
        "atm": atm,
        "otm_call": atm + atr_step,
        "otm_put": atm - atr_step,
        "sr_levels": sr_levels[-6:] if sr_levels else [],
        "patterns": patterns,
    }


# ── BUILD HTML EMAIL ──────────────────────────────────────────────────────────
def build_email(d):
    cs = d["call_score"]
    ps = d["put_score"]
    total = cs + ps or 1
    gap = abs(cs - ps)
    pct = round((max(cs, ps) / total) * 100)
    strength = "Very Strong" if gap >= 8 else "Strong" if gap >= 5 else "Moderate" if gap >= 3 else "Weak"

    if cs > ps:
        verdict_text  = f"🟢 BUY CALL — {strength}"
        verdict_color = "#1d9e75"
        verdict_bg    = "#eaf3de"
        bar_call      = pct
        bar_put       = 100 - pct
    elif ps > cs:
        verdict_text  = f"🔴 BUY PUT — {strength}"
        verdict_color = "#a32d2d"
        verdict_bg    = "#fcebeb"
        bar_call      = 100 - pct
        bar_put       = pct
    else:
        verdict_text  = "🟡 No Clear Signal — Wait"
        verdict_color = "#854f0b"
        verdict_bg    = "#faeeda"
        bar_call = bar_put = 50

    # Group reasons by category
    cats = ["Trend", "Momentum", "MACD", "Volatility", "VWAP", "Volume", "S/R", "Pattern"]
    grouped = {c: [] for c in cats}
    for icon, text, side, cat in d["reasons"]:
        grouped.get(cat, grouped["Trend"]).append((icon, text, side))

    reason_sections = ""
    for cat in cats:
        items = grouped[cat]
        if not items:
            continue
        reason_sections += f"""
        <tr><td colspan="2" style="padding:8px 8px 2px;font-size:10px;color:#aaa;
            text-transform:uppercase;letter-spacing:.08em;border-top:1px solid #f0eeea;">
            {cat}</td></tr>"""
        for icon, text, side in items:
            color = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888"
            reason_sections += f"""
        <tr>
          <td style="padding:5px 8px;font-size:13px;width:26px;">{icon}</td>
          <td style="padding:5px 8px;font-size:13px;color:{color};">{text}</td>
        </tr>"""

    sr_html = ""
    if d["sr_levels"]:
        sr_html = "<div style='font-size:11px;color:#888;margin-top:6px;'>Key S/R levels: "
        sr_html += " | ".join([f"<strong>${lv:,.0f}</strong>" for lv in sorted(d["sr_levels"])])
        sr_html += "</div>"

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    adx_trend = "Strong trend" if d["adx"] > 25 else "Weak/choppy"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f5f5f3;font-family:Arial,sans-serif;">
<div style="max-width:600px;margin:28px auto;background:#fff;border-radius:14px;
     overflow:hidden;border:1px solid #e0dfd8;box-shadow:0 2px 12px rgba(0,0,0,.06);">

  <!-- Header -->
  <div style="background:#0f0f0f;padding:18px 24px;display:flex;
       align-items:center;justify-content:space-between;">
    <div>
      <div style="color:#fff;font-size:19px;font-weight:700;letter-spacing:-.3px;">
        ₿ BTC / USDT Options Signal</div>
      <div style="color:#666;font-size:11px;margin-top:2px;">{now} · KuCoin · 1H · 12 Indicators</div>
    </div>
    <div style="text-align:right;">
      <div style="color:#fff;font-size:24px;font-weight:700;">${d['price']:,.0f}</div>
      <div style="color:#666;font-size:11px;">Live Price</div>
    </div>
  </div>

  <!-- Verdict -->
  <div style="background:{verdict_bg};padding:20px 24px;border-bottom:1px solid #e0dfd8;">
    <div style="font-size:11px;color:#888;text-transform:uppercase;
         letter-spacing:.06em;margin-bottom:8px;">Signal Verdict</div>
    <div style="font-size:26px;font-weight:800;color:{verdict_color};
         letter-spacing:-.5px;">{verdict_text}</div>
    <div style="margin-top:10px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="font-size:11px;color:#3b6d11;width:40px;">CALL {cs}</span>
        <div style="flex:1;background:#e8e7e3;border-radius:4px;height:8px;overflow:hidden;">
          <div style="width:{bar_call}%;background:#1d9e75;height:100%;border-radius:4px;"></div>
        </div>
        <div style="flex:1;background:#e8e7e3;border-radius:4px;height:8px;overflow:hidden;">
          <div style="width:{bar_put}%;background:#e24b4a;height:100%;border-radius:4px;float:right;"></div>
        </div>
        <span style="font-size:11px;color:#a32d2d;width:40px;text-align:right;">PUT {ps}</span>
      </div>
    </div>
    <div style="font-size:12px;color:#888;margin-top:6px;">
      Confidence: <strong style="color:{verdict_color};">{pct}%</strong> &nbsp;·&nbsp; Expiry: 1–3 days
    </div>
    {sr_html}
  </div>

  <!-- Metrics Grid -->
  <div style="padding:16px 24px;border-bottom:1px solid #e0dfd8;">
    <div style="font-size:10px;color:#aaa;text-transform:uppercase;
         letter-spacing:.08em;margin-bottom:10px;">Key Metrics</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">RSI 14</div>
        <div style="font-size:17px;font-weight:700;
             color:{'#1d9e75' if d['rsi']<35 else '#a32d2d' if d['rsi']>65 else '#333'};">
             {d['rsi']:.1f}</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">Stoch RSI</div>
        <div style="font-size:17px;font-weight:700;
             color:{'#1d9e75' if d['stoch_k']<20 else '#a32d2d' if d['stoch_k']>80 else '#333'};">
             {d['stoch_k']:.0f}</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">ADX</div>
        <div style="font-size:17px;font-weight:700;color:#333;">{d['adx']:.0f}</div>
        <div style="font-size:9px;color:#999;">{adx_trend}</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">MACD Hist</div>
        <div style="font-size:17px;font-weight:700;
             color:{'#1d9e75' if d['macd_hist']>0 else '#a32d2d'};">
             {d['macd_hist']:+.1f}</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">BB %</div>
        <div style="font-size:17px;font-weight:700;color:#333;">{d['bb_pct']:.2f}</div>
        <div style="font-size:9px;color:#999;">0=low 1=high</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">Vol Ratio</div>
        <div style="font-size:17px;font-weight:700;
             color:{'#1d9e75' if d['vol_ratio']>1.5 else '#333'};">
             {d['vol_ratio']:.1f}x</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">ATR %</div>
        <div style="font-size:17px;font-weight:700;color:#333;">{d['atr_pct']:.2f}%</div>
      </div>
      <div style="background:#f8f7f5;border-radius:8px;padding:10px;">
        <div style="font-size:10px;color:#999;margin-bottom:3px;">Momentum</div>
        <div style="font-size:17px;font-weight:700;
             color:{'#1d9e75' if d['mom10']>0 else '#a32d2d'};">
             {d['mom10']:+.1f}%</div>
      </div>
    </div>
  </div>

  <!-- EMA Levels -->
  <div style="padding:12px 24px;border-bottom:1px solid #e0dfd8;
       display:flex;gap:12px;flex-wrap:wrap;">
    <div style="font-size:12px;color:#888;">
      EMA9 <strong style="color:#333;">${d['ema20']:,.0f}</strong>
    </div>
    <div style="font-size:12px;color:#888;">
      EMA50 <strong style="color:#333;">${d['ema50']:,.0f}</strong>
    </div>
    <div style="font-size:12px;color:#888;">
      EMA200 <strong style="color:#333;">${d['ema200']:,.0f}</strong>
    </div>
    <div style="font-size:12px;color:#888;">
      VWAP <strong style="color:#333;">${d['vwap']:,.0f}</strong>
    </div>
  </div>

  <!-- Signal Breakdown -->
  <div style="padding:16px 24px;border-bottom:1px solid #e0dfd8;">
    <div style="font-size:10px;color:#aaa;text-transform:uppercase;
         letter-spacing:.08em;margin-bottom:6px;">Full Signal Breakdown</div>
    <table style="width:100%;border-collapse:collapse;">
      {reason_sections}
    </table>
  </div>

  <!-- Strike Hints -->
  <div style="padding:16px 24px;border-bottom:1px solid #e0dfd8;">
    <div style="font-size:10px;color:#aaa;text-transform:uppercase;
         letter-spacing:.08em;margin-bottom:10px;">Strike Price Hints (1.5× ATR)</div>
    <div style="display:flex;gap:10px;">
      <div style="flex:1;background:#fcebeb;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:10px;color:#a32d2d;margin-bottom:4px;">OTM PUT</div>
        <div style="font-size:16px;font-weight:700;color:#a32d2d;">${d['otm_put']:,}</div>
      </div>
      <div style="flex:1;background:#f5f5f3;border:1px solid #d3d1c7;
           border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:10px;color:#888;margin-bottom:4px;">ATM</div>
        <div style="font-size:16px;font-weight:700;color:#1a1a1a;">${d['atm']:,}</div>
      </div>
      <div style="flex:1;background:#eaf3de;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:10px;color:#3b6d11;margin-bottom:4px;">OTM CALL</div>
        <div style="font-size:16px;font-weight:700;color:#3b6d11;">${d['otm_call']:,}</div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div style="padding:14px 24px;background:#f8f7f5;">
    <div style="font-size:10px;color:#bbb;text-align:center;line-height:1.6;">
      Automated analysis using 12 indicators — not financial advice.<br>
      Options can expire worthless. Always manage your risk carefully.
    </div>
  </div>

</div>
</body></html>"""

    return html, verdict_text


# ── SEND EMAIL ────────────────────────────────────────────────────────────────
def send_email(html_body, verdict_text, price):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"BTC Signal: {verdict_text} | ${price:,.0f}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
    print(f"Email sent: {verdict_text} | ${price:,.0f}")


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
