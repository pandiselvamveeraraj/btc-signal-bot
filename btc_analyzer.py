"""
BTC Options Analyzer — Complete Edition with Greeks, IV, Strategy Selection & Risk Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW vs original:
  • Black-Scholes Delta, Gamma, Theta, Vega calculation
  • Historical Volatility (HV20, HV30) as IV proxy
  • IV Rank approximation from 30-day HV range
  • Smart strategy selection: Naked / Spread / Straddle / Strangle / Iron Condor
  • DTE recommendation based on signal strength + volatility
  • Stop-loss, profit target, position sizing (ATR-based)
  • Risk/reward ratio gating (min 1:2)
  • ADX +DI / -DI bug fix
  • Confidence tier: Weak / Moderate / Strong / Very Strong with trade gate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install dependencies:
  pip install requests pandas ta scipy smtplib

Environment variables required:
  GMAIL_USER  — your gmail address
  GMAIL_PASS  — gmail app password (not your main password)
  TO_EMAIL    — recipient email
"""

import requests
import pandas as pd
import ta
import smtplib
import os
import math
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from scipy.stats import norm

# ── CONFIG ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TO_EMAIL   = os.environ["TO_EMAIL"]

SYMBOL           = "BTC-USDT"
INTERVAL         = "15min"
LIMIT            = 200
RISK_FREE_RATE   = 0.05          # annualised, ~US T-bill rate
PORTFOLIO_VALUE  = 10_000        # USD — used for position sizing
RISK_PCT         = 0.02          # max 2% of portfolio per trade
CONTRACT_SIZE    = 0.001         # 1 BTC option contract = 0.001 BTC on most exchanges
MIN_SIGNAL_GAP   = 3             # below this gap → "no trade" recommendation


# ══════════════════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Trend
    df["ema9"]   = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema20"]  = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()

    # Momentum
    df["rsi"]    = ta.momentum.RSIIndicator(c, 14).rsi()
    stoch = ta.momentum.StochRSIIndicator(c, 14, 3, 3)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100

    # MACD
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # Volatility
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"]      = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()

    # ADX — keep adx_pos and adx_neg separately (bug fix)
    adx_ind        = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]      = adx_ind.adx()
    df["adx_pos"]  = adx_ind.adx_pos()
    df["adx_neg"]  = adx_ind.adx_neg()

    # Volume
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]

    # VWAP (rolling 24-bar)
    df["vwap"] = (c * v).rolling(24).sum() / v.rolling(24).sum()

    # Momentum ROC
    df["mom10"] = ta.momentum.ROCIndicator(c, 10).roc()

    # ── Historical Volatility (used as IV proxy) ──────────────────────────────
    log_ret = (c / c.shift(1)).apply(math.log)
    df["hv20"] = log_ret.rolling(20).std() * math.sqrt(252 * 96)   # 96 × 15-min bars/day
    df["hv30"] = log_ret.rolling(30).std() * math.sqrt(252 * 96)

    # HV rank (where current HV sits in its own 60-bar range → IV rank proxy)
    rolling_max = df["hv20"].rolling(60).max()
    rolling_min = df["hv20"].rolling(60).min()
    df["iv_rank"] = (df["hv20"] - rolling_min) / (rolling_max - rolling_min + 1e-9) * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    S     = spot price
    K     = strike price
    T     = time to expiry in years  (e.g. 1 day = 1/365)
    r     = risk-free rate (annualised)
    sigma = implied / historical volatility (annualised, e.g. 0.80 = 80%)

    Returns dict: delta, gamma, theta (per day), vega (per 1% IV move)
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
    else:  # put
        delta = norm.cdf(d1) - 1
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega  = S * pdf_d1 * math.sqrt(T) / 100   # per 1% change in vol
    theta = theta_annual / 365                  # per calendar day

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
    }


def greeks_for_strikes(price, atm, otm_call, otm_put, sigma, dte_days):
    """Compute greeks for ATM call, OTM call, and OTM put."""
    T = max(dte_days, 0.5) / 365  # minimum half-day to avoid division by zero
    r = RISK_FREE_RATE

    atm_call  = black_scholes_greeks(price, atm,      T, r, sigma, "call")
    otmc      = black_scholes_greeks(price, otm_call,  T, r, sigma, "call")
    otmp      = black_scholes_greeks(price, otm_put,   T, r, sigma, "put")
    atm_put   = black_scholes_greeks(price, atm,       T, r, sigma, "put")

    return {
        "atm_call":  atm_call,
        "otm_call":  otmc,
        "atm_put":   atm_put,
        "otm_put":   otmp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════
def find_sr_levels(df, lookback=80, tolerance=0.005):
    highs = df["high"].tail(lookback)
    lows  = df["low"].tail(lookback)
    levels = []
    for i in range(2, len(highs) - 2):
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


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
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
        if (c[i-2] < o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3
                and c[i] > o[i] and c[i] > (o[i-2]+c[i-2])/2):
            patterns.append(("🌅", "Morning Star", "Bullish 3-candle reversal", "call"))
        if (c[i-2] > o[i-2] and abs(c[i-1]-o[i-1]) < abs(c[i-2]-o[i-2])*0.3
                and c[i] < o[i] and c[i] < (o[i-2]+c[i-2])/2):
            patterns.append(("🌆", "Evening Star", "Bearish 3-candle reversal", "put"))
    return patterns


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def select_strategy(call_score, put_score, adx_val, atr_pct, bb_width,
                    iv_rank, atm, otm_call, otm_put):
    """
    Returns (strategy_name, description, legs, side)
    Priorities:
      1. BB squeeze → Straddle (direction unknown, volatility play)
      2. Choppy market (ADX < 20, weak signal) → Iron Condor (sell premium)
      3. Strong trend + high IV → Debit Spread (direction + reduce cost)
      4. Default → Naked ATM / OTM option
    """
    gap = abs(call_score - put_score)
    direction = "call" if call_score > put_score else "put"

    # ── 1. Volatility squeeze → Straddle ─────────────────────────────────────
    if bb_width < 2.0 and iv_rank < 30:
        legs = f"Buy ATM Call ${atm:,} + Buy ATM Put ${atm:,}"
        return {
            "name":        "Long Straddle",
            "icon":        "🤸",
            "side":        "neutral",
            "description": "BB squeeze detected with low IV — buy both sides before the explosion.",
            "legs":        legs,
            "why":         "IV is cheap (rank {:.0f}%). Straddle profits if BTC moves >ATR in either direction.".format(iv_rank),
            "risk":        "Loses if BTC stays flat and IV stays compressed.",
        }

    # ── 2. Squeeze but IV high → Strangle (cheaper than straddle) ────────────
    if bb_width < 2.5 and iv_rank < 50 and gap < 4:
        legs = f"Buy OTM Call ${otm_call:,} + Buy OTM Put ${otm_put:,}"
        return {
            "name":        "Long Strangle",
            "icon":        "🎯",
            "side":        "neutral",
            "description": "Near-squeeze with moderate IV — cheaper than straddle, needs bigger move.",
            "legs":        legs,
            "why":         "OTM options cost less premium. Profits on large directional move.",
            "risk":        "Needs bigger BTC move than straddle to profit.",
        }

    # ── 3. Choppy, weak signal → Iron Condor (sell premium) ──────────────────
    if adx_val < 20 and gap < MIN_SIGNAL_GAP and iv_rank > 50:
        wing_call = round((otm_call + (otm_call - atm)) / 100) * 100
        wing_put  = round((otm_put  - (atm - otm_put))  / 100) * 100
        legs = (f"Sell ${otm_call:,}C / Sell ${otm_put:,}P  "
                f"|  Buy ${wing_call:,}C / Buy ${wing_put:,}P (wings)")
        return {
            "name":        "Iron Condor",
            "icon":        "🦅",
            "side":        "neutral",
            "description": "Choppy market with high IV — collect premium, profit if BTC stays in range.",
            "legs":        legs,
            "why":         "ADX {:.0f} confirms no trend. IV rank {:.0f}% makes selling premium attractive.".format(adx_val, iv_rank),
            "risk":        "Max loss if BTC breaks out strongly in either direction.",
        }

    # ── 4. Strong directional signal + high IV → Debit Spread (reduce cost) ──
    if gap >= 6 and atr_pct > 2.0:
        if direction == "call":
            legs = f"Buy ATM Call ${atm:,}  |  Sell OTM Call ${otm_call:,}"
            name = "Bull Call Spread"
            desc = "Strong bullish signal but high IV inflates premium — spread reduces cost."
        else:
            legs = f"Buy ATM Put ${atm:,}  |  Sell OTM Put ${otm_put:,}"
            name = "Bear Put Spread"
            desc = "Strong bearish signal but high IV inflates premium — spread reduces cost."
        return {
            "name":        name,
            "icon":        "📐",
            "side":        direction,
            "description": desc,
            "legs":        legs,
            "why":         f"Signal gap {gap} is strong. High ATR ({atr_pct:.1f}%) makes naked options expensive.",
            "risk":        "Capped profit at OTM strike. Won't capture full move.",
        }

    # ── 5. Moderate directional signal → Naked OTM option ────────────────────
    if gap >= MIN_SIGNAL_GAP:
        if direction == "call":
            legs = f"Buy OTM Call ${otm_call:,} (cheaper) or ATM Call ${atm:,} (safer)"
            name = "Long Call"
        else:
            legs = f"Buy OTM Put ${otm_put:,} (cheaper) or ATM Put ${atm:,} (safer)"
            name = "Long Put"
        return {
            "name":        name,
            "icon":        "📊",
            "side":        direction,
            "description": f"Standard directional play on signal gap of {gap}.",
            "legs":        legs,
            "why":         "Signal is clear enough for a naked directional option.",
            "risk":        "Full premium at risk if price moves against you.",
        }

    # ── 6. Signal too weak → No trade ────────────────────────────────────────
    return {
        "name":        "No Trade",
        "icon":        "🚫",
        "side":        "neutral",
        "description": f"Signal gap only {gap} — below minimum threshold of {MIN_SIGNAL_GAP}.",
        "legs":        "Wait for a stronger signal before entering any position.",
        "why":         "Entering with weak signal is gambling, not trading.",
        "risk":        "N/A — do not enter.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# DTE RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
def recommend_dte(signal_gap, atr_pct, bb_width, strategy_name):
    """Recommend days-to-expiry based on signal and strategy context."""
    if strategy_name in ("Long Straddle", "Long Strangle"):
        return 3, "3–7 DTE — give the squeeze time to resolve"
    if strategy_name == "Iron Condor":
        return 7, "7–14 DTE — collect theta decay over the ranging period"
    if strategy_name in ("Bull Call Spread", "Bear Put Spread"):
        if signal_gap >= 8 and atr_pct > 2.5:
            return 1, "1–2 DTE — very strong signal, high vol"
        return 3, "2–5 DTE — let the trend develop"
    # Naked options
    if signal_gap >= 8 and atr_pct > 2.0:
        return 1, "0–1 DTE (intraday) — very strong signal + high volatility"
    if signal_gap >= 5:
        return 2, "1–3 DTE — solid signal, short duration"
    return 0, "⚠️ Signal too weak — avoid any entry"


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def risk_management(price, atr, call_score, put_score, strategy):
    """
    Returns stop-loss price, profit target, max contracts, and risk/reward.
    """
    direction = strategy["side"]
    gap       = abs(call_score - put_score)

    stop_distance   = atr * 1.5
    target_distance = atr * 3.0    # minimum 1:2 RR

    if direction == "call":
        stop_price   = round((price - stop_distance) / 100) * 100
        target_price = round((price + target_distance) / 100) * 100
    elif direction == "put":
        stop_price   = round((price + stop_distance) / 100) * 100
        target_price = round((price - target_distance) / 100) * 100
    else:
        stop_price   = round((price - stop_distance) / 100) * 100
        target_price = round((price + target_distance) / 100) * 100

    # Position sizing: risk 2% of portfolio
    risk_per_trade  = PORTFOLIO_VALUE * RISK_PCT
    dollar_per_atr  = atr * CONTRACT_SIZE
    max_contracts   = max(1, int(risk_per_trade / (stop_distance * CONTRACT_SIZE)))

    risk_reward = round(target_distance / stop_distance, 2)

    return {
        "stop_price":     stop_price,
        "target_price":   target_price,
        "stop_distance":  round(stop_distance, 0),
        "target_distance":round(target_distance, 0),
        "max_contracts":  max_contracts,
        "risk_reward":    risk_reward,
        "risk_per_trade": round(risk_per_trade, 0),
        "dollar_per_atr": round(dollar_per_atr, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
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
    hv20  = last["hv20"]
    hv30  = last["hv30"]
    iv_rank = last["iv_rank"] if pd.notna(last["iv_rank"]) else 50.0

    call_score = put_score = 0
    reasons = []

    # ── EMA Stack ─────────────────────────────────────────────────────────────
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
    adx_pos = last["adx_pos"]   # fixed: was using adx twice
    adx_neg = last["adx_neg"]
    if adx_val > 25:
        if adx_pos > adx_neg:
            call_score += 2; reasons.append(("💪",f"ADX {adx_val:.0f} strong BULLISH (+DI:{adx_pos:.0f} > -DI:{adx_neg:.0f})","call","Trend"))
        else:
            put_score += 2;  reasons.append(("💪",f"ADX {adx_val:.0f} strong BEARISH (+DI:{adx_pos:.0f} < -DI:{adx_neg:.0f})","put","Trend"))
    elif adx_val < 20:
        reasons.append(("😴",f"ADX {adx_val:.0f} — choppy, no strong trend","neutral","Trend"))
    else:
        reasons.append(("〰️",f"ADX {adx_val:.0f} — trend developing","neutral","Trend"))

    # ── RSI ───────────────────────────────────────────────────────────────────
    if rsi < 30:
        call_score += 3; reasons.append(("🟢",f"RSI {rsi:.1f} — strongly oversold","call","Momentum"))
    elif rsi < 40:
        call_score += 1; reasons.append(("🟢",f"RSI {rsi:.1f} — oversold zone","call","Momentum"))
    elif rsi > 70:
        put_score += 3;  reasons.append(("🔴",f"RSI {rsi:.1f} — strongly overbought","put","Momentum"))
    elif rsi > 60:
        put_score += 1;  reasons.append(("🔴",f"RSI {rsi:.1f} — overbought zone","put","Momentum"))
    else:
        reasons.append(("⚪",f"RSI {rsi:.1f} — neutral","neutral","Momentum"))

    # ── Stoch RSI ─────────────────────────────────────────────────────────────
    if sk < 20 and sd < 20:
        call_score += 2; reasons.append(("🟢",f"Stoch RSI oversold K:{sk:.0f} D:{sd:.0f}","call","Momentum"))
    elif sk > 80 and sd > 80:
        put_score += 2;  reasons.append(("🔴",f"Stoch RSI overbought K:{sk:.0f} D:{sd:.0f}","put","Momentum"))
    elif sk > sd and sk < 50:
        call_score += 1; reasons.append(("🟡",f"Stoch RSI bullish cross low zone K:{sk:.0f}","call","Momentum"))
    elif sk < sd and sk > 50:
        put_score += 1;  reasons.append(("🟡",f"Stoch RSI bearish cross high zone K:{sk:.0f}","put","Momentum"))

    # ── MACD ──────────────────────────────────────────────────────────────────
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

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_pct   = last["bb_pct"]
    bb_width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
    if price <= last["bb_lower"]:
        call_score += 2; reasons.append(("🟢","Price at lower BB — mean reversion CALL","call","Volatility"))
    elif bb_pct < 0.2:
        call_score += 1; reasons.append(("🟢",f"Price near lower BB ({bb_pct:.2f})","call","Volatility"))
    elif price >= last["bb_upper"]:
        put_score += 2;  reasons.append(("🔴","Price at upper BB — mean reversion PUT","put","Volatility"))
    elif bb_pct > 0.8:
        put_score += 1;  reasons.append(("🔴",f"Price near upper BB ({bb_pct:.2f})","put","Volatility"))

    if bb_width < 2.0:
        reasons.append(("💥",f"BB Squeeze! Big move incoming (width {bb_width:.1f}%) — consider Straddle","neutral","Volatility"))
    elif bb_width < 2.5:
        reasons.append(("🌀",f"BB Near-Squeeze (width {bb_width:.1f}%) — Strangle opportunity","neutral","Volatility"))

    # ── IV / HV signals ───────────────────────────────────────────────────────
    if iv_rank < 20:
        call_score += 1; put_score += 1
        reasons.append(("💰",f"IV Rank {iv_rank:.0f}% — options VERY CHEAP, great to buy","call","Volatility"))
    elif iv_rank < 35:
        reasons.append(("💰",f"IV Rank {iv_rank:.0f}% — options cheap, buying favored","neutral","Volatility"))
    elif iv_rank > 70:
        reasons.append(("🔥",f"IV Rank {iv_rank:.0f}% — options EXPENSIVE, prefer spreads or selling","neutral","Volatility"))
    elif iv_rank > 50:
        reasons.append(("⚠️",f"IV Rank {iv_rank:.0f}% — premiums elevated, consider spreads","neutral","Volatility"))

    hv_diff = (hv20 - hv30) * 100  # positive = vol rising
    if hv_diff > 5:
        reasons.append(("📊",f"HV20 {hv20*100:.1f}% > HV30 {hv30*100:.1f}% — volatility expanding","neutral","Volatility"))
    elif hv_diff < -5:
        reasons.append(("📊",f"HV20 {hv20*100:.1f}% < HV30 {hv30*100:.1f}% — volatility contracting","neutral","Volatility"))

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if pd.notna(last["vwap"]):
        if price > last["vwap"] * 1.005:
            call_score += 1; reasons.append(("✅",f"Above VWAP ${last['vwap']:,.0f} — bullish","call","VWAP"))
        elif price < last["vwap"] * 0.995:
            put_score += 1;  reasons.append(("❌",f"Below VWAP ${last['vwap']:,.0f} — bearish","put","VWAP"))

    # ── Volume ────────────────────────────────────────────────────────────────
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

    # ── Momentum ROC ──────────────────────────────────────────────────────────
    mom = last["mom10"]
    if mom > 3:
        call_score += 1; reasons.append(("🚀",f"Strong upward momentum ROC:{mom:+.1f}%","call","Momentum"))
    elif mom < -3:
        put_score += 1;  reasons.append(("💣",f"Strong downward momentum ROC:{mom:+.1f}%","put","Momentum"))

    # ── Candlestick patterns ──────────────────────────────────────────────────
    patterns = detect_patterns(df)
    for icon, name, desc, side in patterns:
        if side == "call":
            call_score += 1; reasons.append((icon, f"{name}: {desc}", "call", "Pattern"))
        elif side == "put":
            put_score += 1;  reasons.append((icon, f"{name}: {desc}", "put", "Pattern"))
        else:
            reasons.append((icon, f"{name}: {desc}", "neutral", "Pattern"))

    # ── Support / Resistance ──────────────────────────────────────────────────
    sr_levels       = find_sr_levels(df, lookback=80)
    near_support    = [lv for lv in sr_levels if 0 < (price - lv) / price < 0.015]
    near_resistance = [lv for lv in sr_levels if 0 < (lv - price) / price < 0.015]
    if near_support:
        call_score += 1; reasons.append(("🛡️",f"Near support ${near_support[-1]:,.0f} — bounce zone","call","S/R"))
    if near_resistance:
        put_score += 1;  reasons.append(("🧱",f"Near resistance ${near_resistance[0]:,.0f} — rejection","put","S/R"))

    # ── Derived values ────────────────────────────────────────────────────────
    atr      = last["atr"]
    atr_pct  = (atr / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(atr * 1.5 / 100) * 100
    otm_call = atm + atr_step
    otm_put  = atm - atr_step

    # ── Strategy selection ────────────────────────────────────────────────────
    strategy = select_strategy(
        call_score, put_score, adx_val, atr_pct, bb_width,
        iv_rank, atm, otm_call, otm_put
    )

    # ── DTE recommendation ────────────────────────────────────────────────────
    gap = abs(call_score - put_score)
    dte_days, dte_label = recommend_dte(gap, atr_pct, bb_width, strategy["name"])

    # ── Greeks ────────────────────────────────────────────────────────────────
    sigma  = hv20 if pd.notna(hv20) and hv20 > 0 else 0.80
    greeks = greeks_for_strikes(price, atm, otm_call, otm_put, sigma, max(dte_days, 1))

    # ── Risk management ───────────────────────────────────────────────────────
    risk = risk_management(price, atr, call_score, put_score, strategy)

    # ── Chart data ────────────────────────────────────────────────────────────
    chart_closes = df["close"].tail(30).tolist()
    chart_highs  = df["high"].tail(30).tolist()
    chart_lows   = df["low"].tail(30).tolist()
    chart_opens  = df["open"].tail(30).tolist()

    return {
        "call_score": call_score, "put_score": put_score, "reasons": reasons,
        "price": price, "rsi": rsi, "stoch_k": sk, "stoch_d": sd,
        "adx": adx_val, "adx_pos": adx_pos, "adx_neg": adx_neg,
        "atr_pct": atr_pct, "atr": atr,
        "macd_hist": mh, "macd": ml, "macd_signal": ms,
        "bb_pct": bb_pct, "bb_upper": last["bb_upper"], "bb_lower": last["bb_lower"],
        "bb_width": bb_width, "bb_mid": last["bb_mid"],
        "vol_ratio": vol_ratio, "vwap": last["vwap"], "mom10": mom,
        "ema9": last["ema9"], "ema20": last["ema20"],
        "ema50": last["ema50"], "ema200": last["ema200"],
        "hv20": hv20, "hv30": hv30, "iv_rank": iv_rank, "sigma": sigma,
        "atm": atm, "otm_call": otm_call, "otm_put": otm_put,
        "sr_levels": sr_levels, "patterns": patterns,
        "strategy": strategy, "dte_days": dte_days, "dte_label": dte_label,
        "greeks": greeks, "risk": risk,
        "chart_closes": chart_closes, "chart_highs": chart_highs,
        "chart_lows": chart_lows, "chart_opens": chart_opens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SVG HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def svg_gauge(value, min_val, max_val, label,
              low_color="#e24b4a", high_color="#1d9e75", mid_color="#f59e0b"):
    pct = max(0, min(1, (value - min_val) / (max_val - min_val)))
    angle = -140 + pct * 280
    nx = 50 + 36 * math.cos(math.radians(angle - 90))
    ny = 54 + 36 * math.sin(math.radians(angle - 90))
    color = low_color if pct < 0.35 else (high_color if pct > 0.65 else mid_color)
    return f"""
<svg width="110" height="80" viewBox="0 0 110 80" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g{label}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{low_color}"/>
      <stop offset="50%" style="stop-color:{mid_color}"/>
      <stop offset="100%" style="stop-color:{high_color}"/>
    </linearGradient>
  </defs>
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="#e8e7e3" stroke-width="7" stroke-linecap="round"/>
  <path d="M 14 58 A 36 36 0 1 1 96 58" fill="none" stroke="url(#g{label})" stroke-width="7"
        stroke-linecap="round" stroke-dasharray="226" stroke-dashoffset="{226 - pct*226:.1f}"/>
  <line x1="55" y1="54" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#333" stroke-width="2" stroke-linecap="round"/>
  <circle cx="55" cy="54" r="4" fill="#333"/>
  <text x="55" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{value:.1f}</text>
  <text x="55" y="12" text-anchor="middle" font-size="9" fill="#999">{label}</text>
</svg>"""


def svg_candle_chart(closes, highs, lows, opens, width=560, height=120):
    n = len(closes)
    if n == 0:
        return ""
    mn, mx = min(lows), max(highs)
    rng = mx - mn or 1
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 20
    cw    = (width - pad_l - pad_r) / n
    bar_w = max(2, cw * 0.6)

    def fy(v):
        return pad_t + (1 - (v - mn) / rng) * (height - pad_t - pad_b)

    candles = ""
    for i in range(n):
        x    = pad_l + i * cw + cw / 2
        bull = closes[i] >= opens[i]
        col  = "#1d9e75" if bull else "#e24b4a"
        o_y, c_y = fy(opens[i]), fy(closes[i])
        h_y, l_y = fy(highs[i]), fy(lows[i])
        body_y = min(o_y, c_y)
        body_h = max(abs(o_y - c_y), 1)
        candles += f'<line x1="{x:.1f}" y1="{h_y:.1f}" x2="{x:.1f}" y2="{l_y:.1f}" stroke="{col}" stroke-width="1"/>'
        candles += f'<rect x="{x - bar_w/2:.1f}" y="{body_y:.1f}" width="{bar_w:.1f}" height="{body_h:.1f}" fill="{col}"/>'

    labels = ""
    for i in range(3):
        v = mn + rng * i / 2
        y = fy(v)
        labels += f'<text x="{pad_l - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="8" fill="#aaa">${v:,.0f}</text>'
        labels += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#f0eeea" stroke-width="0.5"/>'

    last_y = fy(closes[-1])
    labels += f'<line x1="{pad_l}" y1="{last_y:.1f}" x2="{width - pad_r}" y2="{last_y:.1f}" stroke="#f59e0b" stroke-width="0.8" stroke-dasharray="3,2"/>'
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg" style="background:#fafaf8;border-radius:8px;">
  {labels}{candles}
  <text x="{width - pad_r}" y="{last_y - 3:.1f}" text-anchor="end" font-size="8" fill="#f59e0b">${closes[-1]:,.0f}</text>
</svg>"""


def svg_signal_bar(call_score, put_score):
    total    = call_score + put_score or 1
    call_pct = call_score / total * 100
    put_pct  = put_score  / total * 100
    return f"""<svg width="520" height="36" viewBox="0 0 520 36" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="8" width="520" height="20" rx="10" fill="#e8e7e3"/>
  <rect x="0" y="8" width="{call_pct * 5.2:.1f}" height="20" rx="10" fill="#1d9e75"/>
  <rect x="{520 - put_pct * 5.2:.1f}" y="8" width="{put_pct * 5.2:.1f}" height="20" rx="10" fill="#e24b4a"/>
  <text x="8" y="23" font-size="10" font-weight="700" fill="white">CALL {call_score}</text>
  <text x="512" y="23" text-anchor="end" font-size="10" font-weight="700" fill="white">PUT {put_score}</text>
</svg>"""


# ══════════════════════════════════════════════════════════════════════════════
# GREEKS TABLE HTML
# ══════════════════════════════════════════════════════════════════════════════
def build_greeks_html(greeks, strategy, price):
    """Build a full Greeks table for ATM call, OTM call, ATM put, OTM put."""
    rows = [
        ("ATM Call",  greeks["atm_call"],  "#1d9e75", "#eaf3de"),
        ("OTM Call",  greeks["otm_call"],  "#3b6d11", "#f0faf5"),
        ("ATM Put",   greeks["atm_put"],   "#a32d2d", "#fcebeb"),
        ("OTM Put",   greeks["otm_put"],   "#7b1f1f", "#fff0f0"),
    ]

    def greek_bar(value, max_val, color):
        pct = min(100, abs(value) / max_val * 100)
        return f'<div style="height:4px;background:#eee;border-radius:2px;margin-top:3px;"><div style="width:{pct:.0f}%;height:4px;background:{color};border-radius:2px;"></div></div>'

    header = """
    <tr style="background:#f8f7f5;">
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:left;font-weight:600;">Strike</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">Δ Delta</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">Γ Gamma</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">Θ Theta/day</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">ν Vega/1%</th>
    </tr>"""

    body = ""
    for label, g, color, bg in rows:
        body += f"""
    <tr style="background:{bg};">
      <td style="padding:8px 10px;font-size:12px;font-weight:700;color:{color};">{label}</td>
      <td style="padding:8px 10px;text-align:center;">
        <span style="font-size:12px;font-weight:700;color:{color};">{g['delta']:+.3f}</span>
        {greek_bar(g['delta'], 1.0, color)}
      </td>
      <td style="padding:8px 10px;text-align:center;">
        <span style="font-size:12px;color:#555;">{g['gamma']:.5f}</span>
        {greek_bar(g['gamma'], 0.0005, '#6366f1')}
      </td>
      <td style="padding:8px 10px;text-align:center;">
        <span style="font-size:12px;color:#e24b4a;">${g['theta']:+.1f}</span>
        {greek_bar(g['theta'], 50, '#e24b4a')}
      </td>
      <td style="padding:8px 10px;text-align:center;">
        <span style="font-size:12px;color:#f59e0b;">${g['vega']:.1f}</span>
        {greek_bar(g['vega'], 200, '#f59e0b')}
      </td>
    </tr>"""

    direction_row = ""
    if strategy["side"] in ("call", "put"):
        sel = "atm_call" if strategy["side"] == "call" else "atm_put"
        sg  = greeks[sel]
        direction_row = f"""
    <tr style="background:#fef9eb;border-top:2px solid #f59e0b;">
      <td colspan="5" style="padding:8px 10px;font-size:11px;color:#854f0b;">
        💡 <strong>Your signal direction ({strategy['side'].upper()}):</strong>
        Delta {sg['delta']:+.3f} means the option gains/loses
        <strong>${abs(sg['delta'] * price / 100):.0f}</strong> per $100 BTC move.
        Theta burns <strong>${abs(sg['theta']):.1f}/day</strong>.
        Vega gains <strong>${sg['vega']:.1f}</strong> per 1% IV increase.
      </td>
    </tr>"""

    return f"""
  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
    {header}{body}{direction_row}
  </table>"""


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY CARD HTML
# ══════════════════════════════════════════════════════════════════════════════
def build_strategy_html(strategy, dte_label, risk, price):
    side    = strategy["side"]
    s_color = "#1d9e75" if side == "call" else "#a32d2d" if side == "put" else "#854f0b"
    s_bg    = "#eaf3de" if side == "call" else "#fcebeb" if side == "put" else "#faeeda"
    s_bdr   = "#a3d977" if side == "call" else "#f7a3a3" if side == "put" else "#f5c97e"

    rr_color = "#1d9e75" if risk["risk_reward"] >= 2.0 else "#e24b4a"

    return f"""
  <div style="background:{s_bg};border-left:5px solid {s_bdr};border-radius:8px;padding:16px;margin-bottom:12px;">
    <div style="font-size:22px;margin-bottom:4px;">{strategy['icon']}
      <span style="font-size:18px;font-weight:800;color:{s_color};margin-left:8px;">{strategy['name']}</span>
    </div>
    <div style="font-size:12px;color:#555;margin-bottom:10px;">{strategy['description']}</div>

    <div style="background:rgba(255,255,255,0.7);border-radius:6px;padding:10px;margin-bottom:8px;">
      <div style="font-size:10px;color:#aaa;text-transform:uppercase;margin-bottom:4px;">Trade Legs</div>
      <div style="font-size:13px;font-weight:700;color:{s_color};font-family:monospace;">{strategy['legs']}</div>
    </div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Expiry (DTE)</div>
        <div style="font-size:12px;font-weight:700;color:#333;">{dte_label}</div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Stop Loss</div>
        <div style="font-size:12px;font-weight:700;color:#e24b4a;">${risk['stop_price']:,} <span style="font-size:10px;color:#888;">(−${risk['stop_distance']:,.0f})</span></div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Profit Target</div>
        <div style="font-size:12px;font-weight:700;color:#1d9e75;">${risk['target_price']:,} <span style="font-size:10px;color:#888;">(+${risk['target_distance']:,.0f})</span></div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Risk / Reward</div>
        <div style="font-size:12px;font-weight:700;color:{rr_color};">1 : {risk['risk_reward']} {'✅' if risk['risk_reward'] >= 2.0 else '⚠️ Below 1:2'}</div>
      </div>
    </div>

    <div style="margin-top:10px;padding:8px 12px;background:rgba(255,255,255,0.5);border-radius:6px;font-size:11px;color:#666;">
      <strong>Why this strategy:</strong> {strategy['why']}<br>
      <strong>Risk:</strong> {strategy['risk']}
    </div>

    <div style="margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.5);border-radius:6px;font-size:11px;color:#854f0b;">
      💼 <strong>Position sizing:</strong> Max {risk['max_contracts']} contracts
      (risking ${risk['risk_per_trade']:,.0f} = {RISK_PCT*100:.0f}% of ${PORTFOLIO_VALUE:,} portfolio)
      · Each ATR move = ~${risk['dollar_per_atr']:.2f} per contract
    </div>
  </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# IV / VOLATILITY SECTION HTML
# ══════════════════════════════════════════════════════════════════════════════
def build_volatility_html(d):
    iv_rank  = d["iv_rank"]
    hv20     = d["hv20"] * 100
    hv30     = d["hv30"] * 100
    atr_pct  = d["atr_pct"]
    bb_width = d["bb_width"]

    iv_color = "#1d9e75" if iv_rank < 30 else "#e24b4a" if iv_rank > 70 else "#f59e0b"
    iv_label = "CHEAP — good time to BUY options" if iv_rank < 30 else \
               "EXPENSIVE — consider selling or spreads" if iv_rank > 70 else \
               "FAIR — normal pricing"

    squeeze_label = ""
    if bb_width < 2.0:
        squeeze_label = '<span style="background:#6366f1;color:white;font-size:10px;padding:2px 8px;border-radius:4px;margin-left:6px;">BB SQUEEZE ACTIVE</span>'
    elif bb_width < 2.5:
        squeeze_label = '<span style="background:#f59e0b;color:white;font-size:10px;padding:2px 8px;border-radius:4px;margin-left:6px;">NEAR SQUEEZE</span>'

    return f"""
  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
    <tr>
      <td style="padding:8px 12px;font-size:11px;color:#999;width:35%;">HV20 (annualised)</td>
      <td style="padding:8px 12px;font-size:13px;font-weight:700;color:#333;">{hv20:.1f}%</td>
    </tr>
    <tr style="background:#f8f7f5;">
      <td style="padding:8px 12px;font-size:11px;color:#999;">HV30 (annualised)</td>
      <td style="padding:8px 12px;font-size:13px;font-weight:700;color:#333;">{hv30:.1f}%</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;font-size:11px;color:#999;">IV Rank (60-bar)</td>
      <td style="padding:8px 12px;">
        <span style="font-size:13px;font-weight:700;color:{iv_color};">{iv_rank:.0f}%</span>
        <span style="font-size:11px;color:{iv_color};margin-left:6px;">{iv_label}</span>
      </td>
    </tr>
    <tr style="background:#f8f7f5;">
      <td style="padding:8px 12px;font-size:11px;color:#999;">ATR Volatility</td>
      <td style="padding:8px 12px;font-size:13px;font-weight:700;color:#333;">{atr_pct:.2f}% per candle</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;font-size:11px;color:#999;">BB Width (squeeze)</td>
      <td style="padding:8px 12px;">
        <span style="font-size:13px;font-weight:700;color:#333;">{bb_width:.2f}%</span>
        {squeeze_label}
      </td>
    </tr>
  </table>"""


# ══════════════════════════════════════════════════════════════════════════════
# BUILD COMPLETE EMAIL
# ══════════════════════════════════════════════════════════════════════════════
def build_email(d):
    cs    = d["call_score"]
    ps    = d["put_score"]
    total = cs + ps or 1
    gap   = abs(cs - ps)
    pct   = round(max(cs, ps) / total * 100)
    strength = "Very Strong 🔥" if gap >= 8 else "Strong 💪" if gap >= 5 else "Moderate ⚡" if gap >= 3 else "Weak ⚠️"

    strategy = d["strategy"]
    side     = strategy["side"]

    if side == "call":
        verdict, verdict_emoji = "BUY CALL", "🟢"
        v_color, v_bg, v_border = "#1d9e75", "#eaf3de", "#a3d977"
    elif side == "put":
        verdict, verdict_emoji = "BUY PUT", "🔴"
        v_color, v_bg, v_border = "#a32d2d", "#fcebeb", "#f7a3a3"
    else:
        verdict, verdict_emoji = "NO CLEAR SIGNAL", "🟡"
        v_color, v_bg, v_border = "#854f0b", "#faeeda", "#f5c97e"

    if strategy["name"] == "No Trade":
        verdict, verdict_emoji = "WAIT — NO TRADE", "🚫"
        v_color, v_bg, v_border = "#555", "#f5f5f5", "#ccc"

    # SVG elements
    chart_svg = svg_candle_chart(d["chart_closes"], d["chart_highs"], d["chart_lows"], d["chart_opens"])
    bar_svg   = svg_signal_bar(cs, ps)
    rsi_gauge   = svg_gauge(d["rsi"],     0,   100, "RSI",      "#e24b4a", "#1d9e75", "#f59e0b")
    stoch_gauge = svg_gauge(d["stoch_k"], 0,   100, "Stoch",    "#e24b4a", "#1d9e75", "#f59e0b")
    adx_gauge   = svg_gauge(d["adx"],     0,    60, "ADX",      "#aaa",    "#6366f1", "#6366f1")
    mom_gauge   = svg_gauge(d["mom10"],  -10,   10, "Momentum", "#e24b4a", "#1d9e75", "#f59e0b")
    ivr_gauge   = svg_gauge(d["iv_rank"], 0,   100, "IV Rank",  "#1d9e75", "#e24b4a", "#f59e0b")

    # Strategy + Greeks + Volatility HTML blocks
    strategy_html   = build_strategy_html(strategy, d["dte_label"], d["risk"], d["price"])
    greeks_html     = build_greeks_html(d["greeks"], strategy, d["price"])
    volatility_html = build_volatility_html(d)

    # Reason breakdown by category
    cats = ["Trend","Momentum","MACD","Volatility","VWAP","Volume","S/R","Pattern"]
    grouped = {c: [] for c in cats}
    for icon, text, side_, cat in d["reasons"]:
        grouped.get(cat, grouped["Trend"]).append((icon, text, side_))

    reason_html = ""
    for cat in cats:
        items = grouped[cat]
        if not items:
            continue
        reason_html += f"""
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;
           margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #f0eeea;">{cat}</div>"""
        for icon, text, side_ in items:
            bg    = "#f0faf5" if side_=="call" else "#fdf2f2" if side_=="put" else "#f8f7f5"
            color = "#1d9e75" if side_=="call" else "#a32d2d" if side_=="put" else "#888"
            badge = "CALL" if side_=="call" else "PUT" if side_=="put" else ""
            badge_html = f'<span style="font-size:9px;background:{color};color:white;padding:1px 5px;border-radius:3px;margin-left:6px;">{badge}</span>' if badge else ""
            reason_html += f"""
      <div style="display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
           background:{bg};border-radius:6px;margin-bottom:4px;">
        <span style="font-size:14px;flex-shrink:0;">{icon}</span>
        <span style="font-size:12px;color:{color};line-height:1.4;">{text}{badge_html}</span>
      </div>"""
        reason_html += "</div>"

    # S/R table
    sr_html = ""
    if d["sr_levels"]:
        price_ = d["price"]
        close_levels = sorted(d["sr_levels"], key=lambda x: abs(x - price_))[:8]
        for lv in sorted(close_levels):
            diff = ((lv - price_) / price_) * 100
            typ  = "Resistance 🧱" if lv > price_ else "Support 🛡️"
            col  = "#a32d2d" if lv > price_ else "#1d9e75"
            sr_html += f"""
      <tr>
        <td style="padding:6px 10px;font-size:12px;font-weight:600;color:{col};">${lv:,.0f}</td>
        <td style="padding:6px 10px;font-size:12px;color:{col};">{typ}</td>
        <td style="padding:6px 10px;font-size:12px;color:#888;">{diff:+.2f}%</td>
      </tr>"""

    # Candlestick patterns
    pattern_html = ""
    if d["patterns"]:
        for icon, name, desc, side_ in d["patterns"]:
            col = "#1d9e75" if side_=="call" else "#a32d2d" if side_=="put" else "#888"
            bg  = "#f0faf5" if side_=="call" else "#fdf2f2" if side_=="put" else "#f8f7f5"
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
        pattern_html = '<div style="font-size:12px;color:#aaa;padding:8px;">No major pattern detected.</div>'

    adx_label    = "Strong Trend" if d["adx"] > 25 else "Weak/Choppy" if d["adx"] < 20 else "Developing"
    premium_note = ("🔥 Premiums EXPENSIVE — prefer spreads" if d["atr_pct"] > 3
                    else "😴 Premiums CHEAP — good time to buy" if d["atr_pct"] < 1
                    else "📊 Premiums FAIR")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0efeb;font-family:-apple-system,Arial,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#fff;border-radius:16px;
     overflow:hidden;border:1px solid #e0dfd8;box-shadow:0 4px 24px rgba(0,0,0,.08);">

  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#0f0f0f 0%,#1a1a2e 100%);padding:20px 24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <div style="color:#f59e0b;font-size:12px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;">BTC Options Signal</div>
        <div style="color:#fff;font-size:28px;font-weight:800;margin:4px 0;">₿ BTC/USDT</div>
        <div style="color:#666;font-size:11px;">{now} · KuCoin · 15M · 14 Indicators + Greeks</div>
      </div>
      <div style="text-align:right;">
        <div style="color:#fff;font-size:30px;font-weight:800;">${d['price']:,.0f}</div>
        <div style="color:#888;font-size:11px;">Live Price</div>
      </div>
    </div>
  </div>

  <!-- VERDICT BANNER -->
  <div style="background:{v_bg};border-left:5px solid {v_border};padding:20px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td>
        <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Today's Signal</div>
        <div style="font-size:32px;font-weight:900;color:{v_color};letter-spacing:-1px;">{verdict_emoji} {verdict}</div>
        <div style="font-size:14px;font-weight:600;color:{v_color};margin-top:4px;">{strength} · {pct}% Confidence</div>
      </td>
      <td width="90" style="text-align:center;vertical-align:middle;">
        <div style="font-size:48px;">{verdict_emoji}</div>
        <div style="font-size:10px;color:#888;margin-top:4px;">CALL {cs} vs PUT {ps}</div>
      </td>
    </tr></table>
    <div style="margin-top:12px;">{bar_svg}</div>
  </div>

  <!-- STRATEGY CARD -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🎯 Recommended Strategy</div>
    {strategy_html}
  </div>

  <!-- PRICE CHART -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">📈 Last 30 Candles (15M)</div>
    {chart_svg}
    <div style="display:flex;gap:16px;margin-top:8px;">
      <span style="font-size:10px;color:#1d9e75;">█ Bullish</span>
      <span style="font-size:10px;color:#e24b4a;">█ Bearish</span>
      <span style="font-size:10px;color:#f59e0b;">── Current price</span>
    </div>
  </div>

  <!-- GAUGE METERS -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">🎯 Indicator Gauges</div>
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="20%" style="text-align:center;">{rsi_gauge}</td>
      <td width="20%" style="text-align:center;">{stoch_gauge}</td>
      <td width="20%" style="text-align:center;">{adx_gauge}</td>
      <td width="20%" style="text-align:center;">{mom_gauge}</td>
      <td width="20%" style="text-align:center;">{ivr_gauge}</td>
    </tr></table>
    <div style="display:flex;gap:8px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;padding:2px 8px;border-radius:4px;">RSI &lt;35 = Oversold</span>
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;padding:2px 8px;border-radius:4px;">RSI &gt;65 = Overbought</span>
      <span style="font-size:10px;background:#eef2ff;color:#6366f1;padding:2px 8px;border-radius:4px;">ADX &gt;25 = Strong trend</span>
      <span style="font-size:10px;background:#fcebeb;color:#a32d2d;padding:2px 8px;border-radius:4px;">IV Rank &gt;70 = Expensive</span>
    </div>
  </div>

  <!-- GREEKS TABLE -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">Δ Greeks (Black-Scholes · HV20 as IV proxy)</div>
    {greeks_html}
    <div style="margin-top:8px;padding:8px 12px;background:#faeeda;border-radius:6px;font-size:11px;color:#854f0b;">
      ⚠️ Greeks use Historical Volatility (HV20 = {d['hv20']*100:.1f}%) as an IV proxy since BTC option IV requires a live options feed.
      Real IV from Deribit will give more accurate Greeks.
    </div>
  </div>

  <!-- VOLATILITY INTELLIGENCE -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📊 Volatility Intelligence</div>
    {volatility_html}
  </div>

  <!-- KEY METRICS TABLE -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📊 Key Metrics</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;width:35%;">EMA 9 / 20 / 50 / 200</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
          ${d['ema9']:,.0f} / ${d['ema20']:,.0f} / ${d['ema50']:,.0f} / ${d['ema200']:,.0f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">VWAP (24-bar)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['price']>d['vwap'] else '#a32d2d'};">
            ${d['vwap']:,.0f} — {'above ↑' if d['price']>d['vwap'] else 'below ↓'}</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">Bollinger Bands</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            Lower: ${d['bb_lower']:,.0f} | Upper: ${d['bb_upper']:,.0f} | BB%: {d['bb_pct']:.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">Volume Ratio</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#1d9e75' if d['vol_ratio']>1.5 else '#888'};">
            {d['vol_ratio']:.2f}x average {'🔥 Spike!' if d['vol_ratio']>2 else '✅ Above avg' if d['vol_ratio']>1.5 else '⚠️ Low' if d['vol_ratio']<0.6 else ''}</td>
      </tr>
      <tr style="background:#f8f7f5;">
        <td style="padding:8px 12px;font-size:11px;color:#999;">ATR / Premium</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">
            {d['atr_pct']:.2f}% · {premium_note}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;font-size:11px;color:#999;">ADX (+DI / -DI)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;
            color:{'#6366f1' if d['adx']>25 else '#888'};">
            {d['adx']:.1f} — {adx_label} (+DI: {d['adx_pos']:.0f} / -DI: {d['adx_neg']:.0f})</td>
      </tr>
    </table>
  </div>

  <!-- CANDLESTICK PATTERNS -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🕯️ Candlestick Patterns</div>
    {pattern_html}
  </div>

  <!-- FULL SIGNAL BREAKDOWN -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🔍 Full Signal Breakdown</div>
    {reason_html}
  </div>

  <!-- SUPPORT / RESISTANCE -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🗺️ Key S/R Levels</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Price</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Type</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Distance</th>
      </tr>
      {sr_html}
    </table>
  </div>

  <!-- STRIKE PRICE GUIDE -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;">🎯 Strike Price Guide (ATR-based)</div>
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="33%" style="padding:0 4px 0 0;">
        <div style="background:#fcebeb;border:1px solid #f7a3a3;border-radius:10px;padding:14px;text-align:center;">
          <div style="font-size:10px;color:#a32d2d;margin-bottom:6px;font-weight:600;">OTM PUT STRIKE</div>
          <div style="font-size:20px;font-weight:800;color:#a32d2d;">${d['otm_put']:,}</div>
          <div style="font-size:10px;color:#888;margin-top:4px;">Δ {d['greeks']['otm_put']['delta']:+.3f}</div>
          <div style="font-size:10px;color:#c07070;margin-top:2px;">Θ ${d['greeks']['otm_put']['theta']:+.1f}/day</div>
        </div>
      </td>
      <td width="34%" style="padding:0 4px;">
        <div style="background:#fff;border:2px solid #d3d1c7;border-radius:10px;padding:14px;text-align:center;">
          <div style="font-size:10px;color:#888;margin-bottom:6px;font-weight:600;">ATM STRIKE</div>
          <div style="font-size:20px;font-weight:800;color:#333;">${d['atm']:,}</div>
          <div style="font-size:10px;color:#888;margin-top:4px;">Δ {d['greeks']['atm_call']['delta']:+.3f} (call)</div>
          <div style="font-size:10px;color:#888;margin-top:2px;">Θ ${d['greeks']['atm_call']['theta']:+.1f}/day</div>
        </div>
      </td>
      <td width="33%" style="padding:0 0 0 4px;">
        <div style="background:#eaf3de;border:1px solid #a3d977;border-radius:10px;padding:14px;text-align:center;">
          <div style="font-size:10px;color:#3b6d11;margin-bottom:6px;font-weight:600;">OTM CALL STRIKE</div>
          <div style="font-size:20px;font-weight:800;color:#3b6d11;">${d['otm_call']:,}</div>
          <div style="font-size:10px;color:#6a9b3a;margin-top:4px;">Δ {d['greeks']['otm_call']['delta']:+.3f}</div>
          <div style="font-size:10px;color:#6a9b3a;margin-top:2px;">Θ ${d['greeks']['otm_call']['theta']:+.1f}/day</div>
        </div>
      </td>
    </tr></table>
    <div style="margin-top:10px;padding:8px 12px;background:#fff8e7;border-radius:6px;font-size:11px;color:#854f0b;">
      💡 Delta shows how much the option price moves per $1 BTC move.
      Theta is how much the option loses per day from time decay.
      ATM has highest gamma risk — moves fastest near expiry.
    </div>
  </div>

  <!-- FOOTER -->
  <div style="padding:16px 24px;background:#f8f7f5;">
    <div style="font-size:10px;color:#bbb;text-align:center;line-height:1.8;">
      14 indicators · EMA · RSI · Stoch RSI · MACD · BB · ADX · VWAP · Volume · ATR · Momentum · S/R · Candlesticks · Greeks (B-S) · IV Rank<br>
      <strong style="color:#e24b4a;">Not financial advice.</strong> Options can expire worthless. Manage risk. Never invest more than you can afford to lose.
    </div>
  </div>

</div>
</body></html>"""

    return html, f"{verdict_emoji} {verdict} [{strategy['name']}]"


# ══════════════════════════════════════════════════════════════════════════════
# SEND EMAIL
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Fetching BTC data from KuCoin...")
    df = fetch_ohlcv()
    print(f"Got {len(df)} candles. Price: ${df.iloc[-1]['close']:,.0f}")

    df = add_indicators(df)
    result = analyze(df)

    print(f"Call: {result['call_score']} | Put: {result['put_score']}")
    print(f"Strategy: {result['strategy']['name']}")
    print(f"DTE: {result['dte_label']}")
    print(f"HV20: {result['hv20']*100:.1f}% | IV Rank: {result['iv_rank']:.0f}%")
    print(f"ATM Call Delta: {result['greeks']['atm_call']['delta']:+.3f} | "
          f"Theta: ${result['greeks']['atm_call']['theta']:+.2f}/day")
    print(f"Risk/Reward: 1:{result['risk']['risk_reward']} | "
          f"Max contracts: {result['risk']['max_contracts']}")

    html, verdict = build_email(result)
    send_email(html, verdict, result["price"])


if __name__ == "__main__":
    main()
