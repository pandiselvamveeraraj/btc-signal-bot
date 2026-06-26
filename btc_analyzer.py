"""
BTC Options Analyzer — v2 Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT'S NEW vs v1:
  ✅ Live Deribit IV feed  — real ATM implied volatility via Deribit REST API
  ✅ IV surface snapshot   — shows IV for ATM / OTM calls & puts
  ✅ Greeks now use real IV — Black-Scholes with live Deribit IV, not HV proxy
  ✅ IV vs HV comparison   — detects whether options are cheap or rich
  ✅ Funding rate fetch     — added from KuCoin futures (sentiment signal)
  ✅ New strategies added:
       • Calendar Spread    — when front IV > back IV (term structure steep)
       • Ratio Spread       — when IV is high + strong directional signal
       • Jade Lizard        — low risk sell-side on strong bullish signal
       • Broken Wing Butterfly — skewed risk/reward directional play
  ✅ Strategy gate refactor — clean priority matrix (7 strategies)
  ✅ IV percentile (IVP)    — uses 30-day rolling Deribit IV history estimate
  ✅ Put/Call skew detection — shows if market is hedging up or down
  ✅ Improved email layout   — live IV surface table added to email

Install dependencies:
  pip install requests pandas ta scipy

Environment variables required:
  GMAIL_USER  — your gmail address
  GMAIL_PASS  — gmail app password
  TO_EMAIL    — recipient email
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import requests
import pandas as pd
import ta
import smtplib
import os
import math
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from scipy.stats import norm

# ── CONFIG ────────────────────────────────────────────────────────────────────
GMAIL_USER      = os.environ.get("GMAIL_USER", "")
GMAIL_PASS      = os.environ.get("GMAIL_PASS", "")
TO_EMAIL        = os.environ.get("TO_EMAIL", "")

SYMBOL          = "BTC-USDT"
INTERVAL        = "15min"
LIMIT           = 200
RISK_FREE_RATE  = 0.05          # annualised, ~US T-bill rate
PORTFOLIO_VALUE = 10_000        # USD
RISK_PCT        = 0.02          # max 2% of portfolio per trade
CONTRACT_SIZE   = 0.001         # 1 BTC option contract = 0.001 BTC
MIN_SIGNAL_GAP  = 3             # gap ≥ 3 → directional trade (higher = more accurate intraday filter)

# Deribit base URL (no auth needed for public market data)
DERIBIT_BASE    = "https://www.deribit.com/api/v2/public"


# ══════════════════════════════════════════════════════════════════════════════
# FETCH — PRICE DATA (KuCoin)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ohlcv():
    """
    KuCoin candles endpoint caps at 1500 per call but defaults to 100 if no
    time range is given. We pass startAt so we always get ~200 bars.
    """
    url = "https://api.kucoin.com/api/v1/market/candles"
    # 15-min bars: go back LIMIT * 15 minutes from now to guarantee enough data
    end_ts   = int(time.time())
    start_ts = end_ts - LIMIT * 15 * 60 * 2   # 2× buffer for any gaps
    r = requests.get(url, params={
        "symbol":  SYMBOL,
        "type":    INTERVAL,
        "startAt": start_ts,
        "endAt":   end_ts,
    }, timeout=10)
    r.raise_for_status()
    raw = list(reversed(r.json()["data"]))[-LIMIT:]
    df = pd.DataFrame(raw, columns=["open_time","open","close","high","low","volume","turnover"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FETCH — FUNDING RATE (KuCoin Futures)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_funding_rate():
    """
    Fetch BTC perpetual funding rate from KuCoin Futures.
    Positive rate = longs paying shorts = crowded longs = bearish contrarian signal.
    Negative rate = shorts paying longs = crowded shorts = bullish contrarian signal.
    """
    try:
        r = requests.get(
            "https://api-futures.kucoin.com/api/v1/funding-rate/XBTUSDTM/current",
            timeout=8
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        rate = float(data.get("value", 0))
        return {
            "rate":           rate,
            "rate_pct":       round(rate * 100, 4),
            "annualized_pct": round(rate * 3 * 365 * 100, 1),  # 3 funding events/day
            "bias":           "longs paying (bearish lean)" if rate > 0 else "shorts paying (bullish lean)",
            "error":          None,
        }
    except Exception as e:
        return {"rate": 0, "rate_pct": 0, "annualized_pct": 0, "bias": "neutral", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# FETCH — LIVE IV FROM DERIBIT  (daily-expiry aware)
# ══════════════════════════════════════════════════════════════════════════════
def _deribit_exp_code(ts_ms: int) -> str:
    """
    Convert a Deribit expiration timestamp (ms) to the instrument name code.
    Deribit format: 27JUN26  (day without leading zero, 3-letter month, 2-digit year)
    Portable across Linux / Windows / macOS.
    """
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    day = str(dt.day)                          # no leading zero
    mon = dt.strftime("%b").upper()            # JAN, FEB …
    yr  = dt.strftime("%y")                    # 26
    return f"{day}{mon}{yr}"                   # e.g. 27JUN26


def fetch_deribit_iv(price: float):
    """
    Fetches live implied volatility from Deribit for BTC options.

    BTC has DAILY expiries on Deribit (every calendar day at 08:00 UTC),
    plus weekly (every Friday) and monthly (last Friday of month).

    Term structure we build:
      index 0 — front daily  (today or tomorrow if today already expired)
      index 1 — next daily   (+1 day)
      index 2 — nearest Friday that is ≥4 days out  (weekly anchor)
      index 3 — nearest Friday that is ≥25 days out (monthly anchor)

    This gives a meaningful daily→weekly→monthly IV curve so Calendar
    Spread logic can distinguish a daily/weekly spread from a weekly/monthly.

    Returns dict:
      iv_atm, iv_otm_call, iv_otm_put  — decimals (0.55 = 55%)
      iv_rank        — 0–100 percentile from Deribit DVOL 30-day history
      skew           — OTM put IV minus OTM call IV in %
                       positive = put-heavy = market hedging downside
      term_structure — list of dicts: {expiry, exp_ts, atm_strike, iv,
                                        iv_pct, kind, days_out}
      atm_strike, otm_call_strike, otm_put_strike
      error          — None or error string
    """
    result = {
        "iv_atm":         None,
        "iv_otm_call":    None,
        "iv_otm_put":     None,
        "iv_rank":        50.0,
        "skew":           0.0,
        "term_structure": [],
        "source":         "deribit",
        "error":          None,
    }

    try:
        # ── Step 1: Pull all live BTC option instruments ───────────────────────
        r = requests.get(
            f"{DERIBIT_BASE}/get_instruments",
            params={"currency": "BTC", "kind": "option", "expired": False},
            timeout=12,
        )
        r.raise_for_status()
        instruments = r.json().get("result", [])
        if not instruments:
            result["error"] = "No instruments returned from Deribit"
            return result

        # ── Step 2: Filter to expiries with > 1 hour remaining ────────────────
        now_ts   = int(time.time() * 1000)
        cutoff   = now_ts + 3_600_000
        valid    = [i for i in instruments if i["expiration_timestamp"] > cutoff]
        if not valid:
            result["error"] = "No valid future expiries found on Deribit"
            return result

        all_exp = sorted(set(i["expiration_timestamp"] for i in valid))

        # ── Step 3: Select a representative set of expiries ────────────────────
        # Always take the two nearest dailies
        selected = list(all_exp[:2])

        # Nearest Friday ≥ 4 days out (weekly)
        for ts in all_exp:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            days_out = (ts - now_ts) / 86_400_000
            if dt.weekday() == 4 and days_out >= 4 and ts not in selected:
                selected.append(ts)
                break

        # Nearest Friday ≥ 25 days out (monthly)
        for ts in all_exp:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            days_out = (ts - now_ts) / 86_400_000
            if dt.weekday() == 4 and days_out >= 25 and ts not in selected:
                selected.append(ts)
                break

        selected = sorted(selected)[:4]
        front_exp_ts = selected[0]

        # ── Step 4: For each selected expiry fetch ATM call IV ────────────────
        term_structure = []
        atm_iv_front   = None

        for exp_ts in selected:
            exp_instr = [i for i in valid if i["expiration_timestamp"] == exp_ts]
            strikes   = sorted(set(i["strike"] for i in exp_instr))
            if not strikes:
                continue

            atm_strike = min(strikes, key=lambda s: abs(s - price))
            exp_code   = _deribit_exp_code(exp_ts)
            dt_exp     = datetime.fromtimestamp(exp_ts / 1000, tz=timezone.utc)
            days_out   = (exp_ts - now_ts) / 86_400_000

            if   dt_exp.weekday() == 4 and days_out >= 25: kind = "monthly"
            elif dt_exp.weekday() == 4 and days_out >= 4:  kind = "weekly"
            else:                                            kind = "daily"

            try:
                tr = requests.get(
                    f"{DERIBIT_BASE}/get_order_book",
                    params={"instrument_name": f"BTC-{exp_code}-{int(atm_strike)}-C",
                            "depth": 1},
                    timeout=8,
                )
                tr.raise_for_status()
                iv_raw = tr.json().get("result", {}).get("mark_iv")
                if iv_raw and iv_raw > 0:
                    entry = {
                        "expiry":     exp_code,
                        "exp_ts":     exp_ts,
                        "atm_strike": atm_strike,
                        "iv":         iv_raw / 100.0,
                        "iv_pct":     round(iv_raw, 1),
                        "kind":       kind,
                        "days_out":   round(days_out, 1),
                    }
                    term_structure.append(entry)
                    if exp_ts == front_exp_ts and atm_iv_front is None:
                        atm_iv_front = iv_raw / 100.0
            except Exception:
                continue

        result["term_structure"] = term_structure
        if atm_iv_front is None and term_structure:
            atm_iv_front = term_structure[0]["iv"]
        if atm_iv_front is None:
            result["error"] = "Could not fetch ATM IV from Deribit"
            return result

        result["iv_atm"] = atm_iv_front

        # ── Step 5: OTM call and put IV from the front daily expiry ──────────
        # Daily expiries have a narrow strike range; take next strike above/below ATM
        front_instr    = [i for i in valid if i["expiration_timestamp"] == front_exp_ts]
        strikes_front  = sorted(set(i["strike"] for i in front_instr))
        atm_f          = min(strikes_front, key=lambda s: abs(s - price))
        front_code     = _deribit_exp_code(front_exp_ts)

        otm_c_strikes  = [s for s in strikes_front if s > atm_f]
        otm_p_strikes  = [s for s in strikes_front if s < atm_f]
        otm_c_strike   = min(otm_c_strikes) if otm_c_strikes else atm_f
        otm_p_strike   = max(otm_p_strikes) if otm_p_strikes else atm_f

        def _iv(name: str):
            try:
                rr = requests.get(f"{DERIBIT_BASE}/get_order_book",
                                  params={"instrument_name": name, "depth": 1},
                                  timeout=8)
                rr.raise_for_status()
                v = rr.json().get("result", {}).get("mark_iv")
                return v / 100.0 if v and v > 0 else None
            except Exception:
                return None

        otm_call_iv = _iv(f"BTC-{front_code}-{int(otm_c_strike)}-C")
        otm_put_iv  = _iv(f"BTC-{front_code}-{int(otm_p_strike)}-P")

        result["iv_otm_call"]     = otm_call_iv   or atm_iv_front
        result["iv_otm_put"]      = otm_put_iv    or atm_iv_front
        result["otm_call_strike"] = otm_c_strike
        result["otm_put_strike"]  = otm_p_strike
        result["atm_strike"]      = atm_f

        # ── Step 6: Skew = OTM put IV − OTM call IV (positive = fear) ────────
        if otm_put_iv and otm_call_iv:
            result["skew"] = round((otm_put_iv - otm_call_iv) * 100, 2)

        # ── Step 7: IV Rank from Deribit DVOL 30-day history ─────────────────
        try:
            dr = requests.get(
                f"{DERIBIT_BASE}/get_volatility_index_data",
                params={"currency":        "BTC",
                        "start_timestamp": int((time.time() - 90 * 86400) * 1000),  # 90-day window for reliable IV rank
                        "end_timestamp":   int(time.time() * 1000),
                        "resolution":      "1D"},
                timeout=8,
            )
            dr.raise_for_status()
            dvol_rows = dr.json().get("result", {}).get("data", [])
            if dvol_rows:
                closes = [row[4] for row in dvol_rows if row[4]]
                if closes:
                    lo, hi, cur = min(closes), max(closes), closes[-1]
                    result["iv_rank"]      = round((cur - lo) / (hi - lo + 1e-9) * 100, 1)
                    result["dvol_current"] = cur
                    result["dvol_min"]     = lo
                    result["dvol_max"]     = hi
        except Exception:
            pass  # falls back to HV-based iv_rank

    except Exception as e:
        result["error"] = str(e)

    return result

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
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
    df["atr"]      = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()

    adx_ind        = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"]      = adx_ind.adx()
    df["adx_pos"]  = adx_ind.adx_pos()
    df["adx_neg"]  = adx_ind.adx_neg()

    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    df["vwap"]      = (c * v).rolling(24).sum() / v.rolling(24).sum()
    df["mom10"]     = ta.momentum.ROCIndicator(c, 10).roc()

    # Historical Volatility (fallback if Deribit fails)
    log_ret    = (c / c.shift(1)).apply(math.log)
    df["hv20"] = log_ret.rolling(20).std() * math.sqrt(365 * 96)   # 365d × 96 bars/day (crypto never closes)
    df["hv30"] = log_ret.rolling(30).std() * math.sqrt(365 * 96)

    # HV-based IV rank (used as fallback)
    rolling_max = df["hv20"].rolling(60).max()
    rolling_min = df["hv20"].rolling(60).min()
    df["hv_iv_rank"] = (df["hv20"] - rolling_min) / (rolling_max - rolling_min + 1e-9) * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS  (now using live IV per strike)
# ══════════════════════════════════════════════════════════════════════════════
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "price": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta    = norm.cdf(d1)
        price_bs = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        theta_a  = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * norm.cdf(d2))
    else:
        delta    = norm.cdf(d1) - 1
        price_bs = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta_a  = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * norm.cdf(-d2))

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega  = S * pdf_d1 * math.sqrt(T) / 100
    theta = theta_a / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "price": round(price_bs, 2),
    }


def greeks_for_strikes(price, atm, otm_call, otm_put, deribit_iv, dte_days):
    """
    Compute Greeks using live IV per strike from Deribit.
    Falls back to HV20 (annualised) if Deribit IV is unavailable.
    T is capped at min 1 day to avoid near-zero time blowups.
    """
    T = max(dte_days, 1) / 365   # minimum 1 day
    r = RISK_FREE_RATE

    iv_atm      = deribit_iv.get("iv_atm")
    iv_otm_call = deribit_iv.get("iv_otm_call")
    iv_otm_put  = deribit_iv.get("iv_otm_put")
    fallback    = deribit_iv.get("fallback_iv", 0.80)

    # Use live IV when available, else HV fallback
    iv_atm      = iv_atm      if iv_atm      else fallback
    iv_otm_call = iv_otm_call if iv_otm_call else iv_atm
    iv_otm_put  = iv_otm_put  if iv_otm_put  else iv_atm

    return {
        "atm_call": black_scholes_greeks(price, atm,      T, r, iv_atm,      "call"),
        "otm_call": black_scholes_greeks(price, otm_call, T, r, iv_otm_call, "call"),
        "atm_put":  black_scholes_greeks(price, atm,      T, r, iv_atm,      "put"),
        "otm_put":  black_scholes_greeks(price, otm_put,  T, r, iv_otm_put,  "put"),
        "iv_used": {
            "atm":      round(iv_atm * 100, 1),
            "otm_call": round(iv_otm_call * 100, 1),
            "otm_put":  round(iv_otm_put * 100, 1),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════
def find_sr_levels(df, lookback=80, tolerance=0.005):
    highs, lows = df["high"].tail(lookback), df["low"].tail(lookback)
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
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    i = len(c) - 1
    body, rng = abs(c[i] - o[i]), h[i] - l[i]
    uw = h[i] - max(c[i], o[i])
    lw = min(c[i], o[i]) - l[i]
    pb = abs(c[i-1] - o[i-1])

    if rng > 0 and body / rng < 0.1:
        patterns.append(("🕯️", "Doji", "Indecision — reversal possible", "neutral"))
    if lw > 2 * body and uw < body * 0.5 and c[i-1] < o[i-1]:
        patterns.append(("🔨", "Hammer", "Bullish reversal signal", "call"))
    if uw > 2 * body and lw < body * 0.5 and c[i-1] > o[i-1]:
        patterns.append(("⭐", "Shooting Star", "Bearish reversal signal", "put"))
    if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1] and body > pb:
        patterns.append(("📗", "Bullish Engulfing", "Strong reversal upward", "call"))
    if c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1] and body > pb:
        patterns.append(("📕", "Bearish Engulfing", "Strong reversal downward", "put"))
    if rng > 0 and c[i] > o[i] and body / rng > 0.85:
        patterns.append(("💚", "Bullish Marubozu", "Strong buying pressure", "call"))
    if rng > 0 and c[i] < o[i] and body / rng > 0.85:
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
# IMPROVED STRATEGY SELECTION  (7 strategies with IV-aware logic)
# ══════════════════════════════════════════════════════════════════════════════
def select_strategy(call_score, put_score, adx_val, atr_pct, bb_width,
                    iv_rank, atm, otm_call, otm_put, deribit_iv, hv20):
    """
    Priority matrix (highest priority first):

    1. BB Squeeze + Low IV                    → Long Straddle
    2. BB Near-Squeeze + Moderate IV          → Long Strangle
    3. Term structure steep (front > back IV) → Calendar Spread
    4. Choppy + High IV (ADX < 20)            → Iron Condor
    5. Strong bull + High IV (>60%)           → Jade Lizard (sell OTM put + call spread)
    6. Strong directional + Very High IV      → Ratio Spread (1x2)
    7. Strong directional + High ATR          → Debit Spread (bull/bear)
    8. Strong directional + Low/Normal IV     → Broken Wing Butterfly (skewed)
    9. Moderate directional                   → Naked OTM option
   10. Weak signal                            → No Trade
    """
    gap       = abs(call_score - put_score)
    direction = "call" if call_score >= put_score else "put"

    iv_atm      = (deribit_iv.get("iv_atm") or hv20) * 100       # in %
    iv_otm_call = (deribit_iv.get("iv_otm_call") or hv20) * 100
    iv_otm_put  = (deribit_iv.get("iv_otm_put") or hv20) * 100
    skew        = deribit_iv.get("skew", 0.0)   # put IV - call IV in %

    term = deribit_iv.get("term_structure", [])
    has_term = len(term) >= 2

    # Calendar Spread is only valid when comparing DIFFERENT expiry types:
    #   daily → weekly  (sell today's elevated IV, buy the weekly)
    #   weekly → monthly (sell weekly, buy monthly)
    # Two dailies 1 day apart make a terrible calendar — the spread is tiny.
    #
    # Find the best front/back pair where kinds differ and IV spread ≥ 5%
    cal_front = cal_back = None
    for i, t in enumerate(term):
        for j, b in enumerate(term):
            if j <= i:
                continue
            kind_diff = t["kind"] != b["kind"]   # must be different types
            iv_spread = t["iv_pct"] - b["iv_pct"]
            if kind_diff and iv_spread >= 5.0 and t["iv_pct"] >= 45.0:
                if cal_front is None or iv_spread > (cal_front["iv_pct"] - cal_back["iv_pct"]):
                    cal_front, cal_back = t, b

    term_steep = cal_front is not None and gap < 5

    # ── 1. BB Squeeze + Low IV → Long Straddle ───────────────────────────────
    if bb_width < 2.0 and iv_rank < 30:
        return {
            "name":        "Long Straddle",
            "icon":        "🤸",
            "side":        "neutral",
            "description": "BB squeeze detected with CHEAP options — perfect straddle setup.",
            "legs":        f"Buy ATM Call ${atm:,} + Buy ATM Put ${atm:,}",
            "why":         f"IV Rank {iv_rank:.0f}% = options cheap. BB Width {bb_width:.1f}% = explosion imminent.",
            "risk":        "Full premium lost if BTC stays flat. Best with 3-7 DTE.",
            "iv_context":  f"ATM IV {iv_atm:.1f}% — buying volatility at low cost.",
        }

    # ── 2. BB Near-Squeeze + Moderate IV → Long Strangle ────────────────────
    if bb_width < 2.5 and iv_rank < 50 and gap < 4:
        return {
            "name":        "Long Strangle",
            "icon":        "🎯",
            "side":        "neutral",
            "description": "Near-squeeze with moderate IV — OTM strangle for cheaper entry.",
            "legs":        f"Buy OTM Call ${otm_call:,} + Buy OTM Put ${otm_put:,}",
            "why":         f"BB near-squeeze (width {bb_width:.1f}%). Signal gap {gap} too weak for direction.",
            "risk":        "Needs large BTC move. Loses if BTC stays near current price.",
            "iv_context":  f"OTM Call IV {iv_otm_call:.1f}% / OTM Put IV {iv_otm_put:.1f}%",
        }

    # ── 3. Steep Term Structure (different kinds) → Calendar Spread ──────────
    if term_steep:
        iv_spread = cal_front["iv_pct"] - cal_back["iv_pct"]
        leg_type  = "Put" if direction == "put" else "Call"
        kind_label = f"{cal_front['kind'].capitalize()} → {cal_back['kind'].capitalize()}"
        return {
            "name":        "Calendar Spread",
            "icon":        "📅",
            "side":        direction,
            "description": (f"{kind_label} calendar: sell {cal_front['expiry']} "
                            f"({cal_front['iv_pct']:.1f}% IV) — buy {cal_back['expiry']} "
                            f"({cal_back['iv_pct']:.1f}% IV). IV spread: {iv_spread:.1f}%."),
            "legs":        (f"Sell {cal_front['expiry']} ATM {leg_type} ${atm:,}  |  "
                            f"Buy {cal_back['expiry']} ATM {leg_type} ${atm:,}"),
            "why":         (f"{kind_label} IV spread {iv_spread:.1f}% (need ≥5%, kinds must differ). "
                            f"Sell expensive {cal_front['kind']} vol, own cheaper {cal_back['kind']} — "
                            f"net small debit. Max profit if BTC stays near ${atm:,} at front expiry."),
            "risk":        f"Loses on a large BTC move before {cal_front['expiry']} expires, or if IV collapses.",
            "iv_context":  "Term: " + " → ".join(
                f"{t['expiry']}({t['kind'][0].upper()}) {t['iv_pct']:.1f}%" for t in term),
        }

    # ── 4. Choppy + High IV → Iron Condor ───────────────────────────────────
    if adx_val < 20 and gap < MIN_SIGNAL_GAP and iv_rank > 50:
        wing_call = round((otm_call + (otm_call - atm)) / 100) * 100
        wing_put  = round((otm_put  - (atm - otm_put))  / 100) * 100
        return {
            "name":        "Iron Condor",
            "icon":        "🦅",
            "side":        "neutral",
            "description": "Choppy market + expensive IV — collect premium, profit from decay.",
            "legs":        (f"Sell ${otm_call:,}C / Sell ${otm_put:,}P  |  "
                            f"Buy ${wing_call:,}C / Buy ${wing_put:,}P (wings)"),
            "why":         (f"ADX {adx_val:.0f} = no trend. IV Rank {iv_rank:.0f}% = "
                            f"sell premium. Skew {skew:+.1f}% = "
                            f"{'put-heavy (bearish hedge)' if skew > 2 else 'call-heavy (bullish specul.)' if skew < -2 else 'neutral'}."),
            "risk":        "Max loss if BTC breaks out of the wings in either direction.",
            "iv_context":  f"ATM IV {iv_atm:.1f}% — selling rich premium.",
        }

    # ── 5. Strong Bull + High IV → Jade Lizard ───────────────────────────────
    # Jade Lizard: Buy call spread + Sell OTM put (no upside risk, capped downside)
    if direction == "call" and gap >= 5 and iv_rank > 55:
        call_spread_short = round((atm + (otm_call - atm) * 0.5) / 100) * 100
        return {
            "name":        "Jade Lizard",
            "icon":        "🦎",
            "side":        "call",
            "description": "Strong bull + expensive IV — call spread + sell OTM put. No upside risk.",
            "legs":        (f"Buy ATM Call ${atm:,}  |  Sell OTM Call ${call_spread_short:,}  |  "
                            f"Sell OTM Put ${otm_put:,}"),
            "why":         (f"Gap {gap} = strong bullish. IV Rank {iv_rank:.0f}% = "
                            f"selling OTM put fully funds the call spread. "
                            f"Skew {skew:+.1f}% means put premium {'is rich' if skew > 2 else 'is fair'}."),
            "risk":        f"Downside below ${otm_put:,} put. Net credit collected = zero upside risk.",
            "iv_context":  f"OTM Put IV {iv_otm_put:.1f}% — collecting rich put premium.",
        }

    # ── 6. Strong Directional + Very High IV → 1x2 Ratio Spread ─────────────
    if gap >= 7 and iv_rank > 65:
        if direction == "call":
            return {
                "name":        "1x2 Call Ratio Spread",
                "icon":        "⚖️",
                "side":        "call",
                "description": "Very strong bull + very expensive IV — buy 1 ATM call, sell 2 OTM calls.",
                "legs":        f"Buy 1x ATM Call ${atm:,}  |  Sell 2x OTM Call ${otm_call:,}",
                "why":         (f"Gap {gap} extremely bullish. IV Rank {iv_rank:.0f}% = selling 2 calls "
                                f"at {iv_otm_call:.1f}% IV covers the ATM call cost. Max profit between ATM-OTM."),
                "risk":        f"Uncapped loss if BTC rallies above ${otm_call:,} × 2 aggressively.",
                "iv_context":  f"OTM Call IV {iv_otm_call:.1f}% — high enough to fund 2 short calls.",
            }
        else:
            return {
                "name":        "1x2 Put Ratio Spread",
                "icon":        "⚖️",
                "side":        "put",
                "description": "Very strong bear + very expensive IV — buy 1 ATM put, sell 2 OTM puts.",
                "legs":        f"Buy 1x ATM Put ${atm:,}  |  Sell 2x OTM Put ${otm_put:,}",
                "why":         (f"Gap {gap} extremely bearish. IV Rank {iv_rank:.0f}% = selling 2 puts "
                                f"at {iv_otm_put:.1f}% IV covers the ATM put cost."),
                "risk":        f"Uncapped loss if BTC crashes below ${otm_put:,} aggressively.",
                "iv_context":  f"OTM Put IV {iv_otm_put:.1f}% — high enough to fund 2 short puts.",
            }

    # ── 7. Strong Directional + High ATR → Debit Spread ──────────────────────
    if gap >= 6 and atr_pct > 2.0:
        if direction == "call":
            return {
                "name":        "Bull Call Spread",
                "icon":        "📐",
                "side":        "call",
                "description": "Strong bull + high volatility cost — debit spread reduces premium paid.",
                "legs":        f"Buy ATM Call ${atm:,}  |  Sell OTM Call ${otm_call:,}",
                "why":         f"Gap {gap} bullish. High ATR {atr_pct:.1f}% inflates naked call cost.",
                "risk":        f"Capped profit at ${otm_call:,}. Won't capture full breakout.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% vs OTM Call IV {iv_otm_call:.1f}% — spread earns IV difference.",
            }
        else:
            return {
                "name":        "Bear Put Spread",
                "icon":        "📐",
                "side":        "put",
                "description": "Strong bear + high volatility cost — debit spread reduces premium paid.",
                "legs":        f"Buy ATM Put ${atm:,}  |  Sell OTM Put ${otm_put:,}",
                "why":         f"Gap {gap} bearish. High ATR {atr_pct:.1f}% inflates naked put cost.",
                "risk":        f"Capped profit at ${otm_put:,}. Won't capture full crash.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% vs OTM Put IV {iv_otm_put:.1f}%.",
            }

    # ── 8. Strong Directional + Normal IV → Broken Wing Butterfly ────────────
    if gap >= 5 and iv_rank <= 60:
        skip_strike_call = round((otm_call + (otm_call - atm) * 1.5) / 100) * 100
        skip_strike_put  = round((otm_put  - (atm - otm_put)  * 1.5) / 100) * 100
        if direction == "call":
            return {
                "name":        "Broken Wing Butterfly (Call)",
                "icon":        "🦋",
                "side":        "call",
                "description": "Directional butterfly skewed up — profit zone above ATM, small credit or zero cost.",
                "legs":        (f"Buy 1x ATM Call ${atm:,}  |  Sell 2x OTM Call ${otm_call:,}  |  "
                                f"Buy 1x Skip-Strike ${skip_strike_call:,}"),
                "why":         (f"Gap {gap} bullish. Normal IV (rank {iv_rank:.0f}%) suits a defined-risk "
                                f"structure. BWB gives upside exposure with reduced premium."),
                "risk":        f"Max loss between ATM and lower wing if trade goes wrong.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% — moderate, suits defined-risk structure.",
            }
        else:
            return {
                "name":        "Broken Wing Butterfly (Put)",
                "icon":        "🦋",
                "side":        "put",
                "description": "Directional butterfly skewed down — profit zone below ATM.",
                "legs":        (f"Buy 1x ATM Put ${atm:,}  |  Sell 2x OTM Put ${otm_put:,}  |  "
                                f"Buy 1x Skip-Strike ${skip_strike_put:,}"),
                "why":         (f"Gap {gap} bearish. BWB gives downside exposure with lower cost than naked put."),
                "risk":        "Max loss between ATM and lower wing if BTC reverses.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% — moderate, suits defined-risk structure.",
            }

    # ── 9. Moderate Directional → Naked OTM Option ───────────────────────────
    if gap >= MIN_SIGNAL_GAP:
        if direction == "call":
            return {
                "name":        "Long Call",
                "icon":        "📊",
                "side":        "call",
                "description": f"Standard directional play. Signal gap {gap}.",
                "legs":        f"Buy OTM Call ${otm_call:,} or ATM Call ${atm:,}",
                "why":         f"Clear bullish signal. IV Rank {iv_rank:.0f}% = {'cheap — buy OTM' if iv_rank < 40 else 'fair — ATM safer'}.",
                "risk":        "Full premium at risk if price moves against you.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% / OTM Call IV {iv_otm_call:.1f}%",
            }
        else:
            return {
                "name":        "Long Put",
                "icon":        "📊",
                "side":        "put",
                "description": f"Standard directional play. Signal gap {gap}.",
                "legs":        f"Buy OTM Put ${otm_put:,} or ATM Put ${atm:,}",
                "why":         f"Clear bearish signal. IV Rank {iv_rank:.0f}% = {'cheap — buy OTM' if iv_rank < 40 else 'fair — ATM safer'}. Skew {skew:+.1f}%.",
                "risk":        "Full premium at risk if price moves against you.",
                "iv_context":  f"ATM IV {iv_atm:.1f}% / OTM Put IV {iv_otm_put:.1f}%",
            }

    # ── 10. Signal too weak → No Trade ───────────────────────────────────────
    return {
        "name":        "No Trade",
        "icon":        "🚫",
        "side":        "neutral",
        "description": f"Signal gap {gap} is below the minimum threshold of {MIN_SIGNAL_GAP}.",
        "legs":        "Wait for a stronger, cleaner signal before entering.",
        "why":         "Entering on a weak signal is gambling. Patience is a position.",
        "risk":        "N/A — do not enter.",
        "iv_context":  f"ATM IV {iv_atm:.1f}% — noted for when a signal forms.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# DTE RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
def recommend_dte(signal_gap, atr_pct, bb_width, strategy_name):
    dte_map = {
        "Long Straddle":               (5,  "3–7 DTE — give the squeeze time to resolve"),
        "Long Strangle":               (7,  "5–10 DTE — needs more time than straddle"),
        "Calendar Spread":             (14, "14–21 DTE front / 30+ DTE back — let theta work"),
        "Iron Condor":                 (10, "7–14 DTE — collect theta decay in ranging market"),
        "Jade Lizard":                 (7,  "5–10 DTE — balance between theta & delta exposure"),
        "1x2 Call Ratio Spread":       (3,  "2–5 DTE — strong signal, move quickly"),
        "1x2 Put Ratio Spread":        (3,  "2–5 DTE — strong signal, move quickly"),
        "Bull Call Spread":            (3,  "2–5 DTE — let the trend play out"),
        "Bear Put Spread":             (3,  "2–5 DTE — let the trend play out"),
        "Broken Wing Butterfly (Call)":(5,  "3–7 DTE — defined risk, needs some time"),
        "Broken Wing Butterfly (Put)": (5,  "3–7 DTE — defined risk, needs some time"),
        "Long Call":                   (2,  "1–3 DTE — directional, short duration"),
        "Long Put":                    (2,  "1–3 DTE — directional, short duration"),
        "No Trade":                    (0,  "⚠️ No trade — wait for stronger signal"),
    }
    return dte_map.get(strategy_name, (2, "2–3 DTE (default)"))


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def risk_management(price, atr, call_score, put_score, strategy, greeks_atm=None):
    """
    Stop/target are on the UNDERLYING (BTC spot) — the trigger to exit the option.

    Direction logic:
    - strategy side 'call' → stop below, target above
    - strategy side 'put'  → stop above, target below
    - strategy side 'neutral' → use score majority direction for stop/target display
      (so a PUT-biased neutral strategy like No Trade still shows correct direction)
    """
    direction      = strategy["side"]
    stop_distance  = round(atr * 1.5, 0)
    target_distance= round(atr * 3.0, 0)

    # For neutral strategies, derive direction from raw scores so the
    # stop/target still makes intuitive sense
    if direction == "neutral":
        effective_dir = "put" if put_score > call_score else "call"
    else:
        effective_dir = direction

    if effective_dir == "call":
        stop_price   = round((price - stop_distance) / 100) * 100
        target_price = round((price + target_distance) / 100) * 100
    else:  # put
        stop_price   = round((price + stop_distance) / 100) * 100
        target_price = round((price - target_distance) / 100) * 100

    risk_per_trade = PORTFOLIO_VALUE * RISK_PCT   # e.g. $200 on $10k

    # Sizing: use B-S ATM option price if available
    bs_price = greeks_atm.get("price", 0) if greeks_atm else 0
    if bs_price and bs_price > 0:
        if strategy["name"] in ("Iron Condor", "Calendar Spread",
                                "1x2 Call Ratio Spread", "1x2 Put Ratio Spread"):
            cost_per_contract = stop_distance * CONTRACT_SIZE
        else:
            cost_per_contract = bs_price * CONTRACT_SIZE
        max_contracts = max(1, int(risk_per_trade / cost_per_contract))
    else:
        cost_per_contract = max(stop_distance * CONTRACT_SIZE, 1.0)
        max_contracts = max(1, int(risk_per_trade / cost_per_contract))

    # Hard cap: never more than 50 contracts
    max_contracts = min(max_contracts, 50)

    risk_reward = round(target_distance / stop_distance, 2) if stop_distance > 0 else 0.0

    return {
        "stop_price":        stop_price,
        "target_price":      target_price,
        "stop_distance":     stop_distance,
        "target_distance":   target_distance,
        "max_contracts":     max_contracts,
        "risk_reward":       risk_reward,
        "risk_per_trade":    round(risk_per_trade, 0),
        "cost_per_contract": round(cost_per_contract, 2),
        "effective_dir":     effective_dir,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def analyze(df, deribit_iv, funding=None):
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

    # Use Deribit IV rank if available, else HV-based fallback
    iv_rank = deribit_iv.get("iv_rank") or float(last.get("hv_iv_rank", 50.0))

    call_score = put_score = 0
    reasons = []

    # ── EMA Stack ─────────────────────────────────────────────────────────────
    e9, e20, e50, e200 = last["ema9"], last["ema20"], last["ema50"], last["ema200"]
    if e9 > e20 > e50 > e200:
        call_score += 3; reasons.append(("📈","Full EMA bullish stack (9>20>50>200)","call","Trend"))
    elif e9 < e20 < e50 < e200:
        put_score += 3;  reasons.append(("📉","Full EMA bearish stack (9<20<50<200)","put","Trend"))
    elif e20 > e50 > e200:
        call_score += 2; reasons.append(("📈","EMA 20/50/200 bullish","call","Trend"))
    elif e20 < e50 < e200:
        put_score += 2;  reasons.append(("📉","EMA 20/50/200 bearish","put","Trend"))
    else:
        reasons.append(("⚠️","EMAs mixed — no clear trend","neutral","Trend"))

    if price > e200:
        call_score += 1; reasons.append(("✅",f"Above EMA200 ${e200:,.0f} — macro bull","call","Trend"))
    else:
        put_score += 1;  reasons.append(("❌",f"Below EMA200 ${e200:,.0f} — macro bear","put","Trend"))

    adx_val, adx_pos, adx_neg = last["adx"], last["adx_pos"], last["adx_neg"]
    if adx_val > 25:
        if adx_pos > adx_neg:
            call_score += 2; reasons.append(("💪",f"ADX {adx_val:.0f} strong BULLISH (+DI:{adx_pos:.0f} > -DI:{adx_neg:.0f})","call","Trend"))
        else:
            put_score += 2;  reasons.append(("💪",f"ADX {adx_val:.0f} strong BEARISH (-DI:{adx_neg:.0f} > +DI:{adx_pos:.0f})","put","Trend"))
    elif adx_val < 20:
        reasons.append(("😴",f"ADX {adx_val:.0f} — choppy, no trend","neutral","Trend"))
    else:
        reasons.append(("〰️",f"ADX {adx_val:.0f} — developing trend","neutral","Trend"))

    # ── EMA Fresh Cross (intraday momentum signal) ────────────────────────────
    if pd.notna(prev["ema9"]) and pd.notna(prev["ema20"]):
        if e9 > e20 and prev["ema9"] <= prev["ema20"]:
            call_score += 2; reasons.append(("🔀",f"EMA9 just crossed above EMA20 — fresh bullish cross","call","Trend"))
        elif e9 < e20 and prev["ema9"] >= prev["ema20"]:
            put_score += 2;  reasons.append(("🔀",f"EMA9 just crossed below EMA20 — fresh bearish cross","put","Trend"))

    # ── RSI ───────────────────────────────────────────────────────────────────
    if rsi < 20:
        call_score += 4; reasons.append(("🟢",f"RSI {rsi:.1f} — EXTREME oversold (+4) — high-probability reversal","call","Momentum"))
    elif rsi < 30:
        call_score += 3; reasons.append(("🟢",f"RSI {rsi:.1f} — strongly oversold","call","Momentum"))
    elif rsi < 40:
        call_score += 1; reasons.append(("🟢",f"RSI {rsi:.1f} — oversold zone","call","Momentum"))
    elif rsi > 80:
        put_score += 4;  reasons.append(("🔴",f"RSI {rsi:.1f} — EXTREME overbought (+4) — high-probability reversal","put","Momentum"))
    elif rsi > 70:
        put_score += 3;  reasons.append(("🔴",f"RSI {rsi:.1f} — strongly overbought","put","Momentum"))
    elif rsi > 60:
        put_score += 1;  reasons.append(("🔴",f"RSI {rsi:.1f} — overbought zone","put","Momentum"))
    else:
        reasons.append(("⚪",f"RSI {rsi:.1f} — neutral","neutral","Momentum"))

    # RSI momentum — rising from low / falling from high
    if pd.notna(prev["rsi"]):
        prev_rsi = prev["rsi"]
        if rsi > prev_rsi + 2 and rsi < 55:
            call_score += 1; reasons.append(("📈",f"RSI rising momentum ({prev_rsi:.1f}→{rsi:.1f})","call","Momentum"))
        elif rsi < prev_rsi - 2 and rsi > 45:
            put_score += 1;  reasons.append(("📉",f"RSI falling momentum ({prev_rsi:.1f}→{rsi:.1f})","put","Momentum"))

    # RSI divergence — price vs RSI over last 15 bars
    _rc = df["close"].tail(15).values
    _rr = df["rsi"].tail(15).values
    if not any(pd.isna(_rr)):
        if _rc[-1] < min(_rc[:-1]) and _rr[-1] > min(_rr[:-1]):
            call_score += 2; reasons.append(("🔄","Bullish RSI divergence — price new 15-bar low but RSI higher","call","Momentum"))
        elif _rc[-1] > max(_rc[:-1]) and _rr[-1] < max(_rr[:-1]):
            put_score += 2;  reasons.append(("🔄","Bearish RSI divergence — price new 15-bar high but RSI lower","put","Momentum"))

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

    # MACD zero-line cross — strongest MACD signal for intraday
    if pd.notna(prev["macd"]):
        if ml > 0 and prev["macd"] <= 0:
            call_score += 2; reasons.append(("📈","MACD crossed above zero — strong bullish trend confirmation","call","MACD"))
        elif ml < 0 and prev["macd"] >= 0:
            put_score += 2;  reasons.append(("📉","MACD crossed below zero — strong bearish trend confirmation","put","MACD"))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_pct   = last["bb_pct"]
    bb_width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
    if price <= last["bb_lower"]:
        call_score += 2; reasons.append(("🟢","Price at lower BB — mean reversion CALL","call","Volatility"))
    elif bb_pct < 0.2:
        call_score += 1; reasons.append(("🟢",f"Near lower BB ({bb_pct:.2f})","call","Volatility"))
    elif price >= last["bb_upper"]:
        put_score += 2;  reasons.append(("🔴","Price at upper BB — mean reversion PUT","put","Volatility"))
    elif bb_pct > 0.8:
        put_score += 1;  reasons.append(("🔴",f"Near upper BB ({bb_pct:.2f})","put","Volatility"))
    if bb_width < 2.0:
        reasons.append(("💥",f"BB Squeeze! Big move incoming (width {bb_width:.1f}%)","neutral","Volatility"))
    elif bb_width < 2.5:
        reasons.append(("🌀",f"BB Near-Squeeze (width {bb_width:.1f}%)","neutral","Volatility"))

    # ── IV / Deribit signals ──────────────────────────────────────────────────
    iv_atm_pct = (deribit_iv.get("iv_atm") or hv20) * 100
    hv20_pct   = hv20 * 100
    iv_vs_hv   = iv_atm_pct - hv20_pct

    if iv_rank < 20:
        call_score += 1; put_score += 1
        reasons.append(("💰",f"IV Rank {iv_rank:.0f}% — options VERY CHEAP (buy)","call","Volatility"))
    elif iv_rank < 35:
        reasons.append(("💰",f"IV Rank {iv_rank:.0f}% — options cheap, buying favored","neutral","Volatility"))
    elif iv_rank > 70:
        reasons.append(("🔥",f"IV Rank {iv_rank:.0f}% — options EXPENSIVE (sell/spread)","neutral","Volatility"))
    elif iv_rank > 50:
        reasons.append(("⚠️",f"IV Rank {iv_rank:.0f}% — premiums elevated, consider spreads","neutral","Volatility"))

    skew = deribit_iv.get("skew", 0.0)
    if skew > 3:
        put_score += 1; reasons.append(("📊",f"Put skew {skew:+.1f}% — market hedging DOWN (bearish sentiment)","put","Volatility"))
    elif skew < -3:
        call_score += 1; reasons.append(("📊",f"Call skew {skew:+.1f}% — market hedging UP (bullish sentiment)","call","Volatility"))
    else:
        reasons.append(("📊",f"IV skew {skew:+.1f}% — neutral, no strong bias","neutral","Volatility"))

    if iv_vs_hv > 10:
        reasons.append(("📊",f"IV {iv_atm_pct:.1f}% > HV {hv20_pct:.1f}% by {iv_vs_hv:.1f}% — options RICH vs realized","neutral","Volatility"))
    elif iv_vs_hv < -10:
        call_score += 1; put_score += 1
        reasons.append(("📊",f"IV {iv_atm_pct:.1f}% < HV {hv20_pct:.1f}% by {abs(iv_vs_hv):.1f}% — options CHEAP vs realized","call","Volatility"))

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
    if mom > 5:
        call_score += 2; reasons.append(("🚀",f"Strong upward momentum ROC:{mom:+.1f}%","call","Momentum"))
    elif mom > 3:
        call_score += 1; reasons.append(("🚀",f"Upward momentum ROC:{mom:+.1f}%","call","Momentum"))
    elif mom < -5:
        put_score += 2;  reasons.append(("💣",f"Strong downward momentum ROC:{mom:+.1f}%","put","Momentum"))
    elif mom < -3:
        put_score += 1;  reasons.append(("💣",f"Downward momentum ROC:{mom:+.1f}%","put","Momentum"))

    # ── Funding Rate (KuCoin Futures) ─────────────────────────────────────────
    if funding and not funding.get("error"):
        fr = funding["rate_pct"]
        fr_ann = funding["annualized_pct"]
        if fr > 0.05:
            put_score += 1;  reasons.append(("💸",f"Funding +{fr:.3f}%/8h ({fr_ann:.0f}%/yr) — crowded longs, bearish contrarian","put","Momentum"))
        elif fr < -0.02:
            call_score += 1; reasons.append(("💸",f"Funding {fr:.3f}%/8h ({fr_ann:.0f}%/yr) — crowded shorts, bullish contrarian","call","Momentum"))
        else:
            reasons.append(("💸",f"Funding {fr:+.3f}%/8h — neutral sentiment","neutral","Momentum"))

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
        put_score += 1;  reasons.append(("🧱",f"Near resistance ${near_resistance[0]:,.0f} — rejection zone","put","S/R"))

    # ── Confluence bonus ──────────────────────────────────────────────────────
    # Counts how many distinct signal categories agree with the dominant direction.
    # Multi-factor agreement is a much stronger signal than any single indicator.
    dominant = "call" if call_score >= put_score else "put"
    cat_agree = len(set(cat for _, _, side_, cat in reasons if side_ == dominant))
    if cat_agree >= 5:
        if dominant == "call": call_score += 2
        else:                  put_score  += 2
        reasons.append(("🎯",f"Strong confluence: {cat_agree} categories confirm {dominant.upper()} (+2)",dominant,"Trend"))
    elif cat_agree >= 4:
        if dominant == "call": call_score += 1
        else:                  put_score  += 1
        reasons.append(("🎯",f"Good confluence: {cat_agree} categories confirm {dominant.upper()} (+1)",dominant,"Trend"))

    # ── Derived values ────────────────────────────────────────────────────────
    atr      = last["atr"]
    atr_pct  = (atr / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(atr * 1.5 / 100) * 100
    otm_call = atm + atr_step
    otm_put  = atm - atr_step

    # Use Deribit strikes if available (more accurate OTM)
    if deribit_iv.get("otm_call_strike"):
        otm_call = int(deribit_iv["otm_call_strike"])
    if deribit_iv.get("otm_put_strike"):
        otm_put = int(deribit_iv["otm_put_strike"])

    # ── Strategy selection ────────────────────────────────────────────────────
    strategy = select_strategy(
        call_score, put_score, adx_val, atr_pct, bb_width,
        iv_rank, atm, otm_call, otm_put, deribit_iv, hv20
    )

    # ── DTE recommendation ────────────────────────────────────────────────────
    dte_days, dte_label = recommend_dte(abs(call_score - put_score), atr_pct, bb_width, strategy["name"])

    # ── Greeks using live Deribit IV ──────────────────────────────────────────
    # For Greeks snapshot we always use a short horizon (3 DTE) so values are
    # intuitive regardless of strategy DTE. Calendar/condor show the front-leg Greeks.
    greeks_dte = max(dte_days, 1) if dte_days > 0 else 1  # use actual strategy DTE, not always 3
    deribit_iv["fallback_iv"] = hv20  # ensure fallback set
    greeks = greeks_for_strikes(price, atm, otm_call, otm_put, deribit_iv, greeks_dte)

    # ── Risk management ───────────────────────────────────────────────────────
    risk = risk_management(price, atr, call_score, put_score, strategy,
                           greeks_atm=greeks["atm_call"])

    return {
        "call_score": call_score, "put_score": put_score, "reasons": reasons,
        "price": price, "rsi": rsi, "stoch_k": sk, "stoch_d": sd,
        "adx": adx_val, "adx_pos": adx_pos, "adx_neg": adx_neg,
        "atr_pct": atr_pct, "atr": atr,
        "macd_hist": mh, "macd": ml, "macd_signal": ms,
        "bb_pct": bb_pct, "bb_upper": last["bb_upper"], "bb_lower": last["bb_lower"],
        "bb_width": bb_width, "bb_mid": last["bb_mid"],
        "vol_ratio": vol_ratio, "vwap": last["vwap"], "mom10": mom,
        "ema9": e9, "ema20": e20, "ema50": e50, "ema200": e200,
        "hv20": hv20, "hv30": hv30, "iv_rank": iv_rank,
        "deribit_iv": deribit_iv,
        "atm": atm, "otm_call": otm_call, "otm_put": otm_put,
        "sr_levels": sr_levels, "patterns": patterns,
        "strategy": strategy, "dte_days": dte_days, "dte_label": dte_label,
        "greeks": greeks, "risk": risk,
        "chart_closes": df["close"].tail(30).tolist(),
        "chart_highs":  df["high"].tail(30).tolist(),
        "chart_lows":   df["low"].tail(30).tolist(),
        "chart_opens":  df["open"].tail(30).tolist(),
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
    if n == 0: return ""
    mn, mx = min(lows), max(highs)
    rng = mx - mn or 1
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 20
    cw = (width - pad_l - pad_r) / n
    bar_w = max(2, cw * 0.6)
    def fy(v): return pad_t + (1 - (v - mn) / rng) * (height - pad_t - pad_b)
    candles = ""
    for i in range(n):
        x = pad_l + i * cw + cw / 2
        bull = closes[i] >= opens[i]
        col = "#1d9e75" if bull else "#e24b4a"
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
# LIVE IV SURFACE TABLE HTML  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def build_iv_surface_html(d):
    deribit_iv = d["deribit_iv"]
    hv20_pct   = d["hv20"] * 100
    hv30_pct   = d["hv30"] * 100
    iv_atm     = deribit_iv.get("iv_atm")
    iv_atm_pct = iv_atm * 100 if iv_atm else None
    skew       = deribit_iv.get("skew", 0.0)
    error      = deribit_iv.get("error")
    term       = deribit_iv.get("term_structure", [])
    iv_rank    = d["iv_rank"]

    source_badge = (
        '<span style="background:#1d9e75;color:white;font-size:9px;padding:2px 8px;'
        'border-radius:4px;margin-left:6px;">LIVE DERIBIT</span>'
        if iv_atm and not error else
        '<span style="background:#f59e0b;color:white;font-size:9px;padding:2px 8px;'
        'border-radius:4px;margin-left:6px;">HV FALLBACK</span>'
    )

    if error:
        err_box = f'<div style="background:#fdf2f2;border:1px solid #f7a3a3;border-radius:6px;padding:10px;font-size:11px;color:#a32d2d;margin-bottom:10px;">⚠️ Deribit error: {error} — using HV as IV proxy</div>'
    else:
        err_box = ""

    # IV surface snapshot rows
    iv_rows = ""
    iv_used = d["greeks"].get("iv_used", {})
    rows_data = [
        ("ATM Call",  d["atm"],      iv_used.get("atm",      iv_atm_pct or hv20_pct), "#1d9e75"),
        ("OTM Call",  d["otm_call"], iv_used.get("otm_call", hv20_pct), "#3b6d11"),
        ("ATM Put",   d["atm"],      iv_used.get("atm",      iv_atm_pct or hv20_pct), "#a32d2d"),
        ("OTM Put",   d["otm_put"],  iv_used.get("otm_put",  hv20_pct), "#7b1f1f"),
    ]
    for label, strike, iv_val, color in rows_data:
        iv_val = iv_val if iv_val else hv20_pct
        vs_hv  = iv_val - hv20_pct
        vs_txt = f'+{vs_hv:.1f}% vs HV' if vs_hv >= 0 else f'{vs_hv:.1f}% vs HV'
        vs_col = "#e24b4a" if vs_hv > 5 else "#1d9e75" if vs_hv < -5 else "#888"
        iv_rows += f"""
    <tr>
      <td style="padding:7px 10px;font-size:12px;font-weight:700;color:{color};">{label}</td>
      <td style="padding:7px 10px;font-size:12px;color:#555;">${strike:,}</td>
      <td style="padding:7px 10px;font-size:13px;font-weight:800;color:{color};">{iv_val:.1f}%</td>
      <td style="padding:7px 10px;font-size:11px;color:{vs_col};">{vs_txt}</td>
    </tr>"""

    # Term structure rows
    term_rows = ""
    if term:
        for i, t in enumerate(term):
            bg = "#f8f7f5" if i % 2 else "#fff"
            kind_badge_color = {
                "daily":   "#6366f1",
                "weekly":  "#f59e0b",
                "monthly": "#1d9e75",
            }.get(t.get("kind", "daily"), "#888")
            kind_label = t.get("kind", "daily").capitalize()
            days_out   = t.get("days_out", "?")
            term_rows += f"""
      <tr style="background:{bg};">
        <td style="padding:6px 10px;font-size:12px;font-weight:700;color:#333;">{t['expiry']}</td>
        <td style="padding:6px 10px;font-size:11px;color:#888;">${t['atm_strike']:,.0f}</td>
        <td style="padding:6px 10px;font-size:13px;font-weight:800;color:#6366f1;">{t['iv_pct']:.1f}%</td>
        <td style="padding:6px 10px;">
          <span style="background:{kind_badge_color};color:white;font-size:9px;
                padding:2px 6px;border-radius:3px;">{kind_label}</span>
          <span style="font-size:10px;color:#aaa;margin-left:4px;">{days_out}d</span>
        </td>
      </tr>"""
    else:
        term_rows = '<tr><td colspan="4" style="padding:10px;color:#aaa;font-size:12px;">Term structure unavailable</td></tr>'

    skew_color = "#a32d2d" if skew > 2 else "#1d9e75" if skew < -2 else "#888"
    skew_label = "Put heavy (bearish hedge)" if skew > 2 else "Call heavy (bullish specul.)" if skew < -2 else "Neutral"

    iv_rank_color = "#1d9e75" if iv_rank < 30 else "#e24b4a" if iv_rank > 70 else "#f59e0b"
    iv_rank_label = "CHEAP — buy options" if iv_rank < 30 else "EXPENSIVE — sell/spread" if iv_rank > 70 else "FAIR"

    return f"""
  {err_box}
  <div style="margin-bottom:12px;">
    <div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">
      IV Surface Snapshot {source_badge}
    </div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <th style="padding:7px 10px;font-size:10px;color:#aaa;text-align:left;">Option</th>
        <th style="padding:7px 10px;font-size:10px;color:#aaa;text-align:left;">Strike</th>
        <th style="padding:7px 10px;font-size:10px;color:#aaa;text-align:left;">Impl. Vol</th>
        <th style="padding:7px 10px;font-size:10px;color:#aaa;text-align:left;">vs HV20</th>
      </tr>
      {iv_rows}
    </table>
  </div>

  <div style="margin-bottom:12px;">
    <div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Term Structure (ATM IV by Expiry)</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;">
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">Expiry</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">ATM Strike</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;">ATM IV</th>
        <th style="padding:6px 10px;font-size:10px;color:#aaa;text-align:left;"></th>
      </tr>
      {term_rows}
    </table>
  </div>

  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
    <tr>
      <td style="padding:7px 10px;font-size:11px;color:#999;width:40%;">HV20 / HV30</td>
      <td style="padding:7px 10px;font-size:13px;font-weight:700;color:#333;">{hv20_pct:.1f}% / {hv30_pct:.1f}%</td>
    </tr>
    <tr style="background:#f8f7f5;">
      <td style="padding:7px 10px;font-size:11px;color:#999;">Deribit DVOL IV Rank</td>
      <td style="padding:7px 10px;">
        <span style="font-size:13px;font-weight:800;color:{iv_rank_color};">{iv_rank:.0f}%</span>
        <span style="font-size:11px;color:{iv_rank_color};margin-left:6px;">{iv_rank_label}</span>
      </td>
    </tr>
    <tr>
      <td style="padding:7px 10px;font-size:11px;color:#999;">Put/Call IV Skew</td>
      <td style="padding:7px 10px;">
        <span style="font-size:13px;font-weight:800;color:{skew_color};">{skew:+.1f}%</span>
        <span style="font-size:11px;color:{skew_color};margin-left:6px;">{skew_label}</span>
      </td>
    </tr>
  </table>"""


# ══════════════════════════════════════════════════════════════════════════════
# GREEKS TABLE HTML
# ══════════════════════════════════════════════════════════════════════════════
def build_greeks_html(greeks, strategy, price, deribit_iv):
    iv_used  = greeks.get("iv_used", {})
    iv_src   = "Deribit Live" if deribit_iv.get("iv_atm") and not deribit_iv.get("error") else "HV20 Proxy"
    rows = [
        ("ATM Call",  greeks["atm_call"], "#1d9e75", "#eaf3de", iv_used.get("atm", "–")),
        ("OTM Call",  greeks["otm_call"], "#3b6d11", "#f0faf5", iv_used.get("otm_call", "–")),
        ("ATM Put",   greeks["atm_put"],  "#a32d2d", "#fcebeb", iv_used.get("atm", "–")),
        ("OTM Put",   greeks["otm_put"],  "#7b1f1f", "#fff0f0", iv_used.get("otm_put", "–")),
    ]

    def greek_bar(value, max_val, color):
        pct = min(100, abs(value) / max_val * 100)
        return f'<div style="height:4px;background:#eee;border-radius:2px;margin-top:3px;"><div style="width:{pct:.0f}%;height:4px;background:{color};border-radius:2px;"></div></div>'

    header = f"""
    <tr style="background:#f8f7f5;">
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:left;">Strike (IV via {iv_src})</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">IV Used</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">Δ Delta</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">Θ Theta/day</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">ν Vega/1%</th>
      <th style="padding:8px 10px;font-size:10px;color:#aaa;text-align:center;">B-S Price</th>
    </tr>"""

    body = ""
    for label, g, color, bg, iv_val in rows:
        body += f"""
    <tr style="background:{bg};">
      <td style="padding:7px 10px;font-size:12px;font-weight:700;color:{color};">{label}</td>
      <td style="padding:7px 10px;text-align:center;font-size:11px;color:#6366f1;font-weight:700;">{iv_val}%</td>
      <td style="padding:7px 10px;text-align:center;">
        <span style="font-size:12px;font-weight:700;color:{color};">{g['delta']:+.3f}</span>
        {greek_bar(g['delta'], 1.0, color)}
      </td>
      <td style="padding:7px 10px;text-align:center;">
        <span style="font-size:12px;color:#e24b4a;">${g['theta']:+.1f}</span>
        {greek_bar(g['theta'], 50, '#e24b4a')}
      </td>
      <td style="padding:7px 10px;text-align:center;">
        <span style="font-size:12px;color:#f59e0b;">${g['vega']:.1f}</span>
        {greek_bar(g['vega'], 200, '#f59e0b')}
      </td>
      <td style="padding:7px 10px;text-align:center;font-size:12px;font-weight:700;color:#333;">${g['price']:.0f}</td>
    </tr>"""

    return f"""
  <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
    {header}{body}
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
    iv_ctx   = strategy.get("iv_context", "")
    eff_dir  = risk.get("effective_dir", side)
    stop_label   = "Stop (spot exit)" if eff_dir == "call" else "Stop (spot exit)"
    target_label = f"Target {'↑' if eff_dir == 'call' else '↓'}"

    return f"""
  <div style="background:{s_bg};border-left:5px solid {s_bdr};border-radius:8px;padding:16px;margin-bottom:12px;">
    <div style="font-size:22px;margin-bottom:4px;">{strategy['icon']}
      <span style="font-size:18px;font-weight:800;color:{s_color};margin-left:8px;">{strategy['name']}</span>
    </div>
    <div style="font-size:12px;color:#555;margin-bottom:10px;">{strategy['description']}</div>

    <div style="background:rgba(255,255,255,0.7);border-radius:6px;padding:10px;margin-bottom:8px;">
      <div style="font-size:10px;color:#aaa;margin-bottom:4px;">Trade Legs</div>
      <div style="font-size:13px;font-weight:700;color:{s_color};font-family:monospace;">{strategy['legs']}</div>
    </div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Expiry (DTE)</div>
        <div style="font-size:12px;font-weight:700;color:#333;">{dte_label}</div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Stop Loss (spot)</div>
        <div style="font-size:12px;font-weight:700;color:#e24b4a;">${risk['stop_price']:,} {'↓' if eff_dir=='call' else '↑'}</div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Target (spot)</div>
        <div style="font-size:12px;font-weight:700;color:#1d9e75;">${risk['target_price']:,} {'↑' if eff_dir=='call' else '↓'}</div>
      </div>
      <div style="background:rgba(255,255,255,0.6);border-radius:6px;padding:8px 12px;flex:1;min-width:120px;">
        <div style="font-size:10px;color:#aaa;">Risk/Reward</div>
        <div style="font-size:12px;font-weight:700;color:{rr_color};">1:{risk['risk_reward']} {'✅' if risk['risk_reward'] >= 2 else '⚠️'}</div>
      </div>
    </div>

    <div style="padding:8px 12px;background:rgba(255,255,255,0.5);border-radius:6px;font-size:11px;color:#666;margin-bottom:6px;">
      <strong>Why:</strong> {strategy['why']}<br>
      <strong>Risk:</strong> {strategy['risk']}
    </div>

    {f'<div style="padding:8px 12px;background:rgba(99,102,241,0.08);border-radius:6px;font-size:11px;color:#4f46e5;"><strong>IV Context:</strong> {iv_ctx}</div>' if iv_ctx else ''}

    <div style="margin-top:8px;padding:8px 12px;background:rgba(255,255,255,0.5);border-radius:6px;font-size:11px;color:#854f0b;">
      💼 Max {risk['max_contracts']} contracts · Risk ${risk['risk_per_trade']:,.0f} ({RISK_PCT*100:.0f}% of ${PORTFOLIO_VALUE:,}) · ~${risk['cost_per_contract']:.2f} cost/contract
    </div>
  </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# BUILD COMPLETE EMAIL
# ══════════════════════════════════════════════════════════════════════════════
def build_email(d):
    cs, ps  = d["call_score"], d["put_score"]
    total   = cs + ps or 1
    gap     = abs(cs - ps)
    pct     = round(max(cs, ps) / total * 100)
    strength = "Very Strong 🔥" if gap >= 8 else "Strong 💪" if gap >= 5 else "Moderate ⚡" if gap >= 3 else "Weak ⚠️"
    strategy = d["strategy"]
    side     = strategy["side"]

    if strategy["name"] == "No Trade":
        # Still show the directional lean even when not trading
        lean = "BEARISH LEAN" if ps > cs else "BULLISH LEAN" if cs > ps else "NO CLEAR SIGNAL"
        verdict, verdict_emoji = f"WAIT ({lean})", "🚫"
        v_color = "#a32d2d" if ps > cs else "#1d9e75" if cs > ps else "#854f0b"
        v_bg    = "#fcebeb" if ps > cs else "#eaf3de" if cs > ps else "#faeeda"
        v_border= "#f7a3a3" if ps > cs else "#a3d977" if cs > ps else "#f5c97e"
    elif side == "call":
        verdict, verdict_emoji = "BUY CALL", "🟢"
        v_color, v_bg, v_border = "#1d9e75", "#eaf3de", "#a3d977"
    elif side == "put":
        verdict, verdict_emoji = "BUY PUT", "🔴"
        v_color, v_bg, v_border = "#a32d2d", "#fcebeb", "#f7a3a3"
    else:
        verdict, verdict_emoji = "VOLATILITY PLAY", "🟡"
        v_color, v_bg, v_border = "#854f0b", "#faeeda", "#f5c97e"

    deribit_iv = d["deribit_iv"]
    iv_source_note = "Live Deribit IV" if deribit_iv.get("iv_atm") and not deribit_iv.get("error") else "HV20 (Deribit unavailable)"

    chart_svg     = svg_candle_chart(d["chart_closes"], d["chart_highs"], d["chart_lows"], d["chart_opens"])
    bar_svg       = svg_signal_bar(cs, ps)
    rsi_gauge     = svg_gauge(d["rsi"],     0,   100, "RSI",      "#e24b4a", "#1d9e75")
    stoch_gauge   = svg_gauge(d["stoch_k"], 0,   100, "Stoch")
    adx_gauge     = svg_gauge(d["adx"],     0,    60, "ADX",      "#aaa", "#6366f1", "#6366f1")
    mom_gauge     = svg_gauge(d["mom10"],  -10,   10, "Momentum")
    ivr_gauge     = svg_gauge(d["iv_rank"], 0,   100, "IV Rank",  "#1d9e75", "#e24b4a")

    strategy_html = build_strategy_html(strategy, d["dte_label"], d["risk"], d["price"])
    greeks_html   = build_greeks_html(d["greeks"], strategy, d["price"], deribit_iv)
    iv_surf_html  = build_iv_surface_html(d)

    # Reason breakdown
    cats = ["Trend","Momentum","MACD","Volatility","VWAP","Volume","S/R","Pattern"]
    grouped = {c: [] for c in cats}
    for icon, text, side_, cat in d["reasons"]:
        grouped.get(cat, grouped["Trend"]).append((icon, text, side_))

    reason_html = ""
    for cat in cats:
        items = grouped[cat]
        if not items: continue
        reason_html += f'<div style="margin-bottom:12px;"><div style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #f0eeea;">{cat}</div>'
        for icon, text, side_ in items:
            bg    = "#f0faf5" if side_=="call" else "#fdf2f2" if side_=="put" else "#f8f7f5"
            color = "#1d9e75" if side_=="call" else "#a32d2d" if side_=="put" else "#888"
            badge = "CALL" if side_=="call" else "PUT" if side_=="put" else ""
            badge_html = f'<span style="font-size:9px;background:{color};color:white;padding:1px 5px;border-radius:3px;margin-left:6px;">{badge}</span>' if badge else ""
            reason_html += f'<div style="display:flex;align-items:flex-start;gap:8px;padding:7px 10px;background:{bg};border-radius:6px;margin-bottom:4px;"><span style="font-size:14px;flex-shrink:0;">{icon}</span><span style="font-size:12px;color:{color};line-height:1.4;">{text}{badge_html}</span></div>'
        reason_html += "</div>"

    # S/R table
    sr_html = ""
    if d["sr_levels"]:
        price_ = d["price"]
        for lv in sorted(d["sr_levels"], key=lambda x: abs(x - price_))[:8]:
            diff = ((lv - price_) / price_) * 100
            typ  = "Resistance 🧱" if lv > price_ else "Support 🛡️"
            col  = "#a32d2d" if lv > price_ else "#1d9e75"
            sr_html += f'<tr><td style="padding:6px 10px;font-size:12px;font-weight:600;color:{col};">${lv:,.0f}</td><td style="padding:6px 10px;font-size:12px;color:{col};">{typ}</td><td style="padding:6px 10px;font-size:12px;color:#888;">{diff:+.2f}%</td></tr>'

    # Patterns
    pattern_html = ""
    for icon, name, desc, side_ in (d["patterns"] or []):
        col = "#1d9e75" if side_=="call" else "#a32d2d" if side_=="put" else "#888"
        bg  = "#f0faf5" if side_=="call" else "#fdf2f2" if side_=="put" else "#f8f7f5"
        pattern_html += f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:{bg};border-radius:8px;margin-bottom:6px;"><span style="font-size:22px;">{icon}</span><div><div style="font-size:13px;font-weight:700;color:{col};">{name}</div><div style="font-size:11px;color:#888;">{desc}</div></div></div>'
    if not pattern_html:
        pattern_html = '<div style="font-size:12px;color:#aaa;padding:8px;">No major pattern detected.</div>'

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    adx_label    = "Strong Trend" if d["adx"] > 25 else "Weak/Choppy" if d["adx"] < 20 else "Developing"
    premium_note = ("🔥 Premiums EXPENSIVE" if d["atr_pct"] > 3 else "😴 Premiums CHEAP" if d["atr_pct"] < 1 else "📊 Premiums FAIR")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f0efeb;font-family:-apple-system,Arial,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#fff;border-radius:16px;overflow:hidden;border:1px solid #e0dfd8;box-shadow:0 4px 24px rgba(0,0,0,.08);">

  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#0f0f0f 0%,#1a1a2e 100%);padding:20px 24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <div style="color:#f59e0b;font-size:12px;font-weight:600;letter-spacing:.1em;">BTC Options Signal · v2</div>
        <div style="color:#fff;font-size:28px;font-weight:800;margin:4px 0;">₿ BTC/USDT</div>
        <div style="color:#666;font-size:11px;">{now} · KuCoin 15M · {iv_source_note} · 7 Strategies</div>
      </div>
      <div style="text-align:right;">
        <div style="color:#fff;font-size:30px;font-weight:800;">${d['price']:,.0f}</div>
        <div style="color:#888;font-size:11px;">Live Price</div>
      </div>
    </div>
  </div>

  <!-- VERDICT -->
  <div style="background:{v_bg};border-left:5px solid {v_border};padding:20px 24px;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td>
        <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Signal</div>
        <div style="font-size:32px;font-weight:900;color:{v_color};">{verdict_emoji} {verdict}</div>
        <div style="font-size:14px;font-weight:600;color:{v_color};margin-top:4px;">{strength} · {pct}% Confidence</div>
      </td>
      <td width="90" style="text-align:center;">
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

  <!-- LIVE IV SURFACE (NEW) -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📡 Live IV Surface — Deribit</div>
    {iv_surf_html}
  </div>

  <!-- GREEKS TABLE (with live IV) -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">Δ Greeks (Black-Scholes · {iv_source_note})</div>
    {greeks_html}
  </div>

  <!-- PRICE CHART -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">📈 Last 30 Candles (15M)</div>
    {chart_svg}
  </div>

  <!-- GAUGES -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;background:#fafaf8;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">🎯 Indicator Gauges</div>
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="20%" style="text-align:center;">{rsi_gauge}</td>
      <td width="20%" style="text-align:center;">{stoch_gauge}</td>
      <td width="20%" style="text-align:center;">{adx_gauge}</td>
      <td width="20%" style="text-align:center;">{mom_gauge}</td>
      <td width="20%" style="text-align:center;">{ivr_gauge}</td>
    </tr></table>
  </div>

  <!-- KEY METRICS -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📊 Key Metrics</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#f8f7f5;"><td style="padding:8px 12px;font-size:11px;color:#999;width:35%;">EMA 9/20/50/200</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">${d['ema9']:,.0f} / ${d['ema20']:,.0f} / ${d['ema50']:,.0f} / ${d['ema200']:,.0f}</td></tr>
      <tr><td style="padding:8px 12px;font-size:11px;color:#999;">VWAP (24-bar)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:{'#1d9e75' if d['price']>d['vwap'] else '#a32d2d'};">${d['vwap']:,.0f} — {'above ↑' if d['price']>d['vwap'] else 'below ↓'}</td></tr>
      <tr style="background:#f8f7f5;"><td style="padding:8px 12px;font-size:11px;color:#999;">Bollinger Bands</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">L: ${d['bb_lower']:,.0f} | U: ${d['bb_upper']:,.0f} | BB%: {d['bb_pct']:.2f}</td></tr>
      <tr><td style="padding:8px 12px;font-size:11px;color:#999;">Volume Ratio</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:{'#1d9e75' if d['vol_ratio']>1.5 else '#888'};">{d['vol_ratio']:.2f}x {'🔥 Spike!' if d['vol_ratio']>2 else ''}</td></tr>
      <tr style="background:#f8f7f5;"><td style="padding:8px 12px;font-size:11px;color:#999;">ATR</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#333;">{d['atr_pct']:.2f}% · {premium_note}</td></tr>
      <tr><td style="padding:8px 12px;font-size:11px;color:#999;">ADX (+DI / -DI)</td>
        <td style="padding:8px 12px;font-size:12px;font-weight:600;color:{'#6366f1' if d['adx']>25 else '#888'};">{d['adx']:.1f} — {adx_label} (+DI:{d['adx_pos']:.0f} / -DI:{d['adx_neg']:.0f})</td></tr>
    </table>
  </div>

  <!-- PATTERNS -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🕯️ Candlestick Patterns</div>
    {pattern_html}
  </div>

  <!-- FULL BREAKDOWN -->
  <div style="padding:16px 24px;border-bottom:1px solid #f0eeea;">
    <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🔍 Full Signal Breakdown</div>
    {reason_html}
  </div>

  <!-- S/R -->
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

  <!-- FOOTER -->
  <div style="padding:16px 24px;background:#f8f7f5;">
    <div style="font-size:10px;color:#bbb;text-align:center;line-height:1.8;">
      14 indicators · Greeks via {iv_source_note} · 7 Strategies (Straddle, Strangle, Calendar, Iron Condor, Jade Lizard, Ratio Spread, BWB, Spreads, Naked)<br>
      <strong style="color:#e24b4a;">Not financial advice.</strong> Options can expire worthless. Never invest more than you can afford to lose.
    </div>
  </div>

</div></body></html>"""

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
    print("=" * 60)
    print("BTC Options Analyzer v2 — Live Deribit IV + 7 Strategies")
    print("=" * 60)

    print("\n[1/4] Fetching BTC/USDT price data from KuCoin...")
    df = fetch_ohlcv()
    price = df.iloc[-1]["close"]
    print(f"      Got {len(df)} candles. Current price: ${price:,.0f}")

    print("\n[2/4] Computing technical indicators...")
    df = add_indicators(df)
    print("      Done.")

    print("\n[3/4] Fetching live IV from Deribit...")
    deribit_iv = fetch_deribit_iv(price)
    if deribit_iv.get("error"):
        print(f"      ⚠️  Deribit error: {deribit_iv['error']} — using HV as fallback")
    else:
        iv_atm = deribit_iv.get("iv_atm", 0) * 100
        iv_rank = deribit_iv.get("iv_rank", 50)
        skew = deribit_iv.get("skew", 0)
        print(f"      ATM IV: {iv_atm:.1f}% | IV Rank: {iv_rank:.0f}% | Skew: {skew:+.1f}%")
        term = deribit_iv.get("term_structure", [])
        if term:
            term_str = " | ".join(
                f"{t['expiry']}({t.get('kind','?')[0].upper()}) {t['iv_pct']:.1f}%"
                for t in term
            )
            print(f"      Term: {term_str}")

    print("\n[3.5/4] Fetching funding rate from KuCoin Futures...")
    funding = fetch_funding_rate()
    if funding.get("error"):
        print(f"      ⚠️  Funding rate error: {funding['error']} — skipping signal")
    else:
        print(f"      Funding: {funding['rate_pct']:+.4f}%/8h ({funding['annualized_pct']:+.0f}%/yr) — {funding['bias']}")

    print("\n[4/4] Analyzing signals and building report...")
    result = analyze(df, deribit_iv, funding)

    print(f"\n{'='*60}")
    print(f"  Signal:    CALL {result['call_score']} | PUT {result['put_score']}")
    print(f"  Strategy:  {result['strategy']['name']}")
    print(f"  Direction: {result['risk']['effective_dir'].upper()}")
    print(f"  DTE:       {result['dte_label']}")
    print(f"  Legs:      {result['strategy']['legs']}")
    print(f"  Stop:      ${result['risk']['stop_price']:,}  (−${result['risk']['stop_distance']:,.0f} ATR×1.5)")
    print(f"  Target:    ${result['risk']['target_price']:,}  (+${result['risk']['target_distance']:,.0f} ATR×3.0)")
    print(f"  R/R:       1:{result['risk']['risk_reward']}")
    print(f"  Max Contracts: {result['risk']['max_contracts']}  "
          f"(risk ${result['risk']['risk_per_trade']:,.0f}, "
          f"~${result['risk']['cost_per_contract']:.2f}/contract)")
    # Show the greeks for the SIGNAL direction, not always call
    eff = result['risk']['effective_dir']
    atm_g = result['greeks']['atm_put'] if eff == 'put' else result['greeks']['atm_call']
    opt_label = "ATM Put" if eff == 'put' else "ATM Call"
    print(f"  {opt_label} Greeks (3-DTE snapshot, IV={result['greeks']['iv_used']['atm']}%):")
    print(f"    Delta: {atm_g['delta']:+.3f}  "
          f"Theta: ${atm_g['theta']:+.2f}/day  "
          f"Vega: ${atm_g['vega']:.2f}/1%IV")
    print(f"    B-S Price: ~${atm_g['price']:.0f}  "
          f"(Theta×3DTE = ${abs(atm_g['theta'])*3:.0f} = "
          f"{abs(atm_g['theta'])*3/max(atm_g['price'],1)*100:.0f}% of premium — near-expiry decay is normal)")
    print(f"  IV Rank:   {result['iv_rank']:.0f}%  |  "
          f"Skew: {result['deribit_iv'].get('skew', 0):+.1f}%  |  "
          f"ATM IV: {result['greeks']['iv_used']['atm']}%")
    print(f"{'='*60}\n")

    html, verdict = build_email(result)

    # html_path = r"C:\Users\pandi\Desktop\stock.html"
    # with open(html_path, "w", encoding="utf-8") as f:
    #     f.write(html)
    # print(f"\n✅ Report saved → {html_path}")

    if GMAIL_USER and GMAIL_PASS and TO_EMAIL:
        send_email(html, verdict, result["price"])


if __name__ == "__main__":
    main()
