"""
BTC Options Analyzer — Email Edition
Runs every 1 hour via GitHub Actions and sends signal to your email.
Uses KuCoin API (works from all regions including GitHub Actions US servers).
"""

import requests
import pandas as pd
import ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


# ── CONFIG (set these in GitHub Secrets) ─────────────────────────────────────
GMAIL_USER   = os.environ["GMAIL_USER"]
GMAIL_PASS   = os.environ["GMAIL_PASS"]
TO_EMAIL     = os.environ["TO_EMAIL"]

SYMBOL       = "BTC-USDT"   # KuCoin format
INTERVAL     = "1hour"      # KuCoin interval
LIMIT        = 200


# ── FETCH DATA (KuCoin — no region restrictions) ──────────────────────────────
def fetch_ohlcv():
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"symbol": SYMBOL, "type": INTERVAL}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()["data"]        # KuCoin returns newest first
    raw = list(reversed(raw))     # flip to oldest first
    raw = raw[-LIMIT:]            # keep last 200
    # KuCoin columns: time, open, close, high, low, volume, turnover
    df = pd.DataFrame(raw, columns=["open_time","open","close","high","low","volume","turnover"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["ema20"]  = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"]  = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["atr"]    = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    return df


# ── SCORING ───────────────────────────────────────────────────────────────────
def analyze(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]
    rsi   = last["rsi"]
    mh    = last["macd_hist"]
    mh_p  = prev["macd_hist"]

    call_score = put_score = 0
    reasons = []

    if last["ema20"] > last["ema50"] > last["ema200"]:
        call_score += 2; reasons.append(("📈", "EMA stack bullish", "call"))
    elif last["ema20"] < last["ema50"] < last["ema200"]:
        put_score += 2;  reasons.append(("📉", "EMA stack bearish", "put"))
    else:
        reasons.append(("⚠️", "EMAs mixed — trend unclear", "neutral"))

    if price > last["ema200"]:
        call_score += 1; reasons.append(("✅", "Price above EMA200 (macro bullish)", "call"))
    else:
        put_score += 1;  reasons.append(("✅", "Price below EMA200 (macro bearish)", "put"))

    if rsi < 35:
        call_score += 2; reasons.append(("🟢", f"RSI oversold ({rsi:.1f}) → bounce likely", "call"))
    elif rsi > 65:
        put_score += 2;  reasons.append(("🔴", f"RSI overbought ({rsi:.1f}) → pullback likely", "put"))
    else:
        reasons.append(("⚪", f"RSI neutral ({rsi:.1f})", "neutral"))

    if mh > 0 and mh > mh_p:
        call_score += 2; reasons.append(("📈", "MACD histogram rising (bullish momentum)", "call"))
    elif mh < 0 and mh < mh_p:
        put_score += 2;  reasons.append(("📉", "MACD histogram falling (bearish momentum)", "put"))

    if price <= last["bb_lower"]:
        call_score += 2; reasons.append(("🟢", "Price at lower Bollinger Band → Call setup", "call"))
    elif price >= last["bb_upper"]:
        put_score += 2;  reasons.append(("🔴", "Price at upper Bollinger Band → Put setup", "put"))

    atr_pct  = (last["atr"] / price) * 100
    atm      = round(price / 100) * 100
    atr_step = round(last["atr"] * 1.5 / 100) * 100

    return {
        "call_score": call_score,
        "put_score": put_score,
        "reasons": reasons,
        "price": price,
        "rsi": rsi,
        "atr_pct": atr_pct,
        "atr": last["atr"],
        "atm": atm,
        "otm_call": atm + atr_step,
        "otm_put": atm - atr_step,
        "ema20": last["ema20"],
        "ema50": last["ema50"],
        "macd_hist": mh,
    }


# ── BUILD HTML EMAIL ──────────────────────────────────────────────────────────
def build_email(d):
    call_score = d["call_score"]
    put_score  = d["put_score"]
    gap = abs(call_score - put_score)
    strength = "Strong" if gap >= 4 else "Moderate" if gap >= 2 else "Weak"

    if call_score > put_score:
        verdict_text  = f"BUY CALL — {strength} Signal"
        verdict_color = "#1d9e75"
        verdict_bg    = "#eaf3de"
    elif put_score > call_score:
        verdict_text  = f"BUY PUT — {strength} Signal"
        verdict_color = "#a32d2d"
        verdict_bg    = "#fcebeb"
    else:
        verdict_text  = "No Clear Signal — Wait"
        verdict_color = "#854f0b"
        verdict_bg    = "#faeeda"

    reason_rows = ""
    for icon, text, side in d["reasons"]:
        color = "#1d9e75" if side=="call" else "#a32d2d" if side=="put" else "#888780"
        reason_rows += f"""
        <tr>
          <td style="padding:6px 8px;font-size:14px;">{icon}</td>
          <td style="padding:6px 8px;font-size:14px;color:{color};">{text}</td>
        </tr>"""

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f5f5f3;font-family:Arial,sans-serif;">
  <div style="max-width:560px;margin:32px auto;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e0dfd8;">

    <div style="background:#1a1a1a;padding:20px 24px;">
      <div style="color:#fff;font-size:20px;font-weight:600;">BTC Options Signal</div>
      <div style="color:#888;font-size:12px;margin-top:2px;">{now} &nbsp;·&nbsp; KuCoin · 1H</div>
    </div>

    <div style="background:{verdict_bg};padding:18px 24px;border-bottom:1px solid #e0dfd8;">
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">Signal</div>
      <div style="font-size:24px;font-weight:700;color:{verdict_color};">{verdict_text}</div>
      <div style="font-size:13px;color:#666;margin-top:6px;">
        BTC Price: <strong>${d['price']:,.0f}</strong> &nbsp;·&nbsp;
        Call score: {call_score} &nbsp;·&nbsp; Put score: {put_score} &nbsp;·&nbsp; Expiry: 1–3 days
      </div>
    </div>

    <div style="padding:16px 24px;display:flex;gap:12px;border-bottom:1px solid #e0dfd8;flex-wrap:wrap;">
      <div style="flex:1;min-width:100px;background:#f5f5f3;border-radius:8px;padding:10px 14px;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">RSI (14)</div>
        <div style="font-size:18px;font-weight:600;color:{'#1d9e75' if d['rsi']<35 else '#a32d2d' if d['rsi']>65 else '#1a1a1a'};">{d['rsi']:.1f}</div>
      </div>
      <div style="flex:1;min-width:100px;background:#f5f5f3;border-radius:8px;padding:10px 14px;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">MACD Hist</div>
        <div style="font-size:18px;font-weight:600;color:{'#1d9e75' if d['macd_hist']>0 else '#a32d2d'};">{d['macd_hist']:+.1f}</div>
      </div>
      <div style="flex:1;min-width:100px;background:#f5f5f3;border-radius:8px;padding:10px 14px;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">ATR %</div>
        <div style="font-size:18px;font-weight:600;color:#1a1a1a;">{d['atr_pct']:.2f}%</div>
      </div>
      <div style="flex:1;min-width:100px;background:#f5f5f3;border-radius:8px;padding:10px 14px;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">EMA 20 / 50</div>
        <div style="font-size:13px;font-weight:600;color:#1a1a1a;">${d['ema20']:,.0f} / ${d['ema50']:,.0f}</div>
      </div>
    </div>

    <div style="padding:16px 24px;border-bottom:1px solid #e0dfd8;">
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">Signal Breakdown</div>
      <table style="width:100%;border-collapse:collapse;">{reason_rows}</table>
    </div>

    <div style="padding:16px 24px;border-bottom:1px solid #e0dfd8;">
      <div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">Strike Price Hints</div>
      <div style="display:flex;gap:10px;">
        <div style="flex:1;background:#fcebeb;border-radius:8px;padding:10px;text-align:center;">
          <div style="font-size:11px;color:#a32d2d;margin-bottom:4px;">OTM Put</div>
          <div style="font-size:15px;font-weight:600;color:#a32d2d;">${d['otm_put']:,}</div>
        </div>
        <div style="flex:1;background:#f5f5f3;border:1px solid #d3d1c7;border-radius:8px;padding:10px;text-align:center;">
          <div style="font-size:11px;color:#888;margin-bottom:4px;">ATM</div>
          <div style="font-size:15px;font-weight:600;color:#1a1a1a;">${d['atm']:,}</div>
        </div>
        <div style="flex:1;background:#eaf3de;border-radius:8px;padding:10px;text-align:center;">
          <div style="font-size:11px;color:#3b6d11;margin-bottom:4px;">OTM Call</div>
          <div style="font-size:15px;font-weight:600;color:#3b6d11;">${d['otm_call']:,}</div>
        </div>
      </div>
    </div>

    <div style="padding:14px 24px;background:#f5f5f3;">
      <div style="font-size:11px;color:#aaa;text-align:center;">
        Automated analysis only — not financial advice. Options can expire worthless. Always manage risk.
      </div>
    </div>

  </div>
</body>
</html>
"""
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
    print(f"Got {len(df)} candles. Latest close: {df.iloc[-1]['close']}")
    df = add_indicators(df)
    result = analyze(df)
    html, verdict = build_email(result)
    send_email(html, verdict, result["price"])


if __name__ == "__main__":
    main()
