#!/usr/bin/env python3
"""TightStop Walk-Forward Validation: WF, Regime, Monte Carlo, Random Walk tests."""

import glob, warnings, numpy as np, pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Load data ────────────────────────────────────────────────────────────────
files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
df = pd.concat(
    [pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in files],
    ignore_index=True,
)
df.sort_values("UTC", inplace=True)
df.reset_index(drop=True, inplace=True)
df["hour"] = df["UTC"].dt.hour
df["date"] = df["UTC"].dt.date
df["dow"] = df["UTC"].dt.dayofweek  # 0=Mon

# ── Indicators ───────────────────────────────────────────────────────────────
def add_indicators(d):
    d = d.copy()
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    # ATR14
    d["TR"] = np.maximum(
        d["High"] - d["Low"],
        np.maximum(abs(d["High"] - d["Close"].shift(1)), abs(d["Low"] - d["Close"].shift(1))),
    )
    d["ATR14"] = d["TR"].ewm(span=14, adjust=False).mean()
    # RSI14
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    d["RSI14"] = 100 - 100 / (1 + rs)
    # avg volume (20-bar)
    d["avgvol"] = d["Volume"].rolling(20, min_periods=1).mean()
    # EMA9 cross above EMA21
    d["ema9_prev"] = d["EMA9"].shift(1)
    d["ema21_prev"] = d["EMA21"].shift(1)
    d["ema9_cross_up"] = (d["ema9_prev"] < d["ema21_prev"]) & (d["EMA9"] > d["EMA21"])
    # EMA50 lagged 5 bars for regime
    d["EMA50_5ago"] = d["EMA50"].shift(5)
    return d

# ── ORB per day ──────────────────────────────────────────────────────────────
def add_orb(d):
    orb = d[d["hour"].isin([7, 8])].groupby("date").agg(ORB_High=("High", "max"), ORB_Low=("Low", "min"))
    orb["ORB_range"] = orb["ORB_High"] - orb["ORB_Low"]
    d = d.merge(orb, on="date", how="left")
    return d

# ── Backtest ─────────────────────────────────────────────────────────────────
def backtest(d, capital_start=200.0):
    d = add_indicators(d)
    d = add_orb(d)
    d = d.dropna(subset=["EMA50", "ATR14", "RSI14"]).reset_index(drop=True)

    capital = capital_start
    peak = capital
    max_dd = 0.0
    trades = []
    pos = None  # dict: dir, entry, sl, tp, trail, date
    daily_count = defaultdict(int)

    for i in range(len(d)):
        r = d.iloc[i]
        h, dt, dow = int(r["hour"]), r["date"], int(r["dow"])

        # Skip Fridays (dow=4) and weekends
        if dow > 3:
            # force close any open position at end of allowed days
            if pos is not None:
                pnl_pts = (r["Close"] - pos["entry"]) * pos["dir"]
                pnl = pnl_pts / pos["entry"] * pos["risk_capital"]  # simplified pnl
                capital += pnl
                trades.append({**pos, "exit": r["Close"], "pnl": pnl, "exit_reason": "weekend"})
                pos = None
            continue

        # EOD close
        if pos is not None and h >= 20:
            pnl_pts = (r["Close"] - pos["entry"]) * pos["dir"]
            pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
            capital += pnl
            trades.append({**pos, "exit": r["Close"], "pnl": pnl, "exit_reason": "eod"})
            pos = None

        # Check SL / TP / trail on open position
        if pos is not None:
            if pos["dir"] == 1:  # long
                # update trail
                trail_price = r["High"] - pos["trail_dist"]
                pos["sl"] = max(pos["sl"], trail_price)
                if r["Low"] <= pos["sl"]:
                    exit_p = pos["sl"]
                    pnl_pts = (exit_p - pos["entry"])
                    pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
                    capital += pnl
                    trades.append({**pos, "exit": exit_p, "pnl": pnl, "exit_reason": "sl"})
                    pos = None
                elif r["High"] >= pos["tp"]:
                    exit_p = pos["tp"]
                    pnl_pts = (exit_p - pos["entry"])
                    pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
                    capital += pnl
                    trades.append({**pos, "exit": exit_p, "pnl": pnl, "exit_reason": "tp"})
                    pos = None
            else:  # short
                trail_price = r["Low"] + pos["trail_dist"]
                pos["sl"] = min(pos["sl"], trail_price)
                if r["High"] >= pos["sl"]:
                    exit_p = pos["sl"]
                    pnl_pts = (pos["entry"] - exit_p)
                    pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
                    capital += pnl
                    trades.append({**pos, "exit": exit_p, "pnl": pnl, "exit_reason": "sl"})
                    pos = None
                elif r["Low"] <= pos["tp"]:
                    exit_p = pos["tp"]
                    pnl_pts = (pos["entry"] - exit_p)
                    pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
                    capital += pnl
                    trades.append({**pos, "exit": exit_p, "pnl": pnl, "exit_reason": "tp"})
                    pos = None

        # Update drawdown
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        # No new entry if already in position or max daily trades reached
        if pos is not None or daily_count[dt] >= 3:
            continue

        risk_amt = capital * 0.03
        atr = r["ATR14"]
        close = r["Close"]
        vol_ok = r["Volume"] > r["avgvol"] * 0.8

        orb_high = r.get("ORB_High", np.nan)
        orb_low = r.get("ORB_Low", np.nan)
        orb_range = r.get("ORB_range", np.nan)

        # ORB signals (hour 9-19)
        if 9 <= h < 20 and not np.isnan(orb_range) and orb_range > 20 and vol_ok:
            if close > orb_high and r["EMA9"] > r["EMA21"] and close > r["EMA50"]:
                sl_dist = 0.3 * orb_range
                tp_dist = 1.5 * orb_range
                pos = {
                    "dir": 1, "entry": close, "type": "ORB_Long",
                    "sl": close - sl_dist, "tp": close + tp_dist,
                    "trail_dist": atr, "risk_capital": risk_amt, "date": dt,
                }
                daily_count[dt] += 1
                continue
            elif close < orb_low and r["EMA9"] < r["EMA21"] and close < r["EMA50"]:
                sl_dist = 0.3 * orb_range
                tp_dist = 1.5 * orb_range
                pos = {
                    "dir": -1, "entry": close, "type": "ORB_Short",
                    "sl": close + sl_dist, "tp": close - tp_dist,
                    "trail_dist": atr, "risk_capital": risk_amt, "date": dt,
                }
                daily_count[dt] += 1
                continue

        # Momentum long
        if vol_ok and r["ema9_cross_up"] and close > r["EMA50"] and 40 < r["RSI14"] < 70:
            sl_dist = 1.0 * atr
            tp_dist = 2.0 * atr
            pos = {
                "dir": 1, "entry": close, "type": "Mom_Long",
                "sl": close - sl_dist, "tp": close + tp_dist,
                "trail_dist": atr, "risk_capital": risk_amt, "date": dt,
            }
            daily_count[dt] += 1

    # Force close any remaining position
    if pos is not None:
        r = d.iloc[-1]
        pnl_pts = (r["Close"] - pos["entry"]) * pos["dir"]
        pnl = pnl_pts / pos["entry"] * pos["risk_capital"]
        capital += pnl
        trades.append({**pos, "exit": r["Close"], "pnl": pnl, "exit_reason": "end"})

    peak = max(peak, capital)
    dd = (peak - capital) / peak if peak > 0 else 0
    max_dd = max(max_dd, dd)
    return capital, trades, max_dd


def stats_line(trades, cap_start=200.0, cap_end=None):
    if not trades:
        return {"return%": 0, "maxDD%": 0, "winrate%": 0, "trades": 0, "PF": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    ret = ((cap_end - cap_start) / cap_start * 100) if cap_end else 0
    return {
        "return%": round(ret, 2),
        "maxDD%": 0,  # filled by caller
        "winrate%": round(len(wins) / len(pnls) * 100, 1),
        "trades": len(pnls),
        "PF": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: WALK-FORWARD (4 periods)")
print("=" * 70)

periods = [
    ("P1: Feb-Jul 2024", "2024-02-01", "2024-07-31"),
    ("P2: Aug 2024-Jan 2025", "2024-08-01", "2025-01-31"),
    ("P3: Feb-Jul 2025", "2025-02-01", "2025-07-31"),
    ("P4: Aug 2025-Mar 2026", "2025-08-01", "2026-03-31"),
]

all_trades_full = []  # for later tests
for name, start, end in periods:
    mask = (df["UTC"] >= start) & (df["UTC"] <= end)
    sub = df[mask].reset_index(drop=True)
    cap, trades, mdd = backtest(sub)
    s = stats_line(trades, 200, cap)
    s["maxDD%"] = round(mdd * 100, 2)
    print(f"  {name:30s}  ret={s['return%']:+7.2f}%  maxDD={s['maxDD%']:5.2f}%  "
          f"WR={s['winrate%']:5.1f}%  trades={s['trades']:3d}  PF={s['PF']:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: REGIME ANALYSIS")
print("=" * 70)

df_full = add_indicators(df.copy())
df_full = add_orb(df_full)

# Classify regime at 07:00 per day
regime_map = {}
h7 = df_full[df_full["hour"] == 7].copy()
for _, row in h7.iterrows():
    e50 = row["EMA50"]
    e50_5 = row.get("EMA50_5ago", np.nan)
    c = row["Close"]
    if np.isnan(e50) or np.isnan(e50_5):
        regime_map[row["date"]] = "Flat"
    elif c > e50 and e50 > e50_5:
        regime_map[row["date"]] = "Bull"
    elif c < e50 and e50 < e50_5:
        regime_map[row["date"]] = "Bear"
    else:
        regime_map[row["date"]] = "Flat"

# Full backtest
cap_full, trades_full, mdd_full = backtest(df)
all_trades_full = trades_full

# Tag trades
regime_trades = {"Bull": [], "Bear": [], "Flat": []}
for t in trades_full:
    reg = regime_map.get(t["date"], "Flat")
    t["regime"] = reg
    regime_trades[reg].append(t)

for reg in ["Bull", "Bear", "Flat"]:
    tl = regime_trades[reg]
    if not tl:
        print(f"  {reg:6s}:  no trades")
        continue
    pnls = [t["pnl"] for t in tl]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1e-9
    wr = len(wins) / len(pnls) * 100
    pf = gp / gl if gl > 0 else 999
    net = sum(pnls)
    print(f"  {reg:6s}:  trades={len(tl):3d}  WR={wr:5.1f}%  PF={pf:.2f}  "
          f"net=${net:+.2f}")

print(f"\n  Full backtest: ${200:.0f} -> ${cap_full:.2f}  "
      f"ret={((cap_full-200)/200*100):+.2f}%  maxDD={mdd_full*100:.2f}%  trades={len(trades_full)}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: MONTE CARLO (500 shuffles)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: MONTE CARLO (500 shuffles)")
print("=" * 70)

if trades_full:
    pnl_arr = np.array([t["pnl"] for t in trades_full])
    n_shuffles = 500
    finals = np.empty(n_shuffles)
    worst_dd = 0.0
    below_100 = 0

    for s in range(n_shuffles):
        shuffled = np.random.permutation(pnl_arr)
        equity = 200.0
        peak = 200.0
        mdd_s = 0.0
        hit_100 = False
        for p in shuffled:
            equity += p
            if equity < 100:
                hit_100 = True
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            mdd_s = max(mdd_s, dd)
        finals[s] = equity
        worst_dd = max(worst_dd, mdd_s)
        if hit_100:
            below_100 += 1

    print(f"  Median final equity:    ${np.median(finals):.2f}")
    print(f"  5th percentile:         ${np.percentile(finals, 5):.2f}")
    print(f"  95th percentile:        ${np.percentile(finals, 95):.2f}")
    print(f"  Worst max drawdown:     {worst_dd*100:.2f}%")
    print(f"  P(equity < $100):       {below_100/n_shuffles*100:.1f}%")
else:
    print("  No trades to simulate.")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: RANDOM WALK TEST (20 walks)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: RANDOM WALK TEST (20 walks)")
print("=" * 70)

hourly_returns = df["Close"].pct_change().dropna()
vol = hourly_returns.std()
n_bars = len(df)
n_walks = 20
rw_results = []

for w in range(n_walks):
    # Build synthetic OHLCV with same structure
    rand_rets = np.random.normal(0, vol, n_bars)
    synth_close = np.empty(n_bars)
    synth_close[0] = df["Close"].iloc[0]
    for j in range(1, n_bars):
        synth_close[j] = synth_close[j - 1] * (1 + rand_rets[j])

    synth = df[["UTC", "Volume"]].copy()
    synth["Close"] = synth_close
    # Approximate OHLC from close
    noise = np.abs(np.random.normal(0, vol * synth_close, n_bars))
    synth["High"] = synth_close + noise
    synth["Low"] = synth_close - noise
    synth["Open"] = synth_close * (1 + np.random.normal(0, vol * 0.3, n_bars))
    synth["hour"] = synth["UTC"].dt.hour
    synth["date"] = synth["UTC"].dt.date
    synth["dow"] = synth["UTC"].dt.dayofweek

    cap_rw, trades_rw, mdd_rw = backtest(synth)
    ret_rw = (cap_rw - 200) / 200 * 100
    rw_results.append({"return%": ret_rw, "trades": len(trades_rw), "maxDD%": mdd_rw * 100})

avg_ret = np.mean([r["return%"] for r in rw_results])
avg_trades = np.mean([r["trades"] for r in rw_results])
avg_dd = np.mean([r["maxDD%"] for r in rw_results])
profitable = sum(1 for r in rw_results if r["return%"] > 0)

print(f"  Avg return across {n_walks} walks:  {avg_ret:+.2f}%")
print(f"  Avg trades:                  {avg_trades:.1f}")
print(f"  Avg max DD:                  {avg_dd:.2f}%")
print(f"  Profitable walks:            {profitable}/{n_walks}")
if avg_ret > 5:
    print("  WARNING: Strategy profits on random data - possible overfitting!")
elif avg_ret < -5:
    print("  GOOD: Strategy loses on random data - edge likely real.")
else:
    print("  INCONCLUSIVE: Returns near zero on random data.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
