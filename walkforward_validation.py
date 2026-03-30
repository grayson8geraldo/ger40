#!/usr/bin/env python3
"""Walk-forward validation, regime analysis, Monte Carlo, and random walk tests."""

import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
df = pd.concat(
    [pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in files],
    ignore_index=True,
)
df.sort_values("UTC", inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------
# INDICATORS (vectorised, computed once on full data)
# ---------------------------------------------------------------------------
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

# ATR14
tr = pd.concat(
    [
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ],
    axis=1,
).max(axis=1)
df["ATR14"] = tr.ewm(span=14, adjust=False).mean()

# RSI14
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = (-delta.clip(upper=0))
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["RSI14"] = 100 - 100 / (1 + rs)
df["RSI14"] = df["RSI14"].fillna(50)

# Average volume (rolling 50)
df["AvgVol"] = df["Volume"].rolling(50, min_periods=1).mean()

# Hour & date helpers
df["hour"] = df["UTC"].dt.hour
df["date"] = df["UTC"].dt.date
df["dow"] = df["UTC"].dt.dayofweek  # Mon=0 .. Sun=6

# ORB per day: max high / min low of hours 7,8
orb_mask = df["hour"].isin([7, 8])
orb_grp = df[orb_mask].groupby("date")
orb_high = orb_grp["High"].max().rename("ORB_High")
orb_low = orb_grp["Low"].min().rename("ORB_Low")
orb = pd.concat([orb_high, orb_low], axis=1)
orb["ORB_Range"] = orb["ORB_High"] - orb["ORB_Low"]
df = df.merge(orb, on="date", how="left")

# EMA9 cross above EMA21 flag
df["ema9_above"] = df["EMA9"] > df["EMA21"]
df["ema9_cross_up"] = df["ema9_above"] & ~df["ema9_above"].shift(1, fill_value=False)

# EMA50 lagged 5 bars (for regime)
df["EMA50_lag5"] = df["EMA50"].shift(5)

print(f"Loaded {len(df)} bars from {df['UTC'].min()} to {df['UTC'].max()}")

# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------
def _close_pos(pos, exit_price, trades):
    if pos["dir"] == "long":
        pnl = (exit_price - pos["entry"]) * pos["size"]
    else:
        pnl = (pos["entry"] - exit_price) * pos["size"]
    trades.append(dict(
        dir=pos["dir"], entry=pos["entry"], exit=exit_price,
        pnl=pnl, size=pos["size"], bar_date=pos["bar_date"],
        bar_utc=pos.get("bar_utc"), kind=pos.get("kind", ""),
    ))


def backtest(dfx, starting_capital=200.0):
    """Run TightStop backtest on a dataframe slice. Returns (final_cap, trades, max_dd)."""
    dfx = dfx.reset_index(drop=True)
    capital = starting_capital
    peak = capital
    max_dd = 0.0
    trades = []
    position = None
    daily_count = defaultdict(int)

    for i in range(len(dfx)):
        row = dfx.iloc[i]
        h = int(row["hour"])
        d = row["date"]
        dow = int(row["dow"])

        # Skip Fridays and weekends
        if dow >= 4:
            if position and h >= 20:
                _close_pos(position, row["Close"], trades)
                capital += trades[-1]["pnl"]
                position = None
            continue

        # ---- Manage open position ----
        if position is not None:
            closed = False
            if position["dir"] == "long":
                if row["Low"] <= position["sl"]:
                    _close_pos(position, position["sl"], trades)
                    capital += trades[-1]["pnl"]
                    position = None
                    closed = True
                elif row["High"] >= position["tp"]:
                    _close_pos(position, position["tp"], trades)
                    capital += trades[-1]["pnl"]
                    position = None
                    closed = True
                else:
                    new_trail = row["Close"] - 1.0 * row["ATR14"]
                    if new_trail > position["sl"]:
                        position["sl"] = new_trail
            else:  # short
                if row["High"] >= position["sl"]:
                    _close_pos(position, position["sl"], trades)
                    capital += trades[-1]["pnl"]
                    position = None
                    closed = True
                elif row["Low"] <= position["tp"]:
                    _close_pos(position, position["tp"], trades)
                    capital += trades[-1]["pnl"]
                    position = None
                    closed = True
                else:
                    new_trail = row["Close"] + 1.0 * row["ATR14"]
                    if new_trail < position["sl"]:
                        position["sl"] = new_trail

            # EOD close
            if position is not None and h >= 20:
                _close_pos(position, row["Close"], trades)
                capital += trades[-1]["pnl"]
                position = None
                closed = True

            # Update drawdown
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            continue

        # ---- Check for new entries (no position) ----
        if daily_count[d] >= 3:
            continue
        if capital <= 0:
            continue

        risk_amt = capital * 0.03
        orb_h = row.get("ORB_High", np.nan)
        orb_l = row.get("ORB_Low", np.nan)
        orb_r = row.get("ORB_Range", np.nan)
        atr = row["ATR14"]
        vol = row["Volume"]
        avgvol = row["AvgVol"]
        close = row["Close"]
        ema9 = row["EMA9"]
        ema21 = row["EMA21"]
        ema50 = row["EMA50"]
        rsi = row["RSI14"]

        vol_ok = vol > avgvol * 0.8 if avgvol > 0 else False

        entered = False

        # ORB signals (hour 9-19)
        if 9 <= h < 20 and not np.isnan(orb_h) and orb_r > 20 and vol_ok:
            if close > orb_h and ema9 > ema21 and close > ema50:
                sl_dist = 0.3 * orb_r
                tp_dist = 1.5 * orb_r
                if sl_dist > 0:
                    size = risk_amt / sl_dist
                    position = dict(
                        dir="long", entry=close, sl=close - sl_dist,
                        tp=close + tp_dist, size=size, bar_date=d,
                        bar_utc=row["UTC"], kind="ORB",
                    )
                    entered = True
            elif close < orb_l and ema9 < ema21 and close < ema50:
                sl_dist = 0.3 * orb_r
                tp_dist = 1.5 * orb_r
                if sl_dist > 0:
                    size = risk_amt / sl_dist
                    position = dict(
                        dir="short", entry=close, sl=close + sl_dist,
                        tp=close - tp_dist, size=size, bar_date=d,
                        bar_utc=row["UTC"], kind="ORB",
                    )
                    entered = True

        # Momentum Long (only if no ORB entry)
        if not entered and vol_ok and atr > 0:
            if row.get("ema9_cross_up", False) and close > ema50 and 40 < rsi < 70:
                sl_dist = 1.0 * atr
                tp_dist = 2.0 * atr
                size = risk_amt / sl_dist
                position = dict(
                    dir="long", entry=close, sl=close - sl_dist,
                    tp=close + tp_dist, size=size, bar_date=d,
                    bar_utc=row["UTC"], kind="MOM",
                )
                entered = True

        if entered:
            daily_count[d] += 1

        # Update drawdown
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Close any remaining position at last bar
    if position is not None and len(dfx) > 0:
        _close_pos(position, dfx.iloc[-1]["Close"], trades)
        capital += trades[-1]["pnl"]
        position = None

    if capital > peak:
        peak = capital
    dd = (peak - capital) / peak if peak > 0 else 0
    if dd > max_dd:
        max_dd = dd

    return capital, trades, max_dd


def summarise(cap0, cap_final, trades, max_dd):
    ret = (cap_final - cap0) / cap0 * 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return dict(ret_pct=ret, max_dd_pct=max_dd * 100, win_rate=wr,
                num_trades=len(trades), profit_factor=pf, final_cap=cap_final)


# ===========================================================================
# TEST 1: WALK-FORWARD (4 periods)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 1: WALK-FORWARD ANALYSIS (4 periods)")
print("=" * 70)

periods = [
    ("P1: Feb-Jul 2024", "2024-02-01", "2024-07-31"),
    ("P2: Aug 2024-Jan 2025", "2024-08-01", "2025-01-31"),
    ("P3: Feb-Jul 2025", "2025-02-01", "2025-07-31"),
    ("P4: Aug 2025-Mar 2026", "2025-08-01", "2026-03-31"),
]

for name, start, end in periods:
    mask = (df["UTC"] >= start) & (df["UTC"] <= end)
    sub = df[mask].copy()
    cap, trd, mdd = backtest(sub)
    s = summarise(200, cap, trd, mdd)
    print(f"\n{name} ({len(sub)} bars)")
    print(f"  Return: {s['ret_pct']:+.2f}%  |  Final: ${s['final_cap']:.2f}")
    print(f"  Max DD: {s['max_dd_pct']:.2f}%  |  Win Rate: {s['win_rate']:.1f}%")
    print(f"  Trades: {s['num_trades']}  |  Profit Factor: {s['profit_factor']:.2f}")

# ===========================================================================
# TEST 2: REGIME ANALYSIS
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 2: REGIME ANALYSIS")
print("=" * 70)

# Classify regime per bar
def classify_regime(row):
    if pd.isna(row["EMA50_lag5"]):
        return "Flat"
    if row["Close"] > row["EMA50"] and row["EMA50"] > row["EMA50_lag5"]:
        return "Bull"
    if row["Close"] < row["EMA50"] and row["EMA50"] < row["EMA50_lag5"]:
        return "Bear"
    return "Flat"

df["regime"] = df.apply(classify_regime, axis=1)

# Build regime lookup: for each date, regime at hour 7
regime_at_7 = df[df["hour"] == 7].groupby("date")["regime"].first().to_dict()

# Full backtest
full_cap, full_trades, full_mdd = backtest(df)

# Tag trades with regime
for t in full_trades:
    t["regime"] = regime_at_7.get(t["bar_date"], "Flat")

print(f"\nFull backtest: {len(full_trades)} trades, final ${full_cap:.2f}, max DD {full_mdd*100:.2f}%\n")

for reg in ["Bull", "Bear", "Flat"]:
    rt = [t for t in full_trades if t["regime"] == reg]
    if not rt:
        print(f"  {reg}: No trades")
        continue
    wins = [t for t in rt if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in rt)
    wr = len(wins) / len(rt) * 100
    gp = sum(t["pnl"] for t in wins) if wins else 0
    gl = abs(sum(t["pnl"] for t in rt if t["pnl"] <= 0))
    pf = gp / gl if gl > 0 else float("inf")
    print(f"  {reg}: {len(rt)} trades | PnL ${total_pnl:+.2f} | WR {wr:.1f}% | PF {pf:.2f}")

# ===========================================================================
# TEST 3: MONTE CARLO (500 shuffles)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 3: MONTE CARLO SIMULATION (500 shuffles)")
print("=" * 70)

# Store trade R-multiples (pnl / risk_amount) for proper MC replay.
# Each trade risks 3% of capital, so pnl_as_R = pnl / (capital * 0.03).
# We replay: each shuffled trade changes equity by R * 0.03 * equity.
equity_replay = 200.0
trade_R = []
for t in full_trades:
    risk_at_entry = equity_replay * 0.03
    r_mult = t["pnl"] / risk_at_entry if risk_at_entry > 0 else 0
    trade_R.append(r_mult)
    equity_replay += t["pnl"]
trade_R = np.array(trade_R)

rng = np.random.default_rng(42)

mc_finals = []
mc_max_dds = []
mc_below_100 = 0

for _ in range(500):
    shuffled = rng.permutation(trade_R)
    equity = 200.0
    peak_eq = 200.0
    worst_dd = 0.0
    below_100 = False
    for r in shuffled:
        equity += equity * 0.03 * r  # replicate 3% risk position sizing
        if equity < 100:
            below_100 = True
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq if peak_eq > 0 else 0
        if dd > worst_dd:
            worst_dd = dd
    mc_finals.append(equity)
    mc_max_dds.append(worst_dd)
    if below_100:
        mc_below_100 += 1

mc_finals = np.array(mc_finals)
mc_max_dds = np.array(mc_max_dds)

print(f"\n  Final equity (all paths): ${np.median(mc_finals):.2f}")
print(f"    (Note: final equity is identical across shuffles because")
print(f"     multiplicative returns are commutative. Path metrics vary.)")
print(f"  Max drawdown - median: {np.median(mc_max_dds)*100:.2f}%")
print(f"  Max drawdown - 5th pct (best):  {np.percentile(mc_max_dds, 5)*100:.2f}%")
print(f"  Max drawdown - 95th pct (worst): {np.percentile(mc_max_dds, 95)*100:.2f}%")
print(f"  Worst max drawdown (any path):   {np.max(mc_max_dds)*100:.2f}%")
print(f"  P(equity drops below $100):      {mc_below_100/500*100:.1f}%")

# ===========================================================================
# TEST 4: RANDOM WALK TEST (20 runs)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 4: RANDOM WALK TEST (20 synthetic series)")
print("=" * 70)

# Compute real hourly return volatility
real_returns = df["Close"].pct_change().dropna()
hourly_std = real_returns.std()
print(f"\n  Real data hourly return std: {hourly_std:.6f}")

rw_results = []
for run in range(20):
    n = len(df)
    seed_price = df["Close"].iloc[0]
    log_returns = rng.normal(0, hourly_std, n)
    log_returns[0] = 0  # start at seed price
    syn_close = seed_price * np.exp(np.cumsum(log_returns))

    syn_df = pd.DataFrame({
        "UTC": df["UTC"].values,
        "Open": syn_close * (1 + rng.normal(0, hourly_std * 0.3, n)),
        "High": syn_close * (1 + np.abs(rng.normal(0, hourly_std * 0.5, n))),
        "Low": syn_close * (1 - np.abs(rng.normal(0, hourly_std * 0.5, n))),
        "Close": syn_close,
        "Volume": df["Volume"].values,
    })
    syn_df["UTC"] = pd.to_datetime(syn_df["UTC"])

    syn_df["EMA9"] = syn_df["Close"].ewm(span=9, adjust=False).mean()
    syn_df["EMA21"] = syn_df["Close"].ewm(span=21, adjust=False).mean()
    syn_df["EMA50"] = syn_df["Close"].ewm(span=50, adjust=False).mean()

    tr_s = pd.concat([
        syn_df["High"] - syn_df["Low"],
        (syn_df["High"] - syn_df["Close"].shift(1)).abs(),
        (syn_df["Low"] - syn_df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    syn_df["ATR14"] = tr_s.ewm(span=14, adjust=False).mean()

    d2 = syn_df["Close"].diff()
    g2 = d2.clip(lower=0)
    l2 = (-d2.clip(upper=0))
    ag = g2.ewm(span=14, adjust=False).mean()
    al = l2.ewm(span=14, adjust=False).mean()
    rs2 = ag / al.replace(0, np.nan)
    syn_df["RSI14"] = (100 - 100 / (1 + rs2)).fillna(50)

    syn_df["AvgVol"] = syn_df["Volume"].rolling(50, min_periods=1).mean()
    syn_df["hour"] = syn_df["UTC"].dt.hour
    syn_df["date"] = syn_df["UTC"].dt.date
    syn_df["dow"] = syn_df["UTC"].dt.dayofweek

    orb_m = syn_df["hour"].isin([7, 8])
    orb_g = syn_df[orb_m].groupby("date")
    oh = orb_g["High"].max().rename("ORB_High")
    ol = orb_g["Low"].min().rename("ORB_Low")
    orb2 = pd.concat([oh, ol], axis=1)
    orb2["ORB_Range"] = orb2["ORB_High"] - orb2["ORB_Low"]
    syn_df = syn_df.merge(orb2, on="date", how="left")

    syn_df["ema9_above"] = syn_df["EMA9"] > syn_df["EMA21"]
    syn_df["ema9_cross_up"] = syn_df["ema9_above"] & ~syn_df["ema9_above"].shift(1, fill_value=False)

    cap_rw, trades_rw, mdd_rw = backtest(syn_df)
    ret_rw = (cap_rw - 200) / 200 * 100
    wr_rw = sum(1 for t in trades_rw if t["pnl"] > 0) / len(trades_rw) * 100 if trades_rw else 0
    rw_results.append(dict(ret=ret_rw, trades=len(trades_rw), mdd=mdd_rw * 100, final=cap_rw, win_rate=wr_rw))

med_ret = np.median([r["ret"] for r in rw_results])
med_trades = np.median([r["trades"] for r in rw_results])
med_mdd = np.median([r["mdd"] for r in rw_results])
med_final = np.median([r["final"] for r in rw_results])
profitable_runs = sum(1 for r in rw_results if r["ret"] > 0)

print(f"  Median return across 20 random walks: {med_ret:+.2f}%")
print(f"  Median final equity: ${med_final:.2f}")
print(f"  Median trades: {med_trades:.0f}")
print(f"  Median max DD: {med_mdd:.2f}%")
print(f"  Profitable runs: {profitable_runs}/20")

real_ret = (full_cap - 200) / 200 * 100
real_trades = len(full_trades)
print(f"\n  Real strategy: return {real_ret:+.2f}%, {real_trades} trades")
print(f"  Random walks:  median return {med_ret:+.2f}%, median {med_trades:.0f} trades")
if profitable_runs >= 15:
    print("  --> WARNING: Strategy profits on random data too ({}/20 runs profitable).".format(profitable_runs))
    print("      The asymmetric R:R (5:1 for ORB) creates positive expectancy even")
    print("      without directional edge. Real edge assessment requires comparing")
    print("      win rates: real vs random.")
    # Compare win rates
    real_wr = sum(1 for t in full_trades if t["pnl"] > 0) / len(full_trades) * 100
    random_wrs = []
    for r in rw_results:
        random_wrs.append(r.get("win_rate", 0))
    if random_wrs and any(w > 0 for w in random_wrs):
        print(f"      Real win rate: {real_wr:.1f}% vs Random median WR: {np.median(random_wrs):.1f}%")
elif med_ret < 0 and real_ret > 0:
    print("  --> Strategy DOES outperform random walks. Edge likely exists.")
else:
    print("  --> Strategy does NOT clearly outperform random walks.")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
