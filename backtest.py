#!/usr/bin/env python3
"""
GER40 DayTrader Strategy Backtest
Tests the Pine Script strategy logic on historical hourly data.
Starting capital: $200, leverage typical for CFD indices.
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta

# ============================================================================
# LOAD DATA
# ============================================================================

data_dir = "/home/user/ger40/"
files = sorted(glob.glob(os.path.join(data_dir, "DEU.IDX-EUR_Hour_*.csv")))

dfs = []
for f in files:
    df = pd.read_csv(f, parse_dates=["UTC"], dayfirst=True)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.sort_values("UTC", inplace=True)
data.reset_index(drop=True, inplace=True)
data["Hour"] = data["UTC"].dt.hour
data["Date"] = data["UTC"].dt.date
data["DOW"] = data["UTC"].dt.dayofweek  # 0=Monday

print(f"Loaded {len(data)} candles from {data['UTC'].min()} to {data['UTC'].max()}")
print(f"Price range: {data['Close'].min():.0f} - {data['Close'].max():.0f}")

# ============================================================================
# INDICATORS
# ============================================================================

data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

# ATR
data["TR"] = np.maximum(
    data["High"] - data["Low"],
    np.maximum(
        abs(data["High"] - data["Close"].shift(1)),
        abs(data["Low"] - data["Close"].shift(1))
    )
)
data["ATR14"] = data["TR"].rolling(14).mean()

# RSI
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# Bollinger Bands
data["BB_Mid"] = data["Close"].rolling(20).mean()
data["BB_Std"] = data["Close"].rolling(20).std()
data["BB_Upper"] = data["BB_Mid"] + 2 * data["BB_Std"]
data["BB_Lower"] = data["BB_Mid"] - 2 * data["BB_Std"]

# Consecutive candles
data["Bullish"] = (data["Close"] > data["Open"]).astype(int)
data["Bearish"] = (data["Close"] < data["Open"]).astype(int)

consec_bull = []
consec_bear = []
cb, cbe = 0, 0
for i in range(len(data)):
    if data.iloc[i]["Bullish"]:
        cb += 1
        cbe = 0
    elif data.iloc[i]["Bearish"]:
        cbe += 1
        cb = 0
    else:
        cb = 0
        cbe = 0
    consec_bull.append(cb)
    consec_bear.append(cbe)

data["ConsecBull"] = consec_bull
data["ConsecBear"] = consec_bear

# Volume average
data["AvgVol"] = data["Volume"].rolling(20).mean()

# EMA crossover
data["EMA_Bull_Cross"] = (data["EMA9"] > data["EMA21"]) & (data["EMA9"].shift(1) <= data["EMA21"].shift(1))
data["EMA_Bear_Cross"] = (data["EMA9"] < data["EMA21"]) & (data["EMA9"].shift(1) >= data["EMA21"].shift(1))

# ============================================================================
# ORB CALCULATION
# ============================================================================

# Group by date, find ORB range (hours 7-8 UTC)
orb_data = data[data["Hour"].isin([7, 8])].groupby("Date").agg(
    ORB_High=("High", "max"),
    ORB_Low=("Low", "min")
).reset_index()
orb_data["ORB_Range"] = orb_data["ORB_High"] - orb_data["ORB_Low"]

data = data.merge(orb_data, on="Date", how="left")

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class Trade:
    def __init__(self, direction, entry_price, stop_loss, take_profit, entry_time, trade_type, size_lots):
        self.direction = direction  # 1 = long, -1 = short
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        self.trade_type = trade_type
        self.size_lots = size_lots
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.exit_reason = ""


# Strategy parameters
INITIAL_CAPITAL = 200.0
RISK_PERCENT = 3.0
MAX_DAILY_TRADES = 3
MAX_DAILY_LOSS_PCT = 5.0
ORB_TARGET_MULT = 1.5
ORB_STOP_MULT = 0.5
EURO_OPEN = 7
EURO_CLOSE = 20
ORB_END = 9
TRAIL_ATR_MULT = 1.5
POINT_VALUE = 1.0  # $1 per point per 1 lot for CFD
MIN_LOT = 0.01  # Minimum lot size

# CFD leverage for indices ~20:1, margin ~5%
LEVERAGE = 20
MARGIN_PERCENT = 5.0

capital = INITIAL_CAPITAL
equity_curve = [INITIAL_CAPITAL]
trades = []
current_trade = None
daily_trades = 0
daily_start_equity = INITIAL_CAPITAL
current_date = None
max_equity = INITIAL_CAPITAL
max_drawdown = 0
monthly_returns = {}

# Allowed days (0=Mon to 4=Fri, skip Friday)
allowed_days = {0, 1, 2, 3}  # Mon-Thu

print("\nRunning backtest...")
print("=" * 80)

for i in range(50, len(data)):  # Start after indicator warmup
    row = data.iloc[i]
    prev = data.iloc[i-1]
    hour = row["Hour"]
    date = row["Date"]
    dow = row["DOW"]

    # New day reset
    if date != current_date:
        current_date = date
        daily_trades = 0
        daily_start_equity = capital

    # Track monthly returns
    month_key = f"{date.year}-{date.month:02d}"
    if month_key not in monthly_returns:
        monthly_returns[month_key] = capital

    # Daily loss check
    daily_pnl_pct = ((capital - daily_start_equity) / daily_start_equity * 100) if daily_start_equity > 0 else 0
    daily_limit_hit = daily_pnl_pct <= -MAX_DAILY_LOSS_PCT

    # ---- MANAGE OPEN TRADE ----
    if current_trade is not None:
        # EOD close
        if hour >= EURO_CLOSE:
            current_trade.exit_price = row["Close"]
            current_trade.exit_time = row["UTC"]
            current_trade.exit_reason = "EOD"
            pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
            current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
            capital += current_trade.pnl
            trades.append(current_trade)
            current_trade = None
            equity_curve.append(capital)
            continue

        # Check stop loss (using High/Low for intrabar)
        if current_trade.direction == 1:  # Long
            if row["Low"] <= current_trade.stop_loss:
                current_trade.exit_price = current_trade.stop_loss
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "SL"
                pnl_pts = (current_trade.exit_price - current_trade.entry_price)
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                trades.append(current_trade)
                current_trade = None
                equity_curve.append(capital)
                continue
            if row["High"] >= current_trade.take_profit:
                current_trade.exit_price = current_trade.take_profit
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "TP"
                pnl_pts = (current_trade.exit_price - current_trade.entry_price)
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                trades.append(current_trade)
                current_trade = None
                equity_curve.append(capital)
                continue
            # Trailing stop
            new_trail_stop = row["High"] - row["ATR14"] * TRAIL_ATR_MULT
            if new_trail_stop > current_trade.stop_loss:
                current_trade.stop_loss = new_trail_stop

        else:  # Short
            if row["High"] >= current_trade.stop_loss:
                current_trade.exit_price = current_trade.stop_loss
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "SL"
                pnl_pts = (current_trade.entry_price - current_trade.exit_price)
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                trades.append(current_trade)
                current_trade = None
                equity_curve.append(capital)
                continue
            if row["Low"] <= current_trade.take_profit:
                current_trade.exit_price = current_trade.take_profit
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "TP"
                pnl_pts = (current_trade.entry_price - current_trade.exit_price)
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                trades.append(current_trade)
                current_trade = None
                equity_curve.append(capital)
                continue
            # Trailing stop
            new_trail_stop = row["Low"] + row["ATR14"] * TRAIL_ATR_MULT
            if new_trail_stop < current_trade.stop_loss:
                current_trade.stop_loss = new_trail_stop

    # ---- ENTRY SIGNALS ----
    if current_trade is not None:
        continue
    if daily_limit_hit or daily_trades >= MAX_DAILY_TRADES:
        continue
    if dow not in allowed_days:
        continue
    if not (EURO_OPEN <= hour < EURO_CLOSE):
        continue
    if capital <= 10:  # Blown account
        continue
    if pd.isna(row["ATR14"]) or pd.isna(row["ORB_High"]):
        continue

    # Volume filter
    high_vol = row["Volume"] > row["AvgVol"] * 0.8 if not pd.isna(row["AvgVol"]) else True

    # Trend
    bull_trend = row["EMA9"] > row["EMA21"] and row["Close"] > row["EMA50"]
    bear_trend = row["EMA9"] < row["EMA21"] and row["Close"] < row["EMA50"]

    entry_price = row["Close"]
    atr = row["ATR14"]
    orb_high = row["ORB_High"]
    orb_low = row["ORB_Low"]
    orb_range = row["ORB_Range"]

    signal = None
    stop_loss = 0
    take_profit = 0
    trade_type = ""

    # --- ORB Signal ---
    if hour >= ORB_END and hour < EURO_CLOSE and not pd.isna(orb_range) and orb_range > 20:
        if entry_price > orb_high and bull_trend and high_vol:
            stop_loss = orb_low - orb_range * ORB_STOP_MULT
            take_profit = entry_price + orb_range * ORB_TARGET_MULT
            signal = 1
            trade_type = "ORB_L"
        elif entry_price < orb_low and bear_trend and high_vol:
            stop_loss = orb_high + orb_range * ORB_STOP_MULT
            take_profit = entry_price - orb_range * ORB_TARGET_MULT
            signal = -1
            trade_type = "ORB_S"

    # --- Mean Reversion Signal ---
    if signal is None and not pd.isna(row["BB_Lower"]):
        if row["ConsecBear"] >= 5 and entry_price <= row["BB_Lower"] and EURO_OPEN <= hour < EURO_CLOSE:
            stop_loss = row["Low"] - atr * 0.5
            take_profit = row["BB_Mid"]
            signal = 1
            trade_type = "MR_L"
        elif row["ConsecBull"] >= 5 and entry_price >= row["BB_Upper"] and EURO_OPEN <= hour < EURO_CLOSE:
            stop_loss = row["High"] + atr * 0.5
            take_profit = row["BB_Mid"]
            signal = -1
            trade_type = "MR_S"

    # --- EMA Momentum Signal ---
    if signal is None:
        if row["EMA_Bull_Cross"] and entry_price > row["EMA50"] and 40 < row["RSI"] < 70 and EURO_OPEN <= hour < EURO_CLOSE and high_vol:
            stop_loss = entry_price - atr * 1.5
            take_profit = entry_price + atr * 2.5
            signal = 1
            trade_type = "MOM_L"
        elif row["EMA_Bear_Cross"] and entry_price < row["EMA50"] and 30 < row["RSI"] < 60 and EURO_OPEN <= hour < EURO_CLOSE and high_vol:
            stop_loss = entry_price + atr * 1.5
            take_profit = entry_price - atr * 2.5
            signal = -1
            trade_type = "MOM_S"

    # --- Execute Trade ---
    if signal is not None:
        # Position sizing: risk X% of capital
        risk_amount = capital * RISK_PERCENT / 100
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 5:  # Minimum stop distance
            continue

        # Size in lots (CFD): risk_amount / stop_distance
        size_lots = risk_amount / (stop_distance * POINT_VALUE)
        size_lots = max(MIN_LOT, round(size_lots, 2))

        # Check margin requirement
        margin_required = entry_price * size_lots * POINT_VALUE / LEVERAGE
        if margin_required > capital * 0.95:
            size_lots = (capital * 0.95 * LEVERAGE) / (entry_price * POINT_VALUE)
            size_lots = max(MIN_LOT, round(size_lots, 2))

        current_trade = Trade(
            direction=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=row["UTC"],
            trade_type=trade_type,
            size_lots=size_lots
        )
        daily_trades += 1

    # Track drawdown
    if capital > max_equity:
        max_equity = capital
    dd = (max_equity - capital) / max_equity * 100
    if dd > max_drawdown:
        max_drawdown = dd

# Close any remaining trade
if current_trade is not None:
    last_row = data.iloc[-1]
    current_trade.exit_price = last_row["Close"]
    current_trade.exit_time = last_row["UTC"]
    current_trade.exit_reason = "END"
    pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
    current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
    capital += current_trade.pnl
    trades.append(current_trade)

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"BACKTEST RESULTS: GER40 DayTrader Strategy")
print(f"{'='*80}")
print(f"Period: {data['UTC'].min().strftime('%Y-%m-%d')} to {data['UTC'].max().strftime('%Y-%m-%d')}")
print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
print(f"Final Capital:   ${capital:.2f}")
print(f"Total Return:    {((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):.1f}%")
print(f"Max Drawdown:    {max_drawdown:.1f}%")
print(f"Max Equity:      ${max_equity:.2f}")

if trades:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
    profit_factor = sum(t.pnl for t in wins) / sum(abs(t.pnl) for t in losses) if losses and sum(abs(t.pnl) for t in losses) > 0 else float('inf')

    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades:    {len(trades)}")
    print(f"Winning Trades:  {len(wins)}")
    print(f"Losing Trades:   {len(losses)}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Avg Win:         ${avg_win:.2f}")
    print(f"Avg Loss:        ${avg_loss:.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Avg RR Ratio:    {(avg_win/avg_loss if avg_loss > 0 else 0):.2f}")

    # By trade type
    print(f"\n--- By Trade Type ---")
    types = {}
    for t in trades:
        tt = t.trade_type
        if tt not in types:
            types[tt] = {"trades": [], "wins": 0, "pnl": 0}
        types[tt]["trades"].append(t)
        types[tt]["pnl"] += t.pnl
        if t.pnl > 0:
            types[tt]["wins"] += 1

    for tt, stats in sorted(types.items()):
        n = len(stats["trades"])
        wr = stats["wins"] / n * 100 if n > 0 else 0
        print(f"  {tt:8s}: {n:3d} trades, WR={wr:.1f}%, PnL=${stats['pnl']:.2f}")

    # By exit reason
    print(f"\n--- By Exit Reason ---")
    reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl
    for r, stats in sorted(reasons.items()):
        print(f"  {r:6s}: {stats['count']:3d} trades, PnL=${stats['pnl']:.2f}")

    # Monthly breakdown
    print(f"\n--- Monthly Performance ---")
    monthly_pnl = {}
    for t in trades:
        mk = f"{t.entry_time.year}-{t.entry_time.month:02d}"
        if mk not in monthly_pnl:
            monthly_pnl[mk] = {"pnl": 0, "trades": 0, "wins": 0}
        monthly_pnl[mk]["pnl"] += t.pnl
        monthly_pnl[mk]["trades"] += 1
        if t.pnl > 0:
            monthly_pnl[mk]["wins"] += 1

    profitable_months = 0
    for mk in sorted(monthly_pnl.keys()):
        mp = monthly_pnl[mk]
        wr = mp["wins"] / mp["trades"] * 100 if mp["trades"] > 0 else 0
        status = "✓" if mp["pnl"] > 0 else "✗"
        if mp["pnl"] > 0:
            profitable_months += 1
        print(f"  {mk}: {mp['trades']:3d} trades, WR={wr:.0f}%, PnL=${mp['pnl']:8.2f} {status}")

    print(f"\n  Profitable months: {profitable_months}/{len(monthly_pnl)} ({profitable_months/len(monthly_pnl)*100:.0f}%)")

    # Best/Worst trades
    print(f"\n--- Best/Worst Trades ---")
    trades_sorted = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print("  Top 5 Winners:")
    for t in trades_sorted[:5]:
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.trade_type:8s} PnL=${t.pnl:.2f} (entry={t.entry_price:.0f}, exit={t.exit_price:.0f})")
    print("  Top 5 Losers:")
    for t in trades_sorted[-5:]:
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.trade_type:8s} PnL=${t.pnl:.2f} (entry={t.entry_price:.0f}, exit={t.exit_price:.0f})")

print(f"\n{'='*80}")
print(f"SUMMARY: ${INITIAL_CAPITAL:.0f} → ${capital:.2f} ({((capital-INITIAL_CAPITAL)/INITIAL_CAPITAL*100):.1f}% return)")
print(f"{'='*80}")
