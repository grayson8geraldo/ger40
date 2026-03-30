#!/usr/bin/env python3
"""
GER40 DayTrader Strategy - Optimized Backtest
Removes losing Mean Reversion, tunes ORB and Momentum parameters.
"""

import pandas as pd
import numpy as np
import glob
import os
import itertools

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
data["DOW"] = data["UTC"].dt.dayofweek

# Indicators
data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

data["TR"] = np.maximum(data["High"] - data["Low"],
    np.maximum(abs(data["High"] - data["Close"].shift(1)), abs(data["Low"] - data["Close"].shift(1))))
data["ATR14"] = data["TR"].rolling(14).mean()

delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

data["AvgVol"] = data["Volume"].rolling(20).mean()
data["EMA_Bull_Cross"] = (data["EMA9"] > data["EMA21"]) & (data["EMA9"].shift(1) <= data["EMA21"].shift(1))
data["EMA_Bear_Cross"] = (data["EMA9"] < data["EMA21"]) & (data["EMA9"].shift(1) >= data["EMA21"].shift(1))

# ORB
orb_data = data[data["Hour"].isin([7, 8])].groupby("Date").agg(
    ORB_High=("High", "max"), ORB_Low=("Low", "min")).reset_index()
orb_data["ORB_Range"] = orb_data["ORB_High"] - orb_data["ORB_Low"]
data = data.merge(orb_data, on="Date", how="left")

# VWAP-like: session cumulative
data["TypicalPrice"] = (data["High"] + data["Low"] + data["Close"]) / 3
data["TPxVol"] = data["TypicalPrice"] * data["Volume"]

# Calculate session VWAP (reset at hour 7)
vwap_values = []
cum_tpvol = 0
cum_vol = 0
current_date = None
for i in range(len(data)):
    row = data.iloc[i]
    if row["Date"] != current_date or row["Hour"] == 7:
        cum_tpvol = 0
        cum_vol = 0
        current_date = row["Date"]
    cum_tpvol += row["TPxVol"]
    cum_vol += row["Volume"]
    vwap_values.append(cum_tpvol / cum_vol if cum_vol > 0 else row["Close"])
data["VWAP"] = vwap_values

print(f"Loaded {len(data)} candles")

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

class Trade:
    __slots__ = ['direction', 'entry_price', 'stop_loss', 'take_profit', 'entry_time',
                 'trade_type', 'size_lots', 'exit_price', 'exit_time', 'pnl', 'exit_reason']
    def __init__(self, direction, entry_price, stop_loss, take_profit, entry_time, trade_type, size_lots):
        self.direction = direction
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


def run_backtest(params):
    INITIAL_CAPITAL = 200.0
    RISK_PERCENT = params.get("risk_pct", 3.0)
    MAX_DAILY_TRADES = params.get("max_daily", 3)
    MAX_DAILY_LOSS_PCT = params.get("max_daily_loss", 5.0)
    ORB_TARGET_MULT = params.get("orb_target", 1.5)
    ORB_STOP_MULT = params.get("orb_stop", 0.5)
    MOM_TP_ATR = params.get("mom_tp_atr", 2.5)
    MOM_SL_ATR = params.get("mom_sl_atr", 1.5)
    TRAIL_ATR_MULT = params.get("trail_atr", 1.5)
    USE_ORB = params.get("use_orb", True)
    USE_MOM = params.get("use_mom", True)
    USE_VWAP = params.get("use_vwap", False)
    ORB_MIN_RANGE = params.get("orb_min_range", 20)
    EURO_CLOSE = params.get("euro_close", 20)
    ALLOWED_DAYS = params.get("allowed_days", {0, 1, 2, 3})
    POINT_VALUE = 1.0
    LEVERAGE = 20
    MIN_LOT = 0.01

    capital = INITIAL_CAPITAL
    trades = []
    current_trade = None
    daily_trades = 0
    daily_start_equity = INITIAL_CAPITAL
    current_date = None
    max_equity = INITIAL_CAPITAL
    max_drawdown = 0

    for i in range(50, len(data)):
        row = data.iloc[i]
        hour = row["Hour"]
        date = row["Date"]
        dow = row["DOW"]

        if date != current_date:
            current_date = date
            daily_trades = 0
            daily_start_equity = capital

        daily_pnl_pct = ((capital - daily_start_equity) / daily_start_equity * 100) if daily_start_equity > 0 else 0
        daily_limit_hit = daily_pnl_pct <= -MAX_DAILY_LOSS_PCT

        # Manage open trade
        if current_trade is not None:
            if hour >= EURO_CLOSE:
                current_trade.exit_price = row["Close"]
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "EOD"
                pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                trades.append(current_trade)
                current_trade = None
                continue

            if current_trade.direction == 1:
                if row["Low"] <= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "SL"
                    capital += current_trade.pnl
                    trades.append(current_trade)
                    current_trade = None
                    continue
                if row["High"] >= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "TP"
                    capital += current_trade.pnl
                    trades.append(current_trade)
                    current_trade = None
                    continue
                new_trail = row["High"] - row["ATR14"] * TRAIL_ATR_MULT
                if new_trail > current_trade.stop_loss:
                    current_trade.stop_loss = new_trail
            else:
                if row["High"] >= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "SL"
                    capital += current_trade.pnl
                    trades.append(current_trade)
                    current_trade = None
                    continue
                if row["Low"] <= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "TP"
                    capital += current_trade.pnl
                    trades.append(current_trade)
                    current_trade = None
                    continue
                new_trail = row["Low"] + row["ATR14"] * TRAIL_ATR_MULT
                if new_trail < current_trade.stop_loss:
                    current_trade.stop_loss = new_trail

        if current_trade is not None:
            continue
        if daily_limit_hit or daily_trades >= MAX_DAILY_TRADES or capital <= 10:
            continue
        if dow not in ALLOWED_DAYS:
            continue
        if not (7 <= hour < EURO_CLOSE):
            continue
        if pd.isna(row["ATR14"]) or pd.isna(row["ORB_High"]):
            continue

        high_vol = row["Volume"] > row["AvgVol"] * 0.8 if not pd.isna(row["AvgVol"]) else True
        bull_trend = row["EMA9"] > row["EMA21"] and row["Close"] > row["EMA50"]
        bear_trend = row["EMA9"] < row["EMA21"] and row["Close"] < row["EMA50"]

        entry_price = row["Close"]
        atr = row["ATR14"]
        orb_high = row["ORB_High"]
        orb_low = row["ORB_Low"]
        orb_range = row["ORB_Range"]

        signal = None
        stop_loss = take_profit = 0
        trade_type = ""

        # ORB
        if USE_ORB and signal is None and hour >= 9 and hour < EURO_CLOSE and not pd.isna(orb_range) and orb_range > ORB_MIN_RANGE:
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

        # VWAP mean reversion (new)
        if USE_VWAP and signal is None and not pd.isna(row["VWAP"]) and 9 <= hour < 18:
            vwap_dist = (entry_price - row["VWAP"]) / atr if atr > 0 else 0
            if vwap_dist < -1.5 and bull_trend:
                stop_loss = entry_price - atr * 1.0
                take_profit = row["VWAP"]
                signal = 1
                trade_type = "VWAP_L"
            elif vwap_dist > 1.5 and bear_trend:
                stop_loss = entry_price + atr * 1.0
                take_profit = row["VWAP"]
                signal = -1
                trade_type = "VWAP_S"

        # Momentum
        if USE_MOM and signal is None:
            if row["EMA_Bull_Cross"] and entry_price > row["EMA50"] and 40 < row["RSI"] < 70 and 7 <= hour < EURO_CLOSE and high_vol:
                stop_loss = entry_price - atr * MOM_SL_ATR
                take_profit = entry_price + atr * MOM_TP_ATR
                signal = 1
                trade_type = "MOM_L"
            elif row["EMA_Bear_Cross"] and entry_price < row["EMA50"] and 30 < row["RSI"] < 60 and 7 <= hour < EURO_CLOSE and high_vol:
                stop_loss = entry_price + atr * MOM_SL_ATR
                take_profit = entry_price - atr * MOM_TP_ATR
                signal = -1
                trade_type = "MOM_S"

        if signal is not None:
            risk_amount = capital * RISK_PERCENT / 100
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance < 5:
                continue
            size_lots = risk_amount / (stop_distance * POINT_VALUE)
            size_lots = max(MIN_LOT, round(size_lots, 3))
            margin_required = entry_price * size_lots * POINT_VALUE / LEVERAGE
            if margin_required > capital * 0.95:
                size_lots = (capital * 0.95 * LEVERAGE) / (entry_price * POINT_VALUE)
                size_lots = max(MIN_LOT, round(size_lots, 3))

            current_trade = Trade(signal, entry_price, stop_loss, take_profit, row["UTC"], trade_type, size_lots)
            daily_trades += 1

        if capital > max_equity:
            max_equity = capital
        dd = (max_equity - capital) / max_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

    if current_trade is not None:
        last_row = data.iloc[-1]
        current_trade.exit_price = last_row["Close"]
        pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
        current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
        capital += current_trade.pnl
        trades.append(current_trade)

    return capital, trades, max_drawdown, max_equity


# ============================================================================
# OPTIMIZATION: Test different parameter combinations
# ============================================================================

print("\n" + "="*80)
print("PARAMETER OPTIMIZATION")
print("="*80)

best_capital = 0
best_params = {}
best_sharpe = -999

test_configs = [
    # Base improved (no MR)
    {"name": "Base_NoMR", "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # Higher risk
    {"name": "HighRisk", "risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # ORB only
    {"name": "ORB_Only", "risk_pct": 4.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": False, "use_vwap": False, "max_daily": 2, "euro_close": 19, "allowed_days": {0,1,2,3}},
    # ORB + wider target
    {"name": "ORB_Wide", "risk_pct": 4.0, "orb_target": 2.0, "orb_stop": 0.5, "mom_tp_atr": 3.0, "mom_sl_atr": 1.5, "trail_atr": 2.0, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # Tight stops
    {"name": "TightStop", "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0, "trail_atr": 1.0, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # With VWAP
    {"name": "WithVWAP", "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": True, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # All days incl Friday
    {"name": "AllDays", "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3,4}},
    # Aggressive: high risk + wide target + VWAP
    {"name": "Aggressive", "risk_pct": 5.0, "orb_target": 2.0, "orb_stop": 0.5, "mom_tp_atr": 3.0, "mom_sl_atr": 1.5, "trail_atr": 2.0, "use_orb": True, "use_mom": True, "use_vwap": True, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # Early close
    {"name": "EarlyClose", "risk_pct": 4.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 17, "allowed_days": {0,1,2,3}},
    # Best days only (Mon, Wed)
    {"name": "BestDays", "risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": True, "use_vwap": False, "max_daily": 3, "euro_close": 20, "allowed_days": {0,2}},
    # ORB + VWAP no momentum
    {"name": "ORB_VWAP", "risk_pct": 4.0, "orb_target": 1.5, "orb_stop": 0.5, "mom_tp_atr": 2.5, "mom_sl_atr": 1.5, "trail_atr": 1.5, "use_orb": True, "use_mom": False, "use_vwap": True, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3}},
    # Ultra aggressive
    {"name": "UltraAggr", "risk_pct": 7.0, "orb_target": 2.0, "orb_stop": 0.5, "mom_tp_atr": 3.0, "mom_sl_atr": 1.5, "trail_atr": 2.0, "use_orb": True, "use_mom": True, "use_vwap": True, "max_daily": 5, "max_daily_loss": 8.0, "euro_close": 20, "allowed_days": {0,1,2,3}},
]

results = []
for cfg in test_configs:
    name = cfg.pop("name")
    cap, tds, dd, meq = run_backtest(cfg)
    wins = sum(1 for t in tds if t.pnl > 0)
    wr = wins / len(tds) * 100 if tds else 0
    ret = (cap - 200) / 200 * 100
    avg_pnl = np.mean([t.pnl for t in tds]) if tds else 0
    std_pnl = np.std([t.pnl for t in tds]) if tds else 1
    sharpe_approx = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    results.append((name, cap, ret, dd, len(tds), wr, sharpe_approx, meq))
    cfg["name"] = name
    print(f"  {name:15s}: ${cap:8.2f} ({ret:+6.1f}%), DD={dd:.1f}%, Trades={len(tds):3d}, WR={wr:.1f}%, Sharpe≈{sharpe_approx:.2f}")

print("\n" + "="*80)
results.sort(key=lambda x: x[1], reverse=True)
print("TOP 3 BY FINAL CAPITAL:")
for i, r in enumerate(results[:3]):
    print(f"  #{i+1} {r[0]:15s}: ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, Trades={r[4]}, WR={r[5]:.1f}%")

results_sharpe = sorted(results, key=lambda x: x[6], reverse=True)
print("\nTOP 3 BY RISK-ADJUSTED (Sharpe):")
for i, r in enumerate(results_sharpe[:3]):
    print(f"  #{i+1} {r[0]:15s}: Sharpe≈{r[6]:.2f}, ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%")

# ============================================================================
# Run BEST config with detailed output
# ============================================================================

best_name = results[0][0]
best_cfg = next(c for c in test_configs if c["name"] == best_name)
best_cfg_copy = {k:v for k,v in best_cfg.items() if k != "name"}

print(f"\n{'='*80}")
print(f"DETAILED RESULTS FOR BEST CONFIG: {best_name}")
print(f"{'='*80}")
print(f"Parameters: {best_cfg}")

capital, trades, max_dd, max_eq = run_backtest(best_cfg_copy)

wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl <= 0]
win_rate = len(wins) / len(trades) * 100 if trades else 0
avg_win = np.mean([t.pnl for t in wins]) if wins else 0
avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
gross_profit = sum(t.pnl for t in wins)
gross_loss = sum(abs(t.pnl) for t in losses) if losses else 0
pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

print(f"\nFinal Capital:   ${capital:.2f}")
print(f"Total Return:    {((capital-200)/200*100):.1f}%")
print(f"Max Drawdown:    {max_dd:.1f}%")
print(f"Max Equity:      ${max_eq:.2f}")
print(f"Total Trades:    {len(trades)}")
print(f"Win Rate:        {win_rate:.1f}%")
print(f"Avg Win:         ${avg_win:.2f}")
print(f"Avg Loss:        ${avg_loss:.2f}")
print(f"Profit Factor:   {pf:.2f}")
print(f"RR Ratio:        {(avg_win/avg_loss if avg_loss > 0 else 0):.2f}")

# By type
print(f"\n--- By Trade Type ---")
types = {}
for t in trades:
    tt = t.trade_type
    if tt not in types:
        types[tt] = {"n": 0, "wins": 0, "pnl": 0}
    types[tt]["n"] += 1
    types[tt]["pnl"] += t.pnl
    if t.pnl > 0:
        types[tt]["wins"] += 1
for tt, s in sorted(types.items()):
    print(f"  {tt:8s}: {s['n']:3d} trades, WR={s['wins']/s['n']*100:.1f}%, PnL=${s['pnl']:.2f}")

# Monthly
print(f"\n--- Monthly Performance ---")
monthly = {}
for t in trades:
    mk = f"{t.entry_time.year}-{t.entry_time.month:02d}"
    if mk not in monthly:
        monthly[mk] = {"pnl": 0, "n": 0, "wins": 0}
    monthly[mk]["pnl"] += t.pnl
    monthly[mk]["n"] += 1
    if t.pnl > 0:
        monthly[mk]["wins"] += 1

prof_months = 0
for mk in sorted(monthly.keys()):
    m = monthly[mk]
    wr = m["wins"]/m["n"]*100 if m["n"]>0 else 0
    s = "+" if m["pnl"]>0 else "-"
    if m["pnl"]>0: prof_months += 1
    print(f"  {mk}: {m['n']:3d} trades, WR={wr:.0f}%, PnL=${m['pnl']:+8.2f} {'✓' if m['pnl']>0 else '✗'}")
print(f"\n  Profitable months: {prof_months}/{len(monthly)} ({prof_months/len(monthly)*100:.0f}%)")

# Equity curve stats
print(f"\n--- Growth Milestones ---")
running_cap = 200.0
milestones = [300, 400, 500, 750, 1000]
for t in trades:
    running_cap += t.pnl
    for m in milestones[:]:
        if running_cap >= m:
            print(f"  ${m} reached at {t.entry_time.strftime('%Y-%m-%d')}")
            milestones.remove(m)

print(f"\n{'='*80}")
print(f"FINAL: ${200:.0f} → ${capital:.2f} over {len(trades)} trades")
print(f"{'='*80}")
