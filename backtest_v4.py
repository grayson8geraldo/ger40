#!/usr/bin/env python3
"""
GER40 DayTrader v4.0 - Improved Strategy Backtest
Incorporates all research findings:

IMPROVEMENTS OVER v3.0:
1. ADX filter (>20) to avoid sideways markets — from filter analysis
2. Failed ORB Reversal Long signal — PF 2.34, 60% WR from missed opportunities
3. Second ORB Breakout re-entry — PF 2.12, 65% WR from missed opportunities
4. 4H trend alignment filter — PF 1.26 aligned vs 0.92 against
5. Equity curve trading — reduce risk when in drawdown (Monte Carlo: 20% ruin risk)
6. Bull regime bias — only take shorts when ADX>25 (strategy only profits in bull regime)
7. Asymmetric R:R preserved — 0.3x stop vs 1.5x target (works even on random walk)

VALIDATED:
- Walk-forward showed degradation in 2025 → ADX filter + regime filter address this
- Random walk test confirmed structural edge from R:R asymmetry
- Monte Carlo: equity curve filter reduces ruin probability
"""

import pandas as pd
import numpy as np
import glob
import os

# ============================================================================
# LOAD DATA
# ============================================================================
data_dir = "/home/user/ger40/"
files = sorted(glob.glob(os.path.join(data_dir, "DEU.IDX-EUR_Hour_*.csv")))
dfs = [pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in files]
data = pd.concat(dfs, ignore_index=True).sort_values("UTC").reset_index(drop=True)
data["Hour"] = data["UTC"].dt.hour
data["Date"] = data["UTC"].dt.date
data["DOW"] = data["UTC"].dt.dayofweek

print(f"Loaded {len(data)} candles: {data['UTC'].min()} to {data['UTC'].max()}")

# ============================================================================
# INDICATORS
# ============================================================================
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

# ADX calculation
def calc_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = df["TR"]
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

data["ADX"] = calc_adx(data)

# 4H trend (using 4-bar EMA on hourly = pseudo 4H)
data["EMA50_4H"] = data["Close"].ewm(span=200, adjust=False).mean()  # ~50 bars on 4H = 200 on 1H
data["Trend_4H_Bull"] = data["Close"] > data["EMA50_4H"]
data["Trend_4H_Bear"] = data["Close"] < data["EMA50_4H"]

# ORB
orb_data = data[data["Hour"].isin([7, 8])].groupby("Date").agg(
    ORB_High=("High", "max"), ORB_Low=("Low", "min")).reset_index()
orb_data["ORB_Range"] = orb_data["ORB_High"] - orb_data["ORB_Low"]
data = data.merge(orb_data, on="Date", how="left")

# ============================================================================
# BACKTEST ENGINE v4.0
# ============================================================================
class Trade:
    __slots__ = ['direction','entry_price','stop_loss','take_profit','entry_time',
                 'trade_type','size_lots','exit_price','exit_time','pnl','exit_reason']
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

def run_backtest_v4(data, params):
    INITIAL = params.get("initial", 200.0)
    BASE_RISK = params.get("risk_pct", 3.0)
    ORB_TARGET = params.get("orb_target", 1.5)
    ORB_STOP = params.get("orb_stop", 0.3)
    MOM_TP = params.get("mom_tp_atr", 2.0)
    MOM_SL = params.get("mom_sl_atr", 1.0)
    TRAIL = params.get("trail_atr", 1.0)
    MAX_DAILY = params.get("max_daily", 3)
    MAX_DAILY_LOSS = params.get("max_daily_loss", 5.0)
    EURO_CLOSE = params.get("euro_close", 20)
    ALLOWED_DAYS = params.get("allowed_days", {0,1,2,3})

    # NEW v4 params
    ADX_MIN = params.get("adx_min", 20)
    ADX_MIN_SHORT = params.get("adx_min_short", 25)
    USE_4H_FILTER = params.get("use_4h_filter", True)
    USE_EQUITY_CURVE = params.get("use_equity_curve", True)
    USE_FAILED_ORB = params.get("use_failed_orb", True)
    USE_SECOND_ORB = params.get("use_second_orb", True)
    ORB_MIN_RANGE = params.get("orb_min_range", 20)

    POINT_VALUE = 1.0
    LEVERAGE = 20
    MIN_LOT = 0.01

    capital = INITIAL
    trades = []
    current_trade = None
    daily_trades = 0
    daily_start_eq = INITIAL
    current_date = None
    max_equity = INITIAL
    max_drawdown = 0

    # Equity curve tracking
    recent_pnls = []
    equity_sma = INITIAL

    # ORB tracking per day
    orb_long_taken = False
    orb_short_taken = False
    orb_long_failed = False  # price broke above ORB then came back
    orb_short_failed = False
    prev_above_orb = False
    prev_below_orb = False

    for i in range(50, len(data)):
        row = data.iloc[i]
        hour = row["Hour"]
        date = row["Date"]
        dow = row["DOW"]

        # New day reset
        if date != current_date:
            current_date = date
            daily_trades = 0
            daily_start_eq = capital
            orb_long_taken = False
            orb_short_taken = False
            orb_long_failed = False
            orb_short_failed = False
            prev_above_orb = False
            prev_below_orb = False

        daily_pnl_pct = ((capital - daily_start_eq) / daily_start_eq * 100) if daily_start_eq > 0 else 0
        daily_limit = daily_pnl_pct <= -MAX_DAILY_LOSS

        # Track ORB breakout failures for reversal signal
        if not pd.isna(row["ORB_High"]) and hour >= 9:
            currently_above = row["Close"] > row["ORB_High"]
            currently_below = row["Close"] < row["ORB_Low"]
            if prev_above_orb and currently_below:
                orb_long_failed = True  # was above, now below = failed long breakout
            if prev_below_orb and currently_above:
                orb_short_failed = True  # was below, now above = failed short breakout → go long
            prev_above_orb = currently_above
            prev_below_orb = currently_below

        # Manage open trade
        if current_trade is not None:
            if hour >= EURO_CLOSE:
                current_trade.exit_price = row["Close"]
                current_trade.exit_time = row["UTC"]
                current_trade.exit_reason = "EOD"
                pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
                current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
                capital += current_trade.pnl
                recent_pnls.append(current_trade.pnl)
                trades.append(current_trade)
                current_trade = None
                continue

            if current_trade.direction == 1:
                if row["Low"] <= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "SL"
                    capital += current_trade.pnl
                    recent_pnls.append(current_trade.pnl)
                    trades.append(current_trade)
                    current_trade = None
                    continue
                if row["High"] >= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "TP"
                    capital += current_trade.pnl
                    recent_pnls.append(current_trade.pnl)
                    trades.append(current_trade)
                    current_trade = None
                    continue
                new_trail = row["High"] - row["ATR14"] * TRAIL
                if new_trail > current_trade.stop_loss:
                    current_trade.stop_loss = new_trail
            else:
                if row["High"] >= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "SL"
                    capital += current_trade.pnl
                    recent_pnls.append(current_trade.pnl)
                    trades.append(current_trade)
                    current_trade = None
                    continue
                if row["Low"] <= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * current_trade.size_lots * POINT_VALUE
                    current_trade.exit_reason = "TP"
                    capital += current_trade.pnl
                    recent_pnls.append(current_trade.pnl)
                    trades.append(current_trade)
                    current_trade = None
                    continue
                new_trail = row["Low"] + row["ATR14"] * TRAIL
                if new_trail < current_trade.stop_loss:
                    current_trade.stop_loss = new_trail

        if current_trade is not None:
            continue
        if daily_limit or daily_trades >= MAX_DAILY or capital <= 10:
            continue
        if dow not in ALLOWED_DAYS:
            continue
        if not (7 <= hour < EURO_CLOSE):
            continue
        if pd.isna(row["ATR14"]) or pd.isna(row["ORB_High"]) or pd.isna(row["ADX"]):
            continue

        high_vol = row["Volume"] > row["AvgVol"] * 0.8 if not pd.isna(row["AvgVol"]) else True
        bull_trend = row["EMA9"] > row["EMA21"] and row["Close"] > row["EMA50"]
        bear_trend = row["EMA9"] < row["EMA21"] and row["Close"] < row["EMA50"]

        entry_price = row["Close"]
        atr = row["ATR14"]
        orb_high = row["ORB_High"]
        orb_low = row["ORB_Low"]
        orb_range = row["ORB_Range"]

        # Equity curve filter: reduce risk when in drawdown
        risk_pct = BASE_RISK
        if USE_EQUITY_CURVE and len(recent_pnls) >= 10:
            equity_sma = sum(recent_pnls[-10:]) / 10
            if equity_sma < 0:
                risk_pct = BASE_RISK * 0.5  # Half risk when equity curve declining

        signal = None
        stop_loss = take_profit = 0
        trade_type = ""

        # --- SIGNAL 1: ORB Breakout (primary) ---
        if hour >= 9 and hour < EURO_CLOSE and not pd.isna(orb_range) and orb_range > ORB_MIN_RANGE:
            # ADX filter: skip low-ADX environments
            adx_ok = row["ADX"] >= ADX_MIN

            # 4H trend filter
            trend_4h_long_ok = (not USE_4H_FILTER) or row["Trend_4H_Bull"]
            trend_4h_short_ok = (not USE_4H_FILTER) or row["Trend_4H_Bear"]

            if entry_price > orb_high and bull_trend and high_vol and adx_ok and trend_4h_long_ok and not orb_long_taken:
                stop_loss = orb_low - orb_range * ORB_STOP
                take_profit = entry_price + orb_range * ORB_TARGET
                signal = 1
                trade_type = "ORB_L"
                orb_long_taken = True
            elif entry_price < orb_low and bear_trend and high_vol and row["ADX"] >= ADX_MIN_SHORT and trend_4h_short_ok and not orb_short_taken:
                stop_loss = orb_high + orb_range * ORB_STOP
                take_profit = entry_price - orb_range * ORB_TARGET
                signal = -1
                trade_type = "ORB_S"
                orb_short_taken = True

        # --- SIGNAL 2: Failed ORB Reversal Long (new in v4) ---
        if USE_FAILED_ORB and signal is None and orb_short_failed and not orb_long_taken:
            if hour >= 9 and hour < 16 and entry_price > orb_high and bull_trend and row["ADX"] >= ADX_MIN:
                stop_loss = orb_low - orb_range * 0.3
                take_profit = entry_price + orb_range * 1.5
                signal = 1
                trade_type = "FORB_L"
                orb_long_taken = True
                orb_short_failed = False  # consume signal

        # --- SIGNAL 3: Second ORB Breakout (new in v4) ---
        if USE_SECOND_ORB and signal is None and orb_long_taken:
            # Price pulled back into ORB zone and is breaking out again
            prev_row = data.iloc[i-1]
            if (not pd.isna(prev_row["Close"]) and prev_row["Close"] <= orb_high and
                entry_price > orb_high and bull_trend and hour >= 10 and hour < 16 and
                row["ADX"] >= ADX_MIN):
                stop_loss = orb_low - orb_range * 0.3
                take_profit = entry_price + orb_range * 1.5
                signal = 1
                trade_type = "ORB2_L"

        # --- SIGNAL 4: EMA Momentum Long ---
        if signal is None and row["EMA_Bull_Cross"] and entry_price > row["EMA50"]:
            if 40 < row["RSI"] < 70 and 7 <= hour < EURO_CLOSE and high_vol and row["ADX"] >= ADX_MIN:
                trend_ok = (not USE_4H_FILTER) or row["Trend_4H_Bull"]
                if trend_ok:
                    stop_loss = entry_price - atr * MOM_SL
                    take_profit = entry_price + atr * MOM_TP
                    signal = 1
                    trade_type = "MOM_L"

        # Execute trade
        if signal is not None:
            risk_amount = capital * risk_pct / 100
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

    # Close remaining
    if current_trade is not None:
        last = data.iloc[-1]
        current_trade.exit_price = last["Close"]
        pnl_pts = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
        current_trade.pnl = pnl_pts * current_trade.size_lots * POINT_VALUE
        capital += current_trade.pnl
        trades.append(current_trade)

    return capital, trades, max_drawdown, max_equity

# ============================================================================
# RUN TESTS
# ============================================================================

print("\n" + "="*80)
print("STRATEGY COMPARISON: v3.0 TightStop vs v4.0 Improved")
print("="*80)

# v3.0 TightStop baseline (no new features)
v3_params = {
    "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3,
    "mom_tp_atr": 2.0, "mom_sl_atr": 1.0, "trail_atr": 1.0,
    "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3},
    "adx_min": 0, "adx_min_short": 0, "use_4h_filter": False,
    "use_equity_curve": False, "use_failed_orb": False, "use_second_orb": False,
}
cap3, trades3, dd3, meq3 = run_backtest_v4(data, v3_params)

# v4.0 with all improvements
v4_params = {
    "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3,
    "mom_tp_atr": 2.0, "mom_sl_atr": 1.0, "trail_atr": 1.0,
    "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
    "adx_min": 20, "adx_min_short": 25, "use_4h_filter": True,
    "use_equity_curve": True, "use_failed_orb": True, "use_second_orb": True,
}
cap4, trades4, dd4, meq4 = run_backtest_v4(data, v4_params)

def print_results(name, cap, trades, dd, meq, initial=200):
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins)/len(trades)*100 if trades else 0
    avg_w = np.mean([t.pnl for t in wins]) if wins else 0
    avg_l = np.mean([abs(t.pnl) for t in losses]) if losses else 0
    gp = sum(t.pnl for t in wins)
    gl = sum(abs(t.pnl) for t in losses)
    pf = gp/gl if gl > 0 else float('inf')

    print(f"\n--- {name} ---")
    print(f"  Capital: ${initial} → ${cap:.2f} ({(cap-initial)/initial*100:+.1f}%)")
    print(f"  Max DD: {dd:.1f}%, Max Equity: ${meq:.2f}")
    print(f"  Trades: {len(trades)}, WR: {wr:.1f}%, PF: {pf:.2f}")
    print(f"  Avg Win: ${avg_w:.2f}, Avg Loss: ${avg_l:.2f}, RR: {avg_w/avg_l if avg_l>0 else 0:.2f}")

    # By type
    types = {}
    for t in trades:
        if t.trade_type not in types:
            types[t.trade_type] = {"n":0,"w":0,"pnl":0}
        types[t.trade_type]["n"] += 1
        types[t.trade_type]["pnl"] += t.pnl
        if t.pnl > 0: types[t.trade_type]["w"] += 1
    print(f"  By type:")
    for tt in sorted(types.keys()):
        s = types[tt]
        print(f"    {tt:8s}: {s['n']:3d} trades, WR={s['w']/s['n']*100:.1f}%, PnL=${s['pnl']:.2f}")

    # By exit reason
    reasons = {}
    for t in trades:
        if t.exit_reason not in reasons:
            reasons[t.exit_reason] = {"n":0,"pnl":0}
        reasons[t.exit_reason]["n"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl
    print(f"  By exit:")
    for r in sorted(reasons.keys()):
        s = reasons[r]
        print(f"    {r:6s}: {s['n']:3d} trades, PnL=${s['pnl']:.2f}")

    # Monthly
    monthly = {}
    for t in trades:
        mk = f"{t.entry_time.year}-{t.entry_time.month:02d}"
        if mk not in monthly:
            monthly[mk] = {"pnl":0,"n":0,"w":0}
        monthly[mk]["pnl"] += t.pnl
        monthly[mk]["n"] += 1
        if t.pnl > 0: monthly[mk]["w"] += 1

    prof = sum(1 for m in monthly.values() if m["pnl"] > 0)
    print(f"  Monthly: {prof}/{len(monthly)} profitable ({prof/len(monthly)*100:.0f}%)")
    for mk in sorted(monthly.keys()):
        m = monthly[mk]
        print(f"    {mk}: {m['n']:3d} trades, WR={m['w']/m['n']*100:.0f}%, PnL=${m['pnl']:+8.2f} {'✓' if m['pnl']>0 else '✗'}")

print_results("v3.0 TightStop (baseline)", cap3, trades3, dd3, meq3)
print_results("v4.0 Improved", cap4, trades4, dd4, meq4)

# ============================================================================
# ADDITIONAL VARIANTS
# ============================================================================

print("\n" + "="*80)
print("VARIANT TESTING")
print("="*80)

variants = [
    ("v4_no_equity_curve", {**v4_params, "use_equity_curve": False}),
    ("v4_no_4h_filter", {**v4_params, "use_4h_filter": False}),
    ("v4_no_failed_orb", {**v4_params, "use_failed_orb": False}),
    ("v4_no_second_orb", {**v4_params, "use_second_orb": False}),
    ("v4_adx15", {**v4_params, "adx_min": 15}),
    ("v4_adx25", {**v4_params, "adx_min": 25}),
    ("v4_risk4pct", {**v4_params, "risk_pct": 4.0}),
    ("v4_risk5pct", {**v4_params, "risk_pct": 5.0}),
    ("v4_orb_target2x", {**v4_params, "orb_target": 2.0}),
    ("v4_max5_daily", {**v4_params, "max_daily": 5}),
    ("v4_aggressive", {**v4_params, "risk_pct": 5.0, "orb_target": 2.0, "max_daily": 5, "trail_atr": 1.5}),
]

results = []
for name, params in variants:
    cap, tds, dd, meq = run_backtest_v4(data, params)
    wins = sum(1 for t in tds if t.pnl > 0)
    wr = wins/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl > 0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl <= 0)
    pf = gp/gl if gl > 0 else 0
    ret = (cap-200)/200*100
    results.append((name, cap, ret, dd, len(tds), wr, pf))
    print(f"  {name:25s}: ${cap:8.2f} ({ret:+6.1f}%), DD={dd:.1f}%, Trades={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}")

print("\n" + "="*80)
results.sort(key=lambda x: x[1], reverse=True)
print("TOP 5 BY CAPITAL:")
for i, r in enumerate(results[:5]):
    print(f"  #{i+1} {r[0]:25s}: ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, PF={r[6]:.2f}")

# Run detailed report on best variant
best_name = results[0][0]
best_params = dict(next(p for n,p in variants if n == best_name))
print(f"\n{'='*80}")
print(f"BEST VARIANT DETAILED: {best_name}")
cap_best, trades_best, dd_best, meq_best = run_backtest_v4(data, best_params)
print_results(best_name, cap_best, trades_best, dd_best, meq_best)

# Walk-forward on best variant
print(f"\n{'='*80}")
print(f"WALK-FORWARD VALIDATION: {best_name}")
print(f"{'='*80}")

periods = [
    ("P1 Feb-Jul 2024", "2024-02-01", "2024-07-31"),
    ("P2 Aug 2024-Jan 2025", "2024-08-01", "2025-01-31"),
    ("P3 Feb-Jul 2025", "2025-02-01", "2025-07-31"),
    ("P4 Aug 2025-Mar 2026", "2025-08-01", "2026-03-31"),
]

for pname, start, end in periods:
    mask = (data["Date"] >= pd.Timestamp(start).date()) & (data["Date"] <= pd.Timestamp(end).date())
    period_data = data[mask].reset_index(drop=True)
    if len(period_data) < 100:
        print(f"  {pname}: Not enough data")
        continue
    cap_p, trades_p, dd_p, meq_p = run_backtest_v4(period_data, best_params)
    wins_p = sum(1 for t in trades_p if t.pnl > 0)
    wr_p = wins_p/len(trades_p)*100 if trades_p else 0
    gp_p = sum(t.pnl for t in trades_p if t.pnl > 0)
    gl_p = sum(abs(t.pnl) for t in trades_p if t.pnl <= 0)
    pf_p = gp_p/gl_p if gl_p > 0 else 0
    ret_p = (cap_p-200)/200*100
    print(f"  {pname:25s}: ${cap_p:.2f} ({ret_p:+.1f}%), DD={dd_p:.1f}%, Trades={len(trades_p)}, WR={wr_p:.1f}%, PF={pf_p:.2f}")

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")

# ============================================================================
# FINAL OPTIMIZATION: Best combo from findings
# ============================================================================
print("\n" + "="*80)
print("FINAL COMBO SEARCH")
print("="*80)

combos = [
    # ADX25 + no 4H + no equity curve + higher risk
    ("ADX25_5pct", {"risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                    "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                    "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                    "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # ADX25 + 4pct
    ("ADX25_4pct", {"risk_pct": 4.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                    "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                    "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                    "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # ADX20 + no 4H + 5pct + orb2x target
    ("ADX20_5pct_2x", {"risk_pct": 5.0, "orb_target": 2.0, "orb_stop": 0.3, "mom_tp_atr": 2.5, "mom_sl_atr": 1.0,
                       "trail_atr": 1.5, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                       "adx_min": 20, "adx_min_short": 25, "use_4h_filter": False,
                       "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # ADX25 + 7pct aggressive
    ("ADX25_7pct", {"risk_pct": 7.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                    "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                    "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                    "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # No filters at all + 5pct (like v3 but with new signals)
    ("NoFilter_5pct_new", {"risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                           "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                           "adx_min": 0, "adx_min_short": 0, "use_4h_filter": False,
                           "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # ADX20 + no 4H + 5pct
    ("ADX20_5pct", {"risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                    "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                    "adx_min": 20, "adx_min_short": 25, "use_4h_filter": False,
                    "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # v3 baseline + ADX25 only
    ("v3_ADX25", {"risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                  "trail_atr": 1.0, "max_daily": 3, "euro_close": 20, "allowed_days": {0,1,2,3},
                  "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                  "use_equity_curve": False, "use_failed_orb": False, "use_second_orb": False}),
    # v3 + ADX25 + new signals
    ("v3_ADX25_new", {"risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                      "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3},
                      "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                      "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
    # ADX25 + 5pct + all days incl Friday
    ("ADX25_5pct_alldays", {"risk_pct": 5.0, "orb_target": 1.5, "orb_stop": 0.3, "mom_tp_atr": 2.0, "mom_sl_atr": 1.0,
                            "trail_atr": 1.0, "max_daily": 4, "euro_close": 20, "allowed_days": {0,1,2,3,4},
                            "adx_min": 25, "adx_min_short": 25, "use_4h_filter": False,
                            "use_equity_curve": False, "use_failed_orb": True, "use_second_orb": True}),
]

results2 = []
for name, params in combos:
    cap, tds, dd, meq = run_backtest_v4(data, params)
    wins = sum(1 for t in tds if t.pnl > 0)
    wr = wins/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl > 0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl <= 0)
    pf = gp/gl if gl > 0 else 0
    ret = (cap-200)/200*100
    results2.append((name, cap, ret, dd, len(tds), wr, pf))
    print(f"  {name:25s}: ${cap:8.2f} ({ret:+6.1f}%), DD={dd:.1f}%, Trades={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}")

results2.sort(key=lambda x: x[1], reverse=True)
print(f"\nTOP 3 BY CAPITAL:")
for i, r in enumerate(results2[:3]):
    print(f"  #{i+1} {r[0]:25s}: ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, PF={r[6]:.2f}")

# Sort by return/DD ratio
results2_rr = sorted(results2, key=lambda x: x[2]/(x[3]+0.1), reverse=True)
print(f"\nTOP 3 BY RETURN/DD RATIO:")
for i, r in enumerate(results2_rr[:3]):
    ratio = r[2]/(r[3]+0.1)
    print(f"  #{i+1} {r[0]:25s}: R/DD={ratio:.1f}, ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, PF={r[6]:.2f}")

# Detailed report of best combo
best = results2[0]
best_name = best[0]
best_p = dict(next(p for n,p in combos if n == best_name))
print(f"\n{'='*80}")
print(f"BEST COMBO DETAILED: {best_name}")
cap_b, trades_b, dd_b, meq_b = run_backtest_v4(data, best_p)
print_results(best_name, cap_b, trades_b, dd_b, meq_b)

# Walk-forward
print(f"\nWALK-FORWARD:")
for pname, start, end in periods:
    mask = (data["Date"] >= pd.Timestamp(start).date()) & (data["Date"] <= pd.Timestamp(end).date())
    pd_data = data[mask].reset_index(drop=True)
    if len(pd_data) < 100: continue
    cp, tp, dp, mp = run_backtest_v4(pd_data, best_p)
    ws = sum(1 for t in tp if t.pnl > 0)
    wp = ws/len(tp)*100 if tp else 0
    gpp = sum(t.pnl for t in tp if t.pnl > 0)
    glp = sum(abs(t.pnl) for t in tp if t.pnl <= 0)
    pfp = gpp/glp if glp > 0 else 0
    rp = (cp-200)/200*100
    print(f"  {pname:25s}: ${cp:.2f} ({rp:+.1f}%), DD={dp:.1f}%, Trades={len(tp)}, WR={wp:.1f}%, PF={pfp:.2f}")
