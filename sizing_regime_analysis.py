#!/usr/bin/env python3
"""
Advanced Position Sizing & Market Regime Adaptation Analysis
for GER40 (DAX) hourly data.
"""

import glob
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def load_data():
    files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
    frames = []
    for f in files:
        tmp = pd.read_csv(f)
        if len(tmp) == 0:
            continue
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df["UTC"] = pd.to_datetime(df["UTC"], dayfirst=True, utc=True)
    df.sort_values("UTC", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. INDICATORS
# ─────────────────────────────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # ATR 14
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["ATR14"] = df["TR"].ewm(span=14, adjust=False).mean()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - 100 / (1 + rs)

    # ADX 14
    plus_dm = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    mask_plus = plus_dm > minus_dm
    mask_minus = minus_dm > plus_dm
    plus_dm = plus_dm.where(mask_plus, 0)
    minus_dm = minus_dm.where(mask_minus, 0)
    atr_smooth = df["TR"].ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(span=14, adjust=False).mean()

    # Median ATR 50
    df["ATR_median50"] = df["ATR14"].rolling(50, min_periods=1).median()

    df["hour"] = df["UTC"].dt.hour
    df["date"] = df["UTC"].dt.date
    df["dow"] = df["UTC"].dt.dayofweek
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. V4 BALANCED BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────
def backtest_v4(df, capital=200.0, risk_pct=5.0, orb_target=1.5, orb_stop=0.3,
                mom_tp_atr=2.0, mom_sl_atr=1.0, trail_atr=1.0, max_daily=4,
                euro_close=20, allowed_days=None, sizing_func=None):
    """
    V4 Balanced backtest with optional dynamic sizing_func.
    sizing_func(capital, recent_trades, current_atr, adx) -> risk_pct
    """
    if allowed_days is None:
        allowed_days = {0, 1, 2, 3}

    trades = []
    equity = capital
    peak_equity = capital

    dates = df["date"].unique()
    date_groups = {d: grp for d, grp in df.groupby("date")}

    for date in dates:
        day_df = date_groups[date].copy()
        if len(day_df) == 0:
            continue
        dow = day_df["dow"].iloc[0]
        if dow not in allowed_days:
            continue

        # ORB window: hours 7 and 8
        orb_bars = day_df[day_df["hour"].isin([7, 8])]
        if len(orb_bars) == 0:
            continue
        orb_high = orb_bars["High"].max()
        orb_low = orb_bars["Low"].min()
        orb_range = orb_high - orb_low
        if orb_range < 1:
            continue

        # Post-ORB bars (hour >= 9, before euro_close)
        post_orb = day_df[(day_df["hour"] >= 9) & (day_df["hour"] < euro_close)]
        if len(post_orb) == 0:
            continue

        daily_trades = 0
        orb_long_fired = False
        orb_short_fired = False
        second_orb_long_fired = False
        second_orb_short_fired = False
        failed_orb_long_fired = False
        failed_orb_short_fired = False

        for idx_pos in range(len(post_orb)):
            if daily_trades >= max_daily:
                break

            bar = post_orb.iloc[idx_pos]
            bar_idx = bar.name  # df index

            atr = bar["ATR14"] if not np.isnan(bar["ATR14"]) else orb_range
            adx_val = bar["ADX"] if not np.isnan(bar["ADX"]) else 25.0
            ema9 = bar["EMA9"]
            ema21 = bar["EMA21"]
            ema50 = bar["EMA50"]
            rsi = bar["RSI14"] if not np.isnan(bar["RSI14"]) else 50
            close = bar["Close"]
            high = bar["High"]
            low = bar["Low"]

            # Dynamic sizing
            if sizing_func is not None:
                current_risk = sizing_func(equity, trades, atr, adx_val)
            else:
                current_risk = risk_pct

            risk_amt = equity * current_risk / 100.0
            if risk_amt <= 0 or equity <= 0:
                continue

            # Trend filters
            bullish_trend = ema9 > ema21 > ema50
            bearish_trend = ema9 < ema21 < ema50

            signal = None
            entry = None
            sl = None
            tp = None
            trail = None

            # --- ORB LONG ---
            if not orb_long_fired and high > orb_high and bullish_trend and rsi < 75:
                signal = "ORB_Long"
                entry = orb_high
                sl = entry - orb_range * orb_stop
                tp = entry + orb_range * orb_target
                trail = trail_atr * atr
                orb_long_fired = True

            # --- ORB SHORT ---
            elif not orb_short_fired and low < orb_low and bearish_trend and rsi > 25:
                signal = "ORB_Short"
                entry = orb_low
                sl = entry + orb_range * orb_stop
                tp = entry - orb_range * orb_target
                trail = trail_atr * atr
                orb_short_fired = True

            # --- SECOND ORB LONG ---
            elif orb_long_fired and not second_orb_long_fired and high > orb_high + orb_range * 0.5 and bullish_trend:
                signal = "SecondORB_Long"
                entry = orb_high + orb_range * 0.5
                sl = entry - orb_range * orb_stop * 1.2
                tp = entry + orb_range * orb_target * 0.8
                trail = trail_atr * atr
                second_orb_long_fired = True

            # --- SECOND ORB SHORT ---
            elif orb_short_fired and not second_orb_short_fired and low < orb_low - orb_range * 0.5 and bearish_trend:
                signal = "SecondORB_Short"
                entry = orb_low - orb_range * 0.5
                sl = entry + orb_range * orb_stop * 1.2
                tp = entry - orb_range * orb_target * 0.8
                trail = trail_atr * atr
                second_orb_short_fired = True

            # --- FAILED ORB LONG (reversal after failed short breakout) ---
            elif orb_short_fired and not failed_orb_long_fired and close > orb_high and bullish_trend:
                signal = "FailedORB_Long"
                entry = close
                sl = entry - atr * mom_sl_atr
                tp = entry + atr * mom_tp_atr
                trail = trail_atr * atr
                failed_orb_long_fired = True

            # --- FAILED ORB SHORT (reversal after failed long breakout) ---
            elif orb_long_fired and not failed_orb_short_fired and close < orb_low and bearish_trend:
                signal = "FailedORB_Short"
                entry = close
                sl = entry + atr * mom_sl_atr
                tp = entry - atr * mom_tp_atr
                trail = trail_atr * atr
                failed_orb_short_fired = True

            # --- MOMENTUM LONG (EMA cross, long only) ---
            elif bar_idx > 0:
                prev_idx = bar_idx - 1
                if prev_idx in df.index:
                    prev = df.loc[prev_idx]
                    if (prev["EMA9"] <= prev["EMA21"] and ema9 > ema21
                            and close > ema50 and rsi < 70):
                        signal = "Mom_Long"
                        entry = close
                        sl = entry - atr * mom_sl_atr
                        tp = entry + atr * mom_tp_atr
                        trail = trail_atr * atr

            if signal is None:
                continue

            # Simulate trade using remaining bars
            is_long = "Long" in signal
            remaining = post_orb.iloc[idx_pos + 1:]
            exit_price = None
            trail_stop = sl if is_long else sl

            for _, fut in remaining.iterrows():
                if is_long:
                    # Update trailing stop
                    new_trail = fut["High"] - trail
                    if new_trail > trail_stop:
                        trail_stop = new_trail
                    # Check SL (trail)
                    if fut["Low"] <= trail_stop:
                        exit_price = trail_stop
                        break
                    # Check TP
                    if fut["High"] >= tp:
                        exit_price = tp
                        break
                else:
                    new_trail = fut["Low"] + trail
                    if new_trail < trail_stop:
                        trail_stop = new_trail
                    if fut["High"] >= trail_stop:
                        exit_price = trail_stop
                        break
                    if fut["Low"] <= tp:
                        exit_price = tp
                        break

            # If not closed, close at last bar
            if exit_price is None:
                exit_price = remaining["Close"].iloc[-1] if len(remaining) > 0 else close

            # PnL as fraction of risk distance, scaled to risk_amt
            if is_long:
                risk_dist = abs(entry - sl) if abs(entry - sl) > 0 else 1
                pnl_points = exit_price - entry
            else:
                risk_dist = abs(sl - entry) if abs(sl - entry) > 0 else 1
                pnl_points = entry - exit_price

            pnl = (pnl_points / risk_dist) * risk_amt

            equity += pnl
            if equity > peak_equity:
                peak_equity = equity

            trades.append({
                "date": date,
                "signal": signal,
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
                "equity": equity,
                "risk_pct_used": current_risk,
            })
            daily_trades += 1

    return trades, equity


# ─────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────
def compute_metrics(trades, start_capital=200.0):
    if not trades:
        return {
            "final_capital": start_capital, "max_dd": 0, "n_trades": 0,
            "win_rate": 0, "profit_factor": 0, "sharpe": 0,
        }
    pnls = [t["pnl"] for t in trades]
    equities = [start_capital] + [t["equity"] for t in trades]
    peak = start_capital
    max_dd = 0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    arr = np.array(pnls)
    sharpe = (arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    return {
        "final_capital": round(equities[-1], 2),
        "max_dd": round(max_dd, 2),
        "n_trades": len(pnls),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
    }


# ─────────────────────────────────────────────────────────────────────
# 5. SIZING STRATEGIES
# ─────────────────────────────────────────────────────────────────────

def make_fixed_risk(pct):
    def func(capital, recent_trades, current_atr, adx):
        return pct
    func.__name__ = f"Fixed_{pct}pct"
    return func


def make_kelly(half=True):
    label = "Half-Kelly" if half else "Quarter-Kelly"
    fraction = 0.5 if half else 0.25
    def func(capital, recent_trades, current_atr, adx):
        if len(recent_trades) < 100:
            return 5.0  # default during warmup
        first100 = recent_trades[:100]
        wins = [t["pnl"] for t in first100 if t["pnl"] > 0]
        losses = [t["pnl"] for t in first100 if t["pnl"] <= 0]
        if not wins or not losses:
            return 5.0
        wr = len(wins) / len(first100)
        avg_w = np.mean(wins)
        avg_l = abs(np.mean(losses))
        if avg_l == 0:
            return 5.0
        wl_ratio = avg_w / avg_l
        kelly = wr - (1 - wr) / wl_ratio
        kelly_pct = max(1.0, min(kelly * 100 * fraction, 15.0))
        return kelly_pct
    func.__name__ = label
    return func


def anti_martingale(capital, recent_trades, current_atr, adx):
    risk = 5.0
    for t in recent_trades:
        if t["pnl"] > 0:
            risk = min(risk + 1.0, 10.0)
        else:
            risk = max(risk - 1.0, 2.0)
    return risk
anti_martingale.__name__ = "Anti-Martingale"


def volatility_scaling(capital, recent_trades, current_atr, adx):
    base = 5.0
    # We need median ATR from recent context; use current_atr as fallback
    # The actual median ATR is embedded in the dataframe; approximate from trades
    if current_atr > 0:
        # We'll use a simple approach: collect ATRs from recent context
        recent_atrs = [current_atr]  # placeholder
        median_atr = current_atr  # will be overridden by wrapper
        scaled = base * (median_atr / current_atr) if current_atr > 0 else base
        return max(1.0, min(scaled, 12.0))
    return base
volatility_scaling.__name__ = "Vol-Scaling"


def make_vol_scaling_with_df(df):
    """Create vol scaling that has access to ATR median from df."""
    def func(capital, recent_trades, current_atr, adx):
        base = 5.0
        if current_atr > 0:
            # Use global ATR median from last 50 bars (approximated)
            # The backtest passes current_atr from the bar's ATR14
            # We need median_atr_50. We'll compute it from df globally once.
            med = df["ATR_median50"].median()  # overall median
            scaled = base * (med / current_atr) if current_atr > 0 else base
            return max(1.0, min(scaled, 12.0))
        return base
    func.__name__ = "Vol-Scaling"
    return func


def adx_scaling(capital, recent_trades, current_atr, adx):
    if adx > 30:
        return 7.0
    elif adx >= 20:
        return 5.0
    else:
        return 2.0
adx_scaling.__name__ = "ADX-Scaling"


def win_streak_scaling(capital, recent_trades, current_atr, adx):
    if len(recent_trades) < 2:
        return 5.0
    # count consecutive wins/losses at end
    consec_w = 0
    consec_l = 0
    for t in reversed(recent_trades):
        if t["pnl"] > 0:
            if consec_l > 0:
                break
            consec_w += 1
        else:
            if consec_w > 0:
                break
            consec_l += 1
    if consec_w >= 3:
        return 8.0
    elif consec_l >= 2:
        return 2.0
    return 5.0
win_streak_scaling.__name__ = "WinStreak-Scaling"


def compound_growth(capital, recent_trades, current_atr, adx):
    base = 3.0
    milestones_crossed = max(0, int((capital - 200) // 100))
    return min(base + milestones_crossed * 0.5, 15.0)
compound_growth.__name__ = "Compound-Growth"


def drawdown_adaptive(capital, recent_trades, current_atr, adx):
    if not recent_trades:
        return 5.0
    equities = [200.0] + [t["equity"] for t in recent_trades]
    peak = max(equities)
    current = equities[-1]
    dd_pct = (peak - current) / peak * 100 if peak > 0 else 0
    if dd_pct > 15:
        return 0.0  # stop trading
    elif dd_pct > 10:
        return 2.0
    return 5.0
drawdown_adaptive.__name__ = "DD-Adaptive"


# ─────────────────────────────────────────────────────────────────────
# 6. RUN ALL TESTS
# ─────────────────────────────────────────────────────────────────────
def run_full_test(df, start_capital=200.0):
    results = {}

    # 1. Fixed risk sweep
    fixed_pcts = [2, 3, 4, 5, 6, 7, 8, 10]
    for pct in fixed_pcts:
        name = f"Fixed_{pct}%"
        trades, final = backtest_v4(df, capital=start_capital,
                                     sizing_func=make_fixed_risk(pct))
        results[name] = compute_metrics(trades, start_capital)

    # 2. Kelly
    for half, label in [(True, "Half-Kelly"), (False, "Quarter-Kelly")]:
        trades, final = backtest_v4(df, capital=start_capital,
                                     sizing_func=make_kelly(half))
        results[label] = compute_metrics(trades, start_capital)

    # 3. Anti-Martingale
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=anti_martingale)
    results["Anti-Martingale"] = compute_metrics(trades, start_capital)

    # 4. Volatility scaling
    vol_func = make_vol_scaling_with_df(df)
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=vol_func)
    results["Vol-Scaling"] = compute_metrics(trades, start_capital)

    # 5. ADX Scaling
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=adx_scaling)
    results["ADX-Scaling"] = compute_metrics(trades, start_capital)

    # 6. Win Streak Scaling
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=win_streak_scaling)
    results["WinStreak-Scaling"] = compute_metrics(trades, start_capital)

    # 7. Compound Growth
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=compound_growth)
    results["Compound-Growth"] = compute_metrics(trades, start_capital)

    # 8. Drawdown Adaptive
    trades, final = backtest_v4(df, capital=start_capital,
                                 sizing_func=drawdown_adaptive)
    results["DD-Adaptive"] = compute_metrics(trades, start_capital)

    return results


# ─────────────────────────────────────────────────────────────────────
# 7. WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────
def walk_forward(df, strategy_name, sizing_func, start_capital=200.0):
    periods = [
        ("P1: Feb-Jul 2024", "2024-02-01", "2024-07-31"),
        ("P2: Aug 2024-Jan 2025", "2024-08-01", "2025-01-31"),
        ("P3: Feb-Jul 2025", "2025-02-01", "2025-07-31"),
        ("P4: Aug 2025-Mar 2026", "2025-08-01", "2026-03-31"),
    ]
    wf_results = {}
    running_capital = start_capital
    for label, start, end in periods:
        mask = (df["UTC"] >= pd.Timestamp(start, tz="UTC")) & (
            df["UTC"] <= pd.Timestamp(end, tz="UTC")
        )
        period_df = df[mask].copy().reset_index(drop=True)
        if len(period_df) == 0:
            wf_results[label] = {"final_capital": running_capital, "n_trades": 0}
            continue
        period_df = add_indicators(period_df)
        trades, final = backtest_v4(period_df, capital=running_capital,
                                     sizing_func=sizing_func)
        metrics = compute_metrics(trades, running_capital)
        metrics["start_capital"] = round(running_capital, 2)
        wf_results[label] = metrics
        running_capital = metrics["final_capital"]
    return wf_results


# ─────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("   ADVANCED POSITION SIZING & MARKET REGIME ADAPTATION ANALYSIS")
    print("   GER40 (DAX) Hourly Data")
    print("=" * 80)

    print("\nLoading data...")
    df = load_data()
    print(f"  Loaded {len(df)} bars from {df['UTC'].min()} to {df['UTC'].max()}")

    print("Computing indicators...")
    df = add_indicators(df)

    print("\nRunning all sizing strategies...\n")
    results = run_full_test(df)

    # Print results table
    print("-" * 110)
    print(f"{'Strategy':<22} {'Final($)':>10} {'MaxDD%':>8} {'Trades':>7} "
          f"{'WinRate%':>9} {'ProfFact':>9} {'Sharpe':>8}")
    print("-" * 110)

    # Group: Fixed Risk
    print("  [FIXED RISK SWEEP]")
    for pct in [2, 3, 4, 5, 6, 7, 8, 10]:
        name = f"Fixed_{pct}%"
        r = results[name]
        print(f"  {name:<20} {r['final_capital']:>10.2f} {r['max_dd']:>8.2f} "
              f"{r['n_trades']:>7} {r['win_rate']:>9.2f} {r['profit_factor']:>9.3f} "
              f"{r['sharpe']:>8.3f}")

    print("\n  [DYNAMIC STRATEGIES]")
    for name in ["Half-Kelly", "Quarter-Kelly", "Anti-Martingale", "Vol-Scaling",
                  "ADX-Scaling", "WinStreak-Scaling", "Compound-Growth", "DD-Adaptive"]:
        r = results[name]
        print(f"  {name:<20} {r['final_capital']:>10.2f} {r['max_dd']:>8.2f} "
              f"{r['n_trades']:>7} {r['win_rate']:>9.2f} {r['profit_factor']:>9.3f} "
              f"{r['sharpe']:>8.3f}")

    # Rank by final capital for top 3
    ranked = sorted(results.items(), key=lambda x: x[1]["final_capital"], reverse=True)
    print("\n" + "=" * 80)
    print("   TOP 3 STRATEGIES BY FINAL CAPITAL")
    print("=" * 80)
    for i, (name, r) in enumerate(ranked[:3], 1):
        print(f"  #{i}: {name} -> ${r['final_capital']:.2f} "
              f"(MaxDD: {r['max_dd']:.1f}%, WR: {r['win_rate']:.1f}%, "
              f"PF: {r['profit_factor']:.2f}, Sharpe: {r['sharpe']:.3f})")

    # Walk-forward for top 3
    top3_names = [name for name, _ in ranked[:3]]

    # Map names to sizing funcs
    sizing_map = {}
    for pct in [2, 3, 4, 5, 6, 7, 8, 10]:
        sizing_map[f"Fixed_{pct}%"] = make_fixed_risk(pct)
    sizing_map["Half-Kelly"] = make_kelly(True)
    sizing_map["Quarter-Kelly"] = make_kelly(False)
    sizing_map["Anti-Martingale"] = anti_martingale
    sizing_map["Vol-Scaling"] = make_vol_scaling_with_df(df)
    sizing_map["ADX-Scaling"] = adx_scaling
    sizing_map["WinStreak-Scaling"] = win_streak_scaling
    sizing_map["Compound-Growth"] = compound_growth
    sizing_map["DD-Adaptive"] = drawdown_adaptive

    print("\n" + "=" * 80)
    print("   WALK-FORWARD ANALYSIS (TOP 3)")
    print("   P1: Feb-Jul 2024 | P2: Aug 2024-Jan 2025 | P3: Feb-Jul 2025 | P4: Aug 2025-Mar 2026")
    print("=" * 80)

    for name in top3_names:
        func = sizing_map[name]
        wf = walk_forward(df, name, func)
        print(f"\n  Strategy: {name}")
        print(f"  {'Period':<28} {'Start($)':>10} {'Final($)':>10} {'MaxDD%':>8} "
              f"{'Trades':>7} {'WinRate%':>9} {'PF':>8} {'Sharpe':>8}")
        print(f"  {'-'*92}")
        for period_name, m in wf.items():
            sc = m.get("start_capital", 200)
            print(f"  {period_name:<28} {sc:>10.2f} {m['final_capital']:>10.2f} "
                  f"{m['max_dd']:>8.2f} {m['n_trades']:>7} {m['win_rate']:>9.2f} "
                  f"{m['profit_factor']:>8.3f} {m['sharpe']:>8.3f}")

    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
