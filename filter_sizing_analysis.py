#!/usr/bin/env python3
"""Sideways Market Filters & Position Sizing Analysis for GER40 Hourly Data."""

import glob
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── DATA LOADING ───────────────────────────────────────────────────────────
def load_data():
    files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['UTC'] = pd.to_datetime(df['UTC'], dayfirst=True, utc=True)
    df.sort_values('UTC', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Hour'] = df['UTC'].dt.hour
    df['Date'] = df['UTC'].dt.date
    df['DOW'] = df['UTC'].dt.dayofweek
    return df

# ─── INDICATORS ─────────────────────────────────────────────────────────────
def compute_adx(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.zeros(n)
    plus_di_s = np.zeros(n)
    minus_di_s = np.zeros(n)
    if n > period:
        atr[period] = tr[1:period+1].sum()
        plus_di_s[period] = plus_dm[1:period+1].sum()
        minus_di_s[period] = minus_dm[1:period+1].sum()
        for i in range(period+1, n):
            atr[i] = atr[i-1] - atr[i-1]/period + tr[i]
            plus_di_s[i] = plus_di_s[i-1] - plus_di_s[i-1]/period + plus_dm[i]
            minus_di_s[i] = minus_di_s[i-1] - minus_di_s[i-1]/period + minus_dm[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di_s / np.where(atr == 0, np.nan, atr)
        minus_di = 100 * minus_di_s / np.where(atr == 0, np.nan, atr)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, plus_di + minus_di)
    adx = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        adx[start] = np.nanmean(dx[period:start+1])
        for i in range(start+1, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
            else:
                adx[i] = adx[i-1]
    return adx

def add_indicators(df):
    c = df['Close']
    h, l = df['High'], df['Low']
    df['EMA9'] = c.ewm(span=9, adjust=False).mean()
    df['EMA21'] = c.ewm(span=21, adjust=False).mean()
    df['EMA50'] = c.ewm(span=50, adjust=False).mean()
    # ATR14
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    # RSI14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - 100 / (1 + rs)
    # AvgVol20
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    # ADX14
    df['ADX14'] = compute_adx(df, 14)
    # Bollinger Band width
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_Width'] = (upper - lower) / sma20 * 100
    # ORB per day (hours 7-8)
    orb = df[df['Hour'].isin([7, 8])].groupby('Date').agg(
        ORB_High=('High', 'max'), ORB_Low=('Low', 'min'))
    orb['ORB_Range'] = orb['ORB_High'] - orb['ORB_Low']
    df = df.merge(orb, on='Date', how='left')
    # EMA9 cross above/below EMA21
    df['EMA9_above'] = df['EMA9'] > df['EMA21']
    df['EMA9_cross_up'] = df['EMA9_above'] & ~df['EMA9_above'].shift(1, fill_value=False)
    return df

# ─── BACKTEST ENGINE ────────────────────────────────────────────────────────
def run_backtest(df, risk_pct=3.0, orb_target=1.5, orb_stop=0.3, mom_tp_atr=2.0,
                 mom_sl_atr=1.0, trail_atr=1.0, use_orb=True, use_mom=True,
                 max_daily=3, euro_close=20, allowed_days=None,
                 filter_fn=None, sizing_fn=None):
    if allowed_days is None:
        allowed_days = {0, 1, 2, 3}

    capital = 200.0
    peak = capital
    max_dd = 0.0
    trades = []   # list of pnl values
    daily_count = defaultdict(int)
    equity_curve = [capital]

    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        date = row['Date']
        hour = row['Hour']
        dow = row['DOW']

        if dow not in allowed_days or pd.isna(row['ATR14']) or pd.isna(row['EMA50']):
            i += 1; continue
        if daily_count[date] >= max_daily:
            i += 1; continue

        signal = None
        entry_price = row['Close']
        atr = row['ATR14']
        if atr <= 0 or pd.isna(atr):
            i += 1; continue

        # ORB signals
        if use_orb and hour >= 9 and not pd.isna(row.get('ORB_High', np.nan)):
            orb_range = row['ORB_Range']
            if orb_range > 0:
                vol_ok = row['Volume'] > row['AvgVol20'] * 0.8 if not pd.isna(row['AvgVol20']) else False
                if vol_ok:
                    if (entry_price > row['ORB_High'] and row['EMA9'] > row['EMA21']
                            and entry_price > row['EMA50']):
                        signal = 'long'
                        tp = entry_price + orb_range * orb_target
                        sl = entry_price - orb_range * orb_stop
                    elif (entry_price < row['ORB_Low'] and row['EMA9'] < row['EMA21']
                            and entry_price < row['EMA50']):
                        signal = 'short'
                        tp = entry_price - orb_range * orb_target
                        sl = entry_price + orb_range * orb_stop

        # Momentum signals (only long as specified)
        if signal is None and use_mom and not pd.isna(row.get('RSI14', np.nan)):
            if row['EMA9_cross_up'] and entry_price > row['EMA50'] and 40 < row['RSI14'] < 70:
                signal = 'long'
                tp = entry_price + mom_tp_atr * atr
                sl = entry_price - mom_sl_atr * atr

        if signal is None:
            i += 1; continue

        # Apply filter (True = skip)
        if filter_fn is not None and filter_fn(df, i):
            i += 1; continue

        # Position sizing
        stop_dist = abs(entry_price - sl)
        if stop_dist <= 0:
            i += 1; continue

        if sizing_fn is not None:
            risk = sizing_fn(capital, trades, equity_curve, df, i, risk_pct)
        else:
            risk = risk_pct

        risk_amt = capital * risk / 100.0
        pos_size = risk_amt / stop_dist
        if pos_size <= 0:
            i += 1; continue

        # Simulate trade bar-by-bar
        trail_stop = sl
        pnl = 0.0
        closed = False
        j = i + 1
        while j < n:
            bar = df.iloc[j]
            bh, bl, bc = bar['High'], bar['Low'], bar['Close']
            bar_hour = bar['Hour']
            cur_atr = bar['ATR14'] if not pd.isna(bar['ATR14']) else atr

            if signal == 'long':
                new_trail = bh - trail_atr * cur_atr
                if new_trail > trail_stop:
                    trail_stop = new_trail
                # Check SL first, then trailing, then TP
                if bl <= sl:
                    pnl = (sl - entry_price) * pos_size
                    closed = True; break
                if trail_stop > sl and bl <= trail_stop:
                    pnl = (trail_stop - entry_price) * pos_size
                    closed = True; break
                if bh >= tp:
                    pnl = (tp - entry_price) * pos_size
                    closed = True; break
            else:  # short
                new_trail = bl + trail_atr * cur_atr
                if new_trail < trail_stop:
                    trail_stop = new_trail
                if bh >= sl:
                    pnl = (entry_price - sl) * pos_size
                    closed = True; break
                if trail_stop < sl and bh >= trail_stop:
                    pnl = (entry_price - trail_stop) * pos_size
                    closed = True; break
                if bl <= tp:
                    pnl = (entry_price - tp) * pos_size
                    closed = True; break

            # EOD close
            if bar_hour >= euro_close:
                if signal == 'long':
                    pnl = (bc - entry_price) * pos_size
                else:
                    pnl = (entry_price - bc) * pos_size
                closed = True; break
            j += 1

        if not closed:
            bc = df.iloc[min(j, n-1)]['Close']
            pnl = (bc - entry_price) * pos_size if signal == 'long' else (entry_price - bc) * pos_size

        capital += pnl
        if capital <= 0:
            capital = 0
            trades.append(pnl)
            equity_curve.append(capital)
            break
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)
        trades.append(pnl)
        equity_curve.append(capital)
        daily_count[date] += 1
        i = j + 1 if closed else i + 1

    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) * 100 if trades else 0
    if len(trades) > 1:
        ret = np.array(trades)
        sharpe = ret.mean() / (ret.std() + 1e-10) * np.sqrt(252) if ret.std() > 0 else 0
    else:
        sharpe = 0
    return {
        'final_capital': round(capital, 2),
        'max_dd_pct': round(max_dd, 2),
        'trades': len(trades),
        'win_rate': round(wr, 2),
        'sharpe': round(sharpe, 2),
    }

# ─── FILTER FUNCTIONS (return True = skip trade) ───────────────────────────
def filter_adx20(df, i):
    v = df.iloc[i]['ADX14']
    return v < 20 if not pd.isna(v) else False

def filter_adx25(df, i):
    v = df.iloc[i]['ADX14']
    return v < 25 if not pd.isna(v) else False

def filter_bb_width30(df, i):
    start = max(0, i - 100)
    window = df['BB_Width'].iloc[start:i+1].dropna()
    if len(window) < 5:
        return False
    pct30 = window.quantile(0.3)
    return df.iloc[i]['BB_Width'] < pct30

def filter_atr_pct30(df, i):
    start = max(0, i - 50)
    window = df['ATR14'].iloc[start:i+1].dropna()
    if len(window) < 5:
        return False
    pct30 = window.quantile(0.3)
    return df.iloc[i]['ATR14'] < pct30

def filter_orb_80(df, i):
    v = df.iloc[i]['ORB_Range']
    return v < 80 if not pd.isna(v) else False

def filter_orb_100(df, i):
    v = df.iloc[i]['ORB_Range']
    return v < 100 if not pd.isna(v) else False

# ─── SIZING FUNCTIONS ───────────────────────────────────────────────────────
def make_fixed_risk(pct):
    def fn(capital, trades, eq, df, i, base_risk):
        return pct
    return fn

def sizing_martingale_light(capital, trades, eq, df, i, base_risk):
    if trades and trades[-1] < 0:
        return min(base_risk * 1.5, 15.0)
    return base_risk

def sizing_anti_martingale(capital, trades, eq, df, i, base_risk):
    if trades and trades[-1] > 0:
        return min(base_risk * 1.5, 15.0)
    return base_risk

def sizing_equity_curve(capital, trades, eq, df, i, base_risk):
    if len(eq) >= 20:
        sma = np.mean(eq[-20:])
        if eq[-1] < sma:
            return 1.0
    return base_risk

def sizing_vol_adjusted(capital, trades, eq, df, i, base_risk):
    atr = df.iloc[i]['ATR14']
    if pd.isna(atr) or atr <= 0:
        return base_risk
    start = max(0, i - 50)
    med_atr = df['ATR14'].iloc[start:i+1].median()
    if pd.isna(med_atr) or med_atr <= 0:
        return base_risk
    adjusted = base_risk * med_atr / atr
    return max(0.5, min(adjusted, 15.0))

# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} bars from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print("Computing indicators...")
    df = add_indicators(df)
    print("Done.\n")

    # BASELINE
    sep = "=" * 80
    print(sep)
    print("BASELINE (no filter, 3% risk, $200 start)")
    print(sep)
    baseline = run_backtest(df)
    print(f"  Final Capital: ${baseline['final_capital']:.2f}  |  Max DD: {baseline['max_dd_pct']:.2f}%  |  "
          f"Trades: {baseline['trades']}  |  Win Rate: {baseline['win_rate']:.1f}%  |  Sharpe: {baseline['sharpe']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: SIDEWAYS FILTERS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART 1: SIDEWAYS MARKET FILTERS")
    print(sep)
    filters = [
        ("A) ADX(14) < 20 -> skip",        filter_adx20),
        ("B) ADX(14) < 25 -> skip",        filter_adx25),
        ("C) BB Width < 30th pctl -> skip", filter_bb_width30),
        ("D) ATR14 < 30th pctl(50) -> skip",filter_atr_pct30),
        ("E) ORB Range < 80 pts -> skip",  filter_orb_80),
        ("F) ORB Range < 100 pts -> skip", filter_orb_100),
    ]
    hdr = f"  {'Filter':<35} {'Capital':>10} {'MaxDD%':>8} {'Trades':>7} {'WinR%':>8} {'vs Base':>10}"
    print(f"\n{hdr}")
    print("  " + "-" * 78)
    print(f"  {'(Baseline)':<35} {'$'+str(baseline['final_capital']):>10} "
          f"{baseline['max_dd_pct']:>7.2f}% {baseline['trades']:>7} "
          f"{baseline['win_rate']:>7.1f}% {'--':>10}")

    filter_results = {}
    for name, fn in filters:
        r = run_backtest(df, filter_fn=fn)
        delta = r['final_capital'] - baseline['final_capital']
        print(f"  {name:<35} {'$'+str(r['final_capital']):>10} {r['max_dd_pct']:>7.2f}% "
              f"{r['trades']:>7} {r['win_rate']:>7.1f}% {delta:>+10.2f}")
        filter_results[name] = r

    best_filter_name = max(filter_results, key=lambda k: filter_results[k]['final_capital'])
    best_filter_r = filter_results[best_filter_name]
    best_filter_fn = dict(filters)[best_filter_name]
    print(f"\n  >>> Best filter: {best_filter_name}  (${best_filter_r['final_capital']:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART 2: POSITION SIZING")
    print(sep)
    hdr2 = f"  {'Method':<40} {'Capital':>10} {'MaxDD%':>8} {'Trades':>7} {'Sharpe':>8}"
    print(f"\n{hdr2}")
    print("  " + "-" * 73)

    sizing_results = {}
    sizing_fns = {}

    # A) Fixed risk levels
    for pct in [1, 2, 3, 4, 5, 7, 10]:
        name = f"A) Fixed {pct}% risk"
        r = run_backtest(df, risk_pct=pct)
        print(f"  {name:<38} {'$'+str(r['final_capital']):>10} {r['max_dd_pct']:>7.2f}% "
              f"{r['trades']:>7} {r['sharpe']:>8.2f}")
        sizing_results[name] = r
        sizing_fns[name] = (make_fixed_risk(pct), pct)

    # B-E special sizing
    special = [
        ("B) Martingale Light (3% base)",     sizing_martingale_light),
        ("C) Anti-Martingale (3% base)",      sizing_anti_martingale),
        ("D) Equity Curve SMA20",             sizing_equity_curve),
        ("E) Volatility-Adjusted (3% base)",  sizing_vol_adjusted),
    ]
    for name, fn in special:
        r = run_backtest(df, sizing_fn=fn)
        print(f"  {name:<38} {'$'+str(r['final_capital']):>10} {r['max_dd_pct']:>7.2f}% "
              f"{r['trades']:>7} {r['sharpe']:>8.2f}")
        sizing_results[name] = r
        sizing_fns[name] = (fn, 3.0)

    best_sizing_name = max(sizing_results, key=lambda k: sizing_results[k]['final_capital'])
    best_sizing_r = sizing_results[best_sizing_name]
    best_sz_fn, best_sz_risk = sizing_fns[best_sizing_name]
    print(f"\n  >>> Best sizing: {best_sizing_name}  (${best_sizing_r['final_capital']:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3: BEST COMBO
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("PART 3: BEST COMBINATION")
    print(sep)
    print(f"  Filter: {best_filter_name}")
    print(f"  Sizing: {best_sizing_name}")
    print(f"  {'-'*40}")

    combo = run_backtest(df, risk_pct=best_sz_risk, filter_fn=best_filter_fn, sizing_fn=best_sz_fn)
    print(f"  Final Capital:  ${combo['final_capital']:.2f}")
    print(f"  Max Drawdown:   {combo['max_dd_pct']:.2f}%")
    print(f"  Trades:         {combo['trades']}")
    print(f"  Win Rate:       {combo['win_rate']:.1f}%")
    print(f"  Sharpe:         {combo['sharpe']:.2f}")
    print(f"  {'-'*40}")
    print(f"  vs Baseline:      {combo['final_capital'] - baseline['final_capital']:+.2f}")
    print(f"  vs Best Filter:   {combo['final_capital'] - best_filter_r['final_capital']:+.2f}")
    print(f"  vs Best Sizing:   {combo['final_capital'] - best_sizing_r['final_capital']:+.2f}")
    print(f"\n{sep}")
    print("ANALYSIS COMPLETE")
    print(sep)
