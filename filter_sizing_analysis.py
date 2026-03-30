"""
Sideways Market Filters & Position Sizing Analysis for GER40 Hourly Data
"""
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

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
    # ORB per day
    df['ORB_High'] = np.nan
    df['ORB_Low'] = np.nan
    for date, grp in df.groupby('Date'):
        mask_orb = (grp['Hour'] >= 7) & (grp['Hour'] <= 8)
        if mask_orb.any():
            orb_h = grp.loc[mask_orb, 'High'].max()
            orb_l = grp.loc[mask_orb, 'Low'].min()
            df.loc[grp.index, 'ORB_High'] = orb_h
            df.loc[grp.index, 'ORB_Low'] = orb_l
    df['ORB_Range'] = df['ORB_High'] - df['ORB_Low']
    # EMA9 cross above/below EMA21
    df['EMA9_above'] = df['EMA9'] > df['EMA21']
    df['EMA9_cross_up'] = df['EMA9_above'] & ~df['EMA9_above'].shift(1, fill_value=False)
    df['EMA9_cross_down'] = ~df['EMA9_above'] & df['EMA9_above'].shift(1, fill_value=True)
    # ATR percentile over last 50 bars
    df['ATR_pct50'] = df['ATR14'].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    # BB width percentile (rolling full history approximation via expanding)
    df['BB_Width_pct'] = df['BB_Width'].expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    return df

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
    # Smoothed
    atr = np.zeros(n)
    plus_di_s = np.zeros(n)
    minus_di_s = np.zeros(n)
    atr[period] = tr[1:period+1].sum()
    plus_di_s[period] = plus_dm[1:period+1].sum()
    minus_di_s[period] = minus_dm[1:period+1].sum()
    for i in range(period+1, n):
        atr[i] = atr[i-1] - atr[i-1]/period + tr[i]
        plus_di_s[i] = plus_di_s[i-1] - plus_di_s[i-1]/period + plus_dm[i]
        minus_di_s[i] = minus_di_s[i-1] - minus_di_s[i-1]/period + minus_dm[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di_s / np.where(atr==0, np.nan, atr)
        minus_di = 100 * minus_di_s / np.where(atr==0, np.nan, atr)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di)==0, np.nan, plus_di + minus_di)
    adx = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        adx[start] = np.nanmean(dx[period:start+1])
        for i in range(start+1, n):
            adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
    return adx

# ─── BACKTEST ────────────────────────────────────────────────────────────────
def run_backtest(df, risk_pct=3.0, orb_target=1.5, orb_stop=0.3, mom_tp_atr=2.0,
                 mom_sl_atr=1.0, trail_atr=1.0, use_orb=True, use_mom=True,
                 max_daily=3, euro_close=20, allowed_days={0,1,2,3},
                 filter_fn=None, sizing_fn=None):
    capital = 200.0
    peak = capital
    max_dd = 0.0
    trades = []
    daily_count = defaultdict(int)
    equity_curve = [capital]
    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        date = row['Date']
        hour = row['Hour']
        dow = row['DOW']
        # Skip conditions
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
        if use_orb and hour >= 9 and not pd.isna(row['ORB_High']):
            if (entry_price > row['ORB_High'] and row['EMA9'] > row['EMA21']
                    and entry_price > row['EMA50'] and row['Volume'] > row['AvgVol20'] * 0.8):
                orb_range = row['ORB_Range']
                if orb_range > 0:
                    signal = 'long'
                    tp = entry_price + orb_range * orb_target
                    sl = entry_price - orb_range * orb_stop
            elif (entry_price < row['ORB_Low'] and row['EMA9'] < row['EMA21']
                    and entry_price < row['EMA50'] and row['Volume'] > row['AvgVol20'] * 0.8):
                orb_range = row['ORB_Range']
                if orb_range > 0:
                    signal = 'short'
                    tp = entry_price - orb_range * orb_target
                    sl = entry_price + orb_range * orb_stop
        # Momentum signals (only if no ORB signal)
        if signal is None and use_mom and not pd.isna(row['RSI14']):
            if row['EMA9_cross_up'] and entry_price > row['EMA50'] and 40 < row['RSI14'] < 70:
                signal = 'long'
                tp = entry_price + mom_tp_atr * atr
                sl = entry_price - mom_sl_atr * atr
            elif row['EMA9_cross_down'] and entry_price < row['EMA50'] and 30 < row['RSI14'] < 60:
                signal = 'short'
                tp = entry_price - mom_tp_atr * atr
                sl = entry_price + mom_sl_atr * atr
        if signal is None:
            i += 1; continue
        # Apply filter
        if filter_fn and filter_fn(df, i):
            i += 1; continue
        # Position sizing
        stop_dist = abs(entry_price - sl)
        if stop_dist <= 0:
            i += 1; continue
        if sizing_fn:
            risk = sizing_fn(capital, trades, equity_curve, df, i)
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
            if bar['Date'] != date and False:
                pass  # allow multi-day? No, EOD close
            bh, bl, bc = bar['High'], bar['Low'], bar['Close']
            bar_hour = bar['Hour']
            cur_atr = bar['ATR14'] if not pd.isna(bar['ATR14']) else atr
            if signal == 'long':
                # Update trailing stop
                new_trail = bc - trail_atr * cur_atr
                if new_trail > trail_stop:
                    trail_stop = new_trail
                # Check stop
                if bl <= trail_stop:
                    pnl = (trail_stop - entry_price) * pos_size
                    closed = True; break
                if bl <= sl:
                    pnl = (sl - entry_price) * pos_size
                    closed = True; break
                # Check TP
                if bh >= tp:
                    pnl = (tp - entry_price) * pos_size
                    closed = True; break
            else:  # short
                new_trail = bc + trail_atr * cur_atr
                if new_trail < trail_stop:
                    trail_stop = new_trail
                if bh >= trail_stop:
                    pnl = (entry_price - trail_stop) * pos_size
                    closed = True; break
                if bh >= sl:
                    pnl = (entry_price - sl) * pos_size
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
            # Close at last available bar
            if j < n:
                bc = df.iloc[j-1]['Close']
            else:
                bc = df.iloc[-1]['Close']
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
    # Sharpe approximation (daily-ish returns)
    if len(trades) > 1:
        ret = np.array(trades)
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    else:
        sharpe = 0
    return {
        'final_capital': round(capital, 2),
        'max_dd_pct': round(max_dd, 2),
        'trades': len(trades),
        'win_rate': round(wr, 2),
        'sharpe': round(sharpe, 2),
    }

# ─── FILTERS ────────────────────────────────────────────────────────────────
def filter_adx20(df, i):
    return df.iloc[i]['ADX14'] < 20 if not pd.isna(df.iloc[i]['ADX14']) else False

def filter_adx25(df, i):
    return df.iloc[i]['ADX14'] < 25 if not pd.isna(df.iloc[i]['ADX14']) else False

def filter_bb_width30(df, i):
    v = df.iloc[i]['BB_Width_pct']
    return v < 0.30 if not pd.isna(v) else False

def filter_atr_pct30(df, i):
    v = df.iloc[i]['ATR_pct50']
    return v < 0.30 if not pd.isna(v) else False

def filter_orb_80(df, i):
    v = df.iloc[i]['ORB_Range']
    return v < 80 if not pd.isna(v) else False

def filter_orb_100(df, i):
    v = df.iloc[i]['ORB_Range']
    return v < 100 if not pd.isna(v) else False

# ─── SIZING FUNCTIONS ───────────────────────────────────────────────────────
def make_fixed_risk(pct):
    def fn(capital, trades, eq, df, i):
        return pct
    return fn

def sizing_martingale_light(capital, trades, eq, df, i):
    base = 3.0
    if trades and trades[-1] < 0:
        return min(base * 1.5, 15.0)
    return base

def sizing_anti_martingale(capital, trades, eq, df, i):
    base = 3.0
    if trades and trades[-1] > 0:
        return min(base * 1.5, 15.0)
    return base

def sizing_equity_curve(capital, trades, eq, df, i):
    base = 3.0
    if len(eq) >= 20:
        sma = np.mean(eq[-20:])
        if eq[-1] < sma:
            return 1.0
    return base

def sizing_vol_adjusted(capital, trades, eq, df, i):
    base = 3.0
    atr = df.iloc[i]['ATR14']
    if pd.isna(atr) or atr <= 0:
        return base
    med_atr = df['ATR14'].iloc[max(0, i-50):i+1].median()
    if pd.isna(med_atr) or med_atr <= 0:
        return base
    adjusted = base * med_atr / atr
    return max(0.5, min(adjusted, 15.0))

# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} bars from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print("Computing indicators...")
    df = add_indicators(df)
    print("Done.\n")

    # BASELINE
    print("=" * 80)
    print("BASELINE (no filter, 3% risk)")
    print("=" * 80)
    baseline = run_backtest(df)
    print(f"  Final Capital: ${baseline['final_capital']:.2f}  |  Max DD: {baseline['max_dd_pct']:.2f}%  |  "
          f"Trades: {baseline['trades']}  |  Win Rate: {baseline['win_rate']:.1f}%  |  Sharpe: {baseline['sharpe']:.2f}")

    # PART 1: SIDEWAYS FILTERS
    print("\n" + "=" * 80)
    print("PART 1: SIDEWAYS MARKET FILTERS")
    print("=" * 80)
    filters = [
        ("A) ADX(14) < 20 skip", filter_adx20),
        ("B) ADX(14) < 25 skip", filter_adx25),
        ("C) BB Width < 30th pct skip", filter_bb_width30),
        ("D) ATR14 < 30th pct(50) skip", filter_atr_pct30),
        ("E) ORB Range < 80 pts skip", filter_orb_80),
        ("F) ORB Range < 100 pts skip", filter_orb_100),
    ]
    print(f"\n{'Filter':<32} {'Capital':>10} {'MaxDD%':>8} {'Trades':>7} {'WinRate%':>9} {'vs Base':>10}")
    print("-" * 80)
    filter_results = {}
    for name, fn in filters:
        r = run_backtest(df, filter_fn=fn)
        delta = r['final_capital'] - baseline['final_capital']
        print(f"  {name:<30} ${r['final_capital']:>8.2f} {r['max_dd_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>8.1f}% {delta:>+9.2f}")
        filter_results[name] = r

    # Find best filter
    best_filter_name = max(filter_results, key=lambda k: filter_results[k]['final_capital'])
    best_filter_r = filter_results[best_filter_name]
    print(f"\n  >>> Best filter: {best_filter_name}")
    # Map back to function
    best_filter_fn = dict(filters)[best_filter_name]

    # PART 2: POSITION SIZING
    print("\n" + "=" * 80)
    print("PART 2: POSITION SIZING")
    print("=" * 80)

    # A) Fixed risk levels
    print(f"\n{'Sizing Method':<40} {'Capital':>10} {'MaxDD%':>8} {'Trades':>7} {'Sharpe':>8}")
    print("-" * 80)
    sizing_results = {}
    for pct in [1, 2, 3, 4, 5, 7, 10]:
        name = f"A) Fixed {pct}% risk"
        r = run_backtest(df, risk_pct=pct, sizing_fn=make_fixed_risk(pct))
        print(f"  {name:<38} ${r['final_capital']:>8.2f} {r['max_dd_pct']:>7.2f}% {r['trades']:>6} {r['sharpe']:>7.2f}")
        sizing_results[name] = r

    # B-E
    special_sizing = [
        ("B) Martingale Light (3% base)", sizing_martingale_light),
        ("C) Anti-Martingale (3% base)", sizing_anti_martingale),
        ("D) Equity Curve SMA20", sizing_equity_curve),
        ("E) Volatility-Adjusted (3% base)", sizing_vol_adjusted),
    ]
    for name, fn in special_sizing:
        r = run_backtest(df, sizing_fn=fn)
        print(f"  {name:<38} ${r['final_capital']:>8.2f} {r['max_dd_pct']:>7.2f}% {r['trades']:>6} {r['sharpe']:>7.2f}")
        sizing_results[name] = r

    best_sizing_name = max(sizing_results, key=lambda k: sizing_results[k]['final_capital'])
    best_sizing_r = sizing_results[best_sizing_name]
    print(f"\n  >>> Best sizing: {best_sizing_name}")
    best_sizing_fn = dict(special_sizing).get(best_sizing_name)
    best_sizing_risk = 3.0
    # If it's a fixed risk, extract pct
    if best_sizing_name.startswith("A) Fixed"):
        pct = int(best_sizing_name.split("%")[0].split()[-1])
        best_sizing_fn = make_fixed_risk(pct)
        best_sizing_risk = pct

    # PART 3: BEST COMBO
    print("\n" + "=" * 80)
    print("PART 3: BEST COMBO")
    print(f"  Filter: {best_filter_name}")
    print(f"  Sizing: {best_sizing_name}")
    print("=" * 80)
    combo = run_backtest(df, risk_pct=best_sizing_risk, filter_fn=best_filter_fn, sizing_fn=best_sizing_fn)
    print(f"  Final Capital:  ${combo['final_capital']:.2f}")
    print(f"  Max Drawdown:   {combo['max_dd_pct']:.2f}%")
    print(f"  Trades:         {combo['trades']}")
    print(f"  Win Rate:       {combo['win_rate']:.1f}%")
    print(f"  Sharpe:         {combo['sharpe']:.2f}")
    print(f"\n  vs Baseline:    {combo['final_capital'] - baseline['final_capital']:+.2f} capital")
    print(f"  vs Best Filter: {combo['final_capital'] - best_filter_r['final_capital']:+.2f} capital")
    print(f"  vs Best Sizing: {combo['final_capital'] - best_sizing_r['final_capital']:+.2f} capital")
    print("\nDone.")
