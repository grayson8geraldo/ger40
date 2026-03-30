#!/usr/bin/env python3
"""
GER40 Hourly Data - Advanced Pattern Analysis
Covers: EMA crossovers, RSI, VWAP, Bollinger Bands, Opening Range Breakout,
        Session Gaps, ADX/Trend Strength, Best Indicator Combos
"""

import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
files = sorted(glob.glob('/home/user/ger40/DEU.IDX-EUR_Hour_*.csv'))
dfs = []
for f in files:
    tmp = pd.read_csv(f)
    dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
df['UTC'] = pd.to_datetime(df['UTC'], format='%d.%m.%Y %H:%M:%S.%f UTC')
df.sort_values('UTC', inplace=True)
df.reset_index(drop=True, inplace=True)
df['date'] = df['UTC'].dt.date
df['hour'] = df['UTC'].dt.hour
df['dow'] = df['UTC'].dt.dayofweek  # 0=Mon

print(f"Loaded {len(df)} hourly bars from {df['UTC'].min()} to {df['UTC'].max()}")
print(f"Date range: {df['date'].nunique()} unique trading days")
print("="*80)

# ─────────────────────────────────────────────
# HELPER: Indicator calculations
# ─────────────────────────────────────────────
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df_in, period=14):
    h = df_in['High']
    l = df_in['Low']
    c = df_in['Close'].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df_in, period=14):
    h = df_in['High']
    l = df_in['Low']
    c = df_in['Close']
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val, plus_di, minus_di

def profit_factor(wins, losses):
    total_win = sum(wins) if wins else 0
    total_loss = sum(abs(l) for l in losses) if losses else 0.0001
    return total_win / total_loss if total_loss > 0 else float('inf')

def trade_stats(pnl_list, label=""):
    if not pnl_list:
        print(f"  {label}: No trades found")
        return
    pnl = np.array(pnl_list)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr = len(wins) / len(pnl) * 100
    avg_w = np.mean(wins) if len(wins) > 0 else 0
    avg_l = np.mean(losses) if len(losses) > 0 else 0
    pf = profit_factor(list(wins), list(losses))
    total = np.sum(pnl)
    print(f"  {label}")
    print(f"    Trades: {len(pnl)} | Win Rate: {wr:.1f}% | Avg Win: {avg_w:.1f} | Avg Loss: {avg_l:.1f}")
    print(f"    Profit Factor: {pf:.2f} | Total PnL: {total:.1f} pts | Avg PnL/trade: {np.mean(pnl):.1f}")

# Pre-compute indicators on full series
df['ema9'] = ema(df['Close'], 9)
df['ema21'] = ema(df['Close'], 21)
df['sma20'] = sma(df['Close'], 20)
df['rsi14'] = rsi(df['Close'], 14)
df['atr14'] = atr(df, 14)
df['adx14'], df['plus_di'], df['minus_di'] = adx(df, 14)
df['bb_mid'] = df['sma20']
df['bb_std'] = df['Close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

# ═════════════════════════════════════════════
# 1. EMA(9)/EMA(21) CROSSOVER - INTRADAY ONLY
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("1. EMA(9)/EMA(21) CROSSOVER ANALYSIS - INTRADAY TRADES")
print("="*80)

df['ema_cross_up'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
df['ema_cross_down'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))

long_pnl = []
short_pnl = []

for date, grp in df.groupby('date'):
    grp = grp.sort_values('UTC')
    if len(grp) < 3:
        continue
    last_close = grp['Close'].iloc[-1]
    for i, row in grp.iterrows():
        # Don't enter in last 2 hours
        remaining = grp.loc[grp['UTC'] > row['UTC']]
        if len(remaining) < 1:
            continue
        if row['ema_cross_up']:
            exit_price = last_close
            long_pnl.append(exit_price - row['Close'])
        elif row['ema_cross_down']:
            exit_price = last_close
            short_pnl.append(row['Close'] - exit_price)

trade_stats(long_pnl, "LONG (EMA9 crosses above EMA21, close EOD)")
trade_stats(short_pnl, "SHORT (EMA9 crosses below EMA21, close EOD)")
trade_stats(long_pnl + short_pnl, "COMBINED (all crossover trades)")

# Also test: only take signals during European session 07-15 UTC
long_pnl_eu = []
short_pnl_eu = []
for date, grp in df.groupby('date'):
    grp = grp.sort_values('UTC')
    if len(grp) < 3:
        continue
    last_close = grp['Close'].iloc[-1]
    eu_grp = grp[(grp['hour'] >= 7) & (grp['hour'] <= 15)]
    for i, row in eu_grp.iterrows():
        if row['ema_cross_up']:
            long_pnl_eu.append(last_close - row['Close'])
        elif row['ema_cross_down']:
            short_pnl_eu.append(row['Close'] - last_close)

print("\n  --- Filtered: European Session Only (07-15 UTC) ---")
trade_stats(long_pnl_eu, "LONG (EU session)")
trade_stats(short_pnl_eu, "SHORT (EU session)")
trade_stats(long_pnl_eu + short_pnl_eu, "COMBINED (EU session)")


# ═════════════════════════════════════════════
# 2. RSI(14) OVERSOLD/OVERBOUGHT ANALYSIS
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("2. RSI(14) OVERSOLD/OVERBOUGHT SIGNAL ANALYSIS")
print("="*80)

for label, condition, direction in [
    ("RSI < 30 (Oversold - expect bounce)", df['rsi14'] < 30, 1),
    ("RSI < 25 (Deep Oversold)", df['rsi14'] < 25, 1),
    ("RSI > 70 (Overbought - expect drop)", df['rsi14'] > 70, -1),
    ("RSI > 75 (Deep Overbought)", df['rsi14'] > 75, -1),
]:
    signals = df[condition].index
    print(f"\n  {label}: {len(signals)} signals")
    for n_candles in [1, 2, 3, 5]:
        returns = []
        for idx in signals:
            if idx + n_candles >= len(df):
                continue
            entry = df.loc[idx, 'Close']
            exit_p = df.loc[idx + n_candles, 'Close']
            ret = (exit_p - entry) * direction
            returns.append(ret)
        if returns:
            r = np.array(returns)
            wr = (r > 0).sum() / len(r) * 100
            avg = np.mean(r)
            print(f"    Next {n_candles} candle(s): WR={wr:.1f}% | Avg move={avg:.1f}pts | "
                  f"Median={np.median(r):.1f}pts | N={len(r)}")

# ═════════════════════════════════════════════
# 3. SESSION VWAP MEAN REVERSION
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("3. SESSION VWAP MEAN REVERSION ANALYSIS (from 07:00 UTC)")
print("="*80)

# Calculate session VWAP starting from 07:00 each day
df['vwap'] = np.nan
df['vwap_dev'] = np.nan

for date, grp in df.groupby('date'):
    session = grp[grp['hour'] >= 7].copy()
    if len(session) < 2:
        continue
    typical = (session['High'] + session['Low'] + session['Close']) / 3
    vol = session['Volume'].replace(0, 0.001)
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap_vals = cum_tp_vol / cum_vol
    df.loc[session.index, 'vwap'] = vwap_vals

# Deviation from VWAP in ATR units
df['vwap_atr_dev'] = (df['Close'] - df['vwap']) / df['atr14']

print("\n  Mean Reversion Test: When price is X ATR from VWAP, probability of returning")
print("  (Looking at next 1-5 candles for reversion toward VWAP)\n")

for atr_thresh in [0.5, 1.0, 1.5, 2.0, 2.5]:
    # Price ABOVE VWAP by X ATR
    above = df[df['vwap_atr_dev'] > atr_thresh].index
    below = df[df['vwap_atr_dev'] < -atr_thresh].index

    for label, indices, direction in [
        (f"Price > VWAP + {atr_thresh} ATR (SHORT reversion)", above, -1),
        (f"Price < VWAP - {atr_thresh} ATR (LONG reversion)", below, 1),
    ]:
        reversion_rates = {}
        for n in [1, 3, 5]:
            reverted = 0
            total = 0
            pnl_list = []
            for idx in indices:
                if idx + n >= len(df):
                    continue
                entry = df.loc[idx, 'Close']
                vwap_at_entry = df.loc[idx, 'vwap']
                future_close = df.loc[idx + n, 'Close']
                # Did price move back toward VWAP?
                if direction == -1:  # was above, expect down
                    moved_toward = future_close < entry
                    pnl_list.append(entry - future_close)
                else:  # was below, expect up
                    moved_toward = future_close > entry
                    pnl_list.append(future_close - entry)
                if moved_toward:
                    reverted += 1
                total += 1
            if total > 0:
                reversion_rates[n] = (reverted / total * 100, total, np.mean(pnl_list))

        if reversion_rates:
            parts = [f"{n}bar: {v[0]:.1f}% (N={v[1]}, avg={v[2]:.1f}pts)" for n, v in reversion_rates.items()]
            print(f"  {label}")
            print(f"    Reversion: {' | '.join(parts)}")


# ═════════════════════════════════════════════
# 4. BOLLINGER BAND MEAN REVERSION
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("4. BOLLINGER BAND (20,2) MEAN REVERSION ANALYSIS")
print("="*80)

# Touch/break lower band -> go long
# Touch/break upper band -> go short
for scenario, cond_func, direction, desc in [
    ("Close <= Lower BB (Long)", lambda: df['Close'] <= df['bb_lower'], 1, "LONG after lower band touch"),
    ("Close >= Upper BB (Short)", lambda: df['Close'] >= df['bb_upper'], -1, "SHORT after upper band touch"),
    ("Low <= Lower BB (wick touch, Long)", lambda: df['Low'] <= df['bb_lower'], 1, "LONG after lower wick touch"),
    ("High >= Upper BB (wick touch, Short)", lambda: df['High'] >= df['bb_upper'], -1, "SHORT after upper wick touch"),
]:
    cond = cond_func()
    signals = df[cond].index
    print(f"\n  {scenario}: {len(signals)} signals")
    for n in [1, 2, 3, 5, 10]:
        pnl_list = []
        for idx in signals:
            if idx + n >= len(df):
                continue
            entry = df.loc[idx + 1, 'Close'] if idx + 1 < len(df) else df.loc[idx, 'Close']  # enter next bar open approx
            exit_p = df.loc[idx + n, 'Close']
            pnl = (exit_p - entry) * direction
            pnl_list.append(pnl)
        if pnl_list:
            r = np.array(pnl_list)
            wr = (r > 0).sum() / len(r) * 100
            pf = profit_factor(list(r[r > 0]), list(r[r <= 0]))
            print(f"    Hold {n:2d} bars: WR={wr:.1f}% | AvgPnL={np.mean(r):.1f} | PF={pf:.2f} | N={len(r)}")


# ═════════════════════════════════════════════
# 5. OPENING RANGE BREAKOUT (07-09 UTC)
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("5. OPENING RANGE BREAKOUT ANALYSIS (07-09 UTC)")
print("="*80)

orb_results = {mult: {'long': [], 'short': []} for mult in [1.0, 1.5, 2.0]}

for date, grp in df.groupby('date'):
    grp = grp.sort_values('UTC')
    opening = grp[(grp['hour'] >= 7) & (grp['hour'] < 9)]
    rest = grp[grp['hour'] >= 9]
    if len(opening) < 2 or len(rest) < 1:
        continue
    range_high = opening['High'].max()
    range_low = opening['Low'].min()
    range_size = range_high - range_low
    if range_size < 5:  # skip trivial ranges
        continue

    day_close = grp['Close'].iloc[-1]

    # Check for breakout
    for _, bar in rest.iterrows():
        # Upside breakout
        if bar['High'] > range_high:
            entry = range_high
            for mult in [1.0, 1.5, 2.0]:
                target = entry + mult * range_size
                stop = range_low
                # Check if target or stop hit first
                future = rest[rest['UTC'] >= bar['UTC']]
                hit_target = False
                hit_stop = False
                for _, fb in future.iterrows():
                    if fb['High'] >= target:
                        hit_target = True
                        break
                    if fb['Low'] <= stop:
                        hit_stop = True
                        break
                if hit_target:
                    orb_results[mult]['long'].append(mult * range_size)
                elif hit_stop:
                    orb_results[mult]['long'].append(-(range_size))  # stop = range width
                else:
                    # Close at EOD
                    orb_results[mult]['long'].append(day_close - entry)
            break  # one signal per day

    for _, bar in rest.iterrows():
        # Downside breakout
        if bar['Low'] < range_low:
            entry = range_low
            for mult in [1.0, 1.5, 2.0]:
                target = entry - mult * range_size
                stop = range_high
                future = rest[rest['UTC'] >= bar['UTC']]
                hit_target = False
                hit_stop = False
                for _, fb in future.iterrows():
                    if fb['Low'] <= target:
                        hit_target = True
                        break
                    if fb['High'] >= stop:
                        hit_stop = True
                        break
                if hit_target:
                    orb_results[mult]['short'].append(mult * range_size)
                elif hit_stop:
                    orb_results[mult]['short'].append(-(range_size))
                else:
                    orb_results[mult]['short'].append(entry - day_close)
            break

for mult in [1.0, 1.5, 2.0]:
    print(f"\n  --- Target = {mult}x Range, Stop = Range Low/High (1x range risk) ---")
    trade_stats(orb_results[mult]['long'], f"LONG breakouts ({mult}x target)")
    trade_stats(orb_results[mult]['short'], f"SHORT breakouts ({mult}x target)")
    combined = orb_results[mult]['long'] + orb_results[mult]['short']
    trade_stats(combined, f"COMBINED ({mult}x target)")

# Avg range size
range_sizes = []
for date, grp in df.groupby('date'):
    opening = grp[(grp['hour'] >= 7) & (grp['hour'] < 9)]
    if len(opening) >= 2:
        range_sizes.append(opening['High'].max() - opening['Low'].min())
if range_sizes:
    print(f"\n  Opening Range Stats: Avg={np.mean(range_sizes):.1f}pts | "
          f"Median={np.median(range_sizes):.1f}pts | Std={np.std(range_sizes):.1f}pts")


# ═════════════════════════════════════════════
# 6. SESSION GAP ANALYSIS
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("6. SESSION GAP ANALYSIS (Asian Close vs European Open)")
print("="*80)

gap_data = []
dates = sorted(df['date'].unique())
for date in dates:
    grp = df[df['date'] == date].sort_values('UTC')
    # Asian session: before 07:00 UTC (overnight bars)
    asian = grp[grp['hour'] < 7]
    european = grp[grp['hour'] >= 7]
    if len(asian) == 0 or len(european) == 0:
        continue
    asian_close = asian['Close'].iloc[-1]
    eu_open = european['Open'].iloc[0]
    gap = eu_open - asian_close
    gap_pct = gap / asian_close * 100

    # Does gap fill during European session?
    eu_session = grp[(grp['hour'] >= 7) & (grp['hour'] <= 17)]
    if len(eu_session) == 0:
        continue
    if gap > 0:  # gap up - fill means price goes back to asian_close
        filled = eu_session['Low'].min() <= asian_close
    elif gap < 0:  # gap down
        filled = eu_session['High'].max() >= asian_close
    else:
        filled = True

    # How many hours to fill?
    hours_to_fill = np.nan
    if gap != 0:
        for j, (_, bar) in enumerate(eu_session.iterrows()):
            if gap > 0 and bar['Low'] <= asian_close:
                hours_to_fill = j + 1
                break
            elif gap < 0 and bar['High'] >= asian_close:
                hours_to_fill = j + 1
                break

    gap_data.append({
        'date': date,
        'gap': gap,
        'gap_pct': gap_pct,
        'abs_gap': abs(gap),
        'filled': filled,
        'hours_to_fill': hours_to_fill,
        'direction': 'up' if gap > 0 else ('down' if gap < 0 else 'flat')
    })

gdf = pd.DataFrame(gap_data)
if len(gdf) > 0:
    # Filter out near-zero gaps
    meaningful = gdf[gdf['abs_gap'] > 5]
    print(f"\n  Total gaps analyzed: {len(gdf)} | Meaningful (>5pts): {len(meaningful)}")
    print(f"  Average gap size: {gdf['abs_gap'].mean():.1f} pts ({gdf['gap_pct'].abs().mean():.3f}%)")
    print(f"  Overall gap fill rate: {gdf['filled'].mean()*100:.1f}%")

    if len(meaningful) > 0:
        print(f"\n  Meaningful gaps (>5pts):")
        print(f"    Fill rate: {meaningful['filled'].mean()*100:.1f}%")
        filled_m = meaningful[meaningful['filled']]
        if len(filled_m) > 0:
            print(f"    Avg hours to fill (when filled): {filled_m['hours_to_fill'].mean():.1f}")
        print(f"    Gap UP fill rate: {meaningful[meaningful['direction']=='up']['filled'].mean()*100:.1f}% "
              f"(N={len(meaningful[meaningful['direction']=='up'])})")
        print(f"    Gap DOWN fill rate: {meaningful[meaningful['direction']=='down']['filled'].mean()*100:.1f}% "
              f"(N={len(meaningful[meaningful['direction']=='down'])})")

    # By gap size buckets
    print("\n  Gap Fill Rate by Size:")
    for lo, hi in [(5, 20), (20, 50), (50, 100), (100, 500)]:
        bucket = gdf[(gdf['abs_gap'] >= lo) & (gdf['abs_gap'] < hi)]
        if len(bucket) > 0:
            fr = bucket['filled'].mean() * 100
            avg_h = bucket[bucket['filled']]['hours_to_fill'].mean() if bucket['filled'].any() else np.nan
            print(f"    {lo:3d}-{hi:3d} pts: Fill={fr:.1f}% | N={len(bucket)} | Avg hours={avg_h:.1f}")

    # Trading the gap fade
    print("\n  Gap Fade Trade (enter at EU open, target = gap fill, stop = 1x gap):")
    fade_pnl = []
    for _, row in meaningful.iterrows():
        if row['filled']:
            fade_pnl.append(row['abs_gap'])  # won the gap fill
        else:
            fade_pnl.append(-row['abs_gap'])  # stopped out
    trade_stats(fade_pnl, "Gap fade trades (meaningful gaps)")


# ═════════════════════════════════════════════
# 7. ADX / TREND STRENGTH ANALYSIS
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("7. TREND STRENGTH (ADX) ANALYSIS")
print("="*80)

# Classify market regime
df['regime'] = 'ranging'
df.loc[df['adx14'] > 25, 'regime'] = 'trending'
df.loc[df['adx14'] > 40, 'regime'] = 'strong_trend'

trending = df[df['regime'] == 'trending']
strong = df[df['regime'] == 'strong_trend']
ranging = df[df['regime'] == 'ranging']

print(f"\n  Market Regime Distribution:")
print(f"    Ranging (ADX < 25): {len(ranging)} bars ({len(ranging)/len(df)*100:.1f}%)")
print(f"    Trending (ADX 25-40): {len(trending)} bars ({len(trending)/len(df)*100:.1f}%)")
print(f"    Strong Trend (ADX > 40): {len(strong)} bars ({len(strong)/len(df)*100:.1f}%)")

# Mean Reversion (BB) performance in different regimes
print("\n  Bollinger Band Mean Reversion by Regime (enter after band touch, hold 3 bars):")
for regime_name, regime_mask in [('Ranging (ADX<25)', df['adx14'] < 25),
                                   ('Trending (25<ADX<40)', (df['adx14'] >= 25) & (df['adx14'] < 40)),
                                   ('Strong Trend (ADX>40)', df['adx14'] >= 40)]:
    # Lower band long
    cond = regime_mask & (df['Close'] <= df['bb_lower'])
    signals = df[cond].index
    pnl = []
    for idx in signals:
        if idx + 4 >= len(df):
            continue
        entry = df.loc[idx + 1, 'Close']
        exit_p = df.loc[idx + 3, 'Close']
        pnl.append(exit_p - entry)
    wr = (np.array(pnl) > 0).sum() / max(len(pnl), 1) * 100
    avg_pnl = np.mean(pnl) if pnl else 0
    print(f"    {regime_name}: BB Lower Long -> WR={wr:.1f}% | Avg={avg_pnl:.1f}pts | N={len(pnl)}")

# Trend Following (EMA crossover) in different regimes
print("\n  EMA Crossover Trend Following by Regime (hold 5 bars):")
for regime_name, regime_mask in [('Ranging (ADX<25)', df['adx14'] < 25),
                                   ('Trending (25<ADX<40)', (df['adx14'] >= 25) & (df['adx14'] < 40)),
                                   ('Strong Trend (ADX>40)', df['adx14'] >= 40)]:
    cond = regime_mask & df['ema_cross_up']
    signals = df[cond].index
    pnl = []
    for idx in signals:
        if idx + 5 >= len(df):
            continue
        entry = df.loc[idx, 'Close']
        exit_p = df.loc[idx + 5, 'Close']
        pnl.append(exit_p - entry)
    wr = (np.array(pnl) > 0).sum() / max(len(pnl), 1) * 100
    avg_pnl = np.mean(pnl) if pnl else 0
    print(f"    {regime_name}: EMA Cross Long -> WR={wr:.1f}% | Avg={avg_pnl:.1f}pts | N={len(pnl)}")


# ═════════════════════════════════════════════
# 8. BEST INDICATOR COMBINATIONS
# ═════════════════════════════════════════════
print("\n" + "="*80)
print("8. INDICATOR COMBINATION ANALYSIS")
print("="*80)

combo_results = []

# Combo 1: RSI oversold + BB lower band touch (LONG)
print("\n  Combo 1: RSI<30 + Close <= Lower BB -> LONG")
cond = (df['rsi14'] < 30) & (df['Close'] <= df['bb_lower'])
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx + 1, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(exit_p - entry)
trade_stats(pnl, "RSI<30 + BB Lower (hold 5 bars)")
if pnl:
    combo_results.append(("RSI<30 + BB Lower", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 2: RSI overbought + BB upper band touch (SHORT)
print("\n  Combo 2: RSI>70 + Close >= Upper BB -> SHORT")
cond = (df['rsi14'] > 70) & (df['Close'] >= df['bb_upper'])
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx + 1, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(entry - exit_p)
trade_stats(pnl, "RSI>70 + BB Upper (hold 5 bars)")
if pnl:
    combo_results.append(("RSI>70 + BB Upper", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 3: EMA cross up + ADX trending + RSI not overbought
print("\n  Combo 3: EMA9 cross up + ADX>25 + RSI<65 -> LONG")
cond = df['ema_cross_up'] & (df['adx14'] > 25) & (df['rsi14'] < 65)
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(exit_p - entry)
trade_stats(pnl, "EMA cross + Trending + RSI filter (hold 5)")
if pnl:
    combo_results.append(("EMA+ADX>25+RSI<65", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 4: VWAP reversion + RSI oversold
print("\n  Combo 4: Price < VWAP - 1.5 ATR + RSI<40 -> LONG (mean reversion)")
cond = (df['vwap_atr_dev'] < -1.5) & (df['rsi14'] < 40)
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx + 1, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(exit_p - entry)
trade_stats(pnl, "VWAP deviation + RSI<40 (hold 5)")
if pnl:
    combo_results.append(("VWAP-1.5ATR+RSI<40", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 5: BB lower + ADX<25 (ranging) -> mean reversion long
print("\n  Combo 5: BB Lower touch + ADX<25 (ranging market) -> LONG")
cond = (df['Close'] <= df['bb_lower']) & (df['adx14'] < 25)
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx + 1, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(exit_p - entry)
trade_stats(pnl, "BB Lower + Ranging (hold 5)")
if pnl:
    combo_results.append(("BB Lower+Ranging", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 6: EMA cross down + RSI>60 + ADX>20 (SHORT)
print("\n  Combo 6: EMA9 cross down + ADX>20 + RSI>55 -> SHORT")
cond = df['ema_cross_down'] & (df['adx14'] > 20) & (df['rsi14'] > 55)
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(entry - exit_p)
trade_stats(pnl, "EMA cross down + ADX>20 + RSI>55 (hold 5)")
if pnl:
    combo_results.append(("EMA down+ADX>20+RSI>55", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# Combo 7: VWAP + BB + EU session
print("\n  Combo 7: Price > VWAP + 1.5 ATR + Close >= Upper BB + EU session -> SHORT")
cond = (df['vwap_atr_dev'] > 1.5) & (df['Close'] >= df['bb_upper']) & (df['hour'] >= 7) & (df['hour'] <= 15)
signals = df[cond].index
pnl = []
for idx in signals:
    if idx + 5 >= len(df):
        continue
    entry = df.loc[idx + 1, 'Close']
    exit_p = df.loc[idx + 5, 'Close']
    pnl.append(entry - exit_p)
trade_stats(pnl, "VWAP+BB Upper+EU session SHORT (hold 5)")
if pnl:
    combo_results.append(("VWAP+BB Upper+EU", np.mean(pnl), len(pnl), (np.array(pnl) > 0).sum() / len(pnl) * 100))

# ─────────── RANKING ───────────
print("\n" + "="*80)
print("RANKING: BEST INDICATOR COMBINATIONS (sorted by Avg PnL/trade)")
print("="*80)
combo_results.sort(key=lambda x: x[1], reverse=True)
for i, (name, avg_pnl, n, wr) in enumerate(combo_results, 1):
    edge = "***" if avg_pnl > 10 else "**" if avg_pnl > 5 else "*" if avg_pnl > 0 else ""
    print(f"  {i}. {name:30s} | AvgPnL={avg_pnl:7.1f} pts | WR={wr:.1f}% | N={n:4d} {edge}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
