#!/usr/bin/env python3
"""
GER40 Missed Opportunity Analysis
Identifies profitable signals NOT captured by current ORB + Momentum strategy.
"""

import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("=" * 80)
print("GER40 MISSED OPPORTUNITY ANALYSIS")
print("=" * 80)

files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
dfs = []
for f in files:
    tmp = pd.read_csv(f, parse_dates=['UTC'], dayfirst=True)
    if len(tmp) > 0:
        dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
df.sort_values('UTC', inplace=True)
df.reset_index(drop=True, inplace=True)
df['UTC'] = pd.to_datetime(df['UTC'], utc=True)

# Drop rows with missing OHLC
df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract time features
df['date'] = df['UTC'].dt.date
df['hour'] = df['UTC'].dt.hour
df['weekday'] = df['UTC'].dt.weekday  # 0=Mon

print(f"Loaded {len(df)} hourly bars from {df['UTC'].min()} to {df['UTC'].max()}")
print(f"Trading days: {df['date'].nunique()}")
print()

# ============================================================
# 2. CALCULATE INDICATORS
# ============================================================
print("Calculating indicators...")

# EMAs
df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

# ATR(14)
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    )
)
df['ATR14'] = df['TR'].rolling(14).mean()

# RSI(14)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df['RSI14'] = 100 - (100 / (1 + rs))

# Bollinger Bands(20, 2)
df['BB_mid'] = df['Close'].rolling(20).mean()
df['BB_std'] = df['Close'].rolling(20).std()
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']

# ADX(14)
plus_dm = df['High'].diff()
minus_dm = -df['Low'].diff()
plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
minus_dm_arr = np.where((minus_dm > plus_dm.astype(float)) & (minus_dm > 0), minus_dm, 0.0)

# Smooth with Wilder's method
atr_s = df['TR'].ewm(alpha=1/14, min_periods=14, adjust=False).mean()
plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_s
minus_di = 100 * pd.Series(minus_dm_arr).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_s
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
df['ADX14'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
df['plus_di'] = plus_di
df['minus_di'] = minus_di

# Session VWAP (reset at 07:00 UTC)
df['session_id'] = ((df['hour'] == 7) & (df['hour'].shift(1) != 7)).cumsum()
# Fallback: also reset on date change if no 07:00 bar
df['session_id'] = df.groupby('date').ngroup()  # simpler: daily reset
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
df['cum_tp_vol'] = 0.0
df['cum_vol'] = 0.0

# Compute VWAP per session (reset at 07:00)
session_starts = (df['hour'] >= 7)
df['session_day'] = df['date'].astype(str) + '_' + session_starts.astype(str)

# Simple daily VWAP from 07:00
vol = df['Volume'].replace(0, 0.001)  # avoid div by zero
tp_vol = typical_price * vol
groups = df.groupby('date')
df['VWAP'] = tp_vol.groupby(df['date']).cumsum() / vol.groupby(df['date']).cumsum()

# Volume average (20-bar)
df['Vol_MA20'] = df['Volume'].rolling(20).mean()

print("Indicators calculated.\n")

# ============================================================
# 3. BUILD ORB REFERENCE (07:00-08:00 UTC)
# ============================================================
orb_hours = df[df['hour'].isin([7, 8])].copy()
orb_range = orb_hours.groupby('date').agg(
    ORB_high=('High', 'max'),
    ORB_low=('Low', 'min'),
    ORB_open=('Open', 'first')
).reset_index()
orb_range['ORB_size'] = orb_range['ORB_high'] - orb_range['ORB_low']
df = df.merge(orb_range, on='date', how='left')

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def calc_forward_returns(df_signals, df_all, hold_periods=[1, 2, 3, 4, 5]):
    """Calculate forward returns for signal rows."""
    results = {}
    for h in hold_periods:
        # Entry at signal bar close, exit at close h bars later
        exits = []
        for idx in df_signals.index:
            pos = df_all.index.get_loc(idx)
            if pos + h < len(df_all):
                exits.append(df_all.iloc[pos + h]['Close'])
            else:
                exits.append(np.nan)
        ret = pd.Series(exits, index=df_signals.index) - df_signals['Close']
        results[h] = ret
    return results


def summarize_signal(name, signals_df, df_all, direction='long', hold_periods=[1, 2, 3, 5]):
    """Print summary stats for a signal."""
    n = len(signals_df)
    if n == 0:
        print(f"\n--- {name} ---")
        print(f"  Signals: 0  (insufficient data)")
        return None

    fwd = calc_forward_returns(signals_df, df_all, hold_periods)

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Total signals: {n}")

    best_pf = 0
    best_hold = None
    best_wr = 0

    for h in hold_periods:
        rets = fwd[h].dropna()
        if direction == 'short':
            rets = -rets
        if len(rets) == 0:
            continue
        wins = (rets > 0).sum()
        losses = (rets <= 0).sum()
        wr = wins / len(rets) * 100
        avg_win = rets[rets > 0].mean() if wins > 0 else 0
        avg_loss = abs(rets[rets <= 0].mean()) if losses > 0 else 0.001
        pf = (rets[rets > 0].sum()) / max(abs(rets[rets <= 0].sum()), 0.01)
        avg_ret = rets.mean()
        total = rets.sum()

        print(f"  Hold {h}h: WR={wr:.1f}% | Avg P/L={avg_ret:+.1f}pts | "
              f"AvgWin={avg_win:.1f} AvgLoss={avg_loss:.1f} | PF={pf:.2f} | "
              f"Total={total:+.0f}pts | N={len(rets)}")

        if pf > best_pf:
            best_pf = pf
            best_hold = h
            best_wr = wr

    return {'name': name, 'signals': n, 'best_pf': best_pf, 'best_hold': best_hold, 'best_wr': best_wr}


# ============================================================
# 5. SIGNAL ANALYSIS
# ============================================================
results_summary = []

# ----------------------------------------------------------
# SIGNAL 1: RSI BULLISH DIVERGENCE (European session)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 1: RSI BULLISH DIVERGENCE")
print("#" * 80)
print("Looking for: Price makes new low but RSI doesn't (within European session 08-16 UTC)")

eu_session = df[(df['hour'] >= 8) & (df['hour'] <= 16)].copy()
div_signals = []

lookback = 10  # candles to look back for divergence
for i in range(lookback, len(eu_session)):
    idx = eu_session.index[i]
    row = eu_session.iloc[i]
    window = eu_session.iloc[i-lookback:i+1]

    # Current bar makes a new low relative to window
    if row['Low'] <= window['Low'].min() and pd.notna(row['RSI14']):
        # But RSI is NOT at its lowest
        rsi_min_idx = window['RSI14'].idxmin()
        if pd.notna(rsi_min_idx) and row['RSI14'] > window['RSI14'].min() + 2:
            div_signals.append(idx)

rsi_div_df = df.loc[div_signals].copy()
# Remove duplicate signals on same day
rsi_div_df = rsi_div_df.drop_duplicates(subset='date', keep='first')
r = summarize_signal("RSI Bullish Divergence (EU Session)", rsi_div_df, df, 'long')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 2: FAILED ORB BREAKOUT REVERSAL (Trap Trade)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 2: FAILED ORB BREAKOUT REVERSAL")
print("#" * 80)
print("Looking for: ORB breaks one direction, then reverses to break other side")

trap_signals = []
trap_directions = []

for date, grp in df.groupby('date'):
    orb_h = grp.iloc[0]['ORB_high'] if pd.notna(grp.iloc[0].get('ORB_high', np.nan)) else None
    orb_l = grp.iloc[0]['ORB_low'] if pd.notna(grp.iloc[0].get('ORB_low', np.nan)) else None
    if orb_h is None or orb_l is None:
        continue

    post_orb = grp[(grp['hour'] >= 9) & (grp['hour'] <= 17)]
    broke_high = False
    broke_low = False

    for i in range(len(post_orb)):
        row = post_orb.iloc[i]
        idx = post_orb.index[i]

        if not broke_high and not broke_low:
            if row['High'] > orb_h:
                broke_high = True
            elif row['Low'] < orb_l:
                broke_low = True
        elif broke_high and not broke_low:
            # Broke high first, now check if it reverses to break low
            if row['Low'] < orb_l:
                trap_signals.append(idx)
                trap_directions.append('short')  # failed long, go short
                break
        elif broke_low and not broke_high:
            # Broke low first, now check if it reverses to break high
            if row['High'] > orb_h:
                trap_signals.append(idx)
                trap_directions.append('long')  # failed short, go long
                break

trap_df = df.loc[trap_signals].copy()
trap_df['trap_dir'] = trap_directions

# Separate long and short
trap_long = trap_df[trap_df['trap_dir'] == 'long']
trap_short = trap_df[trap_df['trap_dir'] == 'short']

r = summarize_signal("Failed ORB Reversal - LONG (broke low then high)", trap_long, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Failed ORB Reversal - SHORT (broke high then low)", trap_short, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 3: SECOND ORB BREAKOUT (Re-entry after pullback)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 3: SECOND ORB BREAKOUT")
print("#" * 80)
print("Looking for: Price breaks ORB, pulls back into range, then breaks out again")

reentry_signals = []
reentry_dirs = []

for date, grp in df.groupby('date'):
    orb_h = grp.iloc[0]['ORB_high'] if pd.notna(grp.iloc[0].get('ORB_high', np.nan)) else None
    orb_l = grp.iloc[0]['ORB_low'] if pd.notna(grp.iloc[0].get('ORB_low', np.nan)) else None
    if orb_h is None or orb_l is None:
        continue

    post_orb = grp[(grp['hour'] >= 9) & (grp['hour'] <= 17)]
    state = 'waiting'  # waiting -> broke_up/broke_down -> pullback -> rebreak

    for i in range(len(post_orb)):
        row = post_orb.iloc[i]
        idx = post_orb.index[i]

        if state == 'waiting':
            if row['Close'] > orb_h:
                state = 'broke_up'
            elif row['Close'] < orb_l:
                state = 'broke_down'
        elif state == 'broke_up':
            if row['Close'] < orb_h and row['Close'] > orb_l:
                state = 'pullback_up'
        elif state == 'broke_down':
            if row['Close'] > orb_l and row['Close'] < orb_h:
                state = 'pullback_down'
        elif state == 'pullback_up':
            if row['Close'] > orb_h:
                reentry_signals.append(idx)
                reentry_dirs.append('long')
                break
        elif state == 'pullback_down':
            if row['Close'] < orb_l:
                reentry_signals.append(idx)
                reentry_dirs.append('short')
                break

reentry_df = df.loc[reentry_signals].copy()
reentry_df['dir'] = reentry_dirs

re_long = reentry_df[reentry_df['dir'] == 'long']
re_short = reentry_df[reentry_df['dir'] == 'short']

r = summarize_signal("Second ORB Breakout - LONG", re_long, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Second ORB Breakout - SHORT", re_short, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 4: EMA BOUNCE (Trend Pullback)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 4: EMA BOUNCE IN ESTABLISHED TREND")
print("#" * 80)
print("Looking for: EMA9 > EMA21 > EMA50, price pulls back to EMA21, bounces")

ema_bounce_signals = []

for i in range(2, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if pd.isna(row['EMA50']) or pd.isna(row['ATR14']):
        continue

    # Established uptrend
    if row['EMA9'] > row['EMA21'] > row['EMA50']:
        # Price touched or came within 0.3*ATR of EMA21
        touch_dist = abs(row['Low'] - row['EMA21'])
        if touch_dist < 0.3 * row['ATR14']:
            # Bounced: close above EMA21
            if row['Close'] > row['EMA21']:
                # Previous bar showed pullback (close was closer to EMA21 or low touched)
                if prev['Low'] <= prev['EMA21'] * 1.002:  # within 0.2%
                    ema_bounce_signals.append(df.index[i])

ema_bounce_df = df.loc[ema_bounce_signals].drop_duplicates(subset='date', keep='first')
r = summarize_signal("EMA21 Bounce in Uptrend (Long)", ema_bounce_df, df, 'long')
if r: results_summary.append(r)

# Also test EMA bounce short (downtrend)
ema_bounce_short = []
for i in range(2, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if pd.isna(row['EMA50']) or pd.isna(row['ATR14']):
        continue
    if row['EMA9'] < row['EMA21'] < row['EMA50']:
        touch_dist = abs(row['High'] - row['EMA21'])
        if touch_dist < 0.3 * row['ATR14']:
            if row['Close'] < row['EMA21']:
                if prev['High'] >= prev['EMA21'] * 0.998:
                    ema_bounce_short.append(df.index[i])

ema_bounce_short_df = df.loc[ema_bounce_short].drop_duplicates(subset='date', keep='first')
r = summarize_signal("EMA21 Bounce in Downtrend (Short)", ema_bounce_short_df, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 5: LUNCH BREAKOUT (12:00-13:00 consolidation, 13:00-14:00 breakout)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 5: LUNCH BREAKOUT (12-13 UTC range, 13-14 UTC breakout)")
print("#" * 80)

lunch_long = []
lunch_short = []

for date, grp in df.groupby('date'):
    lunch = grp[grp['hour'].isin([12])]
    breakout_window = grp[grp['hour'].isin([13, 14])]

    if len(lunch) == 0 or len(breakout_window) == 0:
        continue

    lunch_high = lunch['High'].max()
    lunch_low = lunch['Low'].min()
    lunch_range = lunch_high - lunch_low

    if lunch_range < 5:  # skip tiny ranges
        continue

    for i in range(len(breakout_window)):
        row = breakout_window.iloc[i]
        idx = breakout_window.index[i]
        if row['Close'] > lunch_high:
            lunch_long.append(idx)
            break
        elif row['Close'] < lunch_low:
            lunch_short.append(idx)
            break

lunch_long_df = df.loc[lunch_long]
lunch_short_df = df.loc[lunch_short]

r = summarize_signal("Lunch Breakout LONG (12-13 range, break up)", lunch_long_df, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Lunch Breakout SHORT (12-13 range, break down)", lunch_short_df, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 6: POWER HOUR (17:00-19:00 UTC) Trend Continuation
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 6: POWER HOUR (17:00-19:00 UTC)")
print("#" * 80)
print("Testing trend continuation in last 2 hours of active session")

power_signals_long = []
power_signals_short = []

power = df[(df['hour'] >= 17) & (df['hour'] <= 18)].copy()

for i in range(len(power)):
    idx = power.index[i]
    pos = df.index.get_loc(idx)
    if pos < 5:
        continue
    row = power.iloc[i]

    if pd.isna(row['EMA9']) or pd.isna(row['EMA21']):
        continue

    # Trend from earlier in day
    day_bars = df[(df['date'] == row['date']) & (df['hour'] < 17)]
    if len(day_bars) < 3:
        continue

    day_move = day_bars.iloc[-1]['Close'] - day_bars.iloc[0]['Open']

    # Trend continuation: if day was up and close > EMA9 > EMA21
    if day_move > 0 and row['Close'] > row['EMA9'] and row['hour'] == 17:
        power_signals_long.append(idx)
    elif day_move < 0 and row['Close'] < row['EMA9'] and row['hour'] == 17:
        power_signals_short.append(idx)

power_long_df = df.loc[power_signals_long]
power_short_df = df.loc[power_signals_short]

r = summarize_signal("Power Hour Trend Continuation LONG", power_long_df, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Power Hour Trend Continuation SHORT", power_short_df, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 7: GAP FADE AT EUROPEAN OPEN
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 7: GAP FADE AT EUROPEAN OPEN")
print("#" * 80)
print("Gap > 30pts between prev day 20:00 close and current day ~07:00 open")

gap_fade_long = []  # gap down, fade = buy
gap_fade_short = []  # gap up, fade = sell

dates = sorted(df['date'].unique())

for i in range(1, len(dates)):
    prev_date = dates[i - 1]
    curr_date = dates[i]

    # Previous day close around 20:00
    prev_bars = df[(df['date'] == prev_date) & (df['hour'] >= 19) & (df['hour'] <= 21)]
    if len(prev_bars) == 0:
        prev_bars = df[(df['date'] == prev_date)]
        if len(prev_bars) == 0:
            continue
    prev_close = prev_bars.iloc[-1]['Close']

    # Current day open around 07:00-08:00
    curr_open_bars = df[(df['date'] == curr_date) & (df['hour'] >= 7) & (df['hour'] <= 8)]
    if len(curr_open_bars) == 0:
        continue
    curr_open = curr_open_bars.iloc[0]['Open']
    entry_idx = curr_open_bars.index[0]

    gap = curr_open - prev_close

    if gap > 30:
        gap_fade_short.append(entry_idx)
    elif gap < -30:
        gap_fade_long.append(entry_idx)

gap_long_df = df.loc[gap_fade_long]
gap_short_df = df.loc[gap_fade_short]

r = summarize_signal("Gap Fade LONG (gap down > 30pts, buy the fill)", gap_long_df, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Gap Fade SHORT (gap up > 30pts, sell the fill)", gap_short_df, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 8: ADX THRESHOLD FILTER ON ORB BREAKOUTS
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 8: ADX THRESHOLD AS ORB FILTER")
print("#" * 80)
print("Testing: Does requiring ADX > 20/25/30 improve ORB breakout performance?")

# First build baseline ORB breakout signals
orb_break_signals = []
orb_break_dirs = []

for date, grp in df.groupby('date'):
    orb_h = grp.iloc[0]['ORB_high'] if pd.notna(grp.iloc[0].get('ORB_high', np.nan)) else None
    orb_l = grp.iloc[0]['ORB_low'] if pd.notna(grp.iloc[0].get('ORB_low', np.nan)) else None
    if orb_h is None or orb_l is None:
        continue

    post_orb = grp[(grp['hour'] >= 9) & (grp['hour'] <= 12)]
    for i in range(len(post_orb)):
        row = post_orb.iloc[i]
        idx = post_orb.index[i]
        if row['Close'] > orb_h:
            orb_break_signals.append(idx)
            orb_break_dirs.append('long')
            break
        elif row['Close'] < orb_l:
            orb_break_signals.append(idx)
            orb_break_dirs.append('short')
            break

orb_df = df.loc[orb_break_signals].copy()
orb_df['dir'] = orb_break_dirs

print(f"\n  Baseline ORB breakouts: {len(orb_df)}")
r = summarize_signal("ORB Breakout Baseline (no ADX filter)", orb_df, df, 'long')

for adx_thresh in [20, 25, 30]:
    filtered = orb_df[orb_df['ADX14'] > adx_thresh]
    print(f"\n  --- ADX > {adx_thresh} filter ---")
    r = summarize_signal(f"ORB Breakout + ADX > {adx_thresh}", filtered, df, 'long')
    if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 9: VOLUME SPIKE ENTRY
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 9: VOLUME SPIKE AT KEY LEVELS")
print("#" * 80)
print("Testing: Volume > 2x average at ORB break or EMA cross")

# Volume spike at ORB break
vol_orb_signals = []
for idx in orb_break_signals:
    row = df.loc[idx]
    if pd.notna(row['Vol_MA20']) and row['Vol_MA20'] > 0:
        if row['Volume'] > 2 * row['Vol_MA20']:
            vol_orb_signals.append(idx)

vol_orb_df = df.loc[vol_orb_signals]
print(f"\n  ORB breaks with volume spike: {len(vol_orb_df)} / {len(orb_break_signals)} total ORB breaks")
r = summarize_signal("ORB Break + Volume > 2x Average", vol_orb_df, df, 'long')
if r: results_summary.append(r)

# Volume spike at any point during EU session as standalone signal
vol_spike_long = []
vol_spike_short = []
eu = df[(df['hour'] >= 8) & (df['hour'] <= 16)].copy()

for i in range(1, len(eu)):
    row = eu.iloc[i]
    idx = eu.index[i]
    if pd.isna(row['Vol_MA20']) or row['Vol_MA20'] <= 0:
        continue
    if row['Volume'] > 2 * row['Vol_MA20']:
        # Direction based on candle
        if row['Close'] > row['Open']:
            vol_spike_long.append(idx)
        elif row['Close'] < row['Open']:
            vol_spike_short.append(idx)

vol_long_df = df.loc[vol_spike_long].drop_duplicates(subset='date', keep='first')
vol_short_df = df.loc[vol_spike_short].drop_duplicates(subset='date', keep='first')

r = summarize_signal("Volume Spike Bullish Candle (EU Session)", vol_long_df, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("Volume Spike Bearish Candle (EU Session)", vol_short_df, df, 'short')
if r: results_summary.append(r)

# ----------------------------------------------------------
# SIGNAL 10: MULTI-TIMEFRAME (4H trend as filter)
# ----------------------------------------------------------
print("\n" + "#" * 80)
print("# SIGNAL 10: MULTI-TIMEFRAME (4H Trend Filter)")
print("#" * 80)
print("Using 4H EMA alignment as filter for 1H entries")

# Build 4H bars
df_4h = df.set_index('UTC').resample('4h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna().reset_index()
df_4h['EMA9_4h'] = df_4h['Close'].ewm(span=9, adjust=False).mean()
df_4h['EMA21_4h'] = df_4h['Close'].ewm(span=21, adjust=False).mean()
df_4h['trend_4h'] = np.where(df_4h['EMA9_4h'] > df_4h['EMA21_4h'], 1,
                              np.where(df_4h['EMA9_4h'] < df_4h['EMA21_4h'], -1, 0))

# Map 4H trend to 1H bars
df_4h_indexed = df_4h.set_index('UTC')
df['trend_4h'] = np.nan
for i in range(len(df)):
    t = df.iloc[i]['UTC']
    # Find the most recent completed 4H bar
    mask = df_4h_indexed.index <= t
    if mask.any():
        df.iloc[i, df.columns.get_loc('trend_4h')] = df_4h_indexed.loc[mask].iloc[-1]['trend_4h']

# ORB breakouts aligned with 4H trend
orb_df_with_4h = orb_df.copy()
orb_df_with_4h['trend_4h'] = df.loc[orb_df.index, 'trend_4h']
orb_df_with_4h['dir_num'] = orb_df_with_4h['dir'].map({'long': 1, 'short': -1})

aligned = orb_df_with_4h[orb_df_with_4h['trend_4h'] == orb_df_with_4h['dir_num']]
misaligned = orb_df_with_4h[orb_df_with_4h['trend_4h'] != orb_df_with_4h['dir_num']]

print(f"\n  ORB breakouts aligned with 4H trend: {len(aligned)}")
print(f"  ORB breakouts against 4H trend: {len(misaligned)}")

r = summarize_signal("ORB Breakout ALIGNED with 4H Trend", aligned, df, 'long')
if r: results_summary.append(r)
r = summarize_signal("ORB Breakout AGAINST 4H Trend", misaligned, df, 'long')
if r: results_summary.append(r)


# ============================================================
# 6. FINAL SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n\n" + "=" * 80)
print("=" * 80)
print("  FINAL SUMMARY: MISSED OPPORTUNITY RANKING")
print("=" * 80)
print("=" * 80)

# Sort by profit factor
valid_results = [r for r in results_summary if r is not None and r['best_pf'] > 0]
valid_results.sort(key=lambda x: x['best_pf'], reverse=True)

print(f"\n{'Rank':<5} {'Signal':<50} {'N':>5} {'Best WR':>8} {'Best PF':>8} {'Hold':>5}")
print("-" * 85)

for i, r in enumerate(valid_results):
    add_value = "YES" if r['best_pf'] > 1.2 and r['signals'] >= 15 else "MAYBE" if r['best_pf'] > 1.0 else "NO"
    print(f"{i+1:<5} {r['name'][:50]:<50} {r['signals']:>5} {r['best_wr']:>7.1f}% {r['best_pf']:>8.2f} {r['best_hold'] or 'N/A':>5}")

print()
print("=" * 80)
print("  RECOMMENDATIONS FOR STRATEGY ENHANCEMENT")
print("=" * 80)

print("\n  TIER 1 - HIGH CONFIDENCE ADDITIONS (PF > 1.5, sufficient samples):")
tier1 = [r for r in valid_results if r['best_pf'] > 1.5 and r['signals'] >= 15]
if tier1:
    for r in tier1:
        print(f"    * {r['name']} (PF={r['best_pf']:.2f}, WR={r['best_wr']:.1f}%, N={r['signals']})")
else:
    print("    (none met criteria)")

print("\n  TIER 2 - PROMISING ADDITIONS (PF > 1.2, sufficient samples):")
tier2 = [r for r in valid_results if 1.2 < r['best_pf'] <= 1.5 and r['signals'] >= 15]
if tier2:
    for r in tier2:
        print(f"    * {r['name']} (PF={r['best_pf']:.2f}, WR={r['best_wr']:.1f}%, N={r['signals']})")
else:
    print("    (none met criteria)")

print("\n  TIER 3 - NEEDS MORE DATA / MARGINAL:")
tier3 = [r for r in valid_results if r['best_pf'] <= 1.2 or r['signals'] < 15]
if tier3:
    for r in tier3:
        print(f"    * {r['name']} (PF={r['best_pf']:.2f}, WR={r['best_wr']:.1f}%, N={r['signals']})")
else:
    print("    (none)")

print("\n  SIGNALS THAT DO NOT ADD VALUE (PF < 1.0):")
losers = [r for r in valid_results if r['best_pf'] < 1.0]
if losers:
    for r in losers:
        print(f"    * {r['name']} (PF={r['best_pf']:.2f}, AVOID)")
else:
    print("    (none - all signals had PF >= 1.0)")

print("\n" + "=" * 80)
print("  END OF ANALYSIS")
print("=" * 80)
