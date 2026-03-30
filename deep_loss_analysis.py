#!/usr/bin/env python3
"""
Deep Analysis of Losing Trades - GER40 TightStop Strategy
"""

import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

# ============================================================================
# LOAD DATA (exact replica from backtest_optimized.py)
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

# Indicators (exact same as backtest)
data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

data["TR"] = np.maximum(data["High"] - data["Low"],
    np.maximum(abs(data["High"] - data["Close"].shift(1)), abs(data["Low"] - data["Close"].shift(1))))
data["ATR14"] = data["TR"].rolling(14).mean()

delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss_s
data["RSI"] = 100 - (100 / (1 + rs))

data["AvgVol"] = data["Volume"].rolling(20).mean()
data["EMA_Bull_Cross"] = (data["EMA9"] > data["EMA21"]) & (data["EMA9"].shift(1) <= data["EMA21"].shift(1))
data["EMA_Bear_Cross"] = (data["EMA9"] < data["EMA21"]) & (data["EMA9"].shift(1) >= data["EMA21"].shift(1))

# ORB
orb_data = data[data["Hour"].isin([7, 8])].groupby("Date").agg(
    ORB_High=("High", "max"), ORB_Low=("Low", "min")).reset_index()
orb_data["ORB_Range"] = orb_data["ORB_High"] - orb_data["ORB_Low"]
data = data.merge(orb_data, on="Date", how="left")

# VWAP
data["TypicalPrice"] = (data["High"] + data["Low"] + data["Close"]) / 3
data["TPxVol"] = data["TypicalPrice"] * data["Volume"]
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

# Additional indicators for analysis
# ADX (14-period)
plus_dm = data["High"].diff()
minus_dm = -data["Low"].diff()
plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
tr14 = data["TR"].rolling(14).sum()
plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
data["ADX"] = dx.rolling(14).mean()

# Bollinger Band width
data["BB_MA"] = data["Close"].rolling(20).mean()
data["BB_STD"] = data["Close"].rolling(20).std()
data["BB_Upper"] = data["BB_MA"] + 2 * data["BB_STD"]
data["BB_Lower"] = data["BB_MA"] - 2 * data["BB_STD"]
data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_MA"] * 100

# Candle body size
data["Body"] = abs(data["Close"] - data["Open"])
data["BodyATRRatio"] = data["Body"] / data["ATR14"]

# Previous candle direction
data["PrevCandleDir"] = np.sign(data["Close"].shift(1) - data["Open"].shift(1))

# Volume relative to 20-period average
data["VolRatio"] = data["Volume"] / data["AvgVol"]

# EMA50 distance %
data["EMA50_Dist"] = (data["Close"] - data["EMA50"]) / data["EMA50"] * 100

# VWAP distance %
data["VWAP_Dist"] = (data["Close"] - data["VWAP"]) / data["VWAP"] * 100

print(f"Loaded {len(data)} candles from {len(files)} files")

# ============================================================================
# BUILD INDEX for fast candle lookup
# ============================================================================
date_hour_idx = {}
for i in range(len(data)):
    key = (data.iloc[i]["Date"], data.iloc[i]["Hour"])
    date_hour_idx[key] = i

# ============================================================================
# RUN BACKTEST WITH FULL CONTEXT RECORDING (TightStop params)
# ============================================================================

PARAMS = {
    "risk_pct": 3.0, "orb_target": 1.5, "orb_stop": 0.3,
    "mom_tp_atr": 2.0, "mom_sl_atr": 1.0, "trail_atr": 1.0,
    "use_orb": True, "use_mom": True, "use_vwap": False,
    "max_daily": 3, "euro_close": 20, "allowed_days": {0, 1, 2, 3}
}

INITIAL_CAPITAL = 200.0
RISK_PERCENT = PARAMS["risk_pct"]
MAX_DAILY_TRADES = PARAMS["max_daily"]
MAX_DAILY_LOSS_PCT = 5.0
ORB_TARGET_MULT = PARAMS["orb_target"]
ORB_STOP_MULT = PARAMS["orb_stop"]
MOM_TP_ATR = PARAMS["mom_tp_atr"]
MOM_SL_ATR = PARAMS["mom_sl_atr"]
TRAIL_ATR_MULT = PARAMS["trail_atr"]
USE_ORB = PARAMS["use_orb"]
USE_MOM = PARAMS["use_mom"]
USE_VWAP = PARAMS["use_vwap"]
ORB_MIN_RANGE = 20
EURO_CLOSE = PARAMS["euro_close"]
ALLOWED_DAYS = PARAMS["allowed_days"]
POINT_VALUE = 1.0
LEVERAGE = 20
MIN_LOT = 0.01

capital = INITIAL_CAPITAL
trade_records = []
current_trade = None
daily_trades = 0
daily_start_equity = INITIAL_CAPITAL
current_date_bt = None
last_trade_idx = 0  # index of last trade entry for "candles since last trade"

# We need to track entry bar index for MAE/MFE calculation
entry_bar_idx = None
entry_context = {}

for i in range(50, len(data)):
    row = data.iloc[i]
    hour = row["Hour"]
    date = row["Date"]
    dow = row["DOW"]

    if date != current_date_bt:
        current_date_bt = date
        daily_trades = 0
        daily_start_equity = capital

    daily_pnl_pct = ((capital - daily_start_equity) / daily_start_equity * 100) if daily_start_equity > 0 else 0
    daily_limit_hit = daily_pnl_pct <= -MAX_DAILY_LOSS_PCT

    # Manage open trade
    if current_trade is not None:
        if hour >= EURO_CLOSE:
            current_trade["exit_price"] = row["Close"]
            current_trade["exit_time"] = row["UTC"]
            current_trade["exit_reason"] = "EOD"
            direction = current_trade["direction"]
            pnl_pts = (current_trade["exit_price"] - current_trade["entry_price"]) * direction
            current_trade["pnl"] = pnl_pts * current_trade["size_lots"] * POINT_VALUE
            # Compute MAE/MFE from candles between entry and exit
            _compute_mae_mfe(current_trade, entry_bar_idx, i, data)
            current_trade["exit_bar_idx"] = i
            capital += current_trade["pnl"]
            trade_records.append(current_trade)
            current_trade = None
            continue

        direction = current_trade["direction"]
        if direction == 1:
            if row["Low"] <= current_trade["stop_loss"]:
                current_trade["exit_price"] = current_trade["stop_loss"]
                current_trade["pnl"] = (current_trade["exit_price"] - current_trade["entry_price"]) * current_trade["size_lots"] * POINT_VALUE
                current_trade["exit_reason"] = "SL"
                current_trade["exit_time"] = row["UTC"]
                _compute_mae_mfe(current_trade, entry_bar_idx, i, data)
                current_trade["exit_bar_idx"] = i
                capital += current_trade["pnl"]
                trade_records.append(current_trade)
                current_trade = None
                continue
            if row["High"] >= current_trade["take_profit"]:
                current_trade["exit_price"] = current_trade["take_profit"]
                current_trade["pnl"] = (current_trade["exit_price"] - current_trade["entry_price"]) * current_trade["size_lots"] * POINT_VALUE
                current_trade["exit_reason"] = "TP"
                current_trade["exit_time"] = row["UTC"]
                _compute_mae_mfe(current_trade, entry_bar_idx, i, data)
                current_trade["exit_bar_idx"] = i
                capital += current_trade["pnl"]
                trade_records.append(current_trade)
                current_trade = None
                continue
            new_trail = row["High"] - row["ATR14"] * TRAIL_ATR_MULT
            if new_trail > current_trade["stop_loss"]:
                current_trade["stop_loss"] = new_trail
        else:
            if row["High"] >= current_trade["stop_loss"]:
                current_trade["exit_price"] = current_trade["stop_loss"]
                current_trade["pnl"] = (current_trade["entry_price"] - current_trade["exit_price"]) * current_trade["size_lots"] * POINT_VALUE
                current_trade["exit_reason"] = "SL"
                current_trade["exit_time"] = row["UTC"]
                _compute_mae_mfe(current_trade, entry_bar_idx, i, data)
                current_trade["exit_bar_idx"] = i
                capital += current_trade["pnl"]
                trade_records.append(current_trade)
                current_trade = None
                continue
            if row["Low"] <= current_trade["take_profit"]:
                current_trade["exit_price"] = current_trade["take_profit"]
                current_trade["pnl"] = (current_trade["entry_price"] - current_trade["exit_price"]) * current_trade["size_lots"] * POINT_VALUE
                current_trade["exit_reason"] = "TP"
                current_trade["exit_time"] = row["UTC"]
                _compute_mae_mfe(current_trade, entry_bar_idx, i, data)
                current_trade["exit_bar_idx"] = i
                capital += current_trade["pnl"]
                trade_records.append(current_trade)
                current_trade = None
                continue
            new_trail = row["Low"] + row["ATR14"] * TRAIL_ATR_MULT
            if new_trail < current_trade["stop_loss"]:
                current_trade["stop_loss"] = new_trail

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

        # Collect context features at entry
        prev_candle_dir = data.iloc[i-1]["Close"] - data.iloc[i-1]["Open"]
        prev_same_dir = (prev_candle_dir > 0 and signal == 1) or (prev_candle_dir < 0 and signal == -1)
        candles_since_last = i - last_trade_idx

        current_trade = {
            "direction": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": row["UTC"],
            "trade_type": trade_type,
            "size_lots": size_lots,
            "exit_price": None,
            "exit_time": None,
            "pnl": 0,
            "exit_reason": "",
            # Context features
            "atr": atr,
            "rsi": row["RSI"],
            "entry_hour": hour,
            "entry_dow": dow,
            "ema50_dist_pct": row["EMA50_Dist"] if not pd.isna(row["EMA50_Dist"]) else 0,
            "vwap_dist_pct": row["VWAP_Dist"] if not pd.isna(row["VWAP_Dist"]) else 0,
            "orb_range": orb_range if not pd.isna(orb_range) else 0,
            "body_atr_ratio": row["BodyATRRatio"] if not pd.isna(row["BodyATRRatio"]) else 0,
            "prev_same_dir": prev_same_dir,
            "candles_since_last": candles_since_last,
            "adx": row["ADX"] if not pd.isna(row["ADX"]) else 0,
            "bb_width": row["BB_Width"] if not pd.isna(row["BB_Width"]) else 0,
            "vol_ratio": row["VolRatio"] if not pd.isna(row["VolRatio"]) else 1.0,
            "stop_distance": stop_distance,
            # MAE/MFE placeholders
            "mae": 0,
            "mfe": 0,
            "entry_bar_idx": i,
        }
        entry_bar_idx = i
        last_trade_idx = i
        daily_trades += 1

# Close any remaining trade
if current_trade is not None:
    last_row = data.iloc[-1]
    current_trade["exit_price"] = last_row["Close"]
    current_trade["exit_time"] = last_row["UTC"]
    current_trade["exit_reason"] = "OPEN"
    direction = current_trade["direction"]
    pnl_pts = (current_trade["exit_price"] - current_trade["entry_price"]) * direction
    current_trade["pnl"] = pnl_pts * current_trade["size_lots"] * POINT_VALUE
    _compute_mae_mfe(current_trade, entry_bar_idx, len(data)-1, data)
    current_trade["exit_bar_idx"] = len(data)-1
    capital += current_trade["pnl"]
    trade_records.append(current_trade)

print(f"Backtest complete: {len(trade_records)} trades, final capital ${capital:.2f}")
