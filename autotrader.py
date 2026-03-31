#!/usr/bin/env python3
"""
GER40 DayTrader v5.0 — Automated Paper Trading Bot
===================================================
Runs the strategy on virtual balance ($200) using real-time market data.
Fetches hourly candles from Yahoo Finance, applies strategy logic,
and manages a simulated portfolio.

Usage:
  python3 autotrader.py              # Run in live mode
  python3 autotrader.py --status     # Show current status
  python3 autotrader.py --reset      # Reset virtual balance to $200
  python3 autotrader.py --history    # Show trade history

State is saved to autotrader_state.json between restarts.
Trade log saved to autotrader_trades.xlsx
"""

import json
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "autotrader_state.json")
TRADES_FILE = os.path.join(SCRIPT_DIR, "autotrader_trades.xlsx")

# Strategy parameters (v5.0 Balanced profile)
CONFIG = {
    "initial_capital": 200.0,
    "risk_pct": 8.0,
    "orb_stop_mult": 0.25,
    "orb_target_mult": 1.5,
    "trail_atr_mult": 0.5,
    "mom_sl_atr": 1.0,
    "mom_tp_atr": 2.0,
    "max_daily_trades": 4,
    "max_daily_loss_pct": 5.0,
    "orb_start_hour": 7,   # UTC
    "orb_end_hour": 9,     # UTC
    "close_hour": 20,      # UTC
    "allowed_days": [0, 1, 2, 3],  # Mon-Thu (0=Mon)
    "symbols": {
        "GER40": {"yahoo": "^GDAXI", "name": "DAX (GER40)"},
        "USA500": {"yahoo": "^GSPC", "name": "S&P 500 (USA500)"},
    },
    "active_symbol": "GER40",
    "check_interval_seconds": 300,  # Check every 5 min (candle closes once/hour, 5 min buffer is enough)
}


# ============================================================================
# DATA FETCHER
# ============================================================================

def fetch_hourly_data(symbol_key, days=10):
    """Fetch hourly OHLCV data from Yahoo Finance."""
    yahoo_symbol = CONFIG["symbols"][symbol_key]["yahoo"]

    # Yahoo Finance v8 API
    end = int(datetime.now(timezone.utc).timestamp())
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + yahoo_symbol
    params = {
        "period1": start,
        "period2": end,
        "interval": "1h",
        "includePrePost": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        data = resp.json()

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "UTC": pd.to_datetime(timestamps, unit="s", utc=True),
            "Open": quotes["open"],
            "High": quotes["high"],
            "Low": quotes["low"],
            "Close": quotes["close"],
            "Volume": quotes["volume"],
        })
        df.dropna(subset=["Close"], inplace=True)
        df.sort_values("UTC", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Hour"] = df["UTC"].dt.hour
        df["Date"] = df["UTC"].dt.date
        df["DOW"] = df["UTC"].dt.dayofweek
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to fetch data for {symbol_key}: {e}")
        return None


def add_indicators(df):
    """Add all technical indicators."""
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["TR"] = np.maximum(df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift(1)),
                   abs(df["Low"] - df["Close"].shift(1))))
    df["ATR14"] = df["TR"].rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["AvgVol"] = df["Volume"].rolling(20).mean()
    df["EMA_Bull_Cross"] = (df["EMA9"] > df["EMA21"]) & (df["EMA9"].shift(1) <= df["EMA21"].shift(1))

    # ORB per day
    orb = df[df["Hour"].isin([CONFIG["orb_start_hour"], CONFIG["orb_start_hour"] + 1])].groupby("Date").agg(
        ORB_High=("High", "max"), ORB_Low=("Low", "min")).reset_index()
    orb["ORB_Range"] = orb["ORB_High"] - orb["ORB_Low"]
    df = df.merge(orb, on="Date", how="left")

    return df


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def default_state():
    return {
        "capital": CONFIG["initial_capital"],
        "max_equity": CONFIG["initial_capital"],
        "total_pnl": 0,
        "position": None,  # {direction, entry_price, stop, tp, lots, type, entry_time}
        "daily_trades": 0,
        "daily_start_equity": CONFIG["initial_capital"],
        "current_date": None,
        "orb_long_taken": False,
        "orb_short_taken": False,
        "orb_short_failed": False,
        "prev_above_orb": False,
        "prev_below_orb": False,
        "last_processed_candle": None,
        "trades_history": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "symbol": CONFIG["active_symbol"],
    }

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return default_state()

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ============================================================================
# TRADING ENGINE
# ============================================================================

def process_candle(state, row, prev_row, median_price):
    """Process a single closed hourly candle. Returns updated state and any action taken."""
    actions = []
    h = int(row["Hour"])
    d = str(row["Date"])
    dow = int(row["DOW"])

    # Adaptive thresholds
    orb_min = median_price * 0.001
    atr_max = median_price * 0.007
    min_stop = median_price * 0.0003

    # New day reset
    if d != state["current_date"]:
        state["current_date"] = d
        state["daily_trades"] = 0
        state["daily_start_equity"] = state["capital"]
        state["orb_long_taken"] = False
        state["orb_short_taken"] = False
        state["orb_short_failed"] = False
        state["prev_above_orb"] = False
        state["prev_below_orb"] = False

    # Daily loss check
    daily_pnl_pct = 0
    if state["daily_start_equity"] > 0:
        daily_pnl_pct = (state["capital"] - state["daily_start_equity"]) / state["daily_start_equity"] * 100
    daily_limit = daily_pnl_pct <= -CONFIG["max_daily_loss_pct"]

    # Track failed ORB
    if not pd.isna(row.get("ORB_High")) and h >= CONFIG["orb_end_hour"]:
        ca = row["Close"] > row["ORB_High"]
        cb = row["Close"] < row["ORB_Low"]
        if state["prev_below_orb"] and ca:
            state["orb_short_failed"] = True
        state["prev_above_orb"] = ca
        state["prev_below_orb"] = cb

    # ---- MANAGE OPEN POSITION ----
    pos = state["position"]
    if pos is not None:
        pnl = 0
        exit_reason = None
        exit_price = None

        if h >= CONFIG["close_hour"]:
            exit_price = row["Close"]
            exit_reason = "EOD"
        elif pos["direction"] == 1:
            if row["Low"] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "SL"
            elif row["High"] >= pos["tp"]:
                exit_price = pos["tp"]
                exit_reason = "TP"
            else:
                # Trailing stop
                if not pd.isna(row["ATR14"]):
                    new_trail = row["High"] - row["ATR14"] * CONFIG["trail_atr_mult"]
                    if new_trail > pos["stop"]:
                        pos["stop"] = round(new_trail, 2)
                        actions.append(f"  Trail stop → {pos['stop']:.2f}")
        else:  # Short
            if row["High"] >= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "SL"
            elif row["Low"] <= pos["tp"]:
                exit_price = pos["tp"]
                exit_reason = "TP"
            else:
                if not pd.isna(row["ATR14"]):
                    new_trail = row["Low"] + row["ATR14"] * CONFIG["trail_atr_mult"]
                    if new_trail < pos["stop"]:
                        pos["stop"] = round(new_trail, 2)
                        actions.append(f"  Trail stop → {pos['stop']:.2f}")

        if exit_price is not None:
            if pos["direction"] == 1:
                pnl = (exit_price - pos["entry_price"]) * pos["lots"]
            else:
                pnl = (pos["entry_price"] - exit_price) * pos["lots"]

            pnl = round(pnl, 2)
            state["capital"] = round(state["capital"] + pnl, 2)
            state["total_pnl"] = round(state["total_pnl"] + pnl, 2)

            trade_record = {
                "symbol": state["symbol"],
                "type": pos["type"],
                "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 2),
                "lots": pos["lots"],
                "pnl": pnl,
                "exit_reason": exit_reason,
                "entry_time": pos["entry_time"],
                "exit_time": str(row["UTC"]),
                "capital_after": state["capital"],
            }
            state["trades_history"].append(trade_record)
            state["position"] = None

            emoji = "+" if pnl >= 0 else ""
            actions.append(
                f"  CLOSED {trade_record['direction']} {pos['type']} @ {exit_price:.2f} "
                f"({exit_reason}) PnL: {emoji}${pnl:.2f} | Balance: ${state['capital']:.2f}"
            )

            # Update max equity
            if state["capital"] > state["max_equity"]:
                state["max_equity"] = state["capital"]

            return state, actions

    # ---- CHECK FOR NEW ENTRY ----
    if state["position"] is not None:
        return state, actions
    if daily_limit or state["daily_trades"] >= CONFIG["max_daily_trades"]:
        return state, actions
    if state["capital"] <= 10:
        return state, actions
    if dow not in CONFIG["allowed_days"]:
        return state, actions
    if not (CONFIG["orb_start_hour"] <= h < CONFIG["close_hour"]):
        return state, actions

    # Check indicators
    if pd.isna(row.get("ATR14")) or pd.isna(row.get("ORB_High")):
        return state, actions

    vol_ok = True
    if not pd.isna(row.get("AvgVol")) and row["AvgVol"] > 0:
        vol_ok = row["Volume"] > row["AvgVol"] * 0.8

    # Quality filters
    if not pd.isna(row.get("RSI")) and 45 < row["RSI"] < 55:
        return state, actions  # Choppy RSI
    if not pd.isna(row.get("ATR14")) and row["ATR14"] > atr_max:
        return state, actions  # Too volatile

    bull_trend = row["EMA9"] > row["EMA21"] and row["Close"] > row["EMA50"]
    bear_trend = row["EMA9"] < row["EMA21"] and row["Close"] < row["EMA50"]

    ep = row["Close"]
    atr = row["ATR14"]
    oh = row["ORB_High"]
    ol = row["ORB_Low"]
    orng = row["ORB_Range"] if not pd.isna(row.get("ORB_Range")) else 0

    signal = None
    sl = tp = 0
    trade_type = ""

    # Signal 1: ORB Breakout
    if h >= CONFIG["orb_end_hour"] and orng > orb_min:
        if ep > oh and bull_trend and vol_ok and not state["orb_long_taken"]:
            sl = ol - orng * CONFIG["orb_stop_mult"]
            tp = ep + orng * CONFIG["orb_target_mult"]
            signal = 1
            trade_type = "ORB_L"
            state["orb_long_taken"] = True
        elif ep < ol and bear_trend and vol_ok and not state["orb_short_taken"]:
            sl = oh + orng * CONFIG["orb_stop_mult"]
            tp = ep - orng * CONFIG["orb_target_mult"]
            signal = -1
            trade_type = "ORB_S"
            state["orb_short_taken"] = True

    # Signal 2: Failed ORB Reversal
    if signal is None and state["orb_short_failed"] and not state["orb_long_taken"]:
        if h >= CONFIG["orb_end_hour"] and h < 16 and ep > oh and bull_trend:
            sl = ol - orng * CONFIG["orb_stop_mult"]
            tp = ep + orng * CONFIG["orb_target_mult"]
            signal = 1
            trade_type = "FORB_L"
            state["orb_long_taken"] = True
            state["orb_short_failed"] = False

    # Signal 3: Second ORB Breakout
    if signal is None and state["orb_long_taken"] and prev_row is not None:
        if h >= 10 and h < 16 and bull_trend:
            if not pd.isna(prev_row.get("Close")) and prev_row["Close"] <= oh and ep > oh:
                sl = ol - orng * CONFIG["orb_stop_mult"]
                tp = ep + orng * CONFIG["orb_target_mult"]
                signal = 1
                trade_type = "ORB2_L"

    # Signal 4: EMA Momentum
    if signal is None and row.get("EMA_Bull_Cross", False):
        if ep > row["EMA50"] and not pd.isna(row.get("RSI")) and 40 < row["RSI"] < 70 and vol_ok:
            sl = ep - atr * CONFIG["mom_sl_atr"]
            tp = ep + atr * CONFIG["mom_tp_atr"]
            signal = 1
            trade_type = "MOM_L"

    # Execute entry
    if signal is not None:
        stop_dist = abs(ep - sl)
        if stop_dist < min_stop:
            return state, actions

        risk_amount = state["capital"] * CONFIG["risk_pct"] / 100
        lots = round(risk_amount / stop_dist, 3)
        lots = max(0.01, lots)

        margin = ep * lots / 20
        if margin > state["capital"] * 0.95:
            lots = round(state["capital"] * 0.95 * 20 / ep, 3)
            lots = max(0.01, lots)

        state["position"] = {
            "direction": signal,
            "entry_price": round(ep, 2),
            "stop": round(sl, 2),
            "tp": round(tp, 2),
            "lots": lots,
            "type": trade_type,
            "entry_time": str(row["UTC"]),
        }
        state["daily_trades"] += 1

        dir_str = "LONG" if signal == 1 else "SHORT"
        actions.append(
            f"  OPENED {dir_str} {trade_type} @ {ep:.2f} | "
            f"SL: {sl:.2f} | TP: {tp:.2f} | Lots: {lots} | "
            f"Risk: ${risk_amount:.2f}"
        )

    return state, actions


# ============================================================================
# DISPLAY
# ============================================================================

def show_status(state):
    """Display current bot status."""
    cap = state["capital"]
    init = CONFIG["initial_capital"]
    ret = (cap - init) / init * 100
    max_eq = state["max_equity"]
    dd = (max_eq - cap) / max_eq * 100 if max_eq > 0 else 0
    trades = state["trades_history"]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    wr = wins / len(trades) * 100 if trades else 0

    print(f"""
{'='*60}
  GER40 DayTrader v5.0 — Paper Trading Bot
{'='*60}
  Symbol:        {state['symbol']}
  Started:       {state.get('started_at', 'N/A')}
  Balance:       ${cap:.2f} ({ret:+.1f}%)
  Max Equity:    ${max_eq:.2f}
  Drawdown:      {dd:.1f}%
  Total PnL:     ${state['total_pnl']:.2f}
  Trades:        {len(trades)} (W:{wins} L:{losses}, WR:{wr:.1f}%)
  Daily Trades:  {state['daily_trades']}/{CONFIG['max_daily_trades']}
{'='*60}""")

    pos = state["position"]
    if pos:
        dir_str = "LONG" if pos["direction"] == 1 else "SHORT"
        print(f"""  OPEN POSITION:
    {dir_str} {pos['type']} @ {pos['entry_price']:.2f}
    Stop: {pos['stop']:.2f} | TP: {pos['tp']:.2f}
    Lots: {pos['lots']} | Entry: {pos['entry_time']}
{'='*60}""")
    else:
        print(f"  Position:      FLAT (no open trade)")
        print(f"{'='*60}")

    if trades:
        print(f"\n  Last 5 trades:")
        for t in trades[-5:]:
            emoji = "+" if t["pnl"] >= 0 else ""
            print(f"    {t['entry_time'][:16]} {t['direction']:5s} {t['type']:6s} "
                  f"@ {t['entry_price']:.2f} → {t['exit_price']:.2f} "
                  f"({t['exit_reason']:3s}) {emoji}${t['pnl']:.2f}")
    print()


def save_trades_xlsx(state):
    """Save trade history to XLSX."""
    if not state["trades_history"]:
        return
    df = pd.DataFrame(state["trades_history"])
    df.to_excel(TRADES_FILE, index=False, sheet_name="Trades")


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_live(symbol_key=None):
    """Main trading loop."""
    state = load_state()
    if symbol_key:
        state["symbol"] = symbol_key

    sym = state["symbol"]
    sym_name = CONFIG["symbols"][sym]["name"]

    print(f"\n{'='*60}")
    print(f"  GER40 DayTrader v5.0 — LIVE Paper Trading")
    print(f"  Symbol: {sym_name}")
    print(f"  Balance: ${state['capital']:.2f}")
    print(f"  Profile: Balanced (8% risk, trail 0.5 ATR)")
    print(f"  Mode: Sleeps until next hourly candle close")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    last_candle_time = state.get("last_processed_candle")

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Calculate seconds until next hour + 30s buffer
            # (candle closes at :00, we check at :00:30 to ensure data is ready)
            minutes_left = 59 - now.minute
            seconds_left = 60 - now.second
            wait_seconds = minutes_left * 60 + seconds_left + 30  # +30s buffer

            # If we just started or it's close to the hour, check immediately
            if last_candle_time is None or wait_seconds > 3600:
                wait_seconds = 0

            if wait_seconds > 60:
                next_check = now + timedelta(seconds=wait_seconds)
                print(f"[{now.strftime('%H:%M:%S')} UTC] Next candle closes in {minutes_left}m {seconds_left}s. "
                      f"Sleeping until {next_check.strftime('%H:%M:%S')} UTC...")
                time.sleep(wait_seconds)
                continue

            print(f"[{now.strftime('%H:%M:%S')} UTC] Candle closed. Fetching data for {sym}...")

            df = fetch_hourly_data(sym, days=10)
            if df is None or len(df) < 50:
                print("  [!] Data fetch failed or not enough data. Retrying next hour.")
                last_candle_time = "__retry__"  # Force re-check next hour
                time.sleep(300)
                continue

            df = add_indicators(df)
            median_price = df["Close"].median()
            print(f"  Received {len(df)} candles. Median price: {median_price:.0f}")

            # Only process CLOSED candles (not the current forming one)
            if len(df) < 2:
                time.sleep(300)
                continue

            latest_closed = df.iloc[-2]
            latest_time = str(latest_closed["UTC"])

            if latest_time == last_candle_time:
                # Already processed — show status and wait
                cur = df.iloc[-1]
                h = int(cur["Hour"]) if not pd.isna(cur.get("Hour")) else -1
                trend = "BULL" if cur.get("EMA9",0)>cur.get("EMA21",0) and cur["Close"]>cur.get("EMA50",0) else "BEAR/FLAT"
                rsi_val = f"{cur['RSI']:.0f}" if not pd.isna(cur.get("RSI")) else "?"
                atr_val = f"{cur['ATR14']:.1f}" if not pd.isna(cur.get("ATR14")) else "?"

                pos = state["position"]
                if pos:
                    dir_str = "LONG" if pos["direction"] == 1 else "SHORT"
                    unrealized = 0
                    if pos["direction"] == 1:
                        unrealized = (cur["Close"] - pos["entry_price"]) * pos["lots"]
                    else:
                        unrealized = (pos["entry_price"] - cur["Close"]) * pos["lots"]
                    emoji = "+" if unrealized >= 0 else ""
                    print(f"  Price: {cur['Close']:.2f} | {trend} | RSI:{rsi_val} | ATR:{atr_val}")
                    print(f"  Position: {dir_str} {pos['type']} @ {pos['entry_price']:.2f} | "
                          f"Unrealized: {emoji}${unrealized:.2f} | Stop: {pos['stop']:.2f}")
                else:
                    print(f"  Price: {cur['Close']:.2f} | {trend} | RSI:{rsi_val} | ATR:{atr_val}")
                    print(f"  No position | Balance: ${state['capital']:.2f} | "
                          f"Daily trades: {state['daily_trades']}/{CONFIG['max_daily_trades']}")

                    # Show why no entry (if in trading hours)
                    if CONFIG["orb_start_hour"] <= h < CONFIG["close_hour"]:
                        orb_h = cur.get("ORB_High")
                        orb_l = cur.get("ORB_Low")
                        if not pd.isna(orb_h):
                            print(f"  ORB zone: {orb_l:.2f} - {orb_h:.2f} | "
                                  f"Price {'ABOVE' if cur['Close'] > orb_h else 'BELOW' if cur['Close'] < orb_l else 'INSIDE'} ORB")
                continue

            # New closed candle!
            print(f"  New candle: {latest_time} | O:{latest_closed['Open']:.2f} "
                  f"H:{latest_closed['High']:.2f} L:{latest_closed['Low']:.2f} "
                  f"C:{latest_closed['Close']:.2f}")

            # Get previous candle for second ORB signal
            idx = df.index[df["UTC"] == latest_closed["UTC"]]
            prev_row = df.iloc[idx[0] - 1] if len(idx) > 0 and idx[0] > 0 else None

            state, actions = process_candle(state, latest_closed, prev_row, median_price)

            for action in actions:
                print(action)

            if not actions:
                print(f"  No signal. Balance: ${state['capital']:.2f}")

            state["last_processed_candle"] = latest_time
            last_candle_time = latest_time  # Update loop variable too!
            save_state(state)
            save_trades_xlsx(state)

            # Sleep until next hour (don't loop immediately)
            now2 = datetime.now(timezone.utc)
            mins_left = 59 - now2.minute
            secs_left = 60 - now2.second
            sleep_secs = mins_left * 60 + secs_left + 30
            next_check = now2 + timedelta(seconds=sleep_secs)
            print(f"  Next check at {next_check.strftime('%H:%M:%S')} UTC. Sleeping...\n")
            time.sleep(sleep_secs)

        except KeyboardInterrupt:
            print(f"\n\nStopping bot...")
            save_state(state)
            save_trades_xlsx(state)
            show_status(state)
            break
        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(30)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        state = load_state()
        show_status(state)

    elif "--reset" in args:
        state = default_state()
        save_state(state)
        print(f"Balance reset to ${CONFIG['initial_capital']:.2f}")
        print(f"State saved to {STATE_FILE}")

    elif "--history" in args:
        state = load_state()
        if not state["trades_history"]:
            print("No trades yet.")
        else:
            show_status(state)
            save_trades_xlsx(state)
            print(f"Trade history saved to {TRADES_FILE}")

    elif "--symbol" in args:
        idx = args.index("--symbol")
        if idx + 1 < len(args):
            sym = args[idx + 1].upper()
            if sym in CONFIG["symbols"]:
                run_live(sym)
            else:
                print(f"Unknown symbol: {sym}")
                print(f"Available: {', '.join(CONFIG['symbols'].keys())}")
        else:
            print("Usage: --symbol GER40 or --symbol USA500")

    elif "--backtest" in args:
        # Run backtest on historical CSV data for verification
        print("Running backtest on historical data...")
        import glob
        sym = "GER40"
        pattern = "DEU.IDX-EUR_Hour_*.csv"
        if "--symbol" in args:
            idx = args.index("--symbol")
            sym = args[idx + 1].upper() if idx + 1 < len(args) else "GER40"
        if sym == "USA500":
            pattern = "USA500.IDX-USD_Hour_*.csv"

        files = sorted(glob.glob(os.path.join(SCRIPT_DIR, pattern)))
        if not files:
            print(f"No CSV files found for {sym}")
            sys.exit(1)

        dfs = [pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in files]
        data = pd.concat(dfs).sort_values("UTC").reset_index(drop=True)
        data["Hour"] = data["UTC"].dt.hour
        data["Date"] = data["UTC"].dt.date
        data["DOW"] = data["UTC"].dt.dayofweek
        data = add_indicators(data)

        median_price = data["Close"].median()
        state = default_state()
        state["symbol"] = sym

        for i in range(50, len(data)):
            row = data.iloc[i]
            prev = data.iloc[i - 1] if i > 0 else None
            state, actions = process_candle(state, row, prev, median_price)

        show_status(state)
        save_trades_xlsx(state)
        print(f"Backtest complete. Trades saved to {TRADES_FILE}")

    else:
        run_live()
