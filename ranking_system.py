#!/usr/bin/env python3
"""
Strategy Ranking System v1.0
Composite risk-adjusted scoring framework for trading strategies.

SCORING METHODOLOGY (best practices, balanced risk/reward):
============================================================
Score = 0.30 * Norm(Profit Factor)
      + 0.25 * Norm(Return/MaxDD Ratio)
      + 0.20 * Norm(Win Rate)
      + 0.15 * Norm(Walk-Forward Consistency)
      + 0.10 * Norm(Cross-Asset Consistency)

WHY THESE 5 METRICS (no overlap):
1. Profit Factor — core profitability (gross profit / gross loss)
2. Return/MaxDD — risk-adjusted return (single metric combining return AND drawdown)
3. Win Rate — execution quality / signal accuracy
4. Walk-Forward Consistency — robustness across time periods (anti-overfitting)
5. Cross-Asset Consistency — generalization across markets (anti-curve-fitting)

WHAT WAS EXCLUDED (to avoid double-counting):
- Sharpe ratio: overlaps with Return/MaxDD (both measure risk-adjusted return)
- Total return alone: already captured in Return/MaxDD ratio
- Max drawdown alone: already captured in Return/MaxDD ratio
- Average win/loss: already captured in Profit Factor
- Number of trades: not a quality metric, just volume

NORMALIZATION: Each metric scaled 0-100 using reasonable bounds.
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

try:
    import openpyxl
except ImportError:
    os.system("pip install openpyxl -q")
    import openpyxl

# ============================================================================
# DATA LOADING
# ============================================================================

DATA_DIR = "/home/user/ger40/"

def load_all_assets():
    """Load all CSV data, grouped by asset."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Hour_*.csv")))
    assets = {}
    for f in files:
        basename = os.path.basename(f)
        # Extract asset name (everything before _Hour_)
        asset = basename.split("_Hour_")[0]
        if asset not in assets:
            assets[asset] = []
        assets[asset].append(f)

    result = {}
    for asset, flist in assets.items():
        dfs = [pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in flist]
        data = pd.concat(dfs, ignore_index=True).sort_values("UTC").reset_index(drop=True)
        data["Hour"] = data["UTC"].dt.hour
        data["Date"] = data["UTC"].dt.date
        data["DOW"] = data["UTC"].dt.dayofweek
        result[asset] = data
    return result

# ============================================================================
# STRATEGY DEFINITIONS (add new strategies here)
# ============================================================================

STRATEGIES = {
    "GER40_DayTrader_v5": {
        "timeframe": "1H",
        "pine_file": "GER40_DayTrader_Strategy.pine",
        "description": "ORB breakout + EMA momentum, trail 0.5 ATR",
        "profiles": {
            "SafeGrowth": {"risk_pct": 5, "orb_stop": 0.25, "orb_target": 1.5, "trail": 0.5},
            "Balanced":   {"risk_pct": 8, "orb_stop": 0.25, "orb_target": 1.5, "trail": 0.5},
            "MaxGrowth":  {"risk_pct": 10, "orb_stop": 0.20, "orb_target": 1.5, "trail": 0.5},
        },
        "backtest_func": "backtest_daytrader_v5",
    },
}

# ============================================================================
# INDICATORS
# ============================================================================

def add_indicators(data):
    """Add all standard indicators to dataframe."""
    data = data.copy()
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["TR"] = np.maximum(data["High"] - data["Low"],
        np.maximum(abs(data["High"] - data["Close"].shift(1)),
                   abs(data["Low"] - data["Close"].shift(1))))
    data["ATR14"] = data["TR"].rolling(14).mean()
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["RSI"] = 100 - 100 / (1 + gain / loss)
    data["AvgVol"] = data["Volume"].rolling(20).mean()
    data["EMA_Cross"] = (data["EMA9"] > data["EMA21"]) & (data["EMA9"].shift(1) <= data["EMA21"].shift(1))

    orb = data[data["Hour"].isin([7, 8])].groupby("Date").agg(
        ORB_High=("High", "max"), ORB_Low=("Low", "min")).reset_index()
    orb["ORB_Range"] = orb["ORB_High"] - orb["ORB_Low"]
    data = data.merge(orb, on="Date", how="left")
    return data

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class Trade:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.pnl = 0

def backtest_daytrader_v5(data, params):
    """GER40 DayTrader v5.0 backtest — adaptive to any index."""
    rp = params.get("risk_pct", 8)
    sl_m = params.get("orb_stop", 0.25)
    tp_m = params.get("orb_target", 1.5)
    tr = params.get("trail", 0.5)

    # Adaptive thresholds based on asset price level
    median_price = data["Close"].median()
    orb_min = median_price * 0.001   # ~0.1% of price (GER40~20, USA500~5)
    atr_max = median_price * 0.007   # ~0.7% of price (GER40~120, USA500~35)
    min_stop = median_price * 0.0003 # ~0.03% of price (GER40~5, USA500~1.5)

    capital = 200.0
    trades = []
    ct = None
    dt = 0
    dse = 200
    cd = None
    max_eq = 200
    max_dd = 0
    ol_t = os_t = pa = pb = osf = False

    for i in range(50, len(data)):
        r = data.iloc[i]
        h, d, dow = r["Hour"], r["Date"], r["DOW"]

        if d != cd:
            cd = d; dt = 0; dse = capital
            ol_t = os_t = pa = pb = osf = False

        dl = ((capital - dse) / dse * 100) if dse > 0 else 0

        if not pd.isna(r["ORB_High"]) and h >= 9:
            ca = r["Close"] > r["ORB_High"]
            cb = r["Close"] < r["ORB_Low"]
            if pb and ca:
                osf = True
            pa = ca; pb = cb

        if ct is not None:
            if h >= 20:
                ct.pnl = (r["Close"] - ct.ep) * ct.d * ct.l
                capital += ct.pnl; trades.append(ct); ct = None; continue
            if ct.d == 1:
                if r["Low"] <= ct.sl:
                    ct.pnl = (ct.sl - ct.ep) * ct.l; capital += ct.pnl; trades.append(ct); ct = None; continue
                if r["High"] >= ct.tp:
                    ct.pnl = (ct.tp - ct.ep) * ct.l; capital += ct.pnl; trades.append(ct); ct = None; continue
                ns = r["High"] - r["ATR14"] * tr
                if ns > ct.sl:
                    ct.sl = ns
            else:
                if r["High"] >= ct.sl:
                    ct.pnl = (ct.ep - ct.sl) * ct.l; capital += ct.pnl; trades.append(ct); ct = None; continue
                if r["Low"] <= ct.tp:
                    ct.pnl = (ct.ep - ct.tp) * ct.l; capital += ct.pnl; trades.append(ct); ct = None; continue
                ns = r["Low"] + r["ATR14"] * tr
                if ns < ct.sl:
                    ct.sl = ns

        if ct is not None:
            continue
        if dl <= -5 or dt >= 4 or capital <= 10 or dow not in {0, 1, 2, 3}:
            continue
        if not (7 <= h < 20) or pd.isna(r["ATR14"]) or pd.isna(r["ORB_High"]):
            continue

        hv = r["Volume"] > r["AvgVol"] * 0.8 if not pd.isna(r["AvgVol"]) else True
        if not pd.isna(r["RSI"]) and 45 < r["RSI"] < 55:
            continue
        if not pd.isna(r["ATR14"]) and r["ATR14"] > atr_max:
            continue

        bt2 = r["EMA9"] > r["EMA21"] and r["Close"] > r["EMA50"]
        brt = r["EMA9"] < r["EMA21"] and r["Close"] < r["EMA50"]
        ep = r["Close"]; atr = r["ATR14"]
        oh = r["ORB_High"]; ol = r["ORB_Low"]; orng = r["ORB_Range"]

        sig = sl = tp = None
        tt = ""

        if h >= 9 and not pd.isna(orng) and orng > orb_min:
            if ep > oh and bt2 and hv and not ol_t:
                sl = ol - orng * sl_m; tp = ep + orng * tp_m; sig = 1; tt = "ORB_L"; ol_t = True
            elif ep < ol and brt and hv and not os_t:
                sl = oh + orng * sl_m; tp = ep - orng * tp_m; sig = -1; tt = "ORB_S"; os_t = True

        if sig is None and osf and not ol_t and h >= 9 and h < 16 and ep > oh and bt2:
            sl = ol - orng * sl_m; tp = ep + orng * tp_m; sig = 1; tt = "FORB"; ol_t = True; osf = False

        if sig is None and ol_t and h >= 10 and h < 16 and bt2:
            prev = data.iloc[i - 1]
            if not pd.isna(prev["Close"]) and prev["Close"] <= oh and ep > oh:
                sl = ol - orng * sl_m; tp = ep + orng * tp_m; sig = 1; tt = "ORB2"

        if sig is None and r["EMA_Cross"] and ep > r["EMA50"] and not pd.isna(r["RSI"]) and 40 < r["RSI"] < 70 and hv:
            sl = ep - atr * 1.0; tp = ep + atr * 2.0; sig = 1; tt = "MOM"

        if sig is None:
            continue

        sd = abs(ep - sl)
        if sd < min_stop:
            continue
        lots = max(0.01, round(capital * rp / 100 / sd, 3))
        mg = ep * lots / 20
        if mg > capital * 0.95:
            lots = max(0.01, round(capital * 0.95 * 20 / ep, 3))

        ct = Trade(d=sig, ep=ep, sl=sl, tp=tp, l=lots, tt=tt, entry_time=r["UTC"])
        dt += 1

        if capital > max_eq:
            max_eq = capital
        dd = (max_eq - capital) / max_eq * 100
        if dd > max_dd:
            max_dd = dd

    if ct is not None:
        ct.pnl = (data.iloc[-1]["Close"] - ct.ep) * ct.d * ct.l
        capital += ct.pnl
        trades.append(ct)

    return capital, trades, max_dd, max_eq

# ============================================================================
# SCORING SYSTEM
# ============================================================================

def normalize(value, low, high):
    """Normalize value to 0-100 scale."""
    if high == low:
        return 50.0
    return max(0, min(100, (value - low) / (high - low) * 100))

def compute_score(metrics):
    """
    Compute composite score from metrics dict.

    Score = 0.30 * Norm(Profit Factor)       [profitability]
          + 0.25 * Norm(Return/MaxDD)         [risk-adjusted return]
          + 0.20 * Norm(Win Rate)             [signal quality]
          + 0.15 * Norm(WF Consistency)       [robustness over time]
          + 0.10 * Norm(Cross-Asset Score)    [generalization]
    """
    # Normalization bounds (reasonable ranges for any strategy)
    pf_score    = normalize(metrics["profit_factor"], 0.5, 3.0)     # PF: 0.5 (bad) to 3.0 (excellent)
    rdd_score   = normalize(metrics["return_dd_ratio"], 0, 30)      # R/DD: 0 to 30
    wr_score    = normalize(metrics["win_rate"], 30, 70)            # WR: 30% to 70%
    wf_score    = normalize(metrics["wf_consistency"], 0, 100)      # 0-100% periods profitable
    ca_score    = normalize(metrics["cross_asset_score"], 0, 100)   # 0-100

    score = (0.30 * pf_score +
             0.25 * rdd_score +
             0.20 * wr_score +
             0.15 * wf_score +
             0.10 * ca_score)

    return round(score, 2)

def compute_trade_metrics(capital, trades, max_dd, initial=200):
    """Compute standard metrics from trade list."""
    if not trades:
        return {"total_return_pct": 0, "max_drawdown": 0, "profit_factor": 0,
                "win_rate": 0, "total_trades": 0, "avg_win": 0, "avg_loss": 0,
                "return_dd_ratio": 0, "rr_ratio": 0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins) / len(trades) * 100
    gp = sum(t.pnl for t in wins)
    gl = sum(abs(t.pnl) for t in losses)
    pf = gp / gl if gl > 0 else 999
    avg_w = np.mean([t.pnl for t in wins]) if wins else 0
    avg_l = np.mean([abs(t.pnl) for t in losses]) if losses else 0
    ret = (capital - initial) / initial * 100
    rdd = ret / max_dd if max_dd > 0 else ret

    return {
        "total_return_pct": round(ret, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(min(pf, 99), 2),
        "win_rate": round(wr, 2),
        "total_trades": len(trades),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "return_dd_ratio": round(rdd, 2),
        "rr_ratio": round(avg_w / avg_l if avg_l > 0 else 0, 2),
    }

# ============================================================================
# WALK-FORWARD TESTING
# ============================================================================

def walk_forward_test(data, backtest_func, params, n_periods=4):
    """Split data into n periods and test each separately."""
    dates = sorted(data["Date"].unique())
    chunk = len(dates) // n_periods
    results = []

    for p in range(n_periods):
        start = dates[p * chunk]
        end = dates[min((p + 1) * chunk - 1, len(dates) - 1)]
        mask = (data["Date"] >= start) & (data["Date"] <= end)
        period_data = data[mask].reset_index(drop=True)

        if len(period_data) < 100:
            continue

        cap, trades, dd, _ = backtest_func(period_data, params)
        ret = (cap - 200) / 200 * 100
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100 if trades else 0
        gp = sum(t.pnl for t in trades if t.pnl > 0)
        gl = sum(abs(t.pnl) for t in trades if t.pnl <= 0)
        pf = gp / gl if gl > 0 else 0

        results.append({
            "period": f"P{p+1}",
            "start": str(start),
            "end": str(end),
            "return_pct": round(ret, 2),
            "max_dd": round(dd, 2),
            "trades": len(trades),
            "win_rate": round(wr, 2),
            "profit_factor": round(min(pf, 99), 2),
            "profitable": ret > 0,
        })

    return results

# ============================================================================
# MAIN: Run all backtests and generate rankings
# ============================================================================

print("=" * 80)
print("STRATEGY RANKING SYSTEM v1.1")
print("=" * 80)

# Load all assets
print("\nLoading assets...")
assets = load_all_assets()
for asset, adata in assets.items():
    print(f"  {asset}: {len(adata)} candles, {adata['UTC'].min().strftime('%Y-%m-%d')} to {adata['UTC'].max().strftime('%Y-%m-%d')}")
    print(f"    Price range: {adata['Close'].min():.0f} - {adata['Close'].max():.0f}")

num_assets = len(assets)
print(f"\nTotal assets: {num_assets}")

# Phase 1: Run all backtests, collect per-asset results
print("\nPhase 1: Running backtests on all assets...")
raw_results = []  # (strat, profile, asset, metrics, wf_results, wf_consistency)

for strat_name, strat_config in STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"Strategy: {strat_name} | Timeframe: {strat_config['timeframe']}")
    print(f"{'='*60}")

    bt_func = backtest_daytrader_v5

    for profile_name, profile_params in strat_config["profiles"].items():
        print(f"\n  Profile: {profile_name}")

        for asset_name, asset_data in assets.items():
            print(f"    Asset: {asset_name}")
            data_with_ind = add_indicators(asset_data)

            # Full backtest
            cap, trades, dd, meq = bt_func(data_with_ind, profile_params)
            metrics = compute_trade_metrics(cap, trades, dd)
            print(f"      Full: ${cap:.2f} ({metrics['total_return_pct']:+.1f}%), DD={dd:.1f}%, PF={metrics['profit_factor']:.2f}, WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")

            # Walk-forward
            wf_results = walk_forward_test(data_with_ind, bt_func, profile_params)
            wf_profitable = sum(1 for r in wf_results if r["profitable"])
            wf_total = len(wf_results)
            wf_consistency = (wf_profitable / wf_total * 100) if wf_total > 0 else 0
            for wf in wf_results:
                print(f"      {wf['period']}: {wf['return_pct']:+.1f}%, DD={wf['max_dd']:.1f}%, PF={wf['profit_factor']:.2f} {'OK' if wf['profitable'] else 'LOSS'}")

            raw_results.append((strat_name, profile_name, asset_name, metrics, wf_results, wf_consistency))

# Phase 2: Compute cross-asset scores
print(f"\nPhase 2: Computing cross-asset consistency scores...")
all_results = []

# Group by (strategy, profile) to compute cross-asset score
from itertools import groupby
raw_results.sort(key=lambda x: (x[0], x[1]))

for (strat, prof), group in groupby(raw_results, key=lambda x: (x[0], x[1])):
    group_list = list(group)
    # Cross-asset score: average profit factor across all assets, penalized for any losing asset
    n_assets_tested = len(group_list)
    n_profitable = sum(1 for _, _, _, m, _, _ in group_list if m["total_return_pct"] > 0)
    avg_pf = np.mean([m["profit_factor"] for _, _, _, m, _, _ in group_list])
    min_pf = min(m["profit_factor"] for _, _, _, m, _, _ in group_list)

    # Cross-asset score: % profitable assets * min(avg_pf/2, 1) * 100
    # This rewards strategies that work on ALL assets and penalizes single-asset wonders
    cross_pct = n_profitable / n_assets_tested
    cross_asset_score = cross_pct * min(avg_pf / 2, 1) * 100

    for strat_name, profile_name, asset_name, metrics, wf_results, wf_consistency in group_list:
        scoring_metrics = {
            "profit_factor": metrics["profit_factor"],
            "return_dd_ratio": metrics["return_dd_ratio"],
            "win_rate": metrics["win_rate"],
            "wf_consistency": wf_consistency,
            "cross_asset_score": cross_asset_score,
        }
        composite_score = compute_score(scoring_metrics)

        wf_profitable = sum(1 for r in wf_results if r["profitable"])
        wf_total = len(wf_results)

        print(f"  {strat_name}:{profile_name} on {asset_name}: Score={composite_score:.2f}, CA={cross_asset_score:.1f}")

        strat_config = STRATEGIES[strat_name]
        asset_data_ref = assets[asset_name]
        all_results.append({
            "Rank": 0,
            "Strategy": strat_name,
            "Profile": profile_name,
            "Timeframe": strat_config["timeframe"],
            "Asset": asset_name,
            "Composite_Score": composite_score,
            "Total_Return_Pct": metrics["total_return_pct"],
            "Max_Drawdown_Pct": metrics["max_drawdown"],
            "Profit_Factor": metrics["profit_factor"],
            "Win_Rate_Pct": metrics["win_rate"],
            "Return_DD_Ratio": metrics["return_dd_ratio"],
            "RR_Ratio": metrics["rr_ratio"],
            "Total_Trades": metrics["total_trades"],
            "Avg_Win": metrics["avg_win"],
            "Avg_Loss": metrics["avg_loss"],
            "WF_Periods_Profitable": f"{wf_profitable}/{wf_total}",
            "WF_Consistency_Pct": round(wf_consistency, 1),
            "Cross_Asset_Score": round(cross_asset_score, 1),
            "PF_Score_30pct": round(normalize(metrics["profit_factor"], 0.5, 3.0) * 0.30, 2),
            "RDD_Score_25pct": round(normalize(metrics["return_dd_ratio"], 0, 30) * 0.25, 2),
            "WR_Score_20pct": round(normalize(metrics["win_rate"], 30, 70) * 0.20, 2),
            "WF_Score_15pct": round(normalize(wf_consistency, 0, 100) * 0.15, 2),
            "CA_Score_10pct": round(normalize(cross_asset_score, 0, 100) * 0.10, 2),
            "Pine_File": strat_config.get("pine_file", ""),
            "Description": strat_config.get("description", ""),
            "Backtest_Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Data_Range": f"{asset_data_ref['UTC'].min().strftime('%Y-%m-%d')} to {asset_data_ref['UTC'].max().strftime('%Y-%m-%d')}",
        })

# Sort by composite score descending
all_results.sort(key=lambda x: x["Composite_Score"], reverse=True)
for i, r in enumerate(all_results):
    r["Rank"] = i + 1

# ============================================================================
# SAVE TO XLSX
# ============================================================================

xlsx_path = os.path.join(DATA_DIR, "strategy_rankings.xlsx")
df = pd.DataFrame(all_results)

# Column order
cols = ["Rank", "Strategy", "Profile", "Timeframe", "Asset", "Composite_Score",
        "Total_Return_Pct", "Max_Drawdown_Pct", "Profit_Factor", "Win_Rate_Pct",
        "Return_DD_Ratio", "RR_Ratio", "Total_Trades", "Avg_Win", "Avg_Loss",
        "WF_Periods_Profitable", "WF_Consistency_Pct", "Cross_Asset_Score",
        "PF_Score_30pct", "RDD_Score_25pct", "WR_Score_20pct", "WF_Score_15pct",
        "CA_Score_10pct", "Pine_File", "Description", "Backtest_Date", "Data_Range"]
df = df[cols]

# Write to XLSX with formatting
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    # Rankings sheet
    df.to_excel(writer, sheet_name="Rankings", index=False)

    # Walk-forward details sheet
    wf_rows = []
    for strat_name, strat_config in STRATEGIES.items():
        bt_func = backtest_daytrader_v5
        for profile_name, profile_params in strat_config["profiles"].items():
            for asset_name, asset_data in assets.items():
                data_with_ind = add_indicators(asset_data)
                wf_results = walk_forward_test(data_with_ind, bt_func, profile_params)
                for wf in wf_results:
                    wf_rows.append({
                        "Strategy": strat_name,
                        "Profile": profile_name,
                        "Asset": asset_name,
                        "Period": wf["period"],
                        "Start": wf["start"],
                        "End": wf["end"],
                        "Return_Pct": wf["return_pct"],
                        "Max_DD_Pct": wf["max_dd"],
                        "Trades": wf["trades"],
                        "Win_Rate_Pct": wf["win_rate"],
                        "Profit_Factor": wf["profit_factor"],
                        "Profitable": wf["profitable"],
                    })
    if wf_rows:
        pd.DataFrame(wf_rows).to_excel(writer, sheet_name="Walk-Forward", index=False)

    # Scoring methodology sheet
    methodology = pd.DataFrame([
        {"Component": "Profit Factor", "Weight": "30%", "Range": "0.5 - 3.0", "Description": "Gross profit / gross loss. Core profitability metric."},
        {"Component": "Return/MaxDD Ratio", "Weight": "25%", "Range": "0 - 30", "Description": "Total return divided by max drawdown. Single risk-adjusted metric."},
        {"Component": "Win Rate", "Weight": "20%", "Range": "30% - 70%", "Description": "Percentage of winning trades. Signal quality measure."},
        {"Component": "Walk-Forward Consistency", "Weight": "15%", "Range": "0% - 100%", "Description": "% of walk-forward periods that are profitable. Anti-overfitting."},
        {"Component": "Cross-Asset Consistency", "Weight": "10%", "Range": "0 - 100", "Description": "Performance across multiple assets. Anti-curve-fitting."},
        {"Component": "---", "Weight": "---", "Range": "---", "Description": "---"},
        {"Component": "EXCLUDED METRICS", "Weight": "", "Range": "", "Description": ""},
        {"Component": "Sharpe Ratio", "Weight": "N/A", "Range": "", "Description": "Overlaps with Return/MaxDD (both measure risk-adjusted return)"},
        {"Component": "Total Return", "Weight": "N/A", "Range": "", "Description": "Already captured in Return/MaxDD ratio"},
        {"Component": "Max Drawdown", "Weight": "N/A", "Range": "", "Description": "Already captured in Return/MaxDD ratio"},
        {"Component": "Avg Win/Loss", "Weight": "N/A", "Range": "", "Description": "Already captured in Profit Factor"},
    ])
    methodology.to_excel(writer, sheet_name="Methodology", index=False)

    # Format the Rankings sheet
    ws = writer.sheets["Rankings"]
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers

    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'))

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    # Score column (F) highlighting
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    for row in range(2, ws.max_row + 1):
        score_cell = ws.cell(row=row, column=6)  # Composite_Score
        if score_cell.value and score_cell.value >= 70:
            score_cell.fill = green_fill
        elif score_cell.value and score_cell.value >= 50:
            score_cell.fill = yellow_fill
        elif score_cell.value:
            score_cell.fill = red_fill

        for col in range(1, ws.max_column + 1):
            ws.cell(row=row, column=col).border = thin_border
            ws.cell(row=row, column=col).alignment = Alignment(horizontal="center")

    # Auto-width columns
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 30)

print(f"\n{'='*80}")
print(f"RANKING RESULTS")
print(f"{'='*80}")
for r in all_results:
    print(f"  #{r['Rank']} {r['Strategy']}:{r['Profile']:12s} Score={r['Composite_Score']:5.2f}/100  "
          f"Ret={r['Total_Return_Pct']:+8.1f}%  DD={r['Max_Drawdown_Pct']:5.1f}%  PF={r['Profit_Factor']:.2f}  WR={r['Win_Rate_Pct']:.1f}%")

print(f"\nSaved to: {xlsx_path}")
print(f"Sheets: Rankings, Walk-Forward, Methodology")
print(f"\nScoring formula:")
print(f"  Score = 0.30*PF + 0.25*R/DD + 0.20*WR + 0.15*WF + 0.10*CA")
print(f"  (all normalized to 0-100 scale)")
