#!/usr/bin/env python3
"""Combined analysis: losing trade filters + new signals + SL/TP optimization.
Focused and efficient — avoids agent timeout issues."""

import pandas as pd
import numpy as np
import glob, os

# Load data
files = sorted(glob.glob("/home/user/ger40/DEU.IDX-EUR_Hour_*.csv"))
data = pd.concat([pd.read_csv(f, parse_dates=["UTC"], dayfirst=True) for f in files]).sort_values("UTC").reset_index(drop=True)
data["Hour"] = data["UTC"].dt.hour
data["Date"] = data["UTC"].dt.date
data["DOW"] = data["UTC"].dt.dayofweek

# Indicators
data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
data["TR"] = np.maximum(data["High"]-data["Low"], np.maximum(abs(data["High"]-data["Close"].shift(1)), abs(data["Low"]-data["Close"].shift(1))))
data["ATR14"] = data["TR"].rolling(14).mean()
delta = data["Close"].diff()
gain = delta.where(delta>0,0).rolling(14).mean()
loss = (-delta.where(delta<0,0)).rolling(14).mean()
data["RSI"] = 100 - 100/(1+gain/loss)
data["AvgVol"] = data["Volume"].rolling(20).mean()
data["EMA_Cross"] = (data["EMA9"]>data["EMA21"]) & (data["EMA9"].shift(1)<=data["EMA21"].shift(1))

# ADX
plus_dm = data["High"].diff(); minus_dm = -data["Low"].diff()
plus_dm = plus_dm.where((plus_dm>minus_dm)&(plus_dm>0),0)
minus_dm = minus_dm.where((minus_dm>plus_dm)&(minus_dm>0),0)
atr_s = data["TR"].rolling(14).mean()
plus_di = 100*(plus_dm.rolling(14).mean()/atr_s)
minus_di = 100*(minus_dm.rolling(14).mean()/atr_s)
dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di)
data["ADX"] = dx.rolling(14).mean()

# ORB
orb = data[data["Hour"].isin([7,8])].groupby("Date").agg(ORB_High=("High","max"),ORB_Low=("Low","min")).reset_index()
orb["ORB_Range"] = orb["ORB_High"]-orb["ORB_Low"]
data = data.merge(orb, on="Date", how="left")

# Body ratio
data["Body"] = abs(data["Close"]-data["Open"])
data["Range"] = data["High"]-data["Low"]
data["BodyRatio"] = data["Body"]/data["Range"].replace(0,np.nan)

# Distance from EMA50
data["DistEMA50"] = (data["Close"]-data["EMA50"])/data["EMA50"]*100

# Prev candle direction
data["PrevBull"] = (data["Close"].shift(1) > data["Open"].shift(1))

print(f"Data: {len(data)} candles\n")

# ============================================================================
# BACKTEST with configurable params + context recording
# ============================================================================
class Trade:
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self,k,v)
        self.pnl=0; self.exit_reason=""

def backtest(data, risk_pct=5.0, orb_target=1.5, orb_stop=0.3, mom_tp=2.0, mom_sl=1.0,
             trail=1.0, max_daily=4, close_hour=20, allowed_days={0,1,2,3},
             filters=None, record_context=False):
    """Run backtest. filters = dict of filter functions applied at entry."""
    capital=200.0; trades=[]; ct=None; dt=0; dse=200.0; cd=None
    max_eq=200.0; max_dd=0
    orb_lt=False; orb_st=False; prev_above=False; prev_below=False; orb_sf=False

    for i in range(50, len(data)):
        r = data.iloc[i]
        h,d,dow = r["Hour"],r["Date"],r["DOW"]

        if d!=cd:
            cd=d; dt=0; dse=capital; orb_lt=False; orb_st=False
            prev_above=False; prev_below=False; orb_sf=False

        dloss = ((capital-dse)/dse*100) if dse>0 else 0

        # Track failed ORB
        if not pd.isna(r["ORB_High"]) and h>=9:
            ca=r["Close"]>r["ORB_High"]; cb=r["Close"]<r["ORB_Low"]
            if prev_below and ca: orb_sf=True
            prev_above=ca; prev_below=cb

        # Manage open trade
        if ct is not None:
            if h>=close_hour:
                ct.exit_price=r["Close"]; ct.exit_reason="EOD"
                ct.pnl=(ct.exit_price-ct.entry_price)*ct.direction*ct.lots
                capital+=ct.pnl; trades.append(ct); ct=None; continue
            if ct.direction==1:
                if r["Low"]<=ct.stop:
                    ct.exit_price=ct.stop; ct.pnl=(ct.exit_price-ct.entry_price)*ct.lots; ct.exit_reason="SL"
                    capital+=ct.pnl; trades.append(ct); ct=None; continue
                if r["High"]>=ct.tp:
                    ct.exit_price=ct.tp; ct.pnl=(ct.exit_price-ct.entry_price)*ct.lots; ct.exit_reason="TP"
                    capital+=ct.pnl; trades.append(ct); ct=None; continue
                ns=r["High"]-r["ATR14"]*trail
                if ns>ct.stop: ct.stop=ns
            else:
                if r["High"]>=ct.stop:
                    ct.exit_price=ct.stop; ct.pnl=(ct.entry_price-ct.exit_price)*ct.lots; ct.exit_reason="SL"
                    capital+=ct.pnl; trades.append(ct); ct=None; continue
                if r["Low"]<=ct.tp:
                    ct.exit_price=ct.tp; ct.pnl=(ct.entry_price-ct.exit_price)*ct.lots; ct.exit_reason="TP"
                    capital+=ct.pnl; trades.append(ct); ct=None; continue
                ns=r["Low"]+r["ATR14"]*trail
                if ns<ct.stop: ct.stop=ns

        if ct is not None: continue
        if dloss<=-5 or dt>=max_daily or capital<=10 or dow not in allowed_days: continue
        if not (7<=h<close_hour): continue
        if pd.isna(r["ATR14"]) or pd.isna(r["ORB_High"]): continue

        hv = r["Volume"]>r["AvgVol"]*0.8 if not pd.isna(r["AvgVol"]) else True
        bt = r["EMA9"]>r["EMA21"] and r["Close"]>r["EMA50"]
        brt = r["EMA9"]<r["EMA21"] and r["Close"]<r["EMA50"]
        ep=r["Close"]; atr=r["ATR14"]; oh=r["ORB_High"]; ol=r["ORB_Low"]; orng=r["ORB_Range"]

        sig=None; sl=tp=0; tt=""

        # ORB Long
        if h>=9 and h<close_hour and orng>20 and ep>oh and bt and hv and not orb_lt:
            sl=ol-orng*orb_stop; tp=ep+orng*orb_target; sig=1; tt="ORB_L"; orb_lt=True
        # ORB Short
        if sig is None and h>=9 and h<close_hour and orng>20 and ep<ol and brt and hv and not orb_st:
            sl=oh+orng*orb_stop; tp=ep-orng*orb_target; sig=-1; tt="ORB_S"; orb_st=True
        # Failed ORB
        if sig is None and orb_sf and not orb_lt and h>=9 and h<16 and ep>oh and bt:
            sl=ol-orng*0.3; tp=ep+orng*1.5; sig=1; tt="FORB_L"; orb_lt=True; orb_sf=False
        # Second ORB
        if sig is None and orb_lt and h>=10 and h<16 and bt:
            prev=data.iloc[i-1]
            if not pd.isna(prev["Close"]) and prev["Close"]<=oh and ep>oh:
                sl=ol-orng*0.3; tp=ep+orng*1.5; sig=1; tt="ORB2_L"
        # Momentum
        if sig is None and r["EMA_Cross"] and ep>r["EMA50"] and 40<r["RSI"]<70 and 7<=h<close_hour and hv:
            sl=ep-atr*mom_sl; tp=ep+atr*mom_tp; sig=1; tt="MOM_L"

        if sig is None: continue

        # Apply filters
        if filters:
            skip=False
            for fn in filters.values():
                if fn(r, trades, tt):
                    skip=True; break
            if skip: continue

        sd=abs(ep-sl)
        if sd<5: continue
        lots=max(0.01, round(capital*risk_pct/100/sd, 3))
        margin=ep*lots/20
        if margin>capital*0.95:
            lots=max(0.01, round(capital*0.95*20/ep, 3))

        ctx = {}
        if record_context:
            ctx = dict(hour=h, dow=dow, atr=atr, rsi=r["RSI"], adx=r["ADX"],
                       orb_range=orng, vol_ratio=r["Volume"]/r["AvgVol"] if r["AvgVol"]>0 else 1,
                       dist_ema50=r["DistEMA50"], body_ratio=r["BodyRatio"],
                       prev_bull=r["PrevBull"])

        ct = Trade(direction=sig, entry_price=ep, stop=sl, tp=tp, entry_time=r["UTC"],
                   trade_type=tt, lots=lots, ctx=ctx)
        dt+=1

        if capital>max_eq: max_eq=capital
        dd=(max_eq-capital)/max_eq*100
        if dd>max_dd: max_dd=dd

    if ct is not None:
        ct.exit_price=data.iloc[-1]["Close"]
        ct.pnl=(ct.exit_price-ct.entry_price)*ct.direction*ct.lots
        capital+=ct.pnl; trades.append(ct)

    return capital, trades, max_dd

def stats(trades, label="", initial=200):
    if not trades: print(f"  {label}: NO TRADES"); return
    w=[t for t in trades if t.pnl>0]; l=[t for t in trades if t.pnl<=0]
    wr=len(w)/len(trades)*100; gp=sum(t.pnl for t in w); gl=sum(abs(t.pnl) for t in l)
    pf=gp/gl if gl>0 else 999
    cap=initial+sum(t.pnl for t in trades)
    print(f"  {label:35s}: ${cap:8.2f} ({(cap-initial)/initial*100:+6.1f}%), Trades={len(trades):3d}, WR={wr:.1f}%, PF={pf:.2f}")
    return cap

# ============================================================================
# PART 1: BASELINE
# ============================================================================
print("="*80)
print("BASELINE: v4 Balanced (5% risk, no filters)")
print("="*80)
cap0, trades0, dd0 = backtest(data, record_context=True)
stats(trades0, "Baseline", 200)
print(f"  Max DD: {dd0:.1f}%")

# ============================================================================
# PART 2: LOSING TRADE ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("LOSING TRADE PROFILE")
print("="*80)

winners = [t for t in trades0 if t.pnl>0]
losers = [t for t in trades0 if t.pnl<=0]

features = ["hour","atr","rsi","adx","orb_range","vol_ratio","dist_ema50","body_ratio"]
print(f"\n  {'Feature':15s} {'Win_Mean':>10s} {'Loss_Mean':>10s} {'Diff':>10s} {'Separation':>10s}")
print("  "+"-"*60)
for f in features:
    wvals = [t.ctx[f] for t in winners if f in t.ctx and not pd.isna(t.ctx[f])]
    lvals = [t.ctx[f] for t in losers if f in t.ctx and not pd.isna(t.ctx[f])]
    if not wvals or not lvals: continue
    wm,lm = np.mean(wvals), np.mean(lvals)
    combined = wvals+lvals
    std = np.std(combined) if np.std(combined)>0 else 1
    sep = abs(wm-lm)/std
    print(f"  {f:15s} {wm:10.2f} {lm:10.2f} {wm-lm:+10.2f} {sep:10.3f}")

# ============================================================================
# PART 3: FILTER TESTS
# ============================================================================
print(f"\n{'='*80}")
print("FILTER TESTS (each applied independently)")
print("="*80)

filter_defs = {
    "ADX<20 skip":      lambda r,t,tt: r["ADX"]<20 if not pd.isna(r["ADX"]) else False,
    "ADX<25 skip":      lambda r,t,tt: r["ADX"]<25 if not pd.isna(r["ADX"]) else False,
    "RSI 45-55 skip":   lambda r,t,tt: 45<r["RSI"]<55 if not pd.isna(r["RSI"]) else False,
    "ORB<60 skip":      lambda r,t,tt: r["ORB_Range"]<60 if not pd.isna(r["ORB_Range"]) else False,
    "ORB<80 skip":      lambda r,t,tt: r["ORB_Range"]<80 if not pd.isna(r["ORB_Range"]) else False,
    "ORB<100 skip":     lambda r,t,tt: r["ORB_Range"]<100 if not pd.isna(r["ORB_Range"]) else False,
    "Vol<1.0x skip":    lambda r,t,tt: r["Volume"]<r["AvgVol"] if not pd.isna(r["AvgVol"]) else False,
    "Vol<1.5x skip":    lambda r,t,tt: r["Volume"]<r["AvgVol"]*1.5 if not pd.isna(r["AvgVol"]) else False,
    "Hour>15 skip":     lambda r,t,tt: r["Hour"]>15,
    "Hour=9 only":      lambda r,t,tt: r["Hour"]!=9 and tt.startswith("ORB"),
    "ATR>120 skip":     lambda r,t,tt: r["ATR14"]>120 if not pd.isna(r["ATR14"]) else False,
    "|EMA50|>2% skip":  lambda r,t,tt: abs(r["DistEMA50"])>2 if not pd.isna(r["DistEMA50"]) else False,
    "Weak body skip":   lambda r,t,tt: r["BodyRatio"]<0.4 if not pd.isna(r["BodyRatio"]) else False,
    "Strong body req":  lambda r,t,tt: r["BodyRatio"]<0.6 if not pd.isna(r["BodyRatio"]) else False,
}

filter_results = []
for fname, ffunc in filter_defs.items():
    cap, tds, dd = backtest(data, filters={fname: ffunc})
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    filter_results.append((fname, cap, len(tds), wr, pf, dd))
    print(f"  {fname:22s}: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# Sort by return/DD
filter_results.sort(key=lambda x: (x[1]-200)/(x[5]+0.1), reverse=True)
print(f"\n  TOP 5 BY RETURN/DD:")
for i,r in enumerate(filter_results[:5]):
    rdd = (r[1]-200)/(r[5]+0.1)
    print(f"    #{i+1} {r[0]:22s}: R/DD={rdd:.1f}, ${r[1]:.2f}, PF={r[4]:.2f}, DD={r[5]:.1f}%")

# ============================================================================
# PART 4: FILTER COMBOS
# ============================================================================
print(f"\n{'='*80}")
print("BEST FILTER COMBINATIONS")
print("="*80)

combos = [
    ("ADX<25 + ORB<80", {"f1": filter_defs["ADX<25 skip"], "f2": filter_defs["ORB<80 skip"]}),
    ("ADX<25 + WeakBody", {"f1": filter_defs["ADX<25 skip"], "f2": filter_defs["Weak body skip"]}),
    ("ADX<25 + Hour>15", {"f1": filter_defs["ADX<25 skip"], "f2": filter_defs["Hour>15 skip"]}),
    ("ADX<20 + ORB<60", {"f1": filter_defs["ADX<20 skip"], "f2": filter_defs["ORB<60 skip"]}),
    ("ADX<20 + WeakBody", {"f1": filter_defs["ADX<20 skip"], "f2": filter_defs["Weak body skip"]}),
    ("ORB<80 + WeakBody", {"f1": filter_defs["ORB<80 skip"], "f2": filter_defs["Weak body skip"]}),
    ("ORB<80 + Hour>15", {"f1": filter_defs["ORB<80 skip"], "f2": filter_defs["Hour>15 skip"]}),
    ("ADX<25+ORB<80+Weak", {"f1": filter_defs["ADX<25 skip"], "f2": filter_defs["ORB<80 skip"], "f3": filter_defs["Weak body skip"]}),
    ("ADX<20+ORB<60+Weak", {"f1": filter_defs["ADX<20 skip"], "f2": filter_defs["ORB<60 skip"], "f3": filter_defs["Weak body skip"]}),
    ("ADX<25+H>15+Weak", {"f1": filter_defs["ADX<25 skip"], "f2": filter_defs["Hour>15 skip"], "f3": filter_defs["Weak body skip"]}),
]

combo_results = []
for cname, cfilters in combos:
    cap, tds, dd = backtest(data, filters=cfilters)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    combo_results.append((cname, cap, len(tds), wr, pf, dd))
    print(f"  {cname:25s}: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# ============================================================================
# PART 5: STOP LOSS OPTIMIZATION
# ============================================================================
print(f"\n{'='*80}")
print("STOP LOSS OPTIMIZATION")
print("="*80)

for sl_mult in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7]:
    cap, tds, dd = backtest(data, orb_stop=sl_mult)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    print(f"  SL={sl_mult:.2f}x ORB: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# ============================================================================
# PART 6: TAKE PROFIT OPTIMIZATION
# ============================================================================
print(f"\n{'='*80}")
print("TAKE PROFIT OPTIMIZATION")
print("="*80)

for tp_mult in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    cap, tds, dd = backtest(data, orb_target=tp_mult)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    print(f"  TP={tp_mult:.2f}x ORB: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# ============================================================================
# PART 7: TRAILING STOP OPTIMIZATION
# ============================================================================
print(f"\n{'='*80}")
print("TRAILING STOP OPTIMIZATION")
print("="*80)

for tr_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 999]:  # 999 = no trail
    label = f"Trail={tr_mult:.1f}ATR" if tr_mult<100 else "No trail"
    cap, tds, dd = backtest(data, trail=tr_mult)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    print(f"  {label:15s}: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# ============================================================================
# PART 8: RISK % OPTIMIZATION
# ============================================================================
print(f"\n{'='*80}")
print("RISK PER TRADE OPTIMIZATION")
print("="*80)

for rp in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
    cap, tds, dd = backtest(data, risk_pct=rp)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    print(f"  Risk={rp:2d}%: ${cap:8.2f} ({(cap-200)/200*100:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

# ============================================================================
# PART 9: BEST OVERALL COMBO
# ============================================================================
print(f"\n{'='*80}")
print("BEST OVERALL COMBINATIONS")
print("="*80)

# Combine best filter + best SL + best TP + best trail + best risk
mega_combos = [
    ("Baseline 5%", dict(risk_pct=5)),
    ("Best_filter_7%", dict(risk_pct=7, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Best_SL_TP_7%", dict(risk_pct=7, orb_stop=0.25, orb_target=1.5)),
    ("Mega1: ADX25+Weak+7%+SL25", dict(risk_pct=7, orb_stop=0.25, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Mega2: ADX25+Weak+10%+SL25", dict(risk_pct=10, orb_stop=0.25, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Mega3: ADX20+Weak+7%+SL3", dict(risk_pct=7, orb_stop=0.3, filters={"f1":filter_defs["ADX<20 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Mega4: ADX25+ORB80+7%", dict(risk_pct=7, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["ORB<80 skip"]})),
    ("Mega5: ADX25+H15+Weak+7%", dict(risk_pct=7, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Hour>15 skip"],"f3":filter_defs["Weak body skip"]})),
    ("Mega6: ADX25+Weak+8%+SL25", dict(risk_pct=8, orb_stop=0.25, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Mega7: ADX25+Weak+12%+SL25", dict(risk_pct=12, orb_stop=0.25, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
    ("Mega8: NoFilter+8%+SL25", dict(risk_pct=8, orb_stop=0.25)),
    ("Mega9: NoFilter+10%+SL20", dict(risk_pct=10, orb_stop=0.2)),
    ("Mega10: ADX25+Weak+7%+TP2x", dict(risk_pct=7, orb_target=2.0, filters={"f1":filter_defs["ADX<25 skip"],"f2":filter_defs["Weak body skip"]})),
]

mega_results = []
for mname, mparams in mega_combos:
    cap, tds, dd = backtest(data, **mparams)
    w = sum(1 for t in tds if t.pnl>0)
    wr = w/len(tds)*100 if tds else 0
    gp = sum(t.pnl for t in tds if t.pnl>0)
    gl = sum(abs(t.pnl) for t in tds if t.pnl<=0)
    pf = gp/gl if gl>0 else 0
    ret = (cap-200)/200*100
    mega_results.append((mname, cap, ret, dd, len(tds), wr, pf))
    print(f"  {mname:35s}: ${cap:8.2f} ({ret:+6.1f}%), N={len(tds):3d}, WR={wr:.1f}%, PF={pf:.2f}, DD={dd:.1f}%")

mega_results.sort(key=lambda x: x[1], reverse=True)
print(f"\n  TOP 5 BY RETURN:")
for i,r in enumerate(mega_results[:5]):
    rdd = r[2]/(r[3]+0.1)
    print(f"    #{i+1} {r[0]:35s}: ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, PF={r[6]:.2f}, R/DD={rdd:.1f}")

mega_results_rdd = sorted(mega_results, key=lambda x: x[2]/(x[3]+0.1), reverse=True)
print(f"\n  TOP 5 BY RETURN/DD:")
for i,r in enumerate(mega_results_rdd[:5]):
    rdd = r[2]/(r[3]+0.1)
    print(f"    #{i+1} {r[0]:35s}: R/DD={rdd:.1f}, ${r[1]:.2f} ({r[2]:+.1f}%), DD={r[3]:.1f}%, PF={r[6]:.2f}")

# Walk-forward on top combo
best = mega_results[0]
bname = best[0]
bparams = dict(next(p for n,p in mega_combos if n==bname))
print(f"\n{'='*80}")
print(f"WALK-FORWARD: {bname}")
periods = [("P1 Feb-Jul24","2024-02-01","2024-07-31"),("P2 Aug24-Jan25","2024-08-01","2025-01-31"),
           ("P3 Feb-Jul25","2025-02-01","2025-07-31"),("P4 Aug25-Mar26","2025-08-01","2026-03-31")]
for pn,s,e in periods:
    mask=(data["Date"]>=pd.Timestamp(s).date())&(data["Date"]<=pd.Timestamp(e).date())
    pd2=data[mask].reset_index(drop=True)
    if len(pd2)<100: continue
    cp,tp2,dp = backtest(pd2, **bparams)
    ws=sum(1 for t in tp2 if t.pnl>0); wp=ws/len(tp2)*100 if tp2 else 0
    gpp=sum(t.pnl for t in tp2 if t.pnl>0); glp=sum(abs(t.pnl) for t in tp2 if t.pnl<=0)
    pfp=gpp/glp if glp>0 else 0
    print(f"  {pn:20s}: ${cp:.2f} ({(cp-200)/200*100:+.1f}%), DD={dp:.1f}%, N={len(tp2)}, WR={wp:.1f}%, PF={pfp:.2f}")

# Also walk-forward best R/DD combo
best_rdd = mega_results_rdd[0]
bname2 = best_rdd[0]
bparams2 = dict(next(p for n,p in mega_combos if n==bname2))
print(f"\nWALK-FORWARD: {bname2}")
for pn,s,e in periods:
    mask=(data["Date"]>=pd.Timestamp(s).date())&(data["Date"]<=pd.Timestamp(e).date())
    pd2=data[mask].reset_index(drop=True)
    if len(pd2)<100: continue
    cp,tp2,dp = backtest(pd2, **bparams2)
    ws=sum(1 for t in tp2 if t.pnl>0); wp=ws/len(tp2)*100 if tp2 else 0
    gpp=sum(t.pnl for t in tp2 if t.pnl>0); glp=sum(abs(t.pnl) for t in tp2 if t.pnl<=0)
    pfp=gpp/glp if glp>0 else 0
    print(f"  {pn:20s}: ${cp:.2f} ({(cp-200)/200*100:+.1f}%), DD={dp:.1f}%, N={len(tp2)}, WR={wp:.1f}%, PF={pfp:.2f}")

print(f"\n{'='*80}")
print("DONE")
