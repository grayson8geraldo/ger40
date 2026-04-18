#!/usr/bin/env python3
"""
Проверка логики SMC + VP Sniper.

Пайн-скрипт тестировать локально нельзя, но ключевую логику
(структура, Volume Profile, Failed Auction, Breakout, BOS/CHoCH)
можно проверить на часовых данных из репо. Цель — убедиться, что
алгоритм работает без багов и выдаёт осмысленные сигналы.

Ограничения:
  * Данные H1, а индикатор рассчитан на 5M с 15M bias и 1M FVG.
    Поэтому мы тестируем "downscaled"-вариант: структура на H1,
    VP по суткам, без FVG и Kill Zones.
  * Winrate не является итоговым — это санити-чек, что механика
    (пробой → Failed Auction → вход → стоп/тейк) работает.
"""
import glob
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- загрузка ----------
def load_symbol(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{prefix}_Hour_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df["ts"] = pd.to_datetime(df["UTC"].str.replace(" UTC", "", regex=False),
                                   format="%d.%m.%Y %H:%M:%S.%f", utc=True)
        frames.append(df[["ts", "Open", "High", "Low", "Close", "Volume"]])
    out = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    out.rename(columns=str.lower, inplace=True)
    return out


# ---------- структура: фракталы + BOS / CHoCH ----------
def detect_structure(df: pd.DataFrame, swing_len: int = 3, body_break: bool = True):
    n = len(df)
    last_swing_h = np.nan
    last_swing_l = np.nan
    trend = 0  # +1 up, -1 down
    bos_up = np.zeros(n, bool)
    bos_dn = np.zeros(n, bool)
    choch_up = np.zeros(n, bool)
    choch_dn = np.zeros(n, bool)
    bias = np.zeros(n, int)

    highs, lows, closes = df.high.values, df.low.values, df.close.values
    for i in range(swing_len, n - swing_len):
        window_h = highs[i - swing_len:i + swing_len + 1]
        window_l = lows[i - swing_len:i + swing_len + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            last_swing_h = highs[i]
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            last_swing_l = lows[i]

        probe_up = closes[i] if body_break else highs[i]
        probe_dn = closes[i] if body_break else lows[i]

        if not np.isnan(last_swing_h) and probe_up > last_swing_h:
            if trend <= 0:
                choch_up[i] = True
            else:
                bos_up[i] = True
            trend = 1
            last_swing_h = np.nan
        if not np.isnan(last_swing_l) and probe_dn < last_swing_l:
            if trend >= 0:
                choch_dn[i] = True
            else:
                bos_dn[i] = True
            trend = -1
            last_swing_l = np.nan
        bias[i] = trend
    df["bias"] = bias
    df["bos_up"], df["bos_dn"] = bos_up, bos_dn
    df["choch_up"], df["choch_dn"] = choch_up, choch_dn
    return df


# ---------- Volume Profile по предыдущим суткам ----------
def compute_vp(df: pd.DataFrame, row_size: int = 200, va_pct: float = 0.70):
    """Профиль считаем по предыдущим календарным суткам (UTC) и
    прикрепляем VAH/VAL/POC к текущему дню."""
    df["date"] = df.ts.dt.date
    vah_map, val_map, poc_map = {}, {}, {}
    dates = sorted(df.date.unique())
    for i in range(1, len(dates)):
        prev = df[df.date == dates[i - 1]]
        if len(prev) < 3:
            continue
        hi, lo = prev.high.max(), prev.low.min()
        if hi <= lo:
            continue
        bins = np.zeros(row_size)
        step = (hi - lo) / row_size
        for _, r in prev.iterrows():
            i_lo = max(0, int((r.low - lo) / step))
            i_hi = min(row_size - 1, int((r.high - lo) / step))
            span = max(1, i_hi - i_lo + 1)
            share = r.volume / span
            bins[i_lo:i_hi + 1] += share
        poc_idx = int(bins.argmax())
        total = bins.sum()
        target = total * va_pct
        accum = bins[poc_idx]
        lo_i, hi_i = poc_idx, poc_idx
        while accum < target and (lo_i > 0 or hi_i < row_size - 1):
            up = bins[hi_i + 1] if hi_i < row_size - 1 else -1
            dn = bins[lo_i - 1] if lo_i > 0 else -1
            if up >= dn:
                hi_i += 1
                accum += max(up, 0)
            else:
                lo_i -= 1
                accum += max(dn, 0)
        vah_map[dates[i]] = lo + (hi_i + 1) * step
        val_map[dates[i]] = lo + lo_i * step
        poc_map[dates[i]] = lo + (poc_idx + 0.5) * step
    df["vah"] = df.date.map(vah_map)
    df["val"] = df.date.map(val_map)
    df["poc"] = df.date.map(poc_map)
    return df


# ---------- HTF EMA50 (D1) ----------
def add_htf(df: pd.DataFrame):
    daily = df.resample("D", on="ts").close.last().dropna()
    ema = daily.ewm(span=50, adjust=False).mean()
    df["htf_ema"] = df.ts.dt.floor("D").map(ema.to_dict()).ffill()
    df["htf_up"] = df.close > df.htf_ema
    df["htf_dn"] = df.close < df.htf_ema
    return df


# ---------- сетапы + бэктест ----------
@dataclass
class Trade:
    direction: int
    entry_i: int
    entry: float
    sl: float
    tp: float
    exit_i: int = -1
    exit: float = np.nan
    pnl_r: float = 0.0  # в R (1R = risk)


def backtest(df: pd.DataFrame, atr_mult: float = 1.2, tp_rr: float = 2.0):
    # ATR(14)
    tr = np.maximum.reduce([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs(),
    ])
    atr = pd.Series(tr).rolling(14).mean().bfill().values

    above_vah = (df.close > df.vah).values
    below_val = (df.close < df.val).values
    vah, val = df.vah.values, df.val.values
    high, low, close = df.high.values, df.low.values, df.close.values
    bias = df.bias.values
    htf_up = df.htf_up.values
    htf_dn = df.htf_dn.values
    bos_up, bos_dn = df.bos_up.values, df.bos_dn.values
    choch_up, choch_dn = df.choch_up.values, df.choch_dn.values

    trades: list[Trade] = []
    open_t: Trade | None = None

    for i in range(3, len(df)):
        # управление открытой позицией
        if open_t is not None:
            if open_t.direction == 1:
                if low[i] <= open_t.sl:
                    open_t.exit_i, open_t.exit = i, open_t.sl
                    open_t.pnl_r = -1.0
                    trades.append(open_t); open_t = None; continue
                if high[i] >= open_t.tp:
                    open_t.exit_i, open_t.exit = i, open_t.tp
                    open_t.pnl_r = tp_rr
                    trades.append(open_t); open_t = None; continue
            else:
                if high[i] >= open_t.sl:
                    open_t.exit_i, open_t.exit = i, open_t.sl
                    open_t.pnl_r = -1.0
                    trades.append(open_t); open_t = None; continue
                if low[i] <= open_t.tp:
                    open_t.exit_i, open_t.exit = i, open_t.tp
                    open_t.pnl_r = tp_rr
                    trades.append(open_t); open_t = None; continue

        if open_t is not None or np.isnan(vah[i]) or np.isnan(val[i]):
            continue

        # Failed Auction
        fa_long = (low[i] < val[i]) and (close[i] > val[i]) and (close[i] < vah[i])
        fa_short = (high[i] > vah[i]) and (close[i] < vah[i]) and (close[i] > val[i])
        # Breakout (закрепление за VA 3 свечи)
        brk_up = above_vah[i] and above_vah[i - 1] and above_vah[i - 2]
        brk_dn = below_val[i] and below_val[i - 1] and below_val[i - 2]

        buy_bias = (bias[i] > 0) or choch_up[i] or htf_up[i]
        sell_bias = (bias[i] < 0) or choch_dn[i] or htf_dn[i]

        buy_sig = buy_bias and (fa_long or (brk_up and (bos_up[i] or choch_up[i])))
        sell_sig = sell_bias and (fa_short or (brk_dn and (bos_dn[i] or choch_dn[i])))

        if buy_sig:
            sl = close[i] - atr[i] * atr_mult
            tp = close[i] + atr[i] * atr_mult * tp_rr
            open_t = Trade(1, i, close[i], sl, tp)
        elif sell_sig:
            sl = close[i] + atr[i] * atr_mult
            tp = close[i] - atr[i] * atr_mult * tp_rr
            open_t = Trade(-1, i, close[i], sl, tp)
    return trades


def summary(trades: list[Trade], tp_rr: float, symbol: str):
    if not trades:
        return f"{symbol}: сигналов нет"
    closed = [t for t in trades if t.exit_i >= 0]
    wins = [t for t in closed if t.pnl_r > 0]
    losses = [t for t in closed if t.pnl_r < 0]
    wr = len(wins) / len(closed) * 100 if closed else 0
    total_r = sum(t.pnl_r for t in closed)
    gross_win = sum(t.pnl_r for t in wins)
    gross_loss = -sum(t.pnl_r for t in losses) or 1e-9
    pf = gross_win / gross_loss
    # max drawdown в R
    equity = np.cumsum([t.pnl_r for t in closed])
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0])
    dd = (equity - peak).min() if len(equity) else 0
    longs = sum(1 for t in closed if t.direction == 1)
    shorts = len(closed) - longs
    return (
        f"\n=== {symbol} ===\n"
        f"Сделок:   {len(closed)} (L={longs}, S={shorts})\n"
        f"Winrate:  {wr:.1f}%\n"
        f"Итог:     {total_r:+.2f}R\n"
        f"PF:       {pf:.2f}\n"
        f"Max DD:   {dd:.2f}R\n"
        f"Ожидание: {total_r/len(closed):+.2f}R / сделка"
    )


def self_check(df: pd.DataFrame, name: str):
    """Санити-чеки: профиль, структура, bias."""
    checks = []
    # VP корректность
    vp_rows = df.dropna(subset=["vah", "val", "poc"])
    bad_vp = ((vp_rows.vah < vp_rows.val) | (vp_rows.poc < vp_rows.val)
              | (vp_rows.poc > vp_rows.vah)).sum()
    checks.append(f"  VP строк:          {len(vp_rows)}  (сломанных: {bad_vp})")
    # структура
    checks.append(f"  BOS↑/BOS↓:        {df.bos_up.sum()} / {df.bos_dn.sum()}")
    checks.append(f"  CHoCH↑/CHoCH↓:    {df.choch_up.sum()} / {df.choch_dn.sum()}")
    checks.append(f"  bias>0 / bias<0:  {(df.bias>0).sum()} / {(df.bias<0).sum()}")
    return f"\n[self-check] {name}\n" + "\n".join(checks)


def run(symbol_prefix: str, label: str):
    df = load_symbol(symbol_prefix)
    print(f"\n{label}: {len(df)} баров, {df.ts.min().date()} → {df.ts.max().date()}")
    df = detect_structure(df)
    df = compute_vp(df)
    df = add_htf(df)
    print(self_check(df, label))
    trades = backtest(df, tp_rr=2.0)
    print(summary(trades, 2.0, label))
    return trades


if __name__ == "__main__":
    run("USA500.IDX-USD", "S&P 500 (USA500)")
    run("DEU.IDX-EUR",   "DAX    (DEU.IDX)")
