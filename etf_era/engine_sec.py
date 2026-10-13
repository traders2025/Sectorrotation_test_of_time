"""Backtest engine: S&P 500 sector ETF rotation with look-back windows that differ from the
holding period (2000-2024).

Data: data/daily/<TICKER>.csv with columns Date, Close, Adj Close, Volume (Yahoo Finance
adjusted closes, as used in the published fixed-window study); data/tb3ms.csv (FRED TB3MS, %).

Ranking: at the close of each rebalancing date, the nine ETFs are ranked on their total return
over the previous L calendar months (L = 1, 3, 6, 12) ending at that close, or over the previous
ISO week for weekly rebalancing. Winners = ranks 1-3, Median = 4-6, Losers = 7-9,
Ex-Winners = 4-9, EW = all nine. Equal weights at each rebalancing, drifting in between.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'daily')
START, END = '2000-01-01', '2024-12-31'
TICK = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
SECTOR = {'XLB': 'Materials', 'XLE': 'Energy', 'XLF': 'Financials', 'XLI': 'Industrials',
          'XLK': 'Technology', 'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLV': 'Health Care',
          'XLY': 'Consumer Discretionary', 'SPY': 'S&P 500'}
NAMES = {t: f'{SECTOR[t]} Select Sector SPDR Fund' for t in TICK}
NAMES['SPY'] = 'SPDR S&P 500 ETF Trust'
CYC = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLY']
DEF = ['XLP', 'XLU', 'XLV']
SPREAD = 0.00015        # 1.5 bp quoted spread, as in the published study
SPREAD_STRESS = 0.0010  # 10 bp stress case


def _load(t):
    d = pd.read_csv(os.path.join(D, f'{t}.csv'), parse_dates=['Date']).set_index('Date')
    return d['Adj Close'].pct_change()


RA = pd.concat({t: _load(t) for t in TICK}, axis=1)
RA = RA[RA.index <= END].iloc[1:]           # drop first NaN row (1998-12-22)
IDX = RA.index[RA.index >= START]
R = RA.loc[IDX]
assert R.isna().sum().sum() == 0
BENCH = {'SPY': _load('SPY').reindex(RA.index).loc[IDX]}
assert BENCH['SPY'].isna().sum() == 0

_tb = pd.read_csv(os.path.join(HERE, 'data', 'tb3ms.csv'))
_tb = dict(zip(_tb.date, _tb.rate))
RF = pd.Series([_tb[f'{d.year}-{d.month:02d}'] / 100 / 252 for d in IDX], index=IDX)
NY = len(IDX) / 252

FL = {'A': 'Annual', 'S': 'Semiannual', 'Q': 'Quarterly', 'M': 'Monthly', 'W': 'Weekly'}
HOLD_M = {'A': 12, 'S': 6, 'Q': 3, 'M': 1}
CONFIGS = [('A', 1), ('A', 3), ('A', 6), ('A', 12), ('S', 1), ('S', 3), ('S', 6),
           ('Q', 1), ('Q', 3), ('M', 1), ('W', 0)]
CORE = [c for c in CONFIGS if c[0] in 'ASQ']
EXT = [('S', 12), ('Q', 6), ('Q', 12), ('M', 3), ('M', 6), ('M', 12)]
GROUPS = ['W', 'M', 'L', 'XW', 'EW']
GN = {'W': 'Winners', 'M': 'Median', 'L': 'Losers', 'XW': 'Ex-Winners', 'EW': 'Equal-weighted'}


def wlab(f, L):
    return '1 week' if L == 0 else f'{L} month' + ('s' if L > 1 else '')


def clab(c):
    return f"{FL[c[0]]}, {wlab(*c)}"


def matched(c):
    f, L = c
    return (f == 'W') or (HOLD_M.get(f) == L)


def keys(idx, f):
    if f == 'A':
        k = idx.year
    elif f == 'S':
        k = idx.year * 10 + (idx.month - 1) // 6
    elif f == 'Q':
        k = idx.year * 10 + (idx.month - 1) // 3
    elif f == 'M':
        k = idx.year * 100 + idx.month
    elif f == 'W':
        iso = idx.isocalendar()
        k = (iso.year * 100 + iso.week).values
    return np.asarray(k)


_MK = np.asarray(RA.index.year * 12 + RA.index.month - 1)
_OFF = len(RA) - len(IDX)
_cache = {}


def periods(f, L, off=0):
    """(start_pos, end_pos, past_return) for each holding period; off shifts the calendar."""
    if (f, L, off) in _cache:
        return _cache[(f, L, off)]
    k = (_MK - off) // HOLD_M[f] if (f in HOLD_M and off) else keys(RA.index, f)
    ks = pd.unique(k)
    out = []
    for j, kk in enumerate(ks):
        pos = np.where(k == kk)[0]
        pos = pos[RA.index[pos] >= pd.Timestamp(START)]
        if len(pos) == 0:
            continue
        if L == 0:
            prev = RA.iloc[np.where(k == ks[j - 1])[0]]
        else:
            m0 = _MK[pos[0]]
            prev = RA.iloc[np.where((_MK >= m0 - L) & (_MK <= m0 - 1))[0]]
        past = (1 + prev).prod() - 1
        s = pos[0] - _OFF
        out.append((s, s + len(pos), past))
    _cache[(f, L, off)] = out
    return out


def bt(c, group, universe=None, lag=0, spread=0.0, off=0):
    """Backtest configuration c = (freq, months). Groups of three (of nine), or with N funds:
    W = top 3, L = bottom 3, M = the ranks in between, XW = all but W, EW = all."""
    f, L = c
    uni = TICK if universe is None else universe
    N = len(uni)
    k = 3
    sl = {'W': slice(0, k), 'L': slice(N - k, N), 'M': slice(k, N - k),
          'XW': slice(k, N), 'EW': slice(0, N)}[group]
    per = periods(f, L, off)
    n = len(IDX)
    x = R.values
    starts = [p[0] + (lag if j > 0 else 0) for j, p in enumerate(per)]
    ends = starts[1:] + [n]
    gross = np.empty(n)
    cost = np.zeros(n)
    w_prev = np.zeros(len(TICK))
    picks, turns = [], []
    for (s, e, past), s2, e2 in zip(per, starts, ends):
        order = past[uni].sort_values(ascending=False, kind='mergesort').index
        chosen = [TICK.index(t) for t in order[sl]]
        w_t = np.zeros(len(TICK))
        w_t[chosen] = 1 / len(chosen)
        tv = np.abs(w_t - w_prev).sum()
        turns.append(tv)
        picks.append([TICK[i] for i in chosen])
        cost[s2] += spread * tv
        h = w_t.copy()
        for i in range(s2, e2):
            new = h * (1 + x[i])
            gross[i] = new.sum() / h.sum() - 1
            h = new
        w_prev = h / h.sum()
    cost[-1] += spread * 1.0
    g = pd.Series(gross, IDX)
    net = pd.Series((1 - cost) * (1 + gross) - 1, IDX)
    return dict(gross=g, net=net, picks=picks, turn=np.array(turns))


def sr(r):
    e = np.asarray(r) - RF.values
    return e.mean() / e.std() * np.sqrt(252)


def drawdowns(r):
    w = 100 * (1 + r).cumprod()
    mdd = (w / np.maximum(w.cummax(), 100) - 1).min()
    ann, prev_end = [], 100.0
    for y, wy in w.groupby(w.index.year):
        peak = np.maximum(wy.cummax(), prev_end)
        ann.append((wy / peak - 1).min())
        prev_end = wy.iloc[-1]
    ann = np.array(ann)
    return mdd, ann.mean(), ann.min()


def metrics(r):
    w = 100 * (1 + r).cumprod()
    n = len(r)
    fb = w.iloc[-1]
    vol = r.std() * np.sqrt(252)
    ex = r - RF.loc[r.index]
    mdd, avgdd, maxadd = drawdowns(r)
    return dict(final=fb, cagr=(fb / 100) ** (252 / n) - 1, vol=vol, sr0=r.mean() * 252 / vol,
                sr=ex.mean() * 252 / (ex.std() * np.sqrt(252)), mdd=mdd, avgdd=avgdd, maxadd=maxadd)
