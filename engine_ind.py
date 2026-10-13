"""Backtest engine for Winners/Median/Losers rotation among Fama-French industry portfolios,
1928-2025, with look-back (ranking) windows that can differ from the holding period.

Data: Kenneth French data library (data/french/*.zip), parsed by get_french_data.parse_french_csv.
Daily industry returns (value-weighted by default), daily T-bill rate and market return from the
Fama-French 3-factor daily file. Returns in the files are in percent.

Groups: with N industries, Winners = top k, Losers = bottom k, Median = the ranks in between,
where (k, median size) = (3, 4) for N = 10, (3, 3) for N = 9, (4, 4) for N = 12, (16, 17) for N = 49.
Ex-Winners = all but the Winners; EW = all N.
"""
import io
import os
import re
import zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, 'data', 'french')
START, END = '1928-01-01', '2025-12-31'
FL = {'A': 'Annual', 'S': 'Semiannual', 'Q': 'Quarterly', 'M': 'Monthly', 'W': 'Weekly'}
HOLD_M = {'A': 12, 'S': 6, 'Q': 3, 'M': 1}
CONFIGS = [('A', 1), ('A', 3), ('A', 6), ('A', 12), ('S', 1), ('S', 3), ('S', 6),
           ('Q', 1), ('Q', 3), ('M', 1), ('W', 0)]
CORE = [c for c in CONFIGS if c[0] in 'ASQ']
EXT = [('S', 12), ('Q', 6), ('Q', 12), ('M', 3), ('M', 6), ('M', 12)]
GROUPS = ['W', 'M', 'L', 'XW', 'EW']
GN = {'W': 'Winners', 'M': 'Median', 'L': 'Losers', 'XW': 'Ex-Winners', 'EW': 'Equal-weighted'}
IND10 = {'NoDur': 'Consumer Nondurables', 'Durbl': 'Consumer Durables', 'Manuf': 'Manufacturing',
         'Enrgy': 'Energy', 'HiTec': 'Business Equipment (High Tech)', 'Telcm': 'Telecommunications',
         'Shops': 'Wholesale and Retail', 'Hlth': 'Health Care', 'Utils': 'Utilities', 'Other': 'Other'}
SPREAD = 0.0010         # 10 bp base case (pre-ETF history)
SPREAD_ETF = 0.00015    # 1.5 bp (today's sector ETFs)
SPREAD_STRESS = 0.0050  # 50 bp stress case


def _read_block(zipname, block=0):
    """Return block `block` of a Ken French CSV inside data/french/<zipname> as a DataFrame (decimals)."""
    import sys
    sys.path.insert(0, HERE)
    from get_french_data import parse_french_csv
    with zipfile.ZipFile(os.path.join(FD, zipname)) as z:
        name = [m for m in z.namelist() if m.lower().endswith('.csv')][0]
        tabs = parse_french_csv(z.read(name).decode('latin-1'))
    title, df = tabs[block]
    df = df.set_index('Date').apply(pd.to_numeric, errors='coerce').astype(float) / 100
    return df


def group_sizes(N):
    return {10: (3, 4), 9: (3, 3), 12: (4, 4), 49: (16, 17)}.get(N, (N // 3, N - 2 * (N // 3)))


FF3D = _read_block('F-F_Research_Data_Factors_daily_CSV.zip')


class Universe:
    def __init__(self, n_ind=10, weighting='vw', drop=None, start=START, end=END):
        block = 0 if weighting == 'vw' else 1
        ra = _read_block(f'{n_ind}_Industry_Portfolios_daily_CSV.zip', block)
        if drop:
            ra = ra.drop(columns=drop)
        ra = ra[ra.index <= end]
        self.RA = ra
        self.start = start
        self.TICK = list(ra.columns)
        self.N = len(self.TICK)
        self.k, self.km = group_sizes(self.N)
        self.IDX = ra.index[ra.index >= start]
        self.R = ra.loc[self.IDX]
        assert self.R.isna().sum().sum() == 0, 'missing returns in evaluation window'
        self.RF = FF3D['RF'].reindex(self.IDX).fillna(0.0)
        self.MKT = (FF3D['Mkt-RF'] + FF3D['RF']).reindex(self.IDX)
        self.BENCH = {'MKT': self.MKT}
        self.NY = ((self.IDX[-1] - self.IDX[0]).days + 1) / 365.25
        self.PPY = len(self.IDX) / self.NY          # trading days per year (Saturday trading before 1952)
        self._MK = np.asarray(ra.index.year * 12 + ra.index.month - 1)
        self._OFF = len(ra) - len(self.IDX)
        self._cache = {}

    def keys(self, f):
        idx = self.RA.index
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

    def periods(self, f, L, off=0):
        if (f, L, off) in self._cache:
            return self._cache[(f, L, off)]
        k = (self._MK - off) // HOLD_M[f] if (f in HOLD_M and off) else self.keys(f)
        ks = pd.unique(k)
        out = []
        for j, kk in enumerate(ks):
            pos = np.where(k == kk)[0]
            pos = pos[self.RA.index[pos] >= pd.Timestamp(self.start)]
            if len(pos) == 0:
                continue
            if L == 0:
                prev = self.RA.iloc[np.where(k == ks[j - 1])[0]]
            else:
                m0 = self._MK[pos[0]]
                prev = self.RA.iloc[np.where((self._MK >= m0 - L) & (self._MK <= m0 - 1))[0]]
            past = (1 + prev).prod() - 1
            s = pos[0] - self._OFF
            out.append((s, s + len(pos), past))
        self._cache[(f, L, off)] = out
        return out

    def bt(self, c, group, lag=0, spread=0.0, off=0, universe=None):
        f, L = c
        uni = self.TICK if universe is None else universe
        N = len(uni)
        k, km = group_sizes(N)
        sl = {'W': slice(0, k), 'L': slice(N - k, N), 'M': slice(k, N - k),
              'XW': slice(k, N), 'EW': slice(0, N)}[group]
        per = self.periods(f, L, off)
        n = len(self.IDX)
        x = self.R.values
        nT = len(self.TICK)
        starts = [p[0] + (lag if j > 0 else 0) for j, p in enumerate(per)]
        ends = starts[1:] + [n]
        gross = np.empty(n)
        cost = np.zeros(n)
        w_prev = np.zeros(nT)
        picks, turns = [], []
        for (s, e, past), s2, e2 in zip(per, starts, ends):
            order = past[uni].sort_values(ascending=False, kind='mergesort').index
            chosen = [self.TICK.index(t) for t in order[sl]]
            w_t = np.zeros(nT)
            w_t[chosen] = 1 / len(chosen)
            tv = np.abs(w_t - w_prev).sum()
            turns.append(tv)
            picks.append([self.TICK[i] for i in chosen])
            cost[s2] += spread * tv
            h = w_t.copy()
            seg = x[s2:e2]
            if len(seg):
                cum = np.cumprod(1 + seg, axis=0) * h          # value of each holding
                tot = cum.sum(1)
                prevtot = np.concatenate([[h.sum()], tot[:-1]])
                gross[s2:e2] = tot / prevtot - 1
                h = cum[-1]
            w_prev = h / h.sum()
        cost[-1] += spread * 1.0
        g = pd.Series(gross, self.IDX)
        net = pd.Series((1 - cost) * (1 + gross) - 1, self.IDX)
        return dict(gross=g, net=net, picks=picks, turn=np.array(turns))

    def sr(self, r):
        if hasattr(r, 'index'):
            e = np.asarray(r) - self.RF.loc[r.index].values
        else:
            e = np.asarray(r) - self.RF.values
        return e.mean() / e.std() * np.sqrt(self.PPY)

    def metrics(self, r):
        w = 100 * (1 + r).cumprod()
        n = len(r)
        fb = w.iloc[-1]
        ppy = self.PPY
        vol = r.std() * np.sqrt(ppy)
        ex = r - self.RF.loc[r.index]
        mdd = (w / np.maximum(w.cummax(), 100) - 1).min()
        ann, prev_end = [], 100.0
        for y, wy in w.groupby(w.index.year):
            peak = np.maximum(wy.cummax(), prev_end)
            ann.append((wy / peak - 1).min())
            prev_end = wy.iloc[-1]
        ann = np.array(ann)
        yrs = n / ppy
        return dict(final=fb, cagr=(fb / 100) ** (1 / yrs) - 1, vol=vol,
                    sr=ex.mean() * ppy / (ex.std() * np.sqrt(ppy)), mdd=mdd, avgdd=ann.mean(), maxadd=ann.min())


def wlab(f, L):
    return '1 week' if L == 0 else f'{L} month' + ('s' if L > 1 else '')


def clab(c):
    return f"{FL[c[0]]}, {wlab(*c)}"


def matched(c):
    f, L = c
    return (f == 'W') or (HOLD_M.get(f) == L)
