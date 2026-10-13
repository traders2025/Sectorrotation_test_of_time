"""Out-of-sample check after the published ETF sample: 2 January 2025 - 31 August 2026 (end of the
Fama-French data). Same rules for the nine SPDR sector ETFs (data/etf_daily, Yahoo Finance, downloaded
10 October 2026) and the FF9 / FF10 industries. Output: results/oos.pkl"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
from engine_ind import Universe, CONFIGS, GROUPS, FF3D, group_sizes, clab

S0, S1 = '2025-01-01', '2026-08-31'
ETF = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']


class ETFUniverse(Universe):
    """Universe built from the ETF CSV files instead of the French zips."""
    def __init__(self, start, end):
        ra = pd.concat({t: pd.read_csv(f'data/etf_daily/{t}.csv', parse_dates=['Date']).set_index('Date')['Adj Close'].pct_change()
                        for t in ETF}, axis=1).iloc[1:]
        ra = ra[ra.index <= end]
        self.RA, self.start, self.TICK = ra, start, list(ra.columns)
        self.N = len(self.TICK); self.k, self.km = group_sizes(self.N)
        self.IDX = ra.index[ra.index >= start]; self.R = ra.loc[self.IDX]
        assert self.R.isna().sum().sum() == 0
        self.RF = FF3D['RF'].reindex(self.IDX).fillna(0.0)
        spy = pd.read_csv('data/etf_daily/SPY.csv', parse_dates=['Date']).set_index('Date')['Adj Close'].pct_change()
        self.MKT = spy.reindex(self.IDX); self.BENCH = {'SPY': self.MKT}
        self.NY = ((self.IDX[-1] - self.IDX[0]).days + 1) / 365.25
        self.PPY = len(self.IDX) / self.NY
        self._MK = np.asarray(ra.index.year * 12 + ra.index.month - 1)
        self._OFF = len(ra) - len(self.IDX); self._cache = {}


UNIS = {'ETF': ETFUniverse(S0, S1), 'FF9': Universe(10, drop=['Other'], start=S0, end=S1), 'FF10': Universe(10, start=S0, end=S1)}
rows = []
for u, V in UNIS.items():
    print(u, V.N, V.k, V.km, len(V.IDX), V.IDX[0].date(), V.IDX[-1].date())
    for c in CONFIGS:
        for g in GROUPS:
            r = V.bt(c, g)['gross']
            m = V.metrics(r)
            rows.append(dict(univ=u, freq=c[0], L=c[1], group=g, cum=(1 + r).prod() - 1, cagr=m['cagr'], vol=m['vol'], sr=m['sr'], mdd=m['mdd']))
    m = V.metrics(V.MKT)
    rows.append(dict(univ=u, freq='-', L=-1, group='BENCH', cum=(1 + V.MKT).prod() - 1, cagr=m['cagr'], vol=m['vol'], sr=m['sr'], mdd=m['mdd']))
O = pd.DataFrame(rows)
# monthly hit rate of the monthly Median rule vs EW (ETF and FF9)
mret = lambda r: (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
hits = {}
for u in ['ETF', 'FF9', 'FF10']:
    V = UNIS[u]
    for g in ['W', 'M', 'L']:
        d = mret(V.bt(('M', 1), g)['gross']) - mret(V.bt(('M', 1), 'EW')['gross'])
        hits[(u, g)] = dict(months=len(d), above=int((d > 0).sum()), mean=d.mean() * 100)
out = dict(O=O, hits=hits, start=UNIS['ETF'].IDX[0], end=UNIS['ETF'].IDX[-1], ndays=len(UNIS['ETF'].IDX))
pickle.dump(out, open('results/oos.pkl', 'wb'))
pd.set_option('display.width', 250)
print(O[O.freq != '-'].pivot_table(index=['freq', 'L'], columns=['univ', 'group'], values='cum').mul(100).round(1).to_string())
print(O[O.freq != '-'].pivot_table(index=['freq', 'L'], columns=['univ', 'group'], values='sr').round(2).to_string())
print(O[O.freq == '-'][['univ', 'cum', 'sr']])
print(hits)
