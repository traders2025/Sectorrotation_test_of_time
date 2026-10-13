"""Head-to-head comparison over the ETF era (3 Jan 2000 - 31 Dec 2024): the same Winners/Median/Losers
rules applied to (a) the nine Select Sector SPDR ETFs, (b) the nine Fama-French industries excluding
'Other' (groups of three), and (c) the ten Fama-French industries (3/4/3).
Uses the ETF-era engine and data in ./etf_era.
Output: results/bridge.pkl"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
from engine_ind import Universe, CONFIGS, GROUPS, FF3D, clab

SEC = os.path.abspath('etf_era')
sys.path.insert(0, SEC)
import engine_sec as ES                                   # noqa: E402

S0, S1 = '2000-01-01', '2024-12-31'
# --- SPDR ETFs (recomputed with the sector engine, identical to the sector-ETF study)
etf = {(c, g): ES.bt(c, g)['gross'] for c in CONFIGS for g in GROUPS}
etf_spy = ES.BENCH['SPY']
# --- Fama-French industries over the same window
F9 = Universe(10, drop=['Other'], start=S0, end=S1)
F10 = Universe(10, start=S0, end=S1)
ff9 = {(c, g): F9.bt(c, g)['gross'] for c in CONFIGS for g in GROUPS}
ff10 = {(c, g): F10.bt(c, g)['gross'] for c in CONFIGS for g in GROUPS}
dates = etf[(CONFIGS[0], 'W')].index.intersection(F9.IDX)
print('common dates', len(dates), dates[0].date(), dates[-1].date(), 'ETF days', len(ES.IDX), 'FF days', len(F9.IDX))
rf = FF3D['RF'].reindex(dates).fillna(0).values          # one risk-free rate for all three universes
ppy = len(dates) / (((dates[-1] - dates[0]).days + 1) / 365.25)


def ser(d):
    return {k: v.reindex(dates).fillna(0).values - rf for k, v in d.items()}


U = {'ETF': ser(etf), 'FF9': ser(ff9), 'FF10': ser(ff10)}
SR = {u: {k: e.mean() / e.std() * np.sqrt(ppy) for k, e in d.items()} for u, d in U.items()}

# --- bootstrap (common resamples for all universes)
n = len(dates)
B, chunk = 2000, 200
rng = np.random.default_rng(20260924)
ar = np.arange(n)
acc = {(u, k): [] for u in U for k in U[u]}
for b0 in range(0, B, chunk):
    idx = np.empty((chunk, n), dtype=np.int32)
    for i in range(chunk):
        st = rng.integers(0, n, n); nb = rng.random(n) < 1 / 21; nb[0] = True
        last = np.maximum.accumulate(np.where(nb, ar, 0)); idx[i] = (st[last] + ar - last) % n
    for u in U:
        for k, e in U[u].items():
            E = e[idx]; acc[(u, k)].append(E.mean(1) / E.std(1) * np.sqrt(ppy))
SRB = {k: np.concatenate(v) for k, v in acc.items()}

POOLS = {'all11': CONFIGS, 'short1m': [('A', 1), ('S', 1), ('Q', 1)], 'matched5': [('A', 12), ('S', 6), ('Q', 3), ('M', 1), ('W', 0)],
         'M1': [('M', 1)], 'A1': [('A', 1)]}
pool = []
for u in U:
    for pn, cs in POOLS.items():
        for a, b in [('W', 'EW'), ('M', 'EW'), ('L', 'EW'), ('M', 'W')]:
            d = np.mean([SR[u][(c, a)] - SR[u][(c, b)] for c in cs])
            D = np.mean([SRB[(u, (c, a))] - SRB[(u, (c, b))] for c in cs], axis=0)
            pool.append(dict(univ=u, pool=pn, comp=f'{a}-{b}', avg=d, lo=np.percentile(D, 2.5), hi=np.percentile(D, 97.5),
                             p=(np.abs(D - d) >= abs(d)).mean() if len(cs) > 1 else min(1, 2 * min((D <= 0).mean(), (D >= 0).mean()))))
pool = pd.DataFrame(pool)
# difference between universes (ETF minus FF9) in the Median-minus-EW of the monthly rule
dd = []
for c in [('M', 1), ('A', 1), ('Q', 1)]:
    for g in ['W', 'M', 'L']:
        d = (SR['ETF'][(c, g)] - SR['ETF'][(c, 'EW')]) - (SR['FF9'][(c, g)] - SR['FF9'][(c, 'EW')])
        D = (SRB[('ETF', (c, g))] - SRB[('ETF', (c, 'EW'))]) - (SRB[('FF9', (c, g))] - SRB[('FF9', (c, 'EW'))])
        dd.append(dict(cfg=clab(c), group=g, diff=d, p=min(1, 2 * min((D <= 0).mean(), (D >= 0).mean()))))
dd = pd.DataFrame(dd)

# --- monthly return correlations between ETF and FF9 groups, and EW
mret = lambda r: (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
corr = []
for c in CONFIGS:
    for g in GROUPS:
        a, b = mret(etf[(c, g)].reindex(dates)), mret(ff9[(c, g)].reindex(dates))
        corr.append(dict(freq=c[0], L=c[1], group=g, corr=a.corr(b)))
corr = pd.DataFrame(corr)
# correlation of the underlying rankings? overlap is not defined across classifications; report EW/market correlation
spy_vs_mkt = mret(etf_spy.reindex(dates)).corr(mret(F9.MKT.reindex(dates)))

# --- event time, one-month window, months 1-3, for ETFs and FF9 over 2000-2024
def event(ra, tick, k):
    mr = (1 + ra).groupby([ra.index.year, ra.index.month]).prod() - 1
    rows = []
    for j in range(len(mr) - 1):
        y, m = mr.index[j]
        if (y, m) < (1999, 12) or (y, m) > (2024, 11):
            continue
        past = mr.iloc[j][tick]
        order = past.sort_values(ascending=False, kind='mergesort').index
        N = len(tick)
        sets = {'W': list(order[:k]), 'M': list(order[k:N - k]), 'L': list(order[N - k:])}
        nxt = mr.iloc[j + 1]
        rows.append({g: nxt[s].mean() - nxt[tick].mean() for g, s in sets.items()})
    return pd.DataFrame(rows)


import statsmodels.api as sm                               # noqa: E402
ev = []
for lab, ra, tick, k in [('ETF', ES.RA, ES.TICK, 3), ('FF9', F9.RA, F9.TICK, 3), ('FF10', F10.RA, F10.TICK, 3)]:
    d = event(ra, tick, k)
    for g in ['W', 'M', 'L']:
        fit = sm.OLS(d[g].values, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        ev.append(dict(univ=lab, group=g, mean=fit.params[0] * 100, t=fit.tvalues[0], n=len(d)))
ev = pd.DataFrame(ev)

grid = pd.DataFrame([dict(univ=u, freq=c[0], L=c[1], group=g, sr=SR[u][(c, g)]) for u in U for c in CONFIGS for g in GROUPS])
out = dict(grid=grid, pool=pool, dd=dd, corr=corr, ev=ev, spy_vs_mkt=spy_vs_mkt, ndays=len(dates),
           sr_spy=(lambda e: e.mean() / e.std() * np.sqrt(ppy))(etf_spy.reindex(dates).values - rf),
           sr_mkt=(lambda e: e.mean() / e.std() * np.sqrt(ppy))(F9.MKT.reindex(dates).values - rf))
pickle.dump(out, open('results/bridge.pkl', 'wb'))
pd.set_option('display.width', 250)
print(grid.pivot_table(index=['freq', 'L'], columns=['univ', 'group'], values='sr').round(2).to_string())
print(pool[pool.pool.isin(['all11', 'short1m', 'matched5', 'M1'])].round(3).to_string())
print(dd.round(3)); print(ev.round(2)); print(corr.groupby('group')['corr'].mean().round(2), spy_vs_mkt, out['sr_spy'], out['sr_mkt'])
