"""Calendar robustness: shift the rebalancing calendar of every annual, semiannual and
quarterly configuration by 0..H-1 months (12 annual, 6 semiannual, 3 quarterly calendars)
and average the results over calendars. Output: results/offsets.pkl, results/offsets.csv"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
from engine_sec import *

n = len(IDX)
rfv = RF.values
rng = np.random.default_rng(20260924)
B, p = 5000, 1 / 21
idxs = np.empty((B, n), dtype=np.int32)
ar = np.arange(n)
for b in range(B):
    st = rng.integers(0, n, n)
    nb = rng.random(n) < p
    nb[0] = True
    last = np.maximum.accumulate(np.where(nb, ar, 0))
    idxs[b] = (st[last] + ar - last) % n


def srb(r):
    E = (np.asarray(r) - rfv)[idxs]
    return E.mean(1) / E.std(1) * np.sqrt(252)


rows, boot = [], {}
for c in CORE:
    f, L = c
    for off in range(HOLD_M[f]):
        for g in GROUPS:
            r = bt(c, g, off=off)['gross']
            rows.append(dict(freq=f, win=L, off=off, group=g, sr=sr(r)))
            boot[(c, off, g)] = srb(r)
    print(c, flush=True)
D = pd.DataFrame(rows)
D.to_csv('results/offsets.csv', index=False)
summ = []
for c in CORE:
    f, L = c
    offs = range(HOLD_M[f])
    d = D[(D.freq == f) & (D.win == L)].pivot(index='off', columns='group', values='sr')
    row = dict(freq=f, win=L, ncal=len(d), **{f'sr_{g}': d[g].mean() for g in GROUPS},
               W_lt_EW=int((d.W < d.EW).sum()), M_gt_W=int((d.M > d.W).sum()), M_gt_EW=int((d.M > d.EW).sum()), L_gt_W=int((d.L > d.W).sum()),
               XW_gt_EW=int((d.XW > d.EW).sum()), min_WEW=(d.W - d.EW).min(), max_WEW=(d.W - d.EW).max())
    for a, b_ in [('W', 'EW'), ('M', 'W'), ('L', 'W'), ('XW', 'EW'), ('M', 'EW'), ('L', 'EW')]:
        diff = (d[a] - d[b_]).mean()
        bd = np.mean([boot[(c, o, a)] - boot[(c, o, b_)] for o in offs], axis=0)
        row[f'{a}-{b_}'] = diff
        row[f'{a}-{b_}_lo'], row[f'{a}-{b_}_hi'] = np.percentile(bd, [2.5, 97.5])
        row[f'{a}-{b_}_p'] = min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))
    summ.append(row)
S = pd.DataFrame(summ)
# pooled over the three one-month windows and over all nine configurations
pool = []
for pn, cs in [('short1m', [('A', 1), ('S', 1), ('Q', 1)]), ('core9', CORE)]:
    for a, b_ in [('W', 'EW'), ('M', 'W'), ('L', 'W'), ('XW', 'EW'), ('M', 'EW'), ('L', 'EW')]:
        d = np.mean([S[(S.freq == c[0]) & (S.win == c[1])][f'{a}-{b_}'].iloc[0] for c in cs])
        bd = np.mean([np.mean([boot[(c, o, a)] - boot[(c, o, b_)] for o in range(HOLD_M[c[0]])], axis=0) for c in cs], axis=0)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        pool.append(dict(pool=pn, comp=f'{a}-{b_}', avg=d, lo=lo, hi=hi, p=(np.abs(bd - d) >= abs(d)).mean()))
P = pd.DataFrame(pool)
pd.set_option('display.width', 250)
print(S.round(3).to_string())
print(P.round(3))
pickle.dump(dict(D=D, S=S, P=P), open('results/offsets.pkl', 'wb'))
