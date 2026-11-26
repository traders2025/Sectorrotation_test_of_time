"""Calendar robustness: every annual, semiannual and quarterly configuration on all 12/6/3
rebalancing calendars (Fama-French 10 industries, 1928-2025). Output: results/offsets.pkl"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import time
import numpy as np
import pandas as pd
from engine_ind import *

T0 = time.time()
U = Universe(10)
n = len(U.IDX)
rfv = U.RF.values
series = {}
rows = []
for c in CORE:
    f, L = c
    for off in range(HOLD_M[f]):
        for g in GROUPS:
            r = U.bt(c, g, off=off)['gross']
            series[(c, off, g)] = r.values - rfv
            rows.append(dict(freq=f, win=L, off=off, group=g, sr=U.sr(r)))
D = pd.DataFrame(rows)
print('backtests', round(time.time() - T0))
B, chunk = 2000, 200
rng = np.random.default_rng(20260924)
ar = np.arange(n)
acc = {k: [] for k in series}
for b0 in range(0, B, chunk):
    idx = np.empty((chunk, n), dtype=np.int32)
    for i in range(chunk):
        st = rng.integers(0, n, n); nb = rng.random(n) < 1 / 21; nb[0] = True
        last = np.maximum.accumulate(np.where(nb, ar, 0)); idx[i] = (st[last] + ar - last) % n
    for k, e in series.items():
        E = e[idx]; acc[k].append(E.mean(1) / E.std(1) * np.sqrt(U.PPY))
    print('chunk', b0, round(time.time() - T0), flush=True)
boot = {k: np.concatenate(v) for k, v in acc.items()}
COMPS = [('W', 'EW'), ('M', 'EW'), ('L', 'EW'), ('M', 'W'), ('L', 'W'), ('XW', 'EW')]
summ = []
for c in CORE:
    f, L = c
    offs = range(HOLD_M[f])
    d = D[(D.freq == f) & (D.win == L)].pivot(index='off', columns='group', values='sr')
    row = dict(freq=f, win=L, ncal=len(d), **{f'sr_{g}': d[g].mean() for g in GROUPS},
               W_lt_EW=int((d.W < d.EW).sum()), M_gt_EW=int((d.M > d.EW).sum()), L_gt_EW=int((d.L > d.EW).sum()),
               min_WEW=(d.W - d.EW).min(), max_WEW=(d.W - d.EW).max())
    for a, b_ in COMPS:
        diff = (d[a] - d[b_]).mean()
        bd = np.mean([boot[(c, o, a)] - boot[(c, o, b_)] for o in offs], axis=0)
        row[f'{a}-{b_}'] = diff
        row[f'{a}-{b_}_p'] = min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))
    summ.append(row)
S = pd.DataFrame(summ)
pool = []
for pn, cs in [('short1m', [('A', 1), ('S', 1), ('Q', 1)]), ('core9', CORE)]:
    for a, b_ in COMPS:
        d = np.mean([S[(S.freq == c[0]) & (S.win == c[1])][f'{a}-{b_}'].iloc[0] for c in cs])
        bd = np.mean([np.mean([boot[(c, o, a)] - boot[(c, o, b_)] for o in range(HOLD_M[c[0]])], axis=0) for c in cs], axis=0)
        pool.append(dict(pool=pn, comp=f'{a}-{b_}', avg=d, lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5),
                         p=(np.abs(bd - d) >= abs(d)).mean()))
P = pd.DataFrame(pool)
pd.set_option('display.width', 250)
print(S.round(3).to_string()); print(P.round(3))
pickle.dump(dict(D=D, S=S, P=P), open('results/offsets.pkl', 'wb'))
print('DONE', round(time.time() - T0))
