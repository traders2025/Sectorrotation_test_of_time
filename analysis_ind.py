"""All analyses for the industry-rotation study (Fama-French industries, 1928-2025).
Writes results/*.csv and results/out.pkl, results/res.pkl."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results', exist_ok=True)
import pickle
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from engine_ind import *
from engine_ind import _read_block

T0 = time.time()
U = Universe(10)
n = len(U.IDX)
rfv = U.RF.values
PPY = U.PPY
ALL = CONFIGS + EXT
out = dict(N=U.N, k=U.k, km=U.km, NY=U.NY, PPY=PPY, start=U.IDX[0], end=U.IDX[-1], tick=U.TICK)

# ------------------------------------------------------------------ 1. main backtests
res = {}
for c in ALL:
    for g in GROUPS:
        a = U.bt(c, g, spread=SPREAD)
        res[(c, g)] = dict(gross=a['gross'], net=a['net'], picks=a['picks'], turn=a['turn'],
                           net_etf=U.bt(c, g, spread=SPREAD_ETF)['net'], net50=U.bt(c, g, spread=SPREAD_STRESS)['net'])
x = U.R.values
h = np.ones(U.N) / U.N
cum = np.cumprod(1 + x, axis=0) * h
tot = cum.sum(1)
bh = pd.Series(tot / np.concatenate([[1.0], tot[:-1]]) - 1, U.IDX)
rows = []
for (c, g), v in res.items():
    mg, mn, me, m50 = U.metrics(v['gross']), U.metrics(v['net']), U.metrics(v['net_etf']), U.metrics(v['net50'])
    t = v['turn']
    rows.append(dict(freq=c[0], L=c[1], cfg=clab(c), group=g, matched=matched(c), core=c in CONFIGS,
                     **{f'g_{a}': b for a, b in mg.items()}, **{f'n_{a}': b for a, b in mn.items()},
                     **{f'ne_{a}': b for a, b in me.items()}, **{f'n50_{a}': b for a, b in m50.items()},
                     nreb=len(t) - 1, avg_oneway=(t[1:] / 2).mean(), ann_oneway=(t[1:] / 2).sum() / U.NY))
for nm, r in [('BH', bh), ('MKT', U.MKT)]:
    rows.append(dict(freq='-', L=-1, cfg=nm, group=nm, **{f'g_{a}': b for a, b in U.metrics(r).items()}))
MR = pd.DataFrame(rows)
MR.to_csv('results/main_results.csv', index=False)
out['MR'] = MR
print('main done', round(time.time() - T0))
gr = lambda c, g: res[(c, g)]['gross']

# individual industries
ind = []
for t in U.TICK:
    m = U.metrics(U.R[t]); ind.append(dict(t=t, **m))
ind.append(dict(t='MKT', **U.metrics(U.MKT)))
out['ind'] = pd.DataFrame(ind)


# ------------------------------------------------------------------ 2. chunked stationary bootstrap
def boot_indices(B, seed, chunk):
    rng = np.random.default_rng(seed)
    ar = np.arange(n)
    for b0 in range(0, B, chunk):
        m = min(chunk, B - b0)
        idx = np.empty((m, n), dtype=np.int32)
        for i in range(m):
            st = rng.integers(0, n, n)
            nb = rng.random(n) < 1 / 21
            nb[0] = True
            last = np.maximum.accumulate(np.where(nb, ar, 0))
            idx[i] = (st[last] + ar - last) % n
        yield idx


def boot_all(series, B, seed=20260924, chunk=250):
    """series: dict key -> daily return array. Returns dict key -> array (B,) of bootstrap Sharpe ratios."""
    ex = {k: np.asarray(v) - rfv for k, v in series.items()}
    outb = {k: [] for k in series}
    for idx in boot_indices(B, seed, chunk):
        for k, e in ex.items():
            E = e[idx]
            outb[k].append(E.mean(1) / E.std(1) * np.sqrt(PPY))
    return {k: np.concatenate(v) for k, v in outb.items()}


ser = {(c, g): gr(c, g).values for c in ALL for g in GROUPS}
ser['MKT'] = U.MKT.values
SRB = boot_all(ser, 5000)
SR0 = {k: U.sr(pd.Series(v, U.IDX)) for k, v in ser.items()}
print('bootstrap done', round(time.time() - T0))


def family(pairs):
    """pairs: list of (label, key_a, key_b)."""
    d = np.array([SR0[a] - SR0[b] for _, a, b in pairs])
    D = np.column_stack([SRB[a] - SRB[b] for _, a, b in pairs])
    se = D.std(0)
    t = np.abs(d) / se
    tb = np.abs(D - d) / se
    m = len(d)
    p_un = np.array([2 * min((D[:, j] <= 0).mean(), (D[:, j] >= 0).mean()) for j in range(m)]).clip(max=1)
    order = np.argsort(-t)
    p_rw = np.empty(m); prev = 0.0
    for j, ix in enumerate(order):
        pv = max((tb[:, order[j:]].max(1) >= t[ix]).mean(), prev)
        p_rw[ix] = pv; prev = pv
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(label=[q[0] for q in pairs], diff=d, se=se, lo=lo, hi=hi, p=p_un, p_rw=p_rw)), D, d


PAIRS = [('M', 'W'), ('M', 'EW'), ('W', 'EW'), ('L', 'W'), ('L', 'EW'), ('XW', 'EW')]
boot = []
for c in ALL:
    df, _, _ = family([(f'{a}-{b}', (c, a), (c, b)) for a, b in PAIRS])
    for (a, b_), r_ in zip(PAIRS, df.itertuples()):
        boot.append(dict(freq=c[0], L=c[1], cfg=clab(c), a=a, b=b_, diff=r_.diff, lo=r_.lo, hi=r_.hi, p=r_.p))
out['boot'] = pd.DataFrame(boot)
out['boot'].to_csv('results/bootstrap_sharpe_tests.csv', index=False)
bb = []
MRc = MR[MR.core == True]
sel = []
for g in ['W', 'M', 'L', 'XW']:
    rr = MRc[MRc.group == g].sort_values('g_sr').iloc[-1]
    sel.append(((rr.freq, int(rr.L)), g))
sel.append((('M', 1), 'EW'))
for c, g in sel:
    df, _, _ = family([('x', (c, g), 'MKT')])
    bb.append(dict(cfg=clab(c), freq=c[0], L=c[1], group=g, diff=df['diff'][0], lo=df.lo[0], hi=df.hi[0], p=df.p[0]))
out['bootbench'] = pd.DataFrame(bb)
print(out['bootbench'].round(3))

famA = [(f'{clab(c)} {a}-{b}', (c, a), (c, b)) for c in CONFIGS for a, b in PAIRS[:4]]
out['famA'], _, _ = family(famA)
famB = [(f'{clab(c)} {g}-EW', (c, g), (c, 'EW')) for c in CONFIGS for g in ['W', 'M', 'L']]
out['famB'], DB, dB = family(famB)
famC = [(f'{clab(c)} {g}-MKT', (c, g), 'MKT') for c in CONFIGS for g in ['W', 'M', 'L']]
out['famC'], DC, dC = family(famC)
famW = [(f'{clab(c)} W-EW', (c, 'W'), (c, 'EW')) for c in CONFIGS]
out['famW'], DW, dW = family(famW)
famM = [(f'{clab(c)} M-EW', (c, 'M'), (c, 'EW')) for c in CONFIGS]
out['famM'], DMm, dMm = family(famM)


def reality_check(D, d):
    stat = d.max()
    return stat, ((D - d).max(1) >= stat).mean()


out['rc'] = dict(best_vs_EW=reality_check(DB, dB), best_label_EW=out['famB'].label[np.argmax(dB)],
                 best_vs_MKT=reality_check(DC, dC), best_label_MKT=out['famC'].label[np.argmax(dC)],
                 worstW_vs_EW=reality_check(-DW, -dW), worstW_label=out['famW'].label[np.argmin(dW)],
                 bestW_vs_EW=reality_check(DW, dW), bestW_label=out['famW'].label[np.argmax(dW)],
                 bestM_vs_EW=reality_check(DMm, dMm), bestM_label=out['famM'].label[np.argmax(dMm)])
print(out['rc'])
pd.concat([out['famA'].assign(family='A'), out['famB'].assign(family='B'), out['famW'].assign(family='W'),
           out['famM'].assign(family='M')]).to_csv('results/romano_wolf.csv', index=False)


def pooled(D, d):
    a = d.mean(); ab = D.mean(1)
    lo, hi = np.percentile(ab, [2.5, 97.5])
    cm = np.corrcoef(D.T)[np.triu_indices(len(d), 1)] if len(d) > 1 else np.array([np.nan])
    return dict(avg=a, se=ab.std(), lo=lo, hi=hi, p=(np.abs(ab - a) >= abs(a)).mean(),
                npos=int((d > 0).sum()), n=len(d), mean_corr=np.nanmean(cm))


POOLS = {'all11': CONFIGS, 'core9': CORE, 'short1m': [('A', 1), ('S', 1), ('Q', 1)],
         'matched5': [c for c in CONFIGS if matched(c)], 'shorter6': [c for c in CORE if not matched(c)],
         'ext17': CONFIGS + EXT}
pool = []
for pn, cs in POOLS.items():
    for a, b in PAIRS:
        _, D, d = family([(clab(c), (c, a), (c, b)) for c in cs])
        pool.append(dict(pool=pn, comp=f'{a}-{b}', **pooled(D, d)))
out['pool'] = pd.DataFrame(pool)
out['pool'].to_csv('results/pooled_tests.csv', index=False)
print(out['pool'][out['pool'].comp.isin(['W-EW', 'M-EW', 'L-EW'])].round(3).to_string())
wd = []
for f in 'ASQ':
    for g in ['W', 'M', 'L', 'XW']:
        c1, cm = (f, 1), (f, HOLD_M[f])
        df, _, _ = family([('x', (c1, g), (cm, g))])
        wd.append(dict(freq=f, group=g, diff=df['diff'][0], lo=df.lo[0], hi=df.hi[0], p=df.p[0]))
for g in ['W', 'M', 'L']:
    _, D, d = family([(f, ((f, 1), g), ((f, HOLD_M[f]), g)) for f in 'ASQ'])
    pq = pooled(D, d)
    wd.append(dict(freq='avg', group=g, diff=pq['avg'], lo=pq['lo'], hi=pq['hi'], p=pq['p']))
out['windiff'] = pd.DataFrame(wd)
print('multiple testing done', round(time.time() - T0))

# ------------------------------------------------------------------ 3. random portfolios (running moments, no storage)
rng = np.random.default_rng(20260925)
S = 5000
rp = {}
for f in 'ASQMW':
    per = U.periods(f, {'A': 12, 'S': 6, 'Q': 3, 'M': 1, 'W': 0}[f])
    for size in (U.k, U.km):
        s1 = np.zeros(S); s2 = np.zeros(S)
        for (s, e, past) in per:
            G = np.cumprod(1 + x[s:e], axis=0)
            sel_ = np.argsort(rng.random((S, U.N)), axis=1)[:, :size]
            Wm = np.zeros((S, U.N)); np.put_along_axis(Wm, sel_, 1 / size, axis=1)
            path = Wm @ G.T
            prev = np.hstack([np.ones((S, 1)), path[:, :-1]])
            Rr = path / prev - 1 - rfv[s:e]
            s1 += Rr.sum(1); s2 += (Rr ** 2).sum(1)
        mu = s1 / n; sd = np.sqrt(s2 / n - mu ** 2)
        rp[(f, size)] = mu / sd * np.sqrt(PPY)
out['rp'] = rp
rpt = []
for c in ALL:
    f = c[0]
    d = dict(cfg=clab(c), freq=f, win=c[1], med3=np.median(rp[(f, U.k)]), lo3=np.percentile(rp[(f, U.k)], 2.5),
             hi3=np.percentile(rp[(f, U.k)], 97.5), med4=np.median(rp[(f, U.km)]))
    for g in ['W', 'M', 'L', 'EW']:
        s_ = SR0[(c, g)]; d[g] = s_
        dist = rp[(f, U.km)] if g == 'M' else rp[(f, U.k)]
        d[g + '_pct'] = (dist < s_).mean() * 100
    rpt.append(d)
out['rpt'] = pd.DataFrame(rpt)
print('random done', round(time.time() - T0))

# ------------------------------------------------------------------ 4. sub-periods
SUBP = [('1928', '1962'), ('1963', '1998'), ('1999', '2025')]
sp = []
for a, b in SUBP:
    for c in ALL:
        for g in GROUPS:
            r = gr(c, g)[a:b]
            sp.append(dict(per=f'{a}-{b}', freq=c[0], L=c[1], group=g, sr=U.sr(r), cagr=U.metrics(r)['cagr']))
    r = U.MKT[a:b]
    sp.append(dict(per=f'{a}-{b}', freq='-', L=-1, group='MKT', sr=U.sr(r), cagr=U.metrics(r)['cagr']))
out['sp'] = pd.DataFrame(sp)
out['sp'].to_csv('results/subperiods.csv', index=False)
# sub-period bootstrap tests for W-EW and M-EW (pooled over one-month windows and per config) using sub-sample bootstrap
spb = []
for a, b in SUBP:
    mask = (U.IDX >= pd.Timestamp(f'{a}-01-01')) & (U.IDX <= pd.Timestamp(f'{b}-12-31'))
    nn = mask.sum()
    rr = np.random.default_rng(7)
    keys_ = [(c, g) for c in CONFIGS for g in ['W', 'M', 'L', 'EW']]
    exs = {k: (gr(*k).values - rfv)[mask] for k in keys_}
    accum = {k: [] for k in keys_}
    ar = np.arange(nn)
    for b0 in range(0, 2000, 200):
        idx = np.empty((200, nn), dtype=np.int32)
        for i in range(200):
            st = rr.integers(0, nn, nn); nb = rr.random(nn) < 1 / 21; nb[0] = True
            last = np.maximum.accumulate(np.where(nb, ar, 0)); idx[i] = (st[last] + ar - last) % nn
        for k in keys_:
            E = exs[k][idx]; accum[k].append(E.mean(1) / E.std(1) * np.sqrt(PPY))
    sb = {k: np.concatenate(v) for k, v in accum.items()}
    s0 = {k: exs[k].mean() / exs[k].std() * np.sqrt(PPY) for k in keys_}
    for c in CONFIGS:
        for g in ['W', 'M', 'L']:
            d = s0[(c, g)] - s0[(c, 'EW')]; D = sb[(c, g)] - sb[(c, 'EW')]
            spb.append(dict(per=f'{a}-{b}', freq=c[0], L=c[1], group=g, diff=d,
                            p=min(1, 2 * min((D <= 0).mean(), (D >= 0).mean()))))
    for pn in ['short1m', 'matched5', 'all11']:
        for g in ['W', 'M', 'L']:
            cs = POOLS[pn]
            d = np.mean([s0[(c, g)] - s0[(c, 'EW')] for c in cs]); D = np.mean([sb[(c, g)] - sb[(c, 'EW')] for c in cs], axis=0)
            spb.append(dict(per=f'{a}-{b}', freq=pn, L=-1, group=g, diff=d, p=(np.abs(D - d) >= abs(d)).mean(),
                            lo=np.percentile(D, 2.5), hi=np.percentile(D, 97.5)))
out['spb'] = pd.DataFrame(spb)
print(out['spb'][out['spb'].L == -1].round(3).to_string())
print('subperiods done', round(time.time() - T0))

# ------------------------------------------------------------------ 5. lag, leave-one-out, alternative universes
lag = []
for c in CONFIGS:
    for g in GROUPS:
        r = U.bt(c, g, lag=1)['gross']
        lag.append(dict(freq=c[0], L=c[1], group=g, sr=U.sr(r)))
out['lag'] = pd.DataFrame(lag)
loo = []
for drop in U.TICK:
    uni = [t for t in U.TICK if t != drop]
    for c in CONFIGS:
        loo.append(dict(dropped=drop, freq=c[0], win=c[1], **{g: U.sr(U.bt(c, g, universe=uni)['gross']) for g in GROUPS}))
out['loo'] = pd.DataFrame(loo)
alt = []
for lab, kw in [('9 industries (excl. Other)', dict(n_ind=10, drop=['Other'])), ('12 industries', dict(n_ind=12)),
                ('49 industries (from 1973)', dict(n_ind=49, start='1973-01-01')), ('10 industries, equal-weighted', dict(n_ind=10, weighting='ew'))]:
    try:
        V = Universe(**kw)
    except AssertionError as e:
        # 49-industry daily file has missing early values for some industries: start when complete
        V = None
    if V is None:
        continue
    for c in CONFIGS:
        alt.append(dict(universe=lab, N=V.N, k=V.k, km=V.km, freq=c[0], win=c[1],
                        **{g: V.sr(V.bt(c, g)['gross']) for g in GROUPS}))
    print(lab, 'done', round(time.time() - T0))
out['alt'] = pd.DataFrame(alt)
print('robustness done', round(time.time() - T0))

# ------------------------------------------------------------------ 6. composition and overlap
comp = {}
for c in [('A', 1), ('A', 12), ('M', 1), ('Q', 1), ('Q', 3)]:
    cnt = {}
    for g in ['W', 'M', 'L']:
        s_ = pd.Series(0.0, index=U.TICK)
        for pk in res[(c, g)]['picks']:
            s_[pk] += 1
        cnt[g] = s_ / len(res[(c, g)]['picks']) * 100
    comp[c] = pd.DataFrame(cnt)
out['comp'] = comp
ov, rc_ = [], []
for f in 'ASQ':
    Ls = [L for (ff, L) in CONFIGS if ff == f]
    for i, L1 in enumerate(Ls):
        for L2 in Ls[i + 1:]:
            for g in ['W', 'M', 'L']:
                a_ = res[((f, L1), g)]['picks']; b_ = res[((f, L2), g)]['picks']
                ov.append(dict(freq=f, L1=L1, L2=L2, group=g, overlap=np.mean([len(set(u) & set(v)) for u, v in zip(a_, b_)])))
            p1, p2 = U.periods(f, L1), U.periods(f, L2)
            rc_.append(dict(freq=f, L1=L1, L2=L2, rho=np.mean([q1[2][U.TICK].rank().corr(q2[2][U.TICK].rank()) for q1, q2 in zip(p1, p2)])))
out['overlap'] = pd.DataFrame(ov)
out['rankcorr'] = pd.DataFrame(rc_)

# ------------------------------------------------------------------ 7. event time (monthly rankings)
mr = (1 + U.RA).groupby([U.RA.index.year, U.RA.index.month]).prod() - 1
mr.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq='M') for y, m in mr.index])
first_eval = pd.Period('1928-01', 'M')
ev = []
for L in [1, 3, 6, 12]:
    for j in range(len(mr)):
        mend = mr.index[j]
        if mend < first_eval - 1 or j - L + 1 < 1:
            continue
        past = ((1 + mr.iloc[j - L + 1:j + 1]).prod() - 1)[U.TICK]
        order = past.sort_values(ascending=False, kind='mergesort').index
        sets = {'W': list(order[:U.k]), 'M': list(order[U.k:U.N - U.k]), 'L': list(order[U.N - U.k:])}
        for hh in range(1, 13):
            if j + hh >= len(mr) or mr.index[j + hh] > pd.Period('2025-12', 'M'):
                break
            nxt = mr.iloc[j + hh]
            ewr = nxt[U.TICK].mean()
            ev.append(dict(win=L, rank_month=str(mend), h=hh, **{g: nxt[s_].mean() - ewr for g, s_ in sets.items()}))
EV = pd.DataFrame(ev)


def nw_mean(y, lags):
    fit = sm.OLS(np.asarray(y), np.ones(len(y))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return fit.params[0], fit.tvalues[0]


evt = []
for L in [1, 3, 6, 12]:
    for lab, hs in [('1', [1]), ('2', [2]), ('3', [3]), ('4-6', [4, 5, 6]), ('7-12', list(range(7, 13)))]:
        d_ = EV[(EV.win == L) & (EV.h.isin(hs))].groupby('rank_month')[['W', 'M', 'L']].mean()
        row = dict(win=L, h=lab, nobs=len(d_))
        for g in ['W', 'M', 'L']:
            mu, t = nw_mean(d_[g], max(hs) + 2)
            row[g] = mu * 100; row[g + '_t'] = t
        evt.append(row)
out['EV'] = EV
out['evt'] = pd.DataFrame(evt)
out['evt'].to_csv('results/event_time.csv', index=False)
print(out['evt'].round(2).to_string())
# December vs other months, one-month window, horizon 1-12 (annual) and 1-3
E1 = EV[EV.win == 1].copy(); E1['cm'] = E1.rank_month.str[-2:].astype(int)
dec = []
for H in [1, 3, 6, 12]:
    dd = E1[E1.h <= H].groupby(['rank_month', 'cm'])[['W', 'M', 'L']].mean().reset_index()
    for g in ['W', 'M', 'L']:
        for lab, sub in [('Dec', dd[dd.cm == 12]), ('Jun', dd[dd.cm == 6]), ('Other', dd[~dd.cm.isin([6, 12])])]:
            mu, t = nw_mean(sub[g], H + 2) if len(sub) > 5 else (np.nan, np.nan)
            dec.append(dict(H=H, group=g, months=lab, mean=mu * 100, t=t, nobs=len(sub)))
out['dec'] = pd.DataFrame(dec)
print(out['dec'].round(2).to_string())

pickle.dump(res, open('results/res.pkl', 'wb'))
pickle.dump(out, open('results/out_partial.pkl', 'wb'))
# ------------------------------------------------------------------ 8. factor regressions (monthly)
mret = lambda r: (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
ff3 = _read_block('F-F_Research_Data_Factors_CSV.zip')
mom = _read_block('F-F_Momentum_Factor_CSV.zip')
ff5 = _read_block('F-F_Research_Data_5_Factors_2x3_CSV.zip')
for d_ in (ff3, mom, ff5):
    d_.index = pd.MultiIndex.from_arrays([d_.index.year, d_.index.month])
mom.columns = ['MOM']
kidx = mret(U.MKT).index
X4 = sm.add_constant(pd.concat([ff3[['Mkt-RF', 'SMB', 'HML']], mom], axis=1).reindex(kidx))
X6 = sm.add_constant(pd.concat([ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], mom], axis=1).reindex(kidx)).dropna()
MODELS = [('FF3+MOM', X4), ('FF5+MOM', X6)]
out['facmeans'] = pd.concat([ff3, mom], axis=1).reindex(kidx).mean() * 12


def reg(y, X):
    y = y.reindex(X.index)
    fit = sm.OLS(y.values, X.values).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return fit, dict(zip(X.columns, fit.params)), dict(zip(X.columns, fit.tvalues))


fac = []
for c in CONFIGS:
    ew = mret(gr(c, 'EW'))
    for g in ['W', 'M', 'L', 'XW']:
        yd = mret(gr(c, g)) - ew
        for mname, X in MODELS:
            fit, b, t = reg(yd, X)
            fac.append(dict(freq=c[0], L=c[1], group=g, model=mname, alpha_d=b['const'] * 12, t_d=t['const'],
                            nobs=int(fit.nobs), **{f'bd_{k}': v for k, v in b.items() if k != 'const'},
                            **{f'td_{k}': v for k, v in t.items() if k != 'const'}))
out['fac'] = pd.DataFrame(fac)
out['fac'].to_csv('results/factor_regressions.csv', index=False)
fp = []
for pn in ['all11', 'short1m', 'matched5']:
    cs = POOLS[pn]
    for g in ['W', 'M', 'L', 'XW']:
        yd = sum(mret(gr(c, g)) - mret(gr(c, 'EW')) for c in cs) / len(cs)
        for mname, X in MODELS:
            fit, b, t = reg(yd, X)
            fp.append(dict(pool=pn, model=mname, group=g, alpha=b['const'] * 12, t=t['const'], b_MOM=b['MOM'], t_MOM=t['MOM']))
out['fac_pool'] = pd.DataFrame(fp)
print(out['fac_pool'].round(3).to_string())
out['nmonths'] = {m: len(X) for m, X in MODELS}
pickle.dump(res, open('results/res.pkl', 'wb'))
pickle.dump(out, open('results/out.pkl', 'wb'))
print('ALL DONE', round(time.time() - T0))
