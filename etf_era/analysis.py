"""All analyses for 'Rotating Sector ETFs with Dynamic Window Sizes'.

Writes results/*.csv and results/out.pkl. Factor models use SPY and a cyclical-minus-defensive sector spread.

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results', exist_ok=True)
import pickle
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from engine_sec import *

T0 = time.time()
out = {}
n = len(IDX)
rfv = RF.values
ALL = CONFIGS + EXT

# ------------------------------------------------------------------ 1. main backtests
res = {}
for c in ALL:
    for g in GROUPS:
        a = bt(c, g, spread=SPREAD)
        b = bt(c, g, spread=SPREAD_STRESS)
        res[(c, g)] = dict(gross=a['gross'], net=a['net'], net25=b['net'], picks=a['picks'], turn=a['turn'])
# EW buy-and-hold (never rebalanced)
x = R.values
h = np.ones(9) / 9
bh = np.empty(n)
for i in range(n):
    new = h * (1 + x[i]); bh[i] = new.sum() / h.sum() - 1; h = new
bh = pd.Series(bh, IDX)
bhnet = bh.copy(); bhnet.iloc[0] = (1 - SPREAD) * (1 + bh.iloc[0]) - 1; bhnet.iloc[-1] = (1 - SPREAD) * (1 + bh.iloc[-1]) - 1
out['bh'] = bh
rows = []
for (c, g), v in res.items():
    mg, mn, m25 = metrics(v['gross']), metrics(v['net']), metrics(v['net25'])
    t = v['turn']
    rows.append(dict(freq=c[0], L=c[1], cfg=clab(c), group=g, matched=matched(c), core=c in CONFIGS,
                     **{f'g_{a}': b for a, b in mg.items()}, **{f'n_{a}': b for a, b in mn.items()},
                     **{f'n25_{a}': b for a, b in m25.items()},
                     nreb=len(t) - 1, avg_oneway=(t[1:] / 2).mean(), ann_oneway=(t[1:] / 2).sum() / NY))
for nm, r in [('BH', bh), ('SPY', BENCH['SPY'])]:
    m = metrics(r)
    rows.append(dict(freq='-', L=-1, cfg=nm, group=nm, **{f'g_{a}': b for a, b in m.items()}))
MR = pd.DataFrame(rows)
MR.to_csv('results/main_results.csv', index=False)
out['MR'] = MR
print('main done', round(time.time() - T0))
gr = lambda c, g: res[(c, g)]['gross']

# ------------------------------------------------------------------ 2. bootstrap machinery
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
SRB = {}


def srb(key, r):
    if key not in SRB:
        E = (np.asarray(r) - rfv)[idxs]
        SRB[key] = E.mean(1) / E.std(1) * np.sqrt(252)
    return SRB[key]


def family(pairs):
    d = np.array([sr(a) - sr(b) for _, a, ka, b, kb in pairs])
    D = np.column_stack([srb(ka, a) - srb(kb, b) for _, a, ka, b, kb in pairs])
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
    df, _, _ = family([(f'{a}-{b}', gr(c, a), (c, a), gr(c, b), (c, b)) for a, b in PAIRS])
    for (a, b_), r_ in zip(PAIRS, df.itertuples()):
        boot.append(dict(freq=c[0], L=c[1], cfg=clab(c), a=a, b=b_, diff=r_.diff, lo=r_.lo, hi=r_.hi, p=r_.p))
out['boot'] = pd.DataFrame(boot)
out['boot'].to_csv('results/bootstrap_sharpe_tests.csv', index=False)
# selected strategies vs SPY: best Sharpe ratio of each group in the main grid, plus EW quarterly
MRc = MR[MR.core == True]
sel = []
for g in ['W', 'M', 'L', 'XW']:
    rr = MRc[MRc.group == g].sort_values('g_sr').iloc[-1]
    sel.append(((rr.freq, int(rr.L)), g))
sel.append((('Q', 3), 'EW'))
bb = []
for c, g in sel:
    df, _, _ = family([('x', gr(c, g), (c, g), BENCH['SPY'], 'SPY')])
    bb.append(dict(cfg=clab(c), freq=c[0], L=c[1], group=g, diff=df['diff'][0], lo=df.lo[0], hi=df.hi[0], p=df.p[0]))
out['bootbench'] = pd.DataFrame(bb)
print(out['bootbench'].round(3))
print('bootstrap done', round(time.time() - T0))

# ------------------------------------------------------------------ 3. multiple testing
famA = [(f'{clab(c)} {a}-{b}', gr(c, a), (c, a), gr(c, b), (c, b)) for c in CONFIGS for a, b in PAIRS[:4]]
out['famA'], _, _ = family(famA)
famB = [(f'{clab(c)} {g}-EW', gr(c, g), (c, g), gr(c, 'EW'), (c, 'EW')) for c in CONFIGS for g in ['W', 'M', 'L']]
out['famB'], DB, dB = family(famB)
famC = [(f'{clab(c)} {g}-SPY', gr(c, g), (c, g), BENCH['SPY'], 'SPY') for c in CONFIGS for g in ['W', 'M', 'L']]
out['famC'], DC, dC = family(famC)


def reality_check(D, d):
    stat = d.max()
    return stat, ((D - d).max(1) >= stat).mean()


out['rc'] = dict(best_vs_EW=reality_check(DB, dB), best_vs_SPY=reality_check(DC, dC),
                 best_label_EW=out['famB'].label[np.argmax(dB)], best_label_SPY=out['famC'].label[np.argmax(dC)])
# the "worst" side as well: is the worst Winner strategy significantly below EW?
famW = [(f'{clab(c)} W-EW', gr(c, 'W'), (c, 'W'), gr(c, 'EW'), (c, 'EW')) for c in CONFIGS]
out['famW'], DW, dW = family(famW)
out['rc']['worstW_vs_EW'] = reality_check(-DW, -dW)
print('RW min famA', out['famA'].p_rw.min().round(3), 'famB', out['famB'].p_rw.min().round(3), out['rc'])
print(out['famW'].round(3).to_string())
pd.concat([out['famA'].assign(family='A'), out['famB'].assign(family='B'), out['famW'].assign(family='W')]).to_csv(
    'results/romano_wolf.csv', index=False)


def pooled(D, d):
    a = d.mean(); ab = D.mean(1)
    lo, hi = np.percentile(ab, [2.5, 97.5])
    cm = np.corrcoef(D.T)[np.triu_indices(len(d), 1)] if len(d) > 1 else np.array([np.nan])
    return dict(avg=a, se=ab.std(), lo=lo, hi=hi, p=(np.abs(ab - a) >= abs(a)).mean(),
                npos=int((d > 0).sum()), n=len(d), mean_corr=np.nanmean(cm))


POOLS = {'all11': CONFIGS, 'core9': CORE,
         'short1m': [('A', 1), ('S', 1), ('Q', 1)],
         'matched5': [c for c in CONFIGS if matched(c)],
         'shorter6': [c for c in CORE if not matched(c)],
         'ext17': CONFIGS + EXT}
pool = []
for pn, cs in POOLS.items():
    for a, b in PAIRS:
        _, D, d = family([(clab(c), gr(c, a), (c, a), gr(c, b), (c, b)) for c in cs])
        pool.append(dict(pool=pn, comp=f'{a}-{b}', **pooled(D, d)))
out['pool'] = pd.DataFrame(pool)
out['pool'].to_csv('results/pooled_tests.csv', index=False)
print(out['pool'].round(3).to_string())
# difference between windows: 1-month window minus matched window, same group, same frequency
wd = []
for f in 'ASQ':
    for g in ['W', 'M', 'L', 'XW']:
        c1, cm = (f, 1), (f, HOLD_M[f])
        df, _, _ = family([('x', gr(c1, g), (c1, g), gr(cm, g), (cm, g))])
        wd.append(dict(freq=f, group=g, diff=df['diff'][0], lo=df.lo[0], hi=df.hi[0], p=df.p[0]))
for g in ['W', 'L']:
    _, D, d = family([(f, gr((f, 1), g), ((f, 1), g), gr((f, HOLD_M[f]), g), ((f, HOLD_M[f]), g)) for f in 'ASQ'])
    pq = pooled(D, d)
    wd.append(dict(freq='avg', group=g, diff=pq['avg'], lo=pq['lo'], hi=pq['hi'], p=pq['p']))
out['windiff'] = pd.DataFrame(wd)
print(out['windiff'].round(3))
print('multiple testing done', round(time.time() - T0))

# ------------------------------------------------------------------ 4. random portfolios (continue rng as in original)
S = 5000
rp = {}
for f in 'ASQMW':
    per = periods(f, {'A': 12, 'S': 6, 'Q': 3, 'M': 1, 'W': 0}[f])
    V = np.empty((S, n))
    for (s, e, past) in per:
        G = np.cumprod(1 + x[s:e], axis=0)
        sel_ = np.argsort(rng.random((S, 9)), axis=1)[:, :3]
        Wm = np.zeros((S, 9)); np.put_along_axis(Wm, sel_, 1 / 3, axis=1)
        path = Wm @ G.T
        prev = np.hstack([np.ones((S, 1)), path[:, :-1]])
        V[:, s:e] = path / prev - 1
    E = V - rfv
    rp[f] = E.mean(1) / E.std(1) * np.sqrt(252)
out['rp'] = rp
rpt = []
for c in CONFIGS + EXT:
    d = dict(cfg=clab(c), freq=c[0], win=c[1], med=np.median(rp[c[0]]),
             lo=np.percentile(rp[c[0]], 2.5), hi=np.percentile(rp[c[0]], 97.5))
    for g in ['W', 'M', 'L', 'EW']:
        s_ = sr(gr(c, g)); d[g] = s_; d[g + '_pct'] = (rp[c[0]] < s_).mean() * 100
    rpt.append(d)
out['rpt'] = pd.DataFrame(rpt)
print(out['rpt'].round(2).to_string())
print('random done', round(time.time() - T0))

# ------------------------------------------------------------------ 5. sub-periods
sp = []
for a, b in [('2000', '2012'), ('2013', '2024')]:
    for c in CONFIGS:
        for g in GROUPS:
            r = gr(c, g)[a:b]; e = r - RF[a:b]
            sp.append(dict(per=f'{a}-{b}', freq=c[0], L=c[1], group=g, sr=e.mean() / e.std() * np.sqrt(252),
                           cagr=(1 + r).prod() ** (252 / len(r)) - 1))
    r = BENCH['SPY'][a:b]; e = r - RF[a:b]
    sp.append(dict(per=f'{a}-{b}', freq='-', L=-1, group='SPY', sr=e.mean() / e.std() * np.sqrt(252),
                   cagr=(1 + r).prod() ** (252 / len(r)) - 1))
out['sp'] = pd.DataFrame(sp)
out['sp'].to_csv('results/subperiods.csv', index=False)

# ------------------------------------------------------------------ 6. implementation lag, LOO, 12 ETFs
lag = []
for c in CONFIGS:
    for g in GROUPS:
        m = metrics(bt(c, g, lag=1)['gross'])
        lag.append(dict(freq=c[0], L=c[1], group=g, sr=m['sr'], cagr=m['cagr']))
out['lag'] = pd.DataFrame(lag)
out['lag'].to_csv('results/implementation_lag.csv', index=False)
loo = []
for drop in TICK:
    uni = [t for t in TICK if t != drop]
    for c in CONFIGS:
        loo.append(dict(dropped=drop, freq=c[0], win=c[1], **{g: sr(bt(c, g, universe=uni)['gross']) for g in GROUPS}))
out['loo'] = pd.DataFrame(loo)
out['loo'].to_csv('results/leave_one_out.csv', index=False)
print('lag/loo done', round(time.time() - T0))

# ------------------------------------------------------------------ 7. composition and overlap between windows
comp = {}
for c in [('A', 1), ('A', 12), ('Q', 1), ('Q', 3)]:
    cnt = {}
    for g in ['W', 'M', 'L']:
        s_ = pd.Series(0.0, index=TICK)
        for pk in res[(c, g)]['picks']:
            s_[pk] += 1
        cnt[g] = s_ / len(res[(c, g)]['picks']) * 100
    comp[c] = pd.DataFrame(cnt)
out['comp'] = comp
ov = []
for f in 'ASQ':
    Ls = [L for (ff, L) in CONFIGS if ff == f]
    for i, L1 in enumerate(Ls):
        for L2 in Ls[i + 1:]:
            for g in ['W', 'M', 'L']:
                a_ = res[((f, L1), g)]['picks']; b_ = res[((f, L2), g)]['picks']
                ov.append(dict(freq=f, L1=L1, L2=L2, group=g,
                               overlap=np.mean([len(set(u) & set(v)) for u, v in zip(a_, b_)])))
    # rank correlation of the ranking signals
out['overlap'] = pd.DataFrame(ov)
rc_ = []
for f in 'ASQ':
    Ls = [L for (ff, L) in CONFIGS if ff == f]
    for i, L1 in enumerate(Ls):
        for L2 in Ls[i + 1:]:
            p1, p2 = periods(f, L1), periods(f, L2)
            rc_.append(dict(freq=f, L1=L1, L2=L2, rho=np.mean([q1[2][TICK].rank().corr(q2[2][TICK].rank())
                                                                for q1, q2 in zip(p1, p2)])))
out['rankcorr'] = pd.DataFrame(rc_)
print(out['overlap'].round(2).to_string()); print(out['rankcorr'].round(2))

# ------------------------------------------------------------------ 8. event-time profile (monthly rankings)
# At each month end m (Dec 2012 ... ), rank on the past L months; record the equal-weighted
# return of each group minus EW in month m+h, h = 1..12 (each month's ETF returns, equal weights).
mr = (1 + RA).groupby([RA.index.year, RA.index.month]).prod() - 1   # monthly returns, 2012-2025
mr.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq='M') for y, m in mr.index])
first_eval = pd.Period('2000-01', 'M')
ev = []
for L in [1, 3, 6, 12]:
    for j in range(len(mr)):
        mend = mr.index[j]
        if mend < first_eval - 1:
            continue
        win = mr.iloc[max(0, j - L + 1):j + 1]
        past = (1 + win).prod() - 1
        past = past[TICK]
        if past.isna().any():
            continue
        order = past.sort_values(ascending=False, kind='mergesort').index
        sets = {'W': list(order[:3]), 'M': list(order[3:6]), 'L': list(order[6:])}
        for hh in range(1, 13):
            if j + hh >= len(mr):
                break
            nxt = mr.iloc[j + hh]
            ewr = nxt[TICK].mean()
            ev.append(dict(win=L, rank_month=str(mend), h=hh, **{g: nxt[s_].mean() - ewr for g, s_ in sets.items()}))
EV = pd.DataFrame(ev)


def nw_mean(y, lags):
    y = np.asarray(y)
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
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

# ------------------------------------------------------------------ 9. factor regressions
mret = lambda r: (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
rf_m = (1 + RF).groupby([RF.index.year, RF.index.month]).prod() - 1
mkt = mret(BENCH['SPY']) - rf_m
cmd = mret(R[CYC].mean(axis=1)) - mret(R[DEF].mean(axis=1))   # cyclical minus defensive
X1 = sm.add_constant(pd.DataFrame({'MKT': mkt}))
X2 = sm.add_constant(pd.DataFrame({'MKT': mkt, 'CMD': cmd}))
MODELS = [('MKT', X1), ('MKT+CMD', X2)]


def reg(y, X):
    fit = sm.OLS(y.values, X.values).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return fit, dict(zip(X.columns, fit.params)), dict(zip(X.columns, fit.tvalues))


fac = []
for c in CONFIGS:
    ew = mret(gr(c, 'EW'))
    for g in GROUPS:
        y = mret(gr(c, g)) - rf_m
        for mname, X in MODELS:
            fit, b, t = reg(y, X)
            row = dict(freq=c[0], L=c[1], group=g, model=mname, alpha=b['const'] * 12, t=t['const'], r2=fit.rsquared,
                       **{f'b_{k}': v for k, v in b.items() if k != 'const'})
            if g != 'EW':
                fd, bd, td = reg(mret(gr(c, g)) - ew, X)
                row.update(alpha_d=bd['const'] * 12, t_d=td['const'],
                           **{f'bd_{k}': v for k, v in bd.items() if k != 'const'},
                           **{f'td_{k}': v for k, v in td.items() if k != 'const'})
            fac.append(row)
out['fac'] = pd.DataFrame(fac)
out['fac'].to_csv('results/factor_regressions.csv', index=False)
fp = []
for pn in ['all11', 'core9', 'short1m', 'matched5']:
    cs = POOLS[pn]
    for g in ['W', 'M', 'L', 'XW']:
        yd = sum(mret(gr(c, g)) - mret(gr(c, 'EW')) for c in cs) / len(cs)
        for mname, X in MODELS:
            fit, b, t = reg(yd, X)
            fp.append(dict(pool=pn, model=mname, group=g, alpha=b['const'] * 12, t=t['const']))
out['fac_pool'] = pd.DataFrame(fp)
print(out['fac_pool'].pivot_table(index=['pool', 'model'], columns='group', values=['alpha', 't']).round(3))
F = out['fac']
print(F[(F.model == 'MKT+CMD') & (F.group != 'EW')].pivot_table(index=['freq', 'L'], columns='group', values=['alpha_d', 't_d']).round(3))
out['nmonths'] = len(mkt)
out['cmd_mean'] = cmd.mean() * 12
pickle.dump(res, open('results/res.pkl', 'wb'))
pickle.dump(out, open('results/out.pkl', 'wb'))
print('ALL DONE', round(time.time() - T0))
