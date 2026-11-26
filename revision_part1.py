"""Revision analyses, part 1 (no bootstrap): implementation (lag x costs, EW net of costs, break-even
costs), weekly-rule definition, annualization conventions, extreme returns, industry composition and
size, cap-weighted groups. Output: results/rev1.pkl"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import time
import numpy as np
import pandas as pd
from engine_ind import *
from revision_common import *

T0 = time.time()
U = Universe(10)
PPY = U.PPY
out = {}
MATCHED = [('A', 12), ('S', 6), ('Q', 3), ('M', 1), ('W', 0)]

# sanity: generalized engine reproduces the paper's engine
for c in [('M', 1), ('W', 0), ('A', 1)]:
    a = U.bt(c, 'W', spread=0.001, lag=1)['net']; b = bt_general(U, c, 'W', spread=0.001, lag=1)['net']
    assert np.allclose(a.values, b.values), c


# ------------------------------------------------------------------ 1. lag x costs, EW treated identically
def sched_hist(d):
    """Era-specific stress schedule of one-way costs: 100 bp before the end of fixed commissions
    (1 May 1975), 50 bp until decimalization (2001), 10 bp afterwards."""
    if d < pd.Timestamp('1975-05-01'):
        return 0.0100
    if d < pd.Timestamp('2001-04-09'):
        return 0.0050
    return 0.0010


COSTS = [('0', 0.0), ('1.5', 0.00015), ('10', 0.0010), ('50', 0.0050), ('hist', sched_hist)]
imp = []
series = {}
for c in MATCHED + [('Q', 1), ('A', 1)]:
    for lag in [0, 1, 2, 5]:
        if lag in (2, 5) and c not in [('M', 1), ('W', 0)]:
            continue
        for cn, s in COSTS:
            for g in ['W', 'M', 'L', 'EW']:
                r = bt_general(U, c, g, lag=lag, spread=s)
                rr = r['net']
                series[(c, lag, cn, g)] = rr
                m = U.metrics(rr)
                imp.append(dict(freq=c[0], L=c[1], lag=lag, cost=cn, group=g, sr=m['sr'], cagr=m['cagr'],
                                final=m['final'] / 100, turn=(r['turn'][1:] / 2).mean()))
IMP = pd.DataFrame(imp)
out['imp'] = IMP
print(IMP[(IMP.freq.isin(['M', 'W']))].pivot_table(index=['freq', 'lag', 'cost'], columns='group', values='sr').round(2).to_string())


# break-even one-way cost at which net SR(group) = net SR(EW), same lag
def sr_at(c, g, lag, s):
    return U.sr(bt_general(U, c, g, lag=lag, spread=s)['net'])


be = []
for c in MATCHED:
    for lag in [0, 1]:
        for g in ['W']:
            f = lambda s: sr_at(c, g, lag, s) - sr_at(c, 'EW', lag, s)
            lo, hi = 0.0, 0.05
            if f(lo) <= 0:
                be.append(dict(freq=c[0], L=c[1], lag=lag, group=g, be_bp=0.0)); continue
            if f(hi) > 0:
                be.append(dict(freq=c[0], L=c[1], lag=lag, group=g, be_bp=np.inf)); continue
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
            be.append(dict(freq=c[0], L=c[1], lag=lag, group=g, be_bp=(lo + hi) / 2 * 1e4))
out['be'] = pd.DataFrame(be)
print(out['be'].round(1))
# sub-period break-even for monthly Winners
bes = []
for a, b in [('1928', '1962'), ('1963', '1998'), ('1999', '2025')]:
    for c in [('M', 1), ('W', 0)]:
        for lag in [0, 1]:
            def f(s):
                rw = bt_general(U, c, 'W', lag=lag, spread=s)['net'][a:b]
                re = bt_general(U, c, 'EW', lag=lag, spread=s)['net'][a:b]
                return U.sr(rw) - U.sr(re)
            lo, hi = 0.0, 0.05
            if f(lo) <= 0:
                v = 0.0
            else:
                for _ in range(25):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
                v = (lo + hi) / 2 * 1e4
            bes.append(dict(per=f'{a}-{b}', freq=c[0], lag=lag, be_bp=v))
out['be_sub'] = pd.DataFrame(bes)
print(out['be_sub'].round(1))
print('implementation done', round(time.time() - T0))

# ------------------------------------------------------------------ 2. weekly rule definition
perW = U.periods('W', 0)
ra_idx = U.RA.index
iso = ra_idx.isocalendar()
wk = (iso.year * 100 + iso.week).values
wk_days = pd.Series(wk).value_counts()
rank_days = []
for (s, e, past) in perW:
    pos = s + U._OFF
    prevk = wk[pos - 1]
    rank_days.append(dict(date=U.IDX[s], ndays=int((wk == prevk).sum())))
RD = pd.DataFrame(rank_days)
RD['era'] = np.where(RD.date < pd.Timestamp('1952-06-01'), 'pre1952', 'post1952')
out['weekdays'] = RD.groupby('era').ndays.value_counts().unstack(fill_value=0)
print(out['weekdays'])


# fixed five-trading-day ranking window ending at the rebalancing close
def periods_fixed_days(U, base, nd):
    newp = []
    for (s, e, past) in base:
        pos = s + U._OFF
        prev = U.RA.iloc[pos - nd:pos]
        newp.append((s, e, (1 + prev).prod() - 1))
    return newp


wk_alt = []
for lab, per in [('ISO week (paper)', perW), ('Last 5 trading days', periods_fixed_days(U, perW, 5)),
                 ('Last 10 trading days', periods_fixed_days(U, perW, 10))]:
    row = dict(rule=lab)
    for lag in [0, 1]:
        for g in ['W', 'M', 'L', 'EW']:
            row[f'{g}_lag{lag}'] = U.sr(bt_general(U, ('W', 0), g, lag=lag, periods=per)['gross'])
    wk_alt.append(row)
out['wk_alt'] = pd.DataFrame(wk_alt)
print(out['wk_alt'].round(2).to_string())

# ------------------------------------------------------------------ 3. annualization conventions
SUBP = [('1928', '1962'), ('1963', '1998'), ('1999', '2025')]
_G = {}


def gr(c, g):
    if (c, g) not in _G:
        _G[(c, g)] = U.bt(c, g)['gross']
    return _G[(c, g)]
rfm = mret(U.RF)
ann = []
for per in [('1928', '2025')] + SUBP:
    a, b = per
    idx = U.IDX[(U.IDX >= pd.Timestamp(f'{a}-01-01')) & (U.IDX <= pd.Timestamp(f'{b}-12-31'))]
    ppy_sub = len(idx) / (((idx[-1] - idx[0]).days + 1) / 365.25)
    for c in CONFIGS:
        for g in ['W', 'M', 'L', 'EW']:
            r = gr(c, g)[a:b]
            e = r - U.RF.loc[r.index]
            sr262 = e.mean() / e.std() * np.sqrt(PPY)
            srown = e.mean() / e.std() * np.sqrt(ppy_sub)
            m = mret(r); em = m - rfm.loc[m.index]
            srm = em.mean() / em.std() * np.sqrt(12)
            ann.append(dict(per=f'{a}-{b}', ppy=ppy_sub, freq=c[0], L=c[1], group=g, sr262=sr262, srown=srown, srm=srm))
ANN = pd.DataFrame(ann)
out['ann'] = ANN
print(ANN.groupby('per').ppy.first())
print('annualization done', round(time.time() - T0))

# ------------------------------------------------------------------ 4. extreme returns and winsorized rankings
R = U.RA
q = R.stack().quantile([0.001, 0.005, 0.995, 0.999])
ext = dict(min=R.min().min(), max=R.max().max(), q=q,
           n_abs10=int((R.abs() > 0.10).sum().sum()), n_abs10_2933=int((R['1929':'1933'].abs() > 0.10).sum().sum()),
           worst_days=R.min(axis=1).nsmallest(5), best_days=R.max(axis=1).nlargest(5))
out['ext'] = ext
lo_, hi_ = R.stack().quantile([0.005, 0.995])
RW = R.clip(lower=lo_, upper=hi_)


def periods_from(Rsrc, base, c):
    """Recompute ranking returns of the base periods using an alternative daily return panel."""
    f, L = c
    newp = []
    MK = U._MK
    if f == 'W':
        for (s, e, past) in base:
            pos = s + U._OFF
            prevk = wk[pos - 1]
            sel = np.where(wk == prevk)[0]
            newp.append((s, e, (1 + Rsrc.iloc[sel]).prod() - 1))
    else:
        for (s, e, past) in base:
            pos = s + U._OFF
            m0 = MK[pos]
            sel = np.where((MK >= m0 - L) & (MK <= m0 - 1))[0]
            newp.append((s, e, (1 + Rsrc.iloc[sel]).prod() - 1))
    return newp


# rank-based alternative: average daily rank of each industry over the window (insensitive to magnitudes)
RK = R.rank(axis=1, pct=True)
win = []
for c in CONFIGS:
    base = U.periods(*c)
    alts = {'Winsorized (0.5%/99.5%)': periods_from(RW, base, c),
            'Mean daily rank': [(s, e, p) for (s, e, _), (_, _, p) in zip(base, periods_from(RK - 0.5, base, c))]}
    row0 = dict(freq=c[0], L=c[1])
    for g in ['W', 'M', 'L', 'EW']:
        row0[f'base_{g}'] = U.sr(gr(c, g))
    for lab, per in alts.items():
        chg = np.mean([set(p0[U.TICK].sort_values(ascending=False, kind='mergesort').index[:3]) !=
                       set(p1[U.TICK].sort_values(ascending=False, kind='mergesort').index[:3]) for (_, _, p0), (_, _, p1) in zip(base, per)])
        for g in ['W', 'M', 'L', 'EW']:
            row0[f'{lab}_{g}'] = U.sr(bt_general(U, c, g, periods=per)['gross'])
        row0[f'{lab}_chg'] = chg
    win.append(row0)
out['win'] = pd.DataFrame(win)
print(out['win'][['freq', 'L', 'base_W', 'Winsorized (0.5%/99.5%)_W', 'Mean daily rank_W', 'Winsorized (0.5%/99.5%)_chg', 'Mean daily rank_chg']].round(3))
# evaluation excluding crisis years
exc = []
for lab, drop in [('Full sample', []), ('Excluding 1929--1933', [('1929', '1933')]),
                  ('Excluding 1929--1933 and 1987', [('1929', '1933'), ('1987', '1987')]),
                  ('Excluding 2008--2009', [('2008', '2009')])]:
    mask = pd.Series(True, U.IDX)
    for a, b in drop:
        mask[a:b] = False
    row = dict(sample=lab)
    for c in [('M', 1), ('W', 0)]:
        for g in ['W', 'M', 'L', 'EW']:
            r = gr(c, g)[mask.values]
            row[f'{c[0]}_{g}'] = U.sr(r)
    exc.append(row)
out['exc'] = pd.DataFrame(exc)
print(out['exc'].round(2).to_string())
print('outliers done', round(time.time() - T0))

# ------------------------------------------------------------------ 5. industry size and composition over time
cap, nf = month_caps(10)
share = cap.div(cap.sum(1), axis=0)
dates = ['1927-12-31', '1945-12-31', '1962-12-31', '1980-12-31', '1998-12-31', '2025-11-30']
comp = []
for d in dates:
    dd = pd.Timestamp(d)
    row_n = nf.loc[nf.index <= dd + pd.offsets.MonthEnd(1)].iloc[-1]
    row_s = share.loc[:dd].iloc[-1]
    comp.append(dict(date=d, **{f'n_{t}': row_n[t] for t in U.TICK}, **{f's_{t}': row_s[t] for t in U.TICK}, n_tot=row_n.sum()))
out['comp_size'] = pd.DataFrame(comp)
print(out['comp_size'].round(3).T)
out['minfirms'] = nf['1928':'1962'].min()
print(out['minfirms'])

# ------------------------------------------------------------------ 6. cap-weighted groups and benchmarks; no small-industry
capw = cap.copy()
capd = capw  # month-end caps, looked up at the most recent month end before each trade
cw = []
for c in CONFIGS:
    row = dict(freq=c[0], L=c[1])
    for g in ['W', 'M', 'L', 'EW']:
        r = bt_general(U, c, g, weights='cap', caps=capd)['gross']
        row[f'cw_{g}'] = U.sr(r)
        series[('cap', c, g)] = r
        row[f'ew_{g}'] = U.sr(gr(c, g))
        for a, b in SUBP:
            row[f'cw_{g}_{a}'] = U.sr(r[a:b])
    row['mkt'] = U.sr(U.MKT)
    cw.append(row)
out['cw'] = pd.DataFrame(cw)
print(out['cw'][['freq', 'L', 'cw_W', 'cw_M', 'cw_L', 'cw_EW', 'ew_W', 'ew_EW', 'mkt']].round(2).to_string())

# drop the industries that were small in the early decades
small = share['1927-12':'1962-12'].mean().sort_values()
out['early_share'] = small
print(small)
drop2 = list(small.index[:2])
uni8 = [t for t in U.TICK if t not in drop2]
sm8 = []
for c in CONFIGS:
    row = dict(freq=c[0], L=c[1])
    for g in ['W', 'M', 'L', 'EW']:
        r = bt_general(U, c, g, universe=uni8)['gross']
        row[g] = U.sr(r)
        for a, b in SUBP:
            row[f'{g}_{a}'] = U.sr(r[a:b])
    sm8.append(row)
out['drop_small'] = pd.DataFrame(sm8)
out['drop_small_names'] = drop2
print(drop2, out['drop_small'].round(2).to_string())

pickle.dump(out, open('results/rev1.pkl', 'wb'))
pickle.dump(series, open('results/rev1_series.pkl', 'wb'))
print('done', round(time.time() - T0))
