"""Revision analyses, part 2 (bootstrap and time-series tests): block-length sensitivity, a broader
multiple-testing family, net-of-cost tests with a trading lag, structural-break tests for the decline
of industry momentum, and mechanism diagnostics. Output: results/rev2.pkl"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from engine_ind import *
from revision_common import *

T0 = time.time()
U = Universe(10)
PPY = U.PPY
rf = U.RF.values
out = {}
B = 2000
MATCHED = [('A', 12), ('S', 6), ('Q', 3), ('M', 1), ('W', 0)]
ONE = [('A', 1), ('S', 1), ('Q', 1)]

G = {}
for c in CONFIGS + EXT:
    for g in ['W', 'M', 'L', 'EW']:
        G[('g', c, g)] = U.bt(c, g)['gross']
for c in CONFIGS:
    for g in ['W', 'M', 'L', 'EW']:
        G[('lag1', c, g)] = U.bt(c, g, lag=1)['gross']
        G[('net10', c, g)] = U.bt(c, g, spread=0.0010)['net']
G['MKT'] = U.MKT
EX = {k: v.values - rf for k, v in G.items()}
SR0 = {k: e.mean() / e.std() * np.sqrt(PPY) for k, e in EX.items()}
print('series', len(EX), round(time.time() - T0))

# ------------------------------------------------------------------ 1. block length sensitivity
core_keys = [k for k in EX if k == 'MKT' or (k[0] == 'g' and k[1] in CONFIGS)]
bl = []
for Lb in [5, 10, 21, 63, 126, 252]:
    SB = boot_sr({k: EX[k] for k in core_keys}, B, Lb, PPY, seed=1000 + Lb)
    D = lambda a, b: SB[a] - SB[b]
    d = lambda a, b: SR0[a] - SR0[b]
    row = dict(block=Lb)
    for c, g in [(('M', 1), 'W'), (('W', 0), 'W'), (('M', 1), 'M'), (('M', 1), 'L'), (('W', 0), 'L'), (('A', 1), 'W'), (('A', 1), 'M')]:
        row[f'{c[0]}{c[1]}_{g}'] = p2(D(('g', c, g), ('g', c, 'EW')))
    for pn, cs in [('all11', CONFIGS), ('matched5', MATCHED), ('one1m', ONE)]:
        for g in ['W', 'M', 'L']:
            dd = np.mean([d(('g', c, g), ('g', c, 'EW')) for c in cs])
            DD = np.mean([D(('g', c, g), ('g', c, 'EW')) for c in cs], axis=0)
            row[f'pool_{pn}_{g}'] = p_pooled(DD, dd)
    fam = [(('g', c, g), ('g', c, 'EW')) for c in CONFIGS for g in ['W', 'M', 'L']]
    DF = np.column_stack([D(a, b) for a, b in fam]); dF = np.array([d(a, b) for a, b in fam])
    row['rc_best'] = reality_check(DF, dF)
    famM = [(('g', c, 'M'), ('g', c, 'EW')) for c in CONFIGS]
    DM = np.column_stack([D(a, b) for a, b in famM]); dM = np.array([d(a, b) for a, b in famM])
    row['rc_bestM'] = reality_check(DM, dM)
    famMK = [(('g', c, g), 'MKT') for c in CONFIGS for g in ['W', 'M', 'L']]
    DK = np.column_stack([D(a, b) for a, b in famMK]); dK = np.array([d(a, b) for a, b in famMK])
    row['rc_best_mkt'] = reality_check(DK, dK)
    rw = romano_wolf(DF, dF)
    labs = [f'{a[1][0]}{a[1][1]}_{a[2]}' for a, b in fam]
    for key in ['M1_W', 'W0_W', 'M1_L', 'W0_L']:
        row[f'rw_{key}'] = rw[labs.index(key)]
    bl.append(row)
    print(Lb, round(time.time() - T0), {k: round(v, 3) for k, v in row.items()})
out['block'] = pd.DataFrame(bl)

# ------------------------------------------------------------------ 2. broader family (core + extended windows + lag + net of 10 bp)
SB = boot_sr(EX, B, 21, PPY, seed=20261122)
famX = []
for c in CONFIGS:
    for g in ['W', 'M', 'L']:
        famX.append((f'core {c[0]}{c[1]} {g}', ('g', c, g), ('g', c, 'EW')))
for c in EXT:
    for g in ['W', 'M', 'L']:
        famX.append((f'ext {c[0]}{c[1]} {g}', ('g', c, g), ('g', c, 'EW')))
for c in CONFIGS:
    for g in ['W', 'M', 'L']:
        famX.append((f'lag1 {c[0]}{c[1]} {g}', ('lag1', c, g), ('lag1', c, 'EW')))
for c in CONFIGS:
    for g in ['W', 'M', 'L']:
        famX.append((f'net10 {c[0]}{c[1]} {g}', ('net10', c, g), ('net10', c, 'EW')))
DX = np.column_stack([SB[a] - SB[b] for _, a, b in famX]); dX = np.array([SR0[a] - SR0[b] for _, a, b in famX])
rwX = romano_wolf(DX, dX)
FX = pd.DataFrame(dict(label=[f[0] for f in famX], diff=dX, p=[p2(DX[:, j]) for j in range(len(dX))], p_rw=rwX))
out['famX'] = FX
out['famX_rc'] = dict(best=reality_check(DX, dX), best_label=FX.label[np.argmax(dX)], worst=reality_check(-DX, -dX),
                      bestM=reality_check(DX[:, FX.label.str.endswith(' M').values], dX[FX.label.str.endswith(' M').values]))
print(FX[FX.label.str.contains('M1|W0')].round(3).to_string()); print(out['famX_rc'])
print('broad family', round(time.time() - T0))

# ------------------------------------------------------------------ 3. net of costs with a lag (series from part 1)
S1 = pickle.load(open('results/rev1_series.pkl', 'rb'))
keys = [k for k in S1 if k[0] != 'cap' and k[0] in MATCHED and k[1] in (0, 1)]
EX1 = {k: S1[k].values - rf for k in keys}
SB1 = boot_sr(EX1, B, 21, PPY, seed=77)
nt = []
for c in MATCHED:
    for lag in [0, 1]:
        for cn in ['0', '1.5', '10', '50', 'hist']:
            for g in ['W', 'M', 'L']:
                a, b = (c, lag, cn, g), (c, lag, cn, 'EW')
                sa = EX1[a].mean() / EX1[a].std() * np.sqrt(PPY); sb = EX1[b].mean() / EX1[b].std() * np.sqrt(PPY)
                nt.append(dict(freq=c[0], L=c[1], lag=lag, cost=cn, group=g, sr=sa, sr_ew=sb, diff=sa - sb,
                               p=p2(SB1[a] - SB1[b])))
out['nettest'] = pd.DataFrame(nt)
print(out['nettest'][out['nettest'].freq.isin(['M', 'W'])].round(3).to_string())
print('net tests', round(time.time() - T0))

# ------------------------------------------------------------------ 4. structural break in the Winners' advantage
mW = {}
for c in CONFIGS:
    for g in ['W', 'M', 'L']:
        mW[(c, g)] = (mret(G[('g', c, g)]) - mret(G[('g', c, 'EW')])) * 100
mi = mW[(('M', 1), 'W')].index
dates_m = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in mi]) + pd.offsets.MonthEnd(0)
SER = {'Monthly Winners': mW[(('M', 1), 'W')].values,
       'Weekly Winners': mW[(('W', 0), 'W')].values,
       'Matched-window Winners (avg.)': np.mean([mW[(c, 'W')].values for c in MATCHED], axis=0),
       'Monthly Losers': mW[(('M', 1), 'L')].values,
       'Monthly Winners minus Losers': (mW[(('M', 1), 'W')] - mW[(('M', 1), 'L')]).values}


def wald_path(y, trim=0.15, lags=12):
    T = len(y); lo, hi = int(T * trim), int(T * (1 - trim))
    W = np.full(T, np.nan); delta = np.full(T, np.nan)
    for tau in range(lo, hi + 1):
        X = np.column_stack([np.ones(T), (np.arange(T) >= tau).astype(float)])
        fit = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        W[tau] = fit.tvalues[1] ** 2; delta[tau] = fit.params[1]
    return W, delta


def sup_lr(y, trim=0.15):
    """LS (homoskedastic F) statistic path, fast; used inside the bootstrap."""
    T = len(y); lo, hi = int(T * trim), int(T * (1 - trim))
    cs = np.cumsum(y); tot = cs[-1]; ssr0 = ((y - y.mean()) ** 2).sum()
    taus = np.arange(lo, hi + 1)
    n1 = taus; n2 = T - taus
    s1 = cs[taus - 1]; s2 = tot - s1
    ssr1 = ssr0 + T * y.mean() ** 2 - s1 ** 2 / n1 - s2 ** 2 / n2
    F = (ssr0 - ssr1) / (ssr1 / (T - 2))
    return F, taus


brk = []
paths = {}
rng = np.random.default_rng(99)
for nm, y in SER.items():
    T = len(y)
    W, delta = wald_path(y)
    F, taus = sup_lr(y)
    tau_hat = taus[np.argmax(F)]
    # bootstrap the null of a constant mean: stationary block bootstrap (mean block 12 months) of the demeaned series
    yc = y - y.mean(); ar = np.arange(T)
    supF_b = np.empty(B); supW_b = np.empty(200)
    for b in range(B):
        st = rng.integers(0, T, T); nb = rng.random(T) < 1 / 12; nb[0] = True
        last = np.maximum.accumulate(np.where(nb, ar, 0)); yb = yc[(st[last] + ar - last) % T]
        supF_b[b] = sup_lr(yb)[0].max()
    # HAC sup-Wald bootstrap with fewer draws (costly)
    for b in range(200):
        st = rng.integers(0, T, T); nb = rng.random(T) < 1 / 12; nb[0] = True
        last = np.maximum.accumulate(np.where(nb, ar, 0)); yb = yc[(st[last] + ar - last) % T]
        Wb, _ = wald_path(yb)
        supW_b[b] = np.nanmax(Wb)
    i99 = int(np.where(dates_m >= pd.Timestamp('1999-01-31'))[0][0])
    X = np.column_stack([np.ones(T), (np.arange(T) >= i99).astype(float)])
    chow = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    # 90% confidence set for the break date: dates whose F is not rejected against the bootstrap 90% quantile of sup F
    q90 = np.quantile(supF_b, 0.90)
    brk.append(dict(series=nm, mean=y.mean(), supF=F.max(), p_supF=(supF_b >= F.max()).mean(),
                    supW=np.nanmax(W), p_supW=(supW_b >= np.nanmax(W)).mean(), date_hat=dates_m[tau_hat],
                    dateW_hat=dates_m[int(np.nanargmax(W))],
                    pre=y[:tau_hat].mean(), post=y[tau_hat:].mean(),
                    chow1999=chow.params[1], chow1999_t=chow.tvalues[1], chow1999_p=chow.pvalues[1],
                    pre1999=y[:i99].mean(), post1999=y[i99:].mean(),
                    cv10_W=np.quantile(supW_b, 0.90), cv05_W=np.quantile(supW_b, 0.95), cv05_F=np.quantile(supF_b, 0.95)))
    paths[nm] = dict(W=W, delta=delta, F=np.r_[np.full(taus[0], np.nan), F, np.full(T - taus[-1] - 1, np.nan)])
    print(nm, round(time.time() - T0), {k: v for k, v in brk[-1].items() if k != 'series'})
out['brk'] = pd.DataFrame(brk)
out['brk_paths'] = paths
out['dates_m'] = dates_m

# candidate break years: post minus pre mean and Sharpe-ratio difference
cand = []
y = SER['Monthly Winners']; yw = SER['Weekly Winners']; ym = SER['Matched-window Winners (avg.)']
for yr in range(1985, 2013):
    i0 = int(np.where(dates_m >= pd.Timestamp(f'{yr}-01-31'))[0][0])
    row = dict(year=yr)
    for nm, s in [('M', y), ('W', yw), ('X', ym)]:
        X = np.column_stack([np.ones(len(s)), (np.arange(len(s)) >= i0).astype(float)])
        fit = sm.OLS(s, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        row[f'{nm}_pre'] = s[:i0].mean(); row[f'{nm}_post'] = s[i0:].mean()
        row[f'{nm}_d'] = fit.params[1]; row[f'{nm}_lo'] = fit.conf_int()[1][0]; row[f'{nm}_hi'] = fit.conf_int()[1][1]
        row[f'{nm}_t'] = fit.tvalues[1]
        row[f'{nm}_tpost'] = sm.OLS(s[i0:], np.ones(len(s) - i0)).fit(cov_type='HAC', cov_kwds={'maxlags': 12}).tvalues[0]
    cand.append(row)
out['cand'] = pd.DataFrame(cand)
print(out['cand'][['year', 'M_pre', 'M_post', 'M_t', 'M_tpost', 'X_d', 'X_t']].round(2).to_string())

# ------------------------------------------------------------------ 5. mechanisms by sub-period
SUBP = [('1928', '1962'), ('1963', '1998'), ('1999', '2025')]
mr = mret(U.R)
mr.index = dates_m[:len(mr)] if len(mr) == len(dates_m) else pd.to_datetime([f'{a}-{b:02d}-01' for a, b in mr.index]) + pd.offsets.MonthEnd(0)


def nwt(v, lags=6):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    f = sm.OLS(v, np.ones(len(v))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return f.params[0], f.tvalues[0]


mech = []
V_ew = Universe(10, weighting='ew')
ew_d = (mret(V_ew.bt(('M', 1), 'W')['gross']) - mret(V_ew.bt(('M', 1), 'EW')['gross'])) * 100
ew_d.index = mr.index
vw_d = pd.Series(SER['Monthly Winners'], index=mr.index)
wk_d = pd.Series(SER['Weekly Winners'], index=mr.index)
# 49 industries from 1973
V49 = Universe(49, start='1973-01-01')
d49 = (mret(V49.bt(('M', 1), 'W')['gross']) - mret(V49.bt(('M', 1), 'EW')['gross'])) * 100
d49.index = pd.to_datetime([f'{a}-{b:02d}-01' for a, b in d49.index]) + pd.offsets.MonthEnd(0)
# bear-market state (Daniel and Moskowitz 2016): cumulative market return over the previous 24 months < 0
mk = mret(U.MKT); mk.index = mr.index
bear = ((1 + mk).rolling(24).apply(np.prod, raw=True) - 1).shift(1) < 0
for a, b in SUBP + [('1973', '1998')]:
    sl = slice(f'{a}-01-01', f'{b}-12-31')
    X = mr.loc[sl]
    Xd = X.sub(X.mean(1), axis=0)
    disp = X.std(1).mean() * 100
    ic = [X.iloc[t].rank().corr(X.iloc[t + 1].rank()) for t in range(len(X) - 1)]
    icm, ict = nwt(ic)
    # Lo-MacKinlay decomposition of the weighted relative-strength profit, one-month lag
    Xv = X.values; N = Xv.shape[1]
    mu = Xv.mean(0); Z = Xv - mu
    G1 = (Z[:-1].T @ Z[1:]) / (len(Z) - 1)                   # G1[i,j] = cov(r_i,t-1 , r_j,t)
    own = np.trace(G1) * (N - 1) / N ** 2
    cross = -(G1.sum() - np.trace(G1)) / N ** 2
    meanc = ((mu - mu.mean()) ** 2).mean()
    w = Xd.values[:-1] / N; prof = (w * Xv[1:]).sum(1)
    pm, pt = nwt(prof)
    corr = np.corrcoef(Xv.T)[np.triu_indices(N, 1)].mean()
    row = dict(per=f'{a}-{b}', disp=disp, ic=icm, ic_t=ict, wrss=pm * 1e4, wrss_t=pt, own=own * 1e4, cross=cross * 1e4,
               meanc=meanc * 1e4, corr=corr)
    for lab, s in [('vw', vw_d), ('wk', wk_d), ('ew', ew_d), ('i49', d49)]:
        ss = s.loc[sl]
        if len(ss) < 24:
            row[lab] = np.nan; row[lab + '_t'] = np.nan; continue
        m_, t_ = nwt(ss); row[lab] = m_; row[lab + '_t'] = t_
    bb = bear.loc[sl]
    for st, msk in [('bear', bb == True), ('nobear', bb == False)]:
        ss = vw_d.loc[sl][msk.values]
        m_, t_ = nwt(ss) if len(ss) > 12 else (np.nan, np.nan)
        row[f'{st}'] = m_; row[f'{st}_t'] = t_; row[f'{st}_n'] = int(msk.sum())
    mech.append(row)
out['mech'] = pd.DataFrame(mech)
print(out['mech'].round(3).T.to_string())

pickle.dump(out, open('results/rev2.pkl', 'wb'))
print('done', round(time.time() - T0))
