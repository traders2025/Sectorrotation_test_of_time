"""Shared helpers for the analyses added in the revision (revision_*.py)."""
import os
import numpy as np
import pandas as pd
from engine_ind import Universe, group_sizes, FF3D, _read_block

HERE = os.path.dirname(os.path.abspath(__file__))


def bt_general(U, c, group, lag=0, spread=0.0, universe=None, weights='eq', caps=None, periods=None):
    """Generalized version of Universe.bt.

    spread : scalar or callable(Timestamp) -> one-way cost (decimal), applied at each trade date.
    weights: 'eq' (equal weights within the group, as in the paper) or 'cap' (weights proportional to
             the industry's market capitalization at the most recent month end, `caps` DataFrame).
    periods: optional list of (start, end, past) to override the ranking returns (e.g. winsorized).
    Returns dict(gross, net, turn, trade_dates)."""
    f, L = c
    uni = U.TICK if universe is None else universe
    N = len(uni)
    k, km = group_sizes(N)
    sl = {'W': slice(0, k), 'L': slice(N - k, N), 'M': slice(k, N - k),
          'XW': slice(k, N), 'EW': slice(0, N)}[group]
    per = U.periods(f, L) if periods is None else periods
    n = len(U.IDX)
    x = U.R.values
    nT = len(U.TICK)
    starts = [p[0] + (lag if j > 0 else 0) for j, p in enumerate(per)]
    ends = starts[1:] + [n]
    gross = np.empty(n)
    cost = np.zeros(n)
    w_prev = np.zeros(nT)
    turns, tdates = [], []
    sp = spread if callable(spread) else (lambda d, s=spread: s)
    for (s, e, past), s2, e2 in zip(per, starts, ends):
        order = past[uni].sort_values(ascending=False, kind='mergesort').index
        chosen = [U.TICK.index(t) for t in order[sl]]
        w_t = np.zeros(nT)
        if weights == 'cap':
            d0 = U.IDX[max(s2 - 1, 0)] if s2 > 0 else U.IDX[0]
            cap = caps.loc[:d0].iloc[-1]
            cw = cap[[U.TICK[i] for i in chosen]].values
            w_t[chosen] = cw / cw.sum()
        else:
            w_t[chosen] = 1 / len(chosen)
        tv = np.abs(w_t - w_prev).sum()
        turns.append(tv)
        if s2 < n:
            cost[s2] += sp(U.IDX[s2]) * tv
            tdates.append(U.IDX[s2])
        h = w_t.copy()
        seg = x[s2:e2]
        if len(seg):
            cum = np.cumprod(1 + seg, axis=0) * h
            tot = cum.sum(1)
            prevtot = np.concatenate([[h.sum()], tot[:-1]])
            gross[s2:e2] = tot / prevtot - 1
            h = cum[-1]
        w_prev = h / h.sum()
    cost[-1] += sp(U.IDX[-1]) * 1.0
    g = pd.Series(gross, U.IDX)
    net = pd.Series((1 - cost) * (1 + gross) - 1, U.IDX)
    return dict(gross=g, net=net, turn=np.array(turns), trade_dates=tdates)


def month_caps(n_ind=10):
    """Industry market capitalization (USD million) at each month end: number of firms x average firm size.
    French reports both for month t as of the end of month t-1, so the row for month t is dated at the
    end of month t-1."""
    nf = _read_block(f'{n_ind}_Industry_Portfolios_CSV.zip', 4) * 100
    sz = _read_block(f'{n_ind}_Industry_Portfolios_CSV.zip', 5) * 100
    cap = (nf * sz)
    cap.index = cap.index.to_period('M').to_timestamp('M') - pd.offsets.MonthEnd(1)
    return cap, nf


def sb_indices(n, B, L, seed, chunk=200):
    """Stationary-bootstrap index matrices (mean block length L), yielded in chunks."""
    rng = np.random.default_rng(seed)
    ar = np.arange(n)
    for b0 in range(0, B, chunk):
        m = min(chunk, B - b0)
        idx = np.empty((m, n), dtype=np.int32)
        for i in range(m):
            st = rng.integers(0, n, n)
            nb = rng.random(n) < 1 / L
            nb[0] = True
            last = np.maximum.accumulate(np.where(nb, ar, 0))
            idx[i] = (st[last] + ar - last) % n
        yield idx


def boot_sr(ex, B, L, ppy, seed=20260924):
    """ex: dict key -> daily excess return array. Returns dict key -> (B,) bootstrap Sharpe ratios."""
    acc = {k: [] for k in ex}
    n = len(next(iter(ex.values())))
    exf = {k: np.asarray(v, dtype=np.float64) for k, v in ex.items()}
    for idx in sb_indices(n, B, L, seed):
        for k, e in exf.items():
            E = e[idx]
            acc[k].append(E.mean(1) / E.std(1) * np.sqrt(ppy))
    return {k: np.concatenate(v) for k, v in acc.items()}


def p2(D, d=None):
    """Two-sided bootstrap p-value of a difference (percentile method, as in the paper)."""
    return float(min(1, 2 * min((D <= 0).mean(), (D >= 0).mean())))


def p_pooled(D, d):
    return float((np.abs(D - d) >= abs(d)).mean())


def romano_wolf(D, d):
    se = D.std(0)
    t = np.abs(d) / se
    tb = np.abs(D - d) / se
    m = len(d)
    order = np.argsort(-t)
    p_rw = np.empty(m); prev = 0.0
    for j, ix in enumerate(order):
        pv = max((tb[:, order[j:]].max(1) >= t[ix]).mean(), prev)
        p_rw[ix] = pv; prev = pv
    return p_rw


def reality_check(D, d):
    stat = d.max()
    return float(((D - d).max(1) >= stat).mean())


def mret(r):
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
