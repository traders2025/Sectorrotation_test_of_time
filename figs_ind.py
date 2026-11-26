"""Figures for the industry-rotation study. Output: ../paper/*.png"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from engine_ind import *

OUT = os.path.join('..', 'paper')
U = Universe(10)
o = pickle.load(open('results/out.pkl', 'rb'))
res = pickle.load(open('results/res.pkl', 'rb'))
OF = pickle.load(open('results/offsets.pkl', 'rb'))['D']
C = {'M': '#2a78d6', 'L': '#eb6834', 'W': '#1baf7a', 'EW': '#3d3d3a', 'MKT': '#8a8984'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#3d3d3a', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e'})
GRID = '#e6e5df'
gr = lambda c, g: res[(c, g)]['gross']

# ---------------------------------------------------------------- 1. Sharpe ratio dot plot
rp = o['rp']
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ypos, y = {}, 0
for i, c in enumerate(CONFIGS[::-1]):
    if i > 0 and CONFIGS[::-1][i - 1][0] != c[0]:
        y += 0.6
    ypos[c] = y; y += 1
for c, yy in ypos.items():
    d = rp[(c[0], 3)]
    lo, q1, md, q3, hi = np.percentile(d, [2.5, 25, 50, 75, 97.5])
    ax.plot([lo, hi], [yy, yy], color='#d6d5cc', lw=1.2, solid_capstyle='round', zorder=1)
    ax.plot([q1, q3], [yy, yy], color='#bdbcb3', lw=6, solid_capstyle='butt', zorder=1)
    ax.plot([md], [yy], '|', color='#52514e', ms=9, mew=1.3, zorder=2)
    for g, m, ms in [('W', '^', 7.5), ('M', 'o', 7), ('L', 's', 6.5)]:
        ax.plot(U.sr(gr(c, g)), yy, m, color=C[g], ms=ms, mec='#fcfcfb', mew=1, zorder=4)
    ax.plot(U.sr(gr(c, 'EW')), yy, 'D', color=C['EW'], ms=4.5, mec='#fcfcfb', mew=0.8, zorder=3)
ax.axvline(U.sr(U.MKT), color=C['MKT'], ls='--', lw=1, zorder=0)
ax.text(U.sr(U.MKT), max(ypos.values()) + 0.7, 'Market', color='#52514e', fontsize=8, ha='center', va='bottom')
ax.set_yticks(list(ypos.values()))
ax.set_yticklabels([f"{FL[c[0]]}, {wlab(*c)}" + (' *' if matched(c) else '') for c in ypos])
ax.set_ylim(-0.7, max(ypos.values()) + 1.3)
ax.set_xlabel('Sharpe ratio (excess return over one-month T-bill)')
ax.grid(axis='x', color=GRID, lw=0.6)
h = [Line2D([], [], marker='^', ls='', color=C['W'], ms=7.5, label='Winners'),
     Line2D([], [], marker='o', ls='', color=C['M'], ms=7, label='Median'),
     Line2D([], [], marker='s', ls='', color=C['L'], ms=6.5, label='Losers'),
     Line2D([], [], marker='D', ls='', color=C['EW'], ms=4.5, label='Equal-weighted'),
     Line2D([], [], color='#bdbcb3', lw=6, label='Random (3 ind.), 25th–75th pct.'),
     Line2D([], [], color='#d6d5cc', lw=1.2, label='Random (3 ind.), 2.5th–97.5th pct.')]
ax.legend(handles=h, frameon=False, ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, 1.11))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sr_by_window.png'), dpi=300); plt.close()

# ---------------------------------------------------------------- 2. event time
EV = o['EV']
shades = {1: '#0d3f7a', 3: '#1f5fb0', 6: '#4f8fdc', 12: '#8db8ee'}
shades_o = {1: '#7a2e0e', 3: '#c2501f', 6: '#eb6834', 12: '#f4a77e'}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, g, sh, ttl in [(axs[0], 'W', shades, '(a) Winners minus EW'), (axs[1], 'M', shades, '(b) Median minus EW'),
                       (axs[2], 'L', shades_o, '(c) Losers minus EW')]:
    for L in [1, 3, 6, 12]:
        m = EV[EV.win == L].groupby('h')[g].mean() * 100
        cum = np.concatenate([[0], m.cumsum().values])
        ax.plot(range(0, 13), cum, color=sh[L], lw=2 if L == 1 else 1.5, marker='o', ms=3, label=wlab('A', L))
    ax.axhline(0, color='#8a8984', lw=0.8)
    ax.set_xticks([0, 1, 3, 6, 9, 12]); ax.set_xlabel('Months after ranking')
    ax.set_title(ttl, loc='left', fontsize=9); ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc='best', title='Window', title_fontsize=7)
axs[0].set_ylabel('Cumulative return relative to EW (%)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'event_time.png'), dpi=300); plt.close()


# ---------------------------------------------------------------- 3. growth and drawdown
def path(r):
    w = (1 + r).cumprod()
    return pd.concat([pd.Series([1.0], [pd.Timestamp('1927-12-31')]), w])


def growth_fig(ser, fn, title):
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    ends = []
    for lab, short, r, c, ls in ser:
        w = path(r)
        ax[0].plot(w.index, w.values, color=c, lw=1.4 if ls == '-' else 1.3, ls=ls, label=lab)
        ends.append([np.log(w.iloc[-1]), w.iloc[-1], c, short])
        dd = w / w.cummax() - 1
        ax[1].plot(dd.index, dd.values * 100, color=c, lw=1.0, ls=ls)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.62:
            ends[i][0] = ends[i - 1][0] + 0.62
    for ly, v, c, nm in ends:
        ax[0].plot([pd.Timestamp('2026-06-01')], [np.exp(ly)], 'o', color=c, ms=4.5, clip_on=False)
        ax[0].annotate(f'{nm} {v:,.0f}', (pd.Timestamp('2026-06-01'), np.exp(ly)), xytext=(5, 0), textcoords='offset points',
                       va='center', fontsize=7.5, color='#3d3d3a', annotation_clip=False)
    ax[0].set_yscale('log')
    tk = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    ax[0].set_yticks(tk); ax[0].set_yticklabels(['0.1', '1', '10', '100', '1,000', '10,000', '100,000', '1,000,000']); ax[0].minorticks_off()
    ax[0].set_ylim(0.1, 3e6); ax[1].set_ylim(-95, 2)
    ax[0].set_ylabel('Value of USD 1 (log scale)'); ax[1].set_ylabel('Drawdown (%)')
    for a in ax:
        a.grid(axis='y', color=GRID, lw=0.6)
    ax[0].legend(frameon=False, ncol=1, loc='upper left', fontsize=7.5)
    ax[0].set_title(f'(a) Growth of USD 1 — {title}', loc='left', fontsize=9)
    ax[1].set_title('(b) Drawdown from previous peak', loc='left', fontsize=9)
    ax[0].set_xlim(pd.Timestamp('1927-06-01'), pd.Timestamp('2040-01-01'))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fn), dpi=300); plt.close()


growth_fig([('Monthly Winners, 1-month window', 'M1 Winners', gr(('M', 1), 'W'), C['W'], '-'),
            ('Monthly Median, 1-month window', 'M1 Median', gr(('M', 1), 'M'), C['M'], '-'),
            ('Monthly Losers, 1-month window', 'M1 Losers', gr(('M', 1), 'L'), C['L'], '-'),
            ('Annual Winners, 1-month window', 'A1 Winners', gr(('A', 1), 'W'), '#0b6b4a', ':'),
            ('Equal-weighted (monthly)', 'EW', gr(('M', 1), 'EW'), C['EW'], '-'),
            ('Market (CRSP value-weighted)', 'Market', U.MKT, C['MKT'], '--')], 'growth_drawdown.png', 'selected strategies')
names = []
for c in CONFIGS:
    f, L = c
    fn = f"growth_{f}{L}.png"
    growth_fig([('Winners', 'Winners', gr(c, 'W'), C['W'], '-'), ('Median', 'Median', gr(c, 'M'), C['M'], '-'),
                ('Losers', 'Losers', gr(c, 'L'), C['L'], '-'), (f'Equal-weighted ({FL[f].lower()})', 'EW', gr(c, 'EW'), C['EW'], '-'),
                ('Market', 'Market', U.MKT, C['MKT'], '--')], fn, f"{FL[f].lower()} rebalancing, {wlab(f, L)} window")
    names.append(fn)

# ---------------------------------------------------------------- 4. calendar robustness
MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True, gridspec_kw={'width_ratios': [12, 6, 4.2]})
for ax, f in zip(axs, 'ASQ'):
    H = HOLD_M[f]
    for j, (L, col, lab) in enumerate([(1, '#0d3f7a', '1-month window'), (H, '#8db8ee', 'Matched window')]):
        d = OF[(OF.freq == f) & (OF.win == L)].pivot(index='off', columns='group', values='sr')
        ax.bar(np.arange(H) + (j - 0.5) * 0.38, (d.W - d.EW).values, width=0.36, color=col, label=lab)
    labs = []
    for off in range(H):
        ms = sorted({(off - 1 + H * q) % 12 for q in range(12 // H)})
        labs.append('/'.join(MN[m][:1] for m in ms) if H == 3 else ('/'.join(MN[m] for m in ms) if H < 12 else MN[ms[0]]))
    ax.set_xticks(range(H)); ax.set_xticklabels(labs, rotation=90 if H >= 6 else 0, fontsize=7)
    ax.axhline(0, color='#8a8984', lw=0.8)
    ax.set_title({'A': '(a) Annual', 'S': '(b) Semiannual', 'Q': '(c) Quarterly'}[f], loc='left', fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.6)
axs[0].set_ylabel('Sharpe ratio, Winners minus EW')
axs[0].set_xlabel('Rebalancing month', fontsize=8); axs[1].set_xlabel('Rebalancing months', fontsize=8); axs[2].set_xlabel('Rebalancing months', fontsize=8)
axs[0].legend(frameon=False, fontsize=7.5, loc='best')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'calendar.png'), dpi=300); plt.close()

# ---------------------------------------------------------------- 5. rolling 15-year Sharpe ratio differences
win = int(15 * U.PPY)
rf = U.RF


def rsr(r):
    e = r - rf
    return e.rolling(win).mean() / e.rolling(win).std() * np.sqrt(U.PPY)


fig, ax = plt.subplots(figsize=(7.2, 3.9))
for lab, c, g, col, ls in [('Monthly Winners − EW (1-month window)', ('M', 1), 'W', C['W'], '-'),
                           ('Monthly Median − EW (1-month window)', ('M', 1), 'M', C['M'], '-'),
                           ('Annual Winners − EW (12-month window)', ('A', 12), 'W', '#0b6b4a', '--'),
                           ('Annual Winners − EW (1-month window)', ('A', 1), 'W', '#0b6b4a', ':')]:
    d = (rsr(gr(c, g)) - rsr(gr(c, 'EW'))).dropna()
    ax.plot(d.index, d.values, color=col, lw=1.4, ls=ls, label=lab)
ax.axhline(0, color='#8a8984', lw=0.8)
ax.set_ylabel('Difference in 15-year Sharpe ratio')
ax.set_xlabel('End of 15-year window')
ax.grid(axis='y', color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'rolling.png'), dpi=300); plt.close()
print('figures done', names)
