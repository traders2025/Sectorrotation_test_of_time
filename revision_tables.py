"""LaTeX tables and the figure added in the revision. Reads results/rev1.pkl and results/rev2.pkl.
Output: ../paper/tables/rev_*.tex and ../paper/break_test.png"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = os.path.join('..', 'paper', 'tables')
o1 = pickle.load(open('results/rev1.pkl', 'rb'))
o2 = pickle.load(open('results/rev2.pkl', 'rb'))
FL = {'A': 'Annual', 'S': 'Semiannual', 'Q': 'Quarterly', 'M': 'Monthly', 'W': 'Weekly'}
IND = {'NoDur': 'Consumer Nondurables', 'Durbl': 'Consumer Durables', 'Manuf': 'Manufacturing', 'Enrgy': 'Energy',
       'HiTec': 'Business Equipment', 'Telcm': 'Telecommunications', 'Shops': 'Wholesale and Retail',
       'Hlth': 'Health Care', 'Utils': 'Utilities', 'Other': 'Other'}
CONF = [('A', 1), ('A', 3), ('A', 6), ('A', 12), ('S', 1), ('S', 3), ('S', 6), ('Q', 1), ('Q', 3), ('M', 1), ('W', 0)]
MATCHED = [('A', 12), ('S', 6), ('Q', 3), ('M', 1), ('W', 0)]


def n(x, d=2):
    if pd.isna(x):
        return '--'
    s = f'{x:.{d}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    if s in ('$-$0.00', '$-$0.0'):
        s = s.replace('$-$', '')
    return s


def pv(p):
    return '<0.01' if p < 0.005 else f'{p:.2f}'


def pb(p):
    s = pv(p)
    return f'\\textbf{{{s}}}' if p < 0.05 else s


def win(f, L):
    if f == 'W':
        return '1 week'
    return f'{L} month' + ('s' if L > 1 else '')


# revision colours defined in main_mdpi.tex: R1 = RevisionMagenta, R2 = RevisionBlue, Editor = RevisionGreen
COLOR = {'rev_size': 'RevisionBlue', 'rev_impl': 'RevisionMagenta', 'rev_break': 'RevisionMagenta', 'rev_mech': 'RevisionBlue',
         'rev_block': 'RevisionMagenta', 'rev_ann': 'RevisionMagenta', 'rev_weekly': 'RevisionMagenta', 'rev_outliers': 'RevisionBlue',
         'rev_capw': 'RevisionBlue', 'rev_broadmt': 'RevisionMagenta'}


def save(name, s):
    c = COLOR.get(name)
    if c:   # colour the whole new table (caption, body, and notes) for the marked-up revision
        s = s.replace('\\begin{table}[H]\n', '\\begin{table}[H]\n\\color{' + c + '}\n', 1)
        s = s.replace('\\caption{', '\\caption{\\protect\\color{' + c + '}', 1)
    open(os.path.join(T, name + '.tex'), 'w').write(s)


# ---------------------------------------------------------------- 1. industry size over time
cs = o1['comp_size']
dates = ['1927', '1945', '1962', '1980', '1998', '2025']
rows = []
for t in IND:
    cells = [f"{int(cs[f'n_{t}'][i])} ({cs[f's_{t}'][i] * 100:.1f})" for i in range(len(cs))]
    rows.append(f"{IND[t]} & " + ' & '.join(cells) + ' \\\\')
rows.append('\\midrule')
rows.append('All industries & ' + ' & '.join(f"{int(cs['n_tot'][i])} (100)" for i in range(len(cs))) + ' \\\\')
save('rev_size', r"""\begin{table}[H]
\caption{Number of firms and share of total market capitalization (\%) of each Fama--French industry, selected year ends.\label{tab:size}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{lCCCCCC}
\toprule
\textbf{Industry} & \textbf{1927} & \textbf{1945} & \textbf{1962} & \textbf{1980} & \textbf{1998} & \textbf{2025} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Number of NYSE, AMEX, and NASDAQ firms in each value-weighted industry portfolio and, in parentheses, its share of the total market capitalization of the ten portfolios, from the ``Number of Firms in Portfolios'' and ``Average Firm Size'' tables of the monthly ten-industry file in Kenneth French's data library. Values for the end of 2025 are as of 30 November 2025. NASDAQ stocks enter the CRSP universe in December 1972, which explains most of the increase in the number of firms between 1962 and 1980.}}
\end{table}
""")

# ---------------------------------------------------------------- 2. implementation: lag x costs
NT = o2['nettest']
BE = o1['be']
COST = [('0', 'Gross'), ('1.5', '1.5 bp'), ('10', '10 bp'), ('50', '50 bp'), ('hist', 'Era schedule')]
body = []
for f, L, title in [('M', 1, 'Panel A: Monthly rebalancing, one-month window'), ('W', 0, 'Panel B: Weekly rebalancing, one-week window')]:
    body.append(f'\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\')
    for lag in [0, 1]:
        sub = NT[(NT.freq == f) & (NT.L == L) & (NT.lag == lag)]
        lab = 'Same close (paper baseline)' if lag == 0 else 'Next-day close (one-day lag)'
        w = [sub[(sub.cost == c) & (sub.group == 'W')].iloc[0] for c, _ in COST]
        be = BE[(BE.freq == f) & (BE.L == L) & (BE.lag == lag)].be_bp.iloc[0]
        body.append(f'{lab}: Winners & ' + ' & '.join(n(r.sr) for r in w) + ' & \\\\')
        body.append(f'\\quad EW portfolio & ' + ' & '.join(n(r.sr_ew) for r in w) + ' & \\\\')
        body.append(f'\\quad Winners $-$ EW ($p$) & ' + ' & '.join(f'{n(r['diff'])} ({pb(r.p)})' for r in w) + f' & {be:.1f} \\\\')
        m = [sub[(sub.cost == c) & (sub.group == 'M')].iloc[0] for c, _ in COST]
        body.append(f'\\quad Median $-$ EW ($p$) & ' + ' & '.join(f'{n(r['diff'])} ({pb(r.p)})' for r in m) + ' & \\\\')
        if lag == 0:
            body.append('\\addlinespace')
    body.append('\\midrule')
body = body[:-1]
save('rev_impl', r"""\begin{table}[H]
\caption{Implementation lag and transaction costs applied together: net Sharpe ratios, 1928--2025.\label{tab:impl}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\fulllength}{>{\raggedright\arraybackslash}p{5.3cm}CCCCCc}
\toprule
 & \textbf{Gross} & \textbf{1.5 bp} & \textbf{10 bp} & \textbf{50 bp} & \textbf{Era Schedule} & \textbf{\makecell{Break-Even\\Cost (bp)}} \\
\midrule
""" + '\n'.join(body) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Net Sharpe ratios of the Winner portfolio and of the EW portfolio of all ten industries, rebalanced on the same dates and charged the same one-way cost on every trade, including the drift-restoring trades of the EW portfolio, the initial purchase, and the final liquidation. In the one-day-lag rows, the ranking uses the closing prices of the rebalancing date and the trades are executed at the close of the following trading day. The era schedule charges 100~bp per one-way trade before May 1975 (fixed brokerage commissions), 50~bp from May 1975 to April 2001 (decimalization), and 10~bp afterwards; it is a stylized stress case, not an estimate of the costs of a particular investor. Two-sided $p$-values from a stationary bootstrap of the paired daily excess returns (2000 resamples, mean block length of 21 days); bold marks $p < 0.05$. The break-even cost is the one-way cost at which the net Sharpe ratio of Winners equals that of the EW portfolio.}}
\end{table}
""")

# ---------------------------------------------------------------- 3. pre-specified vs post hoc (written by hand; see main text)

# ---------------------------------------------------------------- 4. structural break tests
br = o2['brk'].set_index('series')
order = ['Monthly Winners', 'Weekly Winners', 'Matched-window Winners (avg.)', 'Monthly Losers', 'Monthly Winners minus Losers']
lab = {'Monthly Winners': 'Monthly Winners $-$ EW', 'Weekly Winners': 'Weekly Winners $-$ EW',
       'Matched-window Winners (avg.)': 'Matched-window Winners $-$ EW (average of five)',
       'Monthly Losers': 'Monthly Losers $-$ EW', 'Monthly Winners minus Losers': 'Monthly Winners $-$ Losers'}
rows = []
for s in order:
    r = br.loc[s]
    rows.append(f"{lab[s]} & {n(r.pre1999)} & {n(r.post1999)} & {n(r.chow1999)} ({pb(r.chow1999_p)}) & {r.supF:.1f} ({pb(r.p_supF)}) & {r.supW:.1f} ({pb(r.p_supW)}) & {r.date_hat.strftime('%b %Y')} \\\\")
save('rev_break', r"""\begin{table}[H]
\caption{Tests for a structural break in the monthly return of the ranked portfolios relative to the EW portfolio, 1928--2025.\label{tab:break}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\fulllength}{>{\raggedright\arraybackslash}p{5.2cm}CCCCCC}
\toprule
 & \multicolumn{3}{c}{\textbf{Known Break, January 1999}} & \multicolumn{3}{c}{\textbf{Unknown Break Date}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Series (Monthly Return Difference)} & \textbf{\makecell{Mean\\1928--1998}} & \textbf{\makecell{Mean\\1999--2025}} & \textbf{\makecell{Change\\($p$)}} & \textbf{\makecell{Sup-$F$\\($p$)}} & \textbf{\makecell{Sup-Wald\\HAC ($p$)}} & \textbf{\makecell{Estimated\\Break}} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Means are in percent per month. The known-break test regresses the monthly return difference on a constant and a dummy for January 1999 onward, with Newey--West standard errors (12 lags). The unknown-date tests search over all break dates in the central 70\% of the sample (15\% trimming)~\cite{andrews1993}: sup-$F$ is the largest least-squares $F$-statistic and sup-Wald the largest Newey--West Wald statistic. Their $p$-values come from a stationary bootstrap of the demeaned series (mean block length of 12 months; 2000 resamples for sup-$F$ and 200 for sup-Wald), which imposes the null of a constant mean while preserving serial dependence. The estimated break is the first month of the second regime at the least-squares break date~\cite{baiperron1998}. Bold marks $p < 0.05$.}}
\end{table}
""")

# figure: Wald path and candidate years
dm = o2['dates_m']
P = o2['brk_paths']
cand = o2['cand']
C = {'M': '#1baf7a', 'W': '#2a78d6', 'L': '#eb6834', 'X': '#3d3d3a'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#3d3d3a', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
a = ax[0]
for s, col, nm in [('Monthly Winners', C['M'], 'Monthly W $-$ EW'), ('Weekly Winners', C['W'], 'Weekly W $-$ EW'),
                   ('Monthly Winners minus Losers', C['L'], 'Monthly W $-$ L')]:
    a.plot(dm, P[s]['W'], color=col, lw=1.6, label=nm)
cv = br.loc['Monthly Winners', 'cv05_W']
a.axhline(cv, color='#8a8984', lw=1, ls='--')
a.text(pd.Timestamp('1945-01-31'), cv + 0.35, f'5% critical value, monthly W $-$ EW ({cv:.1f})', fontsize=7, color='#52514e')
a.axvline(pd.Timestamp('1999-01-31'), color='#bdbcb3', lw=0.8, ls=':')
a.set_ylabel('Newey–West Wald statistic'); a.set_title('(a) Break statistic by break date', fontsize=9, loc='left')
a.legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(0.0, 0.58)); a.grid(axis='y', color='#e6e5df', lw=0.6)
a.set_ylim(0, max(13, cv + 2))
b = ax[1]
yrs = cand.year.values
b.fill_between(yrs, cand.M_lo, cand.M_hi, color=C['M'], alpha=0.15, lw=0)
b.plot(yrs, cand.M_d, color=C['M'], lw=1.8, marker='o', ms=3, label='Monthly W $-$ EW')
b.plot(yrs, cand.X_d, color=C['X'], lw=1.6, marker='s', ms=3, label='Matched-window W $-$ EW (avg.)')
b.axhline(0, color='#8a8984', lw=0.8)
b.axvline(1999, color='#bdbcb3', lw=0.8, ls=':')
b.set_xlabel('First year of the second regime'); b.set_ylabel('Change in mean, % per month')
b.set_title('(b) Change in mean for each candidate year', fontsize=9, loc='left')
b.legend(frameon=False, fontsize=7, loc='lower right'); b.grid(axis='y', color='#e6e5df', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join('..', 'paper', 'break_test.png'), dpi=300)

# ---------------------------------------------------------------- 5. mechanisms
me = o2['mech'].set_index('per')
pers = ['1928-1962', '1963-1998', '1999-2025']


def mt(col, p, d=2):
    v = me.loc[p, col]; t = me.loc[p, col + '_t']
    return f'{n(v, d)} ({n(t, 2)})'


rows = [
    'Cross-sectional dispersion of monthly industry returns (\\%) & ' + ' & '.join(n(me.loc[p, 'disp']) for p in pers) + ' \\\\',
    'Average pairwise correlation of monthly industry returns & ' + ' & '.join(n(me.loc[p, 'corr']) for p in pers) + ' \\\\',
    'Rank correlation of industry returns in months $t$ and $t+1$ ($t$) & ' + ' & '.join(mt('ic', p, 3) for p in pers) + ' \\\\',
    '\\addlinespace',
    'Weighted relative-strength profit (bp per month, $t$) & ' + ' & '.join(mt('wrss', p) for p in pers) + ' \\\\',
    '\\quad Own autocovariance component (bp) & ' + ' & '.join(n(me.loc[p, 'own']) for p in pers) + ' \\\\',
    '\\quad Cross-serial covariance component (bp) & ' + ' & '.join(n(me.loc[p, 'cross']) for p in pers) + ' \\\\',
    '\\quad Dispersion of mean returns component (bp) & ' + ' & '.join(n(me.loc[p, 'meanc']) for p in pers) + ' \\\\',
    '\\addlinespace',
    'Monthly Winners $-$ EW, value-weighted industries (\\%, $t$) & ' + ' & '.join(mt('vw', p) for p in pers) + ' \\\\',
    'Monthly Winners $-$ EW, equal-weighted industries (\\%, $t$) & ' + ' & '.join(mt('ew', p) for p in pers) + ' \\\\',
    'Monthly Winners $-$ EW, 49 industries (\\%, $t$)$^{a}$ & -- & ' + mt('i49', '1973-1998') + ' & ' + mt('i49', '1999-2025') + ' \\\\',
    'Monthly Winners $-$ EW, excluding bear-market states (\\%, $t$) & ' + ' & '.join(mt('nobear', p) for p in pers) + ' \\\\',
    'Monthly Winners $-$ EW, bear-market states only (\\%, $t$) & ' + ' & '.join(mt('bear', p) for p in pers) + ' \\\\',
]
save('rev_mech', r"""\begin{table}[H]
\caption{Diagnostics of the weakening of industry momentum, by sub-period.\label{tab:mech}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{>{\raggedright\arraybackslash}p{8.4cm}CCC}
\toprule
 & \textbf{1928--1962} & \textbf{1963--1998} & \textbf{1999--2025} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Monthly returns of the ten value-weighted Fama--French industries unless stated otherwise. The rank correlation is the time-series average of the cross-sectional Spearman correlation between industry returns in consecutive months. The weighted relative-strength profit invests $(r_{i,t-1}-\bar r_{t-1})/N$ in each industry and is decomposed as in Lo and MacKinlay~\cite{lomackinlay1990} and Lewellen~\cite{lewellen2002} into the own-autocovariance, cross-serial covariance, and mean-dispersion components (the components sum to the expected profit; the realized profit differs slightly because of sample means). Bear-market states are months in which the cumulative market return over the previous 24 months was negative, as in Daniel and Moskowitz~\cite{danielmoskowitz2016}. $t$-statistics use Newey--West standard errors with six lags. $^{a}$ The 49-industry daily data are complete from 1973, so the middle column covers 1973--1998.}}
\end{table}
""")

# ---------------------------------------------------------------- 6. appendix: robustness tables
# 6a block length
bl = o2['block']
cols = [('M1_W', 'M W$-$EW'), ('W0_W', 'W W$-$EW'), ('M1_L', 'M L$-$EW'), ('M1_M', 'M M$-$EW'), ('pool_all11_W', 'Avg. W$-$EW'),
        ('pool_all11_M', 'Avg. M$-$EW'), ('pool_all11_L', 'Avg. L$-$EW'), ('rc_best', 'RC best'), ('rc_bestM', 'RC best M'), ('rw_M1_W', 'RW M W')]
rows = [f"{int(r.block)} & " + ' & '.join(pv(r[c]) if c[:2] not in ('rc', 'rw') else (f'{r[c]:.3f}' if r[c] >= 0.001 else '<0.001') for c, _ in cols) + ' \\\\' for _, r in bl.iterrows()]
save('rev_block', r"""\begin{table}[H]
\caption{Sensitivity of bootstrap $p$-values to the mean block length, 1928--2025.\label{tab:block}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\fulllength}{cCCCCCCCCCC}
\toprule
\textbf{\makecell{Block\\(Days)}} & \textbf{\makecell{Monthly\\W$-$EW}} & \textbf{\makecell{Weekly\\W$-$EW}} & \textbf{\makecell{Monthly\\L$-$EW}} & \textbf{\makecell{Monthly\\M$-$EW}} & \textbf{\makecell{Avg.\\W$-$EW}} & \textbf{\makecell{Avg.\\M$-$EW}} & \textbf{\makecell{Avg.\\L$-$EW}} & \textbf{\makecell{RC\\Best}} & \textbf{\makecell{RC Best\\Median}} & \textbf{\makecell{RW\\Monthly W}} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Two-sided $p$-values for differences in Sharpe ratios from stationary bootstraps of the paired daily excess returns with the stated mean block length (2000 resamples each). ``Avg.'' is the average difference over the eleven configurations. RC best: White's reality check $p$-value for the best of the 33 strategies against its EW portfolio; RC best Median: the same for the best of the eleven Median strategies; RW monthly W: Romano--Wolf adjusted $p$-value of monthly Winners minus EW within the family of 33. The paper's baseline is 21 days. Returns before transaction costs.}}
\end{table}
""")

# 6b annualization
an = o1['ann']
rows = []
for p in ['1928-2025', '1928-1962', '1963-1998', '1999-2025']:
    s = an[an.per == p]
    ppy = s.ppy.iloc[0]
    for col, cl in [('sr262', 'Daily, 262 days'), ('srown', f'Daily, {ppy:.0f} days'), ('srm', 'Monthly returns')]:
        g = lambda f, L, grp: s[(s.freq == f) & (s.L == L) & (s.group == grp)][col].iloc[0]
        avg = np.mean([g(f, L, 'W') - g(f, L, 'EW') for f, L in MATCHED])
        first = p.replace('-', '--') if col == 'sr262' else ''
        rows.append(f"{first} & {cl} & {n(g('M', 1, 'W'))} & {n(g('M', 1, 'M'))} & {n(g('M', 1, 'L'))} & {n(g('M', 1, 'EW'))} & {n(g('W', 0, 'W'))} & {n(g('W', 0, 'EW'))} & {n(avg)} \\\\")
    rows.append('\\midrule')
rows = rows[:-1]
save('rev_ann', r"""\begin{table}[H]
\caption{Sharpe ratios under alternative annualization conventions, full sample and sub-periods.\label{tab:ann}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\fulllength}{llCCCCCCC}
\toprule
 & & \multicolumn{4}{c}{\textbf{Monthly Rebalancing}} & \multicolumn{2}{c}{\textbf{Weekly Rebalancing}} & \textbf{\makecell{Matched\\Windows}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-8} \cmidrule(lr){9-9}
\textbf{Period} & \textbf{Convention} & \textbf{W} & \textbf{M} & \textbf{L} & \textbf{EW} & \textbf{W} & \textbf{EW} & \textbf{Avg. W$-$EW} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{``Daily, 262 days'': the paper's convention (average number of trading days per year over 1928--2025). ``Daily, $x$ days'': the average number of trading days per year within the period. ``Monthly returns'': Sharpe ratio of calendar-month excess returns, annualized with $\sqrt{12}$. The last column averages the Sharpe ratio difference between Winners and the EW portfolio over the five matched-window configurations. Returns before transaction costs.}}
\end{table}
""")

# 6c weekly rule
wd = o1['weekdays']
wa = o1['wk_alt']
dist = []
for era, lab_ in [('pre1952', 'Before June 1952 (Saturday trading)'), ('post1952', 'From June 1952')]:
    r = wd.loc[era]
    tot = r.sum()
    dist.append(f"{lab_} & {tot} & " + ' & '.join(f"{r.get(k, 0) / tot * 100:.1f}" for k in [1, 3, 4, 5, 6]) + ' \\\\')
rows = [f"{r.rule} & {n(r.W_lag0)} & {n(r.M_lag0)} & {n(r.L_lag0)} & {n(r.EW_lag0)} & {n(r.W_lag1)} & {n(r.L_lag1)} \\\\" for r in wa.itertuples()]
save('rev_weekly', r"""\begin{table}[H]
\caption{The weekly rule: length of the ranking week and alternative ranking windows, 1928--2025.\label{tab:weekly}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{lCCCCCC}
\toprule
\multicolumn{7}{l}{\textit{Panel A: Distribution of the number of trading days in the ranking week (\% of weeks)}} \\
\textbf{Period} & \textbf{Weeks} & \textbf{1 Day} & \textbf{3 Days} & \textbf{4 Days} & \textbf{5 Days} & \textbf{6 Days} \\
\midrule
""" + '\n'.join(dist) + r"""
\midrule
\multicolumn{7}{l}{\textit{Panel B: Sharpe ratios with alternative ranking windows (rebalancing at the end of each ISO week)}} \\
 & \multicolumn{4}{c}{\textbf{Same-Close Execution}} & \multicolumn{2}{c}{\textbf{One-Day Lag}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7}
\textbf{Ranking Window} & \textbf{W} & \textbf{M} & \textbf{L} & \textbf{EW} & \textbf{W} & \textbf{L} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{The weekly rule rebalances at the close of the last trading day of each ISO week (Monday to Sunday) and ranks the industries on their compounded total return over all trading days of that ISO week. The number of days therefore varies with holidays and, before June 1952, with Saturday sessions. ``Last 5 (10) trading days'' ranks on the return over a fixed number of trading days ending at the rebalancing close. Returns before transaction costs.}}
\end{table}
""")

# 6d extreme returns
wi = o1['win']
ex = o1['exc']
rows = []
for f, L in CONF:
    r = wi[(wi.freq == f) & (wi.L == L)].iloc[0]
    rows.append(f"{FL[f]}, {win(f, L)} & {n(r.base_W - r.base_EW)} & {n(r['Winsorized (0.5%/99.5%)_W'] - r.base_EW)} & {r['Winsorized (0.5%/99.5%)_chg'] * 100:.0f}\\% & {n(r['Mean daily rank_W'] - r.base_EW)} & {n(r.base_L - r.base_EW)} & {n(r['Winsorized (0.5%/99.5%)_L'] - r.base_EW)} \\\\")
rows2 = [f"{r.sample} & {n(r.M_W)} & {n(r.M_M)} & {n(r.M_L)} & {n(r.M_EW)} & {n(r.W_W)} & {n(r.W_EW)} \\\\" for r in ex.itertuples()]
q = o1['ext']['q']
save('rev_outliers', r"""\begin{table}[H]
\caption{Extreme daily returns: winsorized and rank-based ranking signals, and samples excluding crises, 1928--2025.\label{tab:outliers}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{lCCCCCC}
\toprule
\multicolumn{7}{l}{\textit{Panel A: Sharpe ratio difference from the EW portfolio by ranking signal}} \\
 & \multicolumn{4}{c}{\textbf{Winners $-$ EW}} & \multicolumn{2}{c}{\textbf{Losers $-$ EW}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7}
\textbf{Configuration} & \textbf{Baseline} & \textbf{Winsorized} & \textbf{\makecell{Winners\\Changed}} & \textbf{\makecell{Mean Daily\\Rank}} & \textbf{Baseline} & \textbf{Winsorized} \\
\midrule
""" + '\n'.join(rows) + r"""
\midrule
\multicolumn{7}{l}{\textit{Panel B: Sharpe ratios when the evaluation excludes crisis years}} \\
 & \multicolumn{4}{c}{\textbf{Monthly Rebalancing}} & \multicolumn{2}{c}{\textbf{Weekly Rebalancing}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7}
\textbf{Sample} & \textbf{W} & \textbf{M} & \textbf{L} & \textbf{EW} & \textbf{W} & \textbf{EW} \\
\midrule
""" + '\n'.join(rows2) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Panel A: ``Winsorized'' ranks the industries on ranking-window returns compounded from daily returns winsorized at the 0.5th and 99.5th percentiles of the pooled daily industry returns (""" + f"{q[0.005] * 100:.2f}\\% and {q[0.995] * 100:.2f}\\%" + r"""); holding-period returns are not winsorized. ``Winners changed'' is the percentage of rebalancing dates on which winsorization changes the Winner portfolio. ``Mean daily rank'' ranks the industries on their average daily cross-sectional rank over the window, a signal that ignores the magnitude of returns. Panel B drops the stated years from the evaluation (the strategies are run on the full sample). Returns before transaction costs.}}
\end{table}
""")

# 6e cap weights and small industries
cw = o1['cw']
ds = o1['drop_small']
rows = []
for f, L in CONF:
    r = cw[(cw.freq == f) & (cw.L == L)].iloc[0]
    d = ds[(ds.freq == f) & (ds.win == L)].iloc[0]
    rows.append(f"{FL[f]}, {win(f, L)} & {n(r.ew_W - r.ew_EW)} & {n(r.cw_W - r.cw_EW)} & {n(r.cw_M - r.cw_EW)} & {n(r.cw_L - r.cw_EW)} & {n(d.W - d.EW)} & {n(d['W_1928'] - d['EW_1928'])} & {n(d.M - d.EW)} & {n(d.L - d.EW)} \\\\")
r0 = cw.iloc[0]
save('rev_capw', r"""\begin{table}[H]
\caption{Weighting and small early industries: Sharpe ratio differences from the corresponding all-industry portfolio, 1928--2025.\label{tab:capw}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\fulllength}{lCCCCCCCC}
\toprule
 & \textbf{\makecell{Paper\\(Equal\\Weights)}} & \multicolumn{3}{c}{\textbf{Capitalization Weights}} & \multicolumn{4}{c}{\textbf{Excluding Health Care and Business Equipment}} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-5} \cmidrule(lr){6-9}
\textbf{Configuration} & \textbf{W$-$EW} & \textbf{W$-$CW} & \textbf{M$-$CW} & \textbf{L$-$CW} & \textbf{W$-$EW} & \textbf{\makecell{W$-$EW\\1928--1962}} & \textbf{M$-$EW} & \textbf{L$-$EW} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\noindent{\footnotesize{Capitalization weights: within each group, and in the benchmark (CW) that holds all ten industries, industries are weighted by their market capitalization at the most recent month end before the trade (number of firms times average firm size, from Kenneth French's data library). The CW portfolio has a Sharpe ratio of """ + n(r0.cw_EW) + r""", the same as the CRSP value-weighted market (""" + n(r0.mkt) + r"""). Excluding health care and business equipment: the rules are applied to the remaining eight industries (Winners and Losers hold two industries each, Median four), and EW is the equal-weighted portfolio of the eight. Returns before transaction costs.}}
\end{table}
""")

# 6f broader multiple-testing family
FX = o2['famX']
rc = o2['famX_rc']
sel = [('core M1 W', 'Monthly Winners, baseline'), ('core W0 W', 'Weekly Winners, baseline'), ('ext M3 W', 'Monthly Winners, 3-month window'),
       ('ext M12 W', 'Monthly Winners, 12-month window'), ('lag1 M1 W', 'Monthly Winners, one-day lag'), ('lag1 W0 W', 'Weekly Winners, one-day lag'),
       ('net10 M1 W', 'Monthly Winners, net of 10 bp'), ('net10 W0 W', 'Weekly Winners, net of 10 bp'),
       ('core M1 L', 'Monthly Losers, baseline'), ('lag1 M1 L', 'Monthly Losers, one-day lag'), ('core M1 M', 'Monthly Median, baseline')]
F33 = o2['block'].set_index('block').loc[21]
rows = []
for k, lab_ in sel:
    r = FX[FX.label == k].iloc[0]
    rows.append(f"{lab_} & {n(r['diff'])} & {pv(r.p)} & {('<0.001' if r.p_rw < 0.001 else f'{r.p_rw:.3f}')} \\\\")
save('rev_broadmt', r"""\begin{table}[H]
\caption{Multiple testing in a broader family of 117 comparisons with the EW portfolio, 1928--2025.\label{tab:broadmt}}
\small\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{6.2cm}CCC}
\toprule
\textbf{Strategy} & \textbf{$\Delta$SR} & \textbf{$p$ (Unadjusted)} & \textbf{$p$ (Romano--Wolf)} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabularx}
\noindent{\footnotesize{The family contains the 33 baseline comparisons (Winners, Median, and Losers minus EW for the eleven configurations), the 18 comparisons for the six windows longer than the holding period (Table~\ref{tab:longwin}), and the 33 comparisons each with a one-day implementation lag and net of a 10~bp one-way cost (EW net of the same cost). Stationary bootstrap with 2000 resamples and a mean block length of 21 days. White's reality check for the best strategy of the family: $p """ + ('< 0.001' if rc['best'] < 0.001 else f"= {rc['best']:.3f}") + r"""$; for the best Median strategy of the family: $p = """ + f"{rc['bestM']:.2f}" + r"""$.}}
\end{table}
""")
print('tables written')
