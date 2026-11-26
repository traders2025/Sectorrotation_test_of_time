"""LaTeX tables (MDPI style) for the industry-rotation study. Output: ../paper/tables/*.tex"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import numpy as np
import pandas as pd
from engine_ind import CONFIGS, CORE, EXT, GROUPS, GN, FL, HOLD_M, IND10, wlab, clab, matched

TD = os.path.join('..', 'paper', 'tables')
os.makedirs(TD, exist_ok=True)
o = pickle.load(open('results/out.pkl', 'rb'))
OF = pickle.load(open('results/offsets.pkl', 'rb'))
MR = o['MR']
PER = "1928--2025"


def pct(v, d=1):
    s = f"{abs(v) * 100:.{d}f}"
    return f"$-${s}\\%" if (v < 0 and float(s) != 0) else f"{s}\\%"


def num(v, d=2):
    s = f"{abs(v):.{d}f}"
    return f"$-${s}" if (v < 0 and float(s) != 0) else s


def pv(v):
    return '<0.01' if v < 0.005 else f"{v:.2f}"


def B(s, on):
    return f"\\textbf{{{s}}}" if on else s


def W(name, s):
    open(os.path.join(TD, f'{name}.tex'), 'w').write(s)


def note(t):
    return f"\\noindent{{\\footnotesize{{{t}}}}}\n"


def usd1(v):
    """terminal value of USD 1 (v is value of USD 100)"""
    x = v / 100
    return f"{x:,.0f}" if x >= 100 else f"{x:,.1f}"


SRDEF = ("SR is the Sharpe ratio: the annualized mean of daily returns in excess of the one-month Treasury bill rate "
         "divided by their annualized standard deviation.")
DAG = "$^{\\dagger}$"
MATCHNOTE = f"{DAG} Look-back window equal to the holding period (fixed-window design)."
GRP = ("Winners, Median, and Losers hold the industries ranked 1--3, 4--7, and 8--10 on their return over the look-back "
       "window; Ex-Winners holds ranks 4--10; EW holds all ten industries and is rebalanced at the same frequency.")
WIDE = "\\begin{adjustwidth}{-\\extralength}{0cm}\n\\small\n"


def row(c, g):
    return MR[(MR.freq == c[0]) & (MR.L == c[1]) & (MR.group == g)].iloc[0]


def cfg_cells(c, first, cs=CONFIGS):
    f, L = c
    nr = sum(1 for cc in cs if cc[0] == f)
    fcell = (f"\\multirow{{{nr}}}{{*}}{{{FL[f]}}}" if nr > 1 else FL[f]) if first else ''
    return f"{fcell} & {wlab(f, L)}{DAG if matched(c) else ''}"


def blocks(cs):
    for i, c in enumerate(cs):
        first = i == 0 or cs[i - 1][0] != c[0]
        last = i == len(cs) - 1 or cs[i + 1][0] != c[0]
        yield c, first, last


def sr_table(name, cap, getter, cs=CONFIGS, extra_note='', bench=None):
    b = []
    for c, first, last in blocks(cs):
        v = {g: getter(c, g) for g in GROUPS}
        mx = max(round(x, 2) for x in v.values())
        b.append(cfg_cells(c, first, cs) + " & " + " & ".join(B(num(v[g]), round(v[g], 2) == mx) for g in GROUPS) + " \\\\")
        if last and c != cs[-1]:
            b.append("\\midrule")
    if bench is not None:
        b.append(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Market (CRSP value-weighted)}} & \\multicolumn{{5}}{{c}}{{{num(bench)}}} \\\\")
    W(name, "\\begin{table}[H]\n\\caption{" + cap + "\\label{tab:" + name + "}}\n"
      "\\small\\begin{tabularx}{\\textwidth}{llCCCCC}\n\\toprule\n"
      "\\textbf{Frequency} & \\textbf{Window} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} & \\textbf{Ex-Winners} & \\textbf{EW} \\\\\n\\midrule\n"
      + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n" + note(extra_note) + "\\end{table}\n")


# ------------------------------------------------------------ industries
ind = o['ind'].set_index('t')
b = []
for t in o['tick']:
    m = ind.loc[t]
    b.append(f"{t} & {IND10[t]} & {usd1(m['final'])} & {pct(m['cagr'])} & {pct(m['vol'])} & {num(m['sr'])} & {pct(m['mdd'])} \\\\")
m = ind.loc['MKT']
b.append("\\midrule")
b.append(f"MKT & Market (CRSP value-weighted) & {usd1(m['final'])} & {pct(m['cagr'])} & {pct(m['vol'])} & {num(m['sr'])} & {pct(m['mdd'])} \\\\")
W('individual', "\\begin{table}[H]\n\\caption{The ten Fama--French industry portfolios: buy-and-hold performance, January 1928--December 2025.\\label{tab:individual}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCC}\n\\toprule\n"
  "\\textbf{Code} & \\textbf{Industry} & \\textbf{\\makecell{Value of\\\\USD~1}} & \\textbf{\\makecell{Annualized\\\\Return}} & \\textbf{Volatility} & \\textbf{SR} & \\textbf{\\makecell{Maximum\\\\Drawdown}} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("Value-weighted daily returns from Kenneth French's data library, including dividends. Value on 31 December 2025 of USD~1 invested at the close of 31 December 1927. Industries are defined by four-digit SIC codes and reassigned each June. " + SRDEF)
  + "\\end{table}\n")

# ------------------------------------------------------------ design
nreb = {'A': 98, 'S': 196, 'Q': 392, 'M': 1176, 'W': None}
b = []
nW = int(MR[(MR.freq == 'W') & (MR.group == 'W')].nreb.iloc[0]) + 1
for f, hold, wins in [('A', '12 months', '1, 3, 6, 12 months'), ('S', '6 months', '1, 3, 6 months'),
                      ('Q', '3 months', '1, 3 months'), ('M', '1 month', '1 month'), ('W', '1 week', '1 week')]:
    k = sum(1 for c in CONFIGS if c[0] == f)
    b.append(f"{FL[f]} & {hold} & {nreb[f] if nreb[f] else nW:,} & {wins} & {k} & {3 * k} \\\\")
W('design', "\\begin{table}[H]\n\\caption{Design of the experiment: rebalancing frequencies and look-back (ranking) windows.\\label{tab:design}}\n"
  "\\small\\begin{tabularx}{\\textwidth}{lcccCc}\n\\toprule\n"
  "\\textbf{Frequency} & \\textbf{\\makecell{Holding\\\\Period}} & \\textbf{\\makecell{Holding\\\\Periods}} & \\textbf{\\makecell{Look-Back\\\\Windows}} & \\textbf{\\makecell{Configu-\\\\rations}} & \\textbf{\\makecell{Ranked\\\\Strategies}} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\midrule\nTotal & & & & 11 & 33 \\\\\n\\bottomrule\n\\end{tabularx}\n"
  + note("At every rebalancing date, the ten industries are ranked on their total return over the stated number of calendar months ending at that date, and three equal-weighted portfolios are formed: Winners (ranks 1--3), Median (ranks 4--7), and Losers (ranks 8--10). The window equal to the holding period reproduces the fixed-window design of earlier studies. Monthly and weekly rebalancing keep their matched windows and serve as reference points.")
  + "\\end{table}\n")

# ------------------------------------------------------------ grids
sr_table('gridsr', f'Sharpe ratios by rebalancing frequency and look-back window before transaction costs, {PER}.',
         lambda c, g: row(c, g).g_sr, extra_note=f"{GRP} Bold marks the highest value in each row. {MATCHNOTE} {SRDEF}",
         bench=MR[MR.group == 'MKT'].g_sr.iloc[0])

mkt = MR[MR.group == 'MKT'].iloc[0]
b = []
for c, first, last in blocks(CONFIGS):
    r = {g: row(c, g) for g in ['W', 'M', 'L', 'EW']}
    mxr = max(round(r[g].g_cagr, 3) for g in r)
    mxd = max(round(r[g].g_mdd, 3) for g in r)
    b.append(cfg_cells(c, first) + " & " + " & ".join(B(pct(r[g].g_cagr), round(r[g].g_cagr, 3) == mxr) for g in r)
             + " & " + " & ".join(B(pct(r[g].g_mdd), round(r[g].g_mdd, 3) == mxd) for g in r) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
W('gridret', "\\begin{table}[H]\n\\caption{Annualized return and maximum drawdown by rebalancing frequency and look-back window before transaction costs, " + PER + ".\\label{tab:gridret}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{4}{c}{\\textbf{Annualized Return}} & \\multicolumn{4}{c}{\\textbf{Maximum Drawdown}} \\\\\n"
  "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + f"\n\\midrule\n\\multicolumn{{2}}{{l}}{{Market}} & \\multicolumn{{4}}{{c}}{{{pct(mkt.g_cagr)}}} & \\multicolumn{{4}}{{c}}{{{pct(mkt.g_mdd)}}} \\\\\n\\bottomrule\n\\end{{tabularx}}\n\\end{{adjustwidth}}\n"
  + note(f"W = Winners, M = Median, L = Losers, EW = equal-weighted portfolio of all ten industries at the same frequency. The maximum drawdown is the largest peak-to-trough decline over the full sample; all strategies reach it during 1929--1932. Bold marks the highest return and the smallest drawdown in each row and panel. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ bootstrap
BT = o['boot']


def bt_(c, a, b_):
    return BT[(BT.freq == c[0]) & (BT.L == c[1]) & (BT.a == a) & (BT.b == b_)].iloc[0]


b = []
for c, first, last in blocks(CONFIGS):
    cells = []
    for a, b_ in [('W', 'EW'), ('M', 'EW'), ('L', 'EW'), ('M', 'W')]:
        r = bt_(c, a, b_)
        cells.append(f"{num(r['diff'])} & {B(pv(r.p), r.p < 0.05)}")
    b.append(cfg_cells(c, first) + " & " + " & ".join(cells) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
W('boot', "\\begin{table}[H]\n\\caption{Bootstrap tests of Sharpe ratio differences, " + PER + ".\\label{tab:boot}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{2}{c}{\\textbf{Winners $-$ EW}} & \\multicolumn{2}{c}{\\textbf{Median $-$ EW}} & \\multicolumn{2}{c}{\\textbf{Losers $-$ EW}} & \\multicolumn{2}{c}{\\textbf{Median $-$ Winners}} \\\\\n"
  "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}\n"
  "\\textbf{Frequency} & \\textbf{Window} & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"$\\Delta$SR is the difference in Sharpe ratios. Two-sided $p$-values come from a stationary bootstrap~\\cite{{politisromano1994}} of the paired daily excess returns, with 5000 resamples and a mean block length of 21 trading days; bold marks $p < 0.05$ before any correction for multiple testing. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ window differences
WD = o['windiff']
b = []
for f in 'ASQ':
    cells = []
    for g in ['W', 'M', 'L', 'XW']:
        r = WD[(WD.freq == f) & (WD.group == g)].iloc[0]
        cells.append(f"{num(r['diff'])} & {B(pv(r.p), r.p < 0.05)}")
    b.append(f"{FL[f]} (1 vs. {HOLD_M[f]} months) & " + " & ".join(cells) + " \\\\")
cells = []
for g in ['W', 'M', 'L', 'XW']:
    r = WD[(WD.freq == 'avg') & (WD.group == g)]
    cells.append(f"{num(r.iloc[0]['diff'])} & {B(pv(r.iloc[0].p), r.iloc[0].p < 0.05)}" if len(r) else " & ")
b.append("\\midrule\n\\textit{Average of the three} & " + " & ".join(cells) + " \\\\")
W('windiff', "\\begin{table}[H]\n\\caption{Effect of the look-back window: Sharpe ratio with a one-month window minus Sharpe ratio with the matched window, same group and frequency, " + PER + ".\\label{tab:windiff}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{lCCCCCCCC}\n\\toprule\n"
  " & \\multicolumn{2}{c}{\\textbf{Winners}} & \\multicolumn{2}{c}{\\textbf{Median}} & \\multicolumn{2}{c}{\\textbf{Losers}} & \\multicolumn{2}{c}{\\textbf{Ex-Winners}} \\\\\n"
  "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n"
  "\\textbf{Frequency} & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("Stationary bootstrap as in Table~\\ref{tab:boot}. The average row bootstraps the mean of the three differences jointly. Returns before transaction costs.")
  + "\\end{table}\n")

# ------------------------------------------------------------ pooled
PO = o['pool']
PN = {'all11': 'All eleven configurations', 'short1m': 'One-month windows (annual, semiannual, quarterly)',
      'shorter6': 'Windows shorter than the holding period (six)', 'matched5': 'Matched windows (five frequencies)'}
CN = {'W-EW': 'Winners $-$ EW', 'M-EW': 'Median $-$ EW', 'L-EW': 'Losers $-$ EW', 'M-W': 'Median $-$ Winners',
      'L-W': 'Losers $-$ Winners', 'XW-EW': 'Ex-Winners $-$ EW'}
b = []
for pn in ['all11', 'short1m', 'shorter6', 'matched5']:
    b.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{PN[pn]}}}}} \\\\")
    for cn in ['W-EW', 'M-EW', 'L-EW', 'M-W']:
        r = PO[(PO['pool'] == pn) & (PO.comp == cn)].iloc[0]
        b.append(f"\\quad {CN[cn]} & {num(r.avg)} & [{num(r.lo)}, {num(r.hi)}] & {B(pv(r.p), r.p < 0.05)} & {r.npos} of {r.n} & {num(r.mean_corr)} \\\\")
W('pooled', "\\begin{table}[H]\n\\caption{Pooled tests: average Sharpe ratio differences across configurations, " + PER + ".\\label{tab:pooled}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{lCCCCC}\n\\toprule\n"
  "\\textbf{Comparison} & \\textbf{\\makecell{Average\\\\$\\Delta$SR}} & \\textbf{\\makecell{95\\%\\\\Interval}} & \\textbf{$p$-Value} & \\textbf{\\makecell{Positive\\\\Differences}} & \\textbf{\\makecell{Average\\\\Correlation}} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("The average $\\Delta$SR is the mean of the Sharpe ratio differences across the configurations in each block; its 95\\% interval and two-sided $p$-value come from the stationary bootstrap of Table~\\ref{tab:boot}, applied jointly, so they account for the dependence between configurations. The last column is the average pairwise correlation of the bootstrapped differences. Bold marks $p < 0.05$.")
  + "\\end{table}\n")

# ------------------------------------------------------------ costs
b = []
for c, first, last in blocks(CONFIGS):
    r = {g: row(c, g) for g in ['W', 'M', 'L', 'EW']}
    b.append(cfg_cells(c, first) + " & " + " & ".join(f"{r[g].avg_oneway * 100:.0f}\\%" for g in ['W', 'M', 'L'])
             + " & " + " & ".join(num(r[g].ne_sr) for g in ['W', 'M', 'L'])
             + " & " + " & ".join(num(r[g].n_sr) for g in ['W', 'M', 'L'])
             + " & " + " & ".join(num(r[g].n50_sr) for g in ['W', 'M', 'L']) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
W('costs', "\\begin{table}[H]\n\\caption{Turnover and Sharpe ratios after transaction costs, " + PER + ".\\label{tab:costs}}\n"
  + WIDE + "\\setlength{\\tabcolsep}{3pt}\\begin{tabularx}{\\fulllength}{llCCCCCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{3}{c}{\\textbf{Turnover}} & \\multicolumn{3}{c}{\\textbf{Net SR, 1.5 bp}} & \\multicolumn{3}{c}{\\textbf{Net SR, 10 bp}} & \\multicolumn{3}{c}{\\textbf{Net SR, 50 bp}} \\\\\n"
  "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{W} & \\textbf{M} & \\textbf{L} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Turnover is one-way per rebalancing, computed from drifted weights. Every one-way trade is charged the stated spread: 1.5 basis points (bp), typical of today's sector ETFs; 10 bp; and 50 bp, a stress case for earlier decades. W = Winners, M = Median, L = Losers. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ random
RP = o['rpt']
b = []
for c, first, last in blocks(CONFIGS):
    r = RP[(RP.freq == c[0]) & (RP.win == c[1])].iloc[0]
    b.append(cfg_cells(c, first) + f" & {num(r.med3)} & [{num(r.lo3)}, {num(r.hi3)}] & {num(r.med4)} & "
             + " & ".join(f"{num(r[g])} ({r[g + '_pct']:.0f})" for g in ['W', 'M', 'L', 'EW']) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
W('random', "\\begin{table}[H]\n\\caption{Rotation portfolios versus 5000 random portfolios, " + PER + ".\\label{tab:random}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{3}{c}{\\textbf{Random Portfolios}} & \\multicolumn{4}{c}{\\textbf{SR (Percentile among Random Portfolios)}} \\\\\n"
  "\\cmidrule(lr){3-5} \\cmidrule(lr){6-9}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{\\makecell{50th Pct.\\\\(3 Ind.)}} & \\textbf{\\makecell{95\\% Range\\\\(3 Ind.)}} & \\textbf{\\makecell{50th Pct.\\\\(4 Ind.)}} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"At every rebalancing date, each random portfolio draws three (or four) of the ten industries at random and holds them in equal weights until the next date. Winners, Losers, and EW are compared with the three-industry distribution and Median with the four-industry distribution. The percentile is the share of random portfolios with a lower Sharpe ratio. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ sub-periods
SP, SPB = o['sp'], o['spb']
b = []
for per in ['1928-1962', '1963-1998', '1999-2025']:
    b.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{per.replace('-', '--')}}}}} \\\\")
    for c, first, last in blocks(CONFIGS):
        v = {g: SP[(SP.per == per) & (SP.freq == c[0]) & (SP.L == c[1]) & (SP.group == g)].sr.iloc[0] for g in GROUPS}
        mx = max(round(x, 2) for x in v.values())
        b.append(cfg_cells(c, first) + " & " + " & ".join(B(num(v[g]), round(v[g], 2) == mx) for g in GROUPS) + " \\\\")
    ax = SP[(SP.per == per) & (SP.group == 'MKT')].sr.iloc[0]
    b.append(f"\\multicolumn{{2}}{{l}}{{Market}} & \\multicolumn{{5}}{{c}}{{{num(ax)}}} \\\\")
    if per != '1999-2025':
        b.append("\\midrule")
W('subperiods', "\\begin{table}[H]\n\\caption{Sharpe ratios in three sub-periods.\\label{tab:subperiods}}\n"
  "\\small\\begin{tabularx}{\\textwidth}{llCCCCC}\n\\toprule\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} & \\textbf{Ex-Winners} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n"
  + note(f"The full-sample daily return series of each strategy is split at the end of 1962 and 1998. 1999--2025 is the era of the Select Sector SPDR ETFs. Bold marks the highest value in each row. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")
b = []
for per in ['1928-1962', '1963-1998', '1999-2025']:
    cells = []
    for pn in ['matched5', 'short1m']:
        for g in ['W', 'M', 'L']:
            r = SPB[(SPB.per == per) & (SPB.freq == pn) & (SPB.group == g)].iloc[0]
            cells.append(f"{num(r['diff'])} ({B(pv(r.p), r.p < 0.05)})")
    b.append(f"{per.replace('-', '--')} & " + " & ".join(cells) + " \\\\")
W('subpooled', "\\begin{table}[H]\n\\caption{Average Sharpe ratio difference from the EW portfolio by sub-period ($p$-value).\\label{tab:subpooled}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{lCCCCCC}\n\\toprule\n"
  " & \\multicolumn{3}{c}{\\textbf{Matched Windows (Five Frequencies)}} & \\multicolumn{3}{c}{\\textbf{One-Month Windows (A, S, Q)}} \\\\\n"
  "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
  "\\textbf{Period} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("Average across the configurations of each set of the Sharpe ratio of the group minus that of the EW portfolio at the same frequency, within each sub-period. Two-sided $p$-values from a stationary bootstrap of the sub-period daily returns (2000 resamples, mean block length of 21 days), applied jointly to all configurations. Bold marks $p < 0.05$.")
  + "\\end{table}\n")

# ------------------------------------------------------------ event time
E = o['evt']
b = []
hl = {'1': 'Month 1', '2': 'Month 2', '3': 'Month 3', '4-6': 'Months 4--6', '7-12': 'Months 7--12'}
for L in [1, 3, 6, 12]:
    b.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Look-back window: {wlab('A', L)}}}}} \\\\")
    for h in ['1', '2', '3', '4-6', '7-12']:
        r = E[(E.win == L) & (E.h == h)].iloc[0]
        b.append(f"\\quad {hl[h]} & " + " & ".join(B(f"{num(r[g])} ({num(r[g + '_t'])})", abs(r[g + '_t']) >= 1.96) for g in ['W', 'M', 'L']) + " \\\\")
W('eventtime', "\\begin{table}[H]\n\\caption{Event-time returns of the ranked groups relative to the equal-weighted portfolio, percent per month ($t$-statistic), monthly rankings, " + PER + ".\\label{tab:eventtime}}\n"
  "\\small\\begin{tabularx}{\\textwidth}{lCCC}\n\\toprule\n"
  "\\textbf{Months after Ranking} & \\textbf{Winners $-$ EW} & \\textbf{Median $-$ EW} & \\textbf{Losers $-$ EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n"
  + note("At the end of every month from December 1927 to November 2025, the ten industries are ranked on their return over the stated window. The table reports the average return, $h$ months later, of an equal-weighted portfolio of each group minus that of all ten industries, in percent per month. Multi-month horizons are first averaged within each ranking date. $t$-statistics use Newey--West standard errors with $h_{\\max}+2$ lags. Bold marks $|t| \\geq 1.96$.")
  + "\\end{table}\n")

# ------------------------------------------------------------ December
DC = o['dec']
b = []
for H in [1, 3, 12]:
    cells = []
    for g in ['W', 'M', 'L']:
        for mo in ['Dec', 'Jun', 'Other']:
            r = DC[(DC.H == H) & (DC.group == g) & (DC.months == mo)].iloc[0]
            cells.append(B(f"{num(r['mean'])} ({num(r.t)})", abs(r.t) >= 1.96))
    b.append(f"{H} & " + " & ".join(cells) + " \\\\")
W('december', "\\begin{table}[H]\n\\caption{One-month window: average monthly return relative to the EW portfolio over the first $H$ months after the ranking, by ranking month, percent ($t$-statistic), " + PER + ".\\label{tab:december}}\n"
  + WIDE + "\\setlength{\\tabcolsep}{3pt}\\begin{tabularx}{\\fulllength}{cCCCCCCCCC}\n\\toprule\n"
  " & \\multicolumn{3}{c}{\\textbf{Winners $-$ EW}} & \\multicolumn{3}{c}{\\textbf{Median $-$ EW}} & \\multicolumn{3}{c}{\\textbf{Losers $-$ EW}} \\\\\n"
  "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
  "\\textbf{$H$} & \\textbf{Dec} & \\textbf{Jun} & \\textbf{Other} & \\textbf{Dec} & \\textbf{Jun} & \\textbf{Other} & \\textbf{Dec} & \\textbf{Jun} & \\textbf{Other} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("The ten industries are ranked on their return in the ranking month (December, June, or any of the other ten months), and the table reports the average monthly return of each group minus the EW portfolio over the following $H$ months. $t$-statistics use Newey--West standard errors with $H+2$ lags. Bold marks $|t| \\geq 1.96$.")
  + "\\end{table}\n")

# ------------------------------------------------------------ calendar robustness
S_, P_ = OF['S'], OF['P']
b = []
for c, first, last in blocks(CORE):
    r = S_[(S_.freq == c[0]) & (S_.win == c[1])].iloc[0]
    b.append(cfg_cells(c, first, CORE) + f" & {int(r.ncal)} & " + " & ".join(num(r[f'sr_{g}']) for g in ['W', 'M', 'L', 'EW'])
             + f" & {int(r.W_lt_EW)} & {num(r['W-EW'])} & {B(pv(r['W-EW_p']), r['W-EW_p'] < 0.05)}"
             + f" & {num(r['M-EW'])} & {B(pv(r['M-EW_p']), r['M-EW_p'] < 0.05)}"
             + f" & {num(r['L-EW'])} & {B(pv(r['L-EW_p']), r['L-EW_p'] < 0.05)} \\\\")
    if last and c != CORE[-1]:
        b.append("\\midrule")
b.append("\\midrule")
for pn, lab_ in [('short1m', 'Average, one-month windows'), ('core9', 'Average, all nine')]:
    cells = []
    for cn in ['W-EW', 'M-EW', 'L-EW']:
        r = P_[(P_['pool'] == pn) & (P_.comp == cn)].iloc[0]
        cells.append(f"{num(r.avg)} & {B(pv(r.p), r.p < 0.05)}")
    b.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab_}}}}} & " + " & ".join(cells) + " \\\\")
W('offsets', "\\begin{table}[H]\n\\caption{Calendar robustness: Sharpe ratios averaged over all rebalancing calendars, " + PER + ".\\label{tab:offsets}}\n"
  + WIDE + "\\setlength{\\tabcolsep}{3pt}\\begin{tabularx}{\\fulllength}{llcCCCCcCCCCCC}\n\\toprule\n"
  " & & & \\multicolumn{4}{c}{\\textbf{Average SR}} & & \\multicolumn{2}{c}{\\textbf{W $-$ EW}} & \\multicolumn{2}{c}{\\textbf{M $-$ EW}} & \\multicolumn{2}{c}{\\textbf{L $-$ EW}} \\\\\n"
  "\\cmidrule(lr){4-7} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-14}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{\\makecell{Calen-\\\\dars}} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} & \\textbf{\\makecell{W below\\\\EW}} & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ & $\\Delta$SR & $p$ \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Each configuration is re-run with its rebalancing calendar shifted by 0 to $H-1$ months, where $H$ is the holding period in months: twelve annual, six semiannual, and three quarterly calendars. The look-back window always ends at the rebalancing date. Average SR is the mean across calendars; W below EW counts the calendars on which Winners have a lower Sharpe ratio than EW. $\\Delta$SR is the average difference across calendars, with two-sided $p$-values from a stationary bootstrap (2000 resamples) of the average. Bold marks $p < 0.05$. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ lag
LG = o['lag']
sr_table('laguni', f'Sharpe ratios with a one-day implementation lag, {PER}.',
         lambda c, g: LG[(LG.freq == c[0]) & (LG.L == c[1]) & (LG.group == g)].sr.iloc[0],
         extra_note=f"The industries are ranked at the close of each rebalancing date but traded at the close of the following trading day. Bold marks the highest value in each row. Returns before transaction costs. {MATCHNOTE}")

# ------------------------------------------------------------ leave-one-out
L_ = o['loo']
b = []
for c, first, last in blocks(CONFIGS):
    d = L_[(L_.freq == c[0]) & (L_.win == c[1])]
    dd = d.W - d.EW
    b.append(cfg_cells(c, first) + f" & {(d.W > d.EW).sum()} & [{num(dd.min())}, {num(dd.max())}] & {(d.M > d.EW).sum()} & {(d.L > d.EW).sum()} & {(d.M > d.W).sum()} \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
W('loo', "\\begin{table}[H]\n\\caption{Leave-one-out analysis: number of the ten nine-industry universes in which each comparison holds, " + PER + ".\\label{tab:loo}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCC}\n\\toprule\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{\\makecell{Winners\\\\above EW}} & \\textbf{\\makecell{Range of $\\Delta$SR,\\\\Winners $-$ EW}} & \\textbf{\\makecell{Median\\\\above EW}} & \\textbf{\\makecell{Losers\\\\above EW}} & \\textbf{\\makecell{Median\\\\above Winners}} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Each industry is dropped in turn and the strategies are re-estimated on the remaining nine, with groups of three. Counts are out of 10. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ alternative universes
AL = o['alt']
UNIS = list(AL.universe.unique())
b = []
for c, first, last in blocks(CONFIGS):
    cells = []
    for u in UNIS:
        r = AL[(AL.universe == u) & (AL.freq == c[0]) & (AL.win == c[1])].iloc[0]
        cells.append(f"{num(r.W - r.EW)} & {num(r.M - r.EW)} & {num(r.L - r.EW)}")
    b.append(cfg_cells(c, first) + " & " + " & ".join(cells) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{\\makecell{{{u.replace(' (', '\\\\(')}}}}}}}" for u in UNIS)
cm = " ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(UNIS)))
W('altuni', "\\begin{table}[H]\n\\caption{Alternative industry universes: Sharpe ratio of each group minus that of the EW portfolio.\\label{tab:altuni}}\n"
  + WIDE + "\\setlength{\\tabcolsep}{2.5pt}\\footnotesize\\begin{tabularx}{\\fulllength}{ll" + "CCC" * len(UNIS) + "}\n\\toprule\n"
  " & & " + hdr + " \\\\\n" + cm + "\n"
  "\\textbf{Frequency} & \\textbf{Window} & " + " & ".join(["W & M & L"] * len(UNIS)) + " \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Groups: nine industries (Fama--French ten excluding Other), 3/3/3; twelve industries, 4/4/4; 49 industries, 16/17/16 (sample 1973--2025, when all 49 have complete data); ten equal-weighted industry portfolios, 3/4/3. W = Winners, M = Median, L = Losers. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ longer windows
EXTALL = [('S', 6), ('S', 12), ('Q', 3), ('Q', 6), ('Q', 12), ('M', 1), ('M', 3), ('M', 6), ('M', 12)]
sr_table('longwin', f'Sharpe ratios with look-back windows longer than the holding period, {PER}.',
         lambda c, g: row(c, g).g_sr, cs=EXTALL,
         extra_note=f"The matched windows are repeated from Table~\\ref{{tab:gridsr}} for comparison. Bold marks the highest value in each row. Returns before transaction costs. {MATCHNOTE}")

# ------------------------------------------------------------ overlap
OV, RC = o['overlap'], o['rankcorr']
b = []
for f in 'ASQ':
    Ls = [L for (ff, L) in CONFIGS if ff == f]
    for i, L1 in enumerate(Ls):
        for L2 in Ls[i + 1:]:
            ov = {g: OV[(OV.freq == f) & (OV.L1 == L1) & (OV.L2 == L2) & (OV.group == g)].overlap.iloc[0] for g in ['W', 'M', 'L']}
            rho = RC[(RC.freq == f) & (RC.L1 == L1) & (RC.L2 == L2)].rho.iloc[0]
            b.append(f"{FL[f]} & {L1} vs. {L2} months & {num(rho)} & " + " & ".join(f"{ov[g]:.1f}" for g in ['W', 'M', 'L']) + " \\\\")
    if f != 'Q':
        b.append("\\midrule")
W('overlap', "\\begin{table}[H]\n\\caption{Rank correlation and overlap of the groups between look-back windows, " + PER + ".\\label{tab:overlap}}\n"
  "\\small\\begin{tabularx}{\\textwidth}{llCCCC}\n\\toprule\n"
  " & & & \\multicolumn{3}{c}{\\textbf{Common Industries}} \\\\\n\\cmidrule(lr){4-6}\n"
  "\\textbf{Frequency} & \\textbf{Windows} & \\textbf{\\makecell{Rank\\\\Correlation}} & \\textbf{\\makecell{Winners\\\\(of 3)}} & \\textbf{\\makecell{Median\\\\(of 4)}} & \\textbf{\\makecell{Losers\\\\(of 3)}} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n"
  + note("Spearman rank correlation between the rankings of the ten industries under the two windows, averaged over rebalancing dates, and the average number of industries that the same group holds under both windows. Two random selections of three (four) of ten share 0.9 (1.6) industries on average.")
  + "\\end{table}\n")

# ------------------------------------------------------------ composition (monthly, 1-month)
cm1 = o['comp'][('M', 1)]
cq1 = o['comp'][('Q', 1)]
b = []
for t in o['tick']:
    b.append(f"{t} & {IND10[t]} & " + " & ".join(f"{cm1.loc[t, g]:.0f}" for g in ['W', 'M', 'L']) + " & "
             + " & ".join(f"{cq1.loc[t, g]:.0f}" for g in ['W', 'M', 'L']) + " \\\\")
W('composition', "\\begin{table}[H]\n\\caption{Share of holding periods in which each industry belongs to each group, one-month window (\\%).\\label{tab:composition}}\n"
  "\\small\\begin{tabularx}{\\textwidth}{llCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{3}{c}{\\textbf{Monthly}} & \\multicolumn{3}{c}{\\textbf{Quarterly}} \\\\\n"
  "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
  "\\textbf{Code} & \\textbf{Industry} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{W} & \\textbf{M} & \\textbf{L} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n"
  + note("Monthly rebalancing, 1176 holding periods; quarterly rebalancing, 392. W = Winners (3 industries), M = Median (4), L = Losers (3). Each row sums to 100\\% within a frequency.")
  + "\\end{table}\n")

# ------------------------------------------------------------ factor tables
F, FP = o['fac'], o['fac_pool']


def fc(c, g, model):
    r = F[(F.freq == c[0]) & (F.L == c[1]) & (F.group == g) & (F.model == model)].iloc[0]
    return B(f"{num(r.alpha_d * 100)} ({num(r.t_d)})", abs(r.t_d) >= 1.96)


def fm(c, g, model):
    r = F[(F.freq == c[0]) & (F.L == c[1]) & (F.group == g) & (F.model == model)].iloc[0]
    return B(f"{num(r.bd_MOM)} ({num(r.td_MOM)})", abs(r.td_MOM) >= 1.96)


for model, name, cap, nm in [('FF3+MOM', 'factor', 'Alphas and momentum loadings relative to the EW portfolio in the Fama--French three-factor model with momentum, monthly returns, 1928--2025.', 1176),
                             ('FF5+MOM', 'ff', 'Alphas and momentum loadings relative to the EW portfolio in the Fama--French five-factor model with momentum, monthly returns, July 1963--December 2025.', o['nmonths']['FF5+MOM'])]:
    b = ["\\multicolumn{8}{l}{\\textit{Annualized alpha, \\% ($t$-statistic) and loading on the momentum factor ($t$-statistic)}} \\\\"]
    for c, first, last in blocks(CONFIGS):
        b.append(cfg_cells(c, first) + " & " + " & ".join(fc(c, g, model) for g in ['W', 'M', 'L']) + " & "
                 + " & ".join(fm(c, g, model) for g in ['W', 'M', 'L']) + " \\\\")
        if last and c != CONFIGS[-1]:
            b.append("\\midrule")
    b.append("\\midrule")
    for pn, lab_ in [('all11', 'Average, all eleven'), ('short1m', 'Average, one-month windows'), ('matched5', 'Average, matched windows')]:
        cells, cells2 = [], []
        for g in ['W', 'M', 'L']:
            r = FP[(FP['pool'] == pn) & (FP.group == g) & (FP.model == model)].iloc[0]
            cells.append(B(f"{num(r.alpha * 100)} ({num(r.t)})", abs(r.t) >= 1.96))
            cells2.append(B(f"{num(r.b_MOM)} ({num(r.t_MOM)})", abs(r.t_MOM) >= 1.96))
        b.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{lab_}}}}} & " + " & ".join(cells) + " & " + " & ".join(cells2) + " \\\\")
    W(name, "\\begin{table}[H]\n\\caption{" + cap + "\\label{tab:" + name + "}}\n"
      + WIDE + "\\setlength{\\tabcolsep}{3pt}\\footnotesize\\begin{tabularx}{\\fulllength}{llCCCCCC}\n\\toprule\n"
      " & & \\multicolumn{3}{c}{\\textbf{Alpha relative to EW}} & \\multicolumn{3}{c}{\\textbf{Momentum Loading}} \\\\\n"
      "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
      "\\textbf{Frequency} & \\textbf{Window} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} & \\textbf{Winners} & \\textbf{Median} & \\textbf{Losers} \\\\\n\\midrule\n"
      + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
      + note(f"Regressions of the monthly return of each strategy minus that of the EW portfolio rebalanced at the same frequency on the Fama--French factors and the momentum factor (MOM) from Kenneth French's data library; {nm} months. Alphas are annualized (multiplied by 12). $t$-statistics use Newey--West standard errors with three lags. Bold marks $|t| \\geq 1.96$. The average rows regress the mean of the strategy-minus-EW returns across the configurations in each set. {MATCHNOTE}")
      + "\\end{table}\n")


# ------------------------------------------------------------ appendix: performance tables
def perf_table(cs, label, cap):
    b = []
    for c in cs:
        for gi, g in enumerate(['W', 'M', 'L', 'XW', 'EW']):
            x = row(c, g)
            lead = f"{FL[c[0]]}, {wlab(*c)}{DAG if matched(c) else ''}" if gi == 0 else ''
            b.append(f"{lead} & {GN[g]} & {usd1(x.g_final)} & {pct(x.g_cagr)} & {pct(x.g_vol)} & {num(x.g_sr)} & {pct(x.g_avgdd)} & {pct(x.g_maxadd)} & {pct(x.g_mdd)} & {usd1(x.n_final)} & {num(x.n_sr)} \\\\")
        if c != cs[-1]:
            b.append("\\midrule")
    return ("\\begin{table}[H]\n\\caption{" + cap + "\\label{" + label + "}}\n" + WIDE.replace('\\small', '\\footnotesize\\setlength{\\tabcolsep}{3pt}')
            + "\\begin{tabularx}{\\fulllength}{llCCCCCCCCC}\n\\toprule\n"
            "\\textbf{Configuration} & \\textbf{Portfolio} & \\textbf{\\makecell{Value\\\\of\\\\USD~1}} & \\textbf{\\makecell{Annual-\\\\ized\\\\Return}} & \\textbf{\\makecell{Vola-\\\\tility}} & \\textbf{SR} & \\textbf{\\makecell{Avg.\\\\Annual\\\\Draw-\\\\down}} & \\textbf{\\makecell{Max.\\\\Annual\\\\Draw-\\\\down}} & \\textbf{\\makecell{Max.\\\\Full-\\\\Sample\\\\Draw-\\\\down}} & \\textbf{\\makecell{Net\\\\Value\\\\of\\\\USD~1}} & \\textbf{\\makecell{Net\\\\SR}} \\\\\n\\midrule\n"
            + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
            + note(f"USD~1 invested at the close of 31 December 1927; daily returns, 3 January 1928 to 31 December 2025. Columns 3--9 are before transaction costs; the last two columns charge 10~bp on actual turnover. {MATCHNOTE} {SRDEF}")
            + "\\end{table}\n")


W('perfA', perf_table([c for c in CONFIGS if c[0] == 'A'], 'tab:perfA', 'Performance and risk of the annual strategies, 1928--2025.'))
W('perfB', perf_table([c for c in CONFIGS if c[0] in 'SQ'], 'tab:perfB', 'Performance and risk of the semiannual and quarterly strategies, 1928--2025.'))
W('perfC', perf_table([c for c in CONFIGS if c[0] in 'MW'], 'tab:perfC', 'Performance and risk of the monthly and weekly reference strategies, 1928--2025.'))
print(sorted(os.listdir(TD)))

# ------------------------------------------------------------ ETF era (sector ETF study, 2000-2024)
SECR = os.path.join('etf_era', 'results')
so = pickle.load(open(os.path.join(SECR, 'out.pkl'), 'rb'))
sof = pickle.load(open(os.path.join(SECR, 'offsets.pkl'), 'rb'))['S']
SMR = so['MR']
b = []
for c, first, last in blocks(CONFIGS):
    r = {g: SMR[(SMR.freq == c[0]) & (SMR.L == c[1]) & (SMR.group == g)].iloc[0] for g in ['W', 'M', 'L', 'EW']}
    mx = max(round(r[g].g_sr, 2) for g in r)
    cells = " & ".join(B(num(r[g].g_sr), round(r[g].g_sr, 2) == mx) for g in ['W', 'M', 'L', 'EW'])
    if c[0] in 'ASQ':
        q = sof[(sof.freq == c[0]) & (sof.win == c[1])].iloc[0]
        cal = " & ".join(num(q[f'sr_{g}']) for g in ['W', 'M', 'L', 'EW'])
    else:
        cal = " & ".join(['--'] * 4)
    b.append(cfg_cells(c, first) + f" & {cells} & {cal} \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
spy = SMR[SMR.group == 'SPY'].iloc[0]
b.append(f"\\midrule\n\\multicolumn{{2}}{{l}}{{SPY}} & \\multicolumn{{4}}{{c}}{{{num(spy.g_sr)}}} & \\multicolumn{{4}}{{c}}{{}} \\\\")
W('etf_era', "\\begin{table}[H]\n\\caption{The ETF era: Sharpe ratios of Winners, Median, and Losers among the nine Select Sector SPDR ETFs, January 2000--December 2024.\\label{tab:etf_era}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{4}{c}{\\textbf{Standard Calendar}} & \\multicolumn{4}{c}{\\textbf{Average over All Calendars}} \\\\\n"
  "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Nine Select Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY); Winners, Median, and Losers hold the ETFs ranked 1--3, 4--6, and 7--9; EW holds all nine. The standard calendar rebalances at the end of calendar years, half-years, quarters, months, and weeks; the matched windows reproduce the published study~\\cite{{saiteja2026sector}}. The last four columns average each annual, semiannual, and quarterly configuration over all 12, 6, or 3 possible rebalancing calendars. Daily adjusted closes from Yahoo Finance; risk-free rate TB3MS. Bold marks the highest value in each row of the standard calendar. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")

# ------------------------------------------------------------ head-to-head, 2000-2024
BR = pickle.load(open('results/bridge.pkl', 'rb'))
G_, PO_ = BR['grid'], BR['pool']
b = []
for c, first, last in blocks(CONFIGS):
    cells = []
    for u in ['ETF', 'FF9']:
        v = {g: G_[(G_.univ == u) & (G_.freq == c[0]) & (G_.L == c[1]) & (G_.group == g)].sr.iloc[0] for g in ['W', 'M', 'L', 'EW']}
        cells.append(" & ".join(num(v[g]) for g in ['W', 'M', 'L', 'EW']))
    b.append(cfg_cells(c, first) + " & " + " & ".join(cells) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
bp = []
PNL = {'all11': 'All eleven configurations', 'short1m': 'One-month windows (A, S, Q)', 'matched5': 'Matched windows (five)', 'M1': 'Monthly rebalancing, one-month window'}
for pn in ['all11', 'short1m', 'matched5', 'M1']:
    cells = []
    for u in ['ETF', 'FF9', 'FF10']:
        for cmp_ in ['W-EW', 'M-EW']:
            r = PO_[(PO_.univ == u) & (PO_['pool'] == pn) & (PO_.comp == cmp_)].iloc[0]
            cells.append(f"{num(r.avg)} ({B(pv(r.p), r.p < 0.05)})")
    bp.append(f"{PNL[pn]} & " + " & ".join(cells) + " \\\\")
W('bridgepool', "\\begin{table}[H]\n\\caption{Same rules, same years: average Sharpe ratio difference from the EW portfolio ($p$-value), January 2000--December 2024.\\label{tab:bridgepool}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{lCCCCCC}\n\\toprule\n"
  " & \\multicolumn{2}{c}{\\textbf{SPDR Sector ETFs}} & \\multicolumn{2}{c}{\\textbf{FF9 Industries}} & \\multicolumn{2}{c}{\\textbf{FF10 Industries}} \\\\\n"
  "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
  "\\textbf{Configurations} & \\textbf{W $-$ EW} & \\textbf{M $-$ EW} & \\textbf{W $-$ EW} & \\textbf{M $-$ EW} & \\textbf{W $-$ EW} & \\textbf{M $-$ EW} \\\\\n\\midrule\n"
  + "\n".join(bp) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note("The average is taken across the configurations of each set; two-sided $p$-values from a stationary bootstrap (2000 resamples, mean block length of 21 days) with common resamples for all strategies and universes, on the trading days common to both data sets, with the Fama--French one-month bill rate as the risk-free rate. W = Winners, M = Median, EW = equal-weighted portfolio of all ETFs or industries. FF9 = ten Fama--French industries excluding Other, groups of three; FF10 = all ten, 3/4/3. Bold marks $p < 0.05$. Returns before transaction costs.")
  + "\\end{table}\n")
W('bridge', "\\begin{table}[H]\n\\caption{Same rules, same years, two classifications: Select Sector SPDR ETFs versus Fama--French industries, January 2000--December 2024.\\label{tab:bridge}}\n"
  + WIDE + "\\setlength{\\tabcolsep}{3pt}\\begin{tabularx}{\\fulllength}{llCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{4}{c}{\\textbf{SPDR Sector ETFs (3/3/3)}} & \\multicolumn{4}{c}{\\textbf{FF9 Industries (3/3/3)}} \\\\\n"
  "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Both universes are evaluated on the {BR['ndays']} trading days common to the two data sets, with the Fama--French one-month Treasury bill rate as the risk-free rate, so the ETF Sharpe ratios can differ by about 0.01 from Table~\\ref{{tab:etf_era}}. FF9 = the ten Fama--French industries excluding Other, groups of three. W = Winners, M = Median, L = Losers, EW = equal-weighted portfolio. Returns before transaction costs. {MATCHNOTE}")
  + "\\end{table}\n")
print('etf_era and bridge done')

# ------------------------------------------------------------ out of sample, Jan 2025 - Aug 2026
OS = pickle.load(open('results/oos.pkl', 'rb'))
O_ = OS['O']
b = []
for c, first, last in blocks(CONFIGS):
    cells = []
    for u in ['ETF', 'FF9']:
        v = {g: O_[(O_.univ == u) & (O_.freq == c[0]) & (O_.L == c[1]) & (O_.group == g)].cum.iloc[0] for g in ['W', 'M', 'L', 'EW']}
        mx = max(round(x, 3) for x in v.values())
        cells.append(" & ".join(B(pct(v[g]), round(v[g], 3) == mx) for g in ['W', 'M', 'L', 'EW']))
    b.append(cfg_cells(c, first) + " & " + " & ".join(cells) + " \\\\")
    if last and c != CONFIGS[-1]:
        b.append("\\midrule")
spy_ = O_[(O_.univ == 'ETF') & (O_.group == 'BENCH')].cum.iloc[0]
mk_ = O_[(O_.univ == 'FF9') & (O_.group == 'BENCH')].cum.iloc[0]
b.append(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Benchmark}} & \\multicolumn{{4}}{{c}}{{SPY: {pct(spy_)}}} & \\multicolumn{{4}}{{c}}{{Market: {pct(mk_)}}} \\\\")
W('oos', "\\begin{table}[H]\n\\caption{After the published sample: cumulative return, 2 January 2025--31 August 2026.\\label{tab:oos}}\n"
  + WIDE + "\\begin{tabularx}{\\fulllength}{llCCCCCCCC}\n\\toprule\n"
  " & & \\multicolumn{4}{c}{\\textbf{SPDR Sector ETFs (3/3/3)}} & \\multicolumn{4}{c}{\\textbf{FF9 Industries (3/3/3)}} \\\\\n"
  "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}\n"
  "\\textbf{Frequency} & \\textbf{Window} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} & \\textbf{W} & \\textbf{M} & \\textbf{L} & \\textbf{EW} \\\\\n\\midrule\n"
  + "\n".join(b) + "\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n"
  + note(f"Cumulative total return over the {OS['ndays']} trading days from 2 January 2025 to 31 August 2026 (the last date of the Fama--French data), before transaction costs; the first portfolios are formed at the close of 31 December 2024 with look-back windows that end on that date. ETF prices are adjusted closes from Yahoo Finance downloaded on 10 October 2026. Annual and semiannual rules rebalance only once or twice and quarterly rules six times in this window, so their results reflect very few decisions. Bold marks the highest value in each row and universe. {MATCHNOTE}")
  + "\\end{table}\n")
print('oos table done')
