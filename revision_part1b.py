"""Revision, part 1b: rotation without the two industries that the reviewer flagged as small before 1963
(Health care and Business equipment/High tech). Adds results to results/rev1.pkl."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import pandas as pd
from engine_ind import *
from revision_common import *
U = Universe(10)
o = pickle.load(open('results/rev1.pkl', 'rb'))
SUBP = [('1928', '1962'), ('1963', '1998'), ('1999', '2025')]
drop2 = ['Hlth', 'HiTec']
uni8 = [t for t in U.TICK if t not in drop2]
rows = []
for c in CONFIGS:
    row = dict(freq=c[0], win=c[1])
    for g in ['W', 'M', 'L', 'EW']:
        r = bt_general(U, c, g, universe=uni8)['gross']
        row[g] = U.sr(r)
        for a, b in SUBP:
            row[f'{g}_{a}'] = U.sr(r[a:b])
    rows.append(row)
o['drop_small'] = pd.DataFrame(rows)
o['drop_small_names'] = drop2
print(o['drop_small'].round(2).to_string())
pickle.dump(o, open('results/rev1.pkl', 'wb'))
