"""Reproduce everything (about 35 minutes): ETF era -> century analyses -> calendar robustness -> head-to-head -> out of sample -> tables -> figures."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for s in ['etf_era/analysis.py', 'etf_era/offsets.py', 'analysis_ind.py', 'offsets_ind.py', 'bridge.py', 'oos.py', 'gen_tables_ind.py', 'figs_ind.py']:
    print('==', s, flush=True)
    subprocess.run([sys.executable, os.path.join(here, s)], check=True, cwd=os.path.dirname(os.path.join(here, s)))
print('Done: results/*.csv, ../paper/tables/*.tex, ../paper/*.png')
