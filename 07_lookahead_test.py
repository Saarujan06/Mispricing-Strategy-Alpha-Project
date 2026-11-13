"""Step 7: look-ahead test.
If no decision uses future data, then rebuilding everything from data that END in December 2010 must give exactly the
same strategy returns up to December 2010 as the full run. This script:
  1. keeps only CRSP months up to 2010-12 and Compustat fiscal years ending by 2010-12-31,
  2. rebuilds the panel and all signals from these truncated files (step 1),
  3. re-runs the main strategy (step 4 rules: real-time score, 10/20 rule, Lecture 3 costs),
  4. compares month by month with the full-sample run (work/realtime.pkl).
Any difference larger than rounding error would mean that information from after 2010 leaked into earlier decisions.
Writes output/lookahead_test.txt. Takes a few minutes (it rebuilds the panel)."""
import os, subprocess, sys, pickle, numpy as np, pandas as pd
from config import *
CUT = 201012
TMP = os.path.join(WORK, 'lookahead_test'); RAWT = os.path.join(TMP, 'raw'); OUTT = os.path.join(TMP, 'out')
for d in (TMP, RAWT, OUTT): os.makedirs(d, exist_ok=True)
print('1/4 truncating CRSP and Compustat at', CUT)
c = pd.read_csv(CRSP_CLEAN); c[c.YYYYMM <= CUT].to_csv(os.path.join(TMP, 'crsp_clean.csv.gz'), index=False); del c
k = pd.read_csv(COMP_RAW, low_memory=False)
k[pd.to_datetime(k.datadate) <= f'{CUT // 100}-12-31'].to_csv(os.path.join(RAWT, os.path.basename(COMP_RAW)), index=False); del k
env = dict(os.environ, WORK_DIR=TMP, OUT_DIR=OUTT, RAW_DIR=RAWT)
print('2/4 rebuilding the panel from the truncated data')
subprocess.run([sys.executable, '-W', 'ignore', '01_build_panel.py'], cwd=HERE, env=env, check=True)
print('3/4 re-running the main strategy on the truncated panel')
code = ("import importlib; rt = importlib.import_module('04_realtime_strategy'); "
        f"rt.run('ew_pub', buffer=True)[0].to_csv(r'{os.path.join(OUTT, 'main_truncated.csv')}')")
subprocess.run([sys.executable, '-W', 'ignore', '-c', code], cwd=HERE, env=env, check=True)
print('4/4 comparing with the full-sample run')
full = pickle.load(open(os.path.join(WORK, 'realtime.pkl'), 'rb'))[('ew_pub', 'buf')][0]
trunc = pd.read_csv(os.path.join(OUTT, 'main_truncated.csv'), index_col=0)
cols = ['rL', 'rS', 'costL', 'costS', 'turnL', 'turnS', 'nL', 'nS', 'ls_net']
a, b = full.loc[:CUT, cols], trunc.loc[:CUT, cols]
same_months = a.index.equals(b.index)
diff = float(np.nanmax(np.abs(a.values - b.reindex(a.index).values))) if same_months else float('nan')
ok = same_months and diff < 1e-9
lines = [f'Look-ahead test: rebuild from data ending {CUT}, compare {len(a)} months ({a.index[0]}-{a.index[-1]}) with the full run.',
         f'  same months: {same_months}; largest absolute difference in returns, costs, turnover and holdings: {diff:.2e}',
         '  RESULT: PASS - no information from after the cut-off is used in earlier decisions.' if ok else
         '  RESULT: FAIL - results before the cut-off change when later data are removed (look-ahead).']
print('\n'.join(lines)); open(os.path.join(OUT, 'lookahead_test.txt'), 'w').write('\n'.join(lines) + '\n')
import json
json.dump({'cut': CUT, 'n_months': len(a), 'first': int(a.index[0]), 'last': int(a.index[-1]), 'max_diff': diff, 'pass': bool(ok)},
          open(os.path.join(OUT, 'lookahead_test.json'), 'w'), indent=1)
