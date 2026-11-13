"""Step 0: shrink the 2.4 GB CRSP CIZ monthly file to US common stocks and the columns we need.
Reads in chunks, so it runs on a laptop with ~3 GB free RAM (takes a few minutes).
Output: work/crsp_clean.csv.gz  (~80 MB)

Filters (equivalent to the legacy shrcd 10/11 + exchcd 1/2/3 screen):
  SecurityType == 'EQTY', SecuritySubType == 'COM', ShareType == 'NS', USIncFlg == 'Y', PrimaryExch in {N, A, Q}.
The CIZ file has one row per distribution event, so a stock-month can appear more than once: we keep one.
In CIZ, MthRet already includes delisting returns -- do NOT merge a delisting file on top.
Optional: python3 00_prep_crsp.py 500000   -> only reads the first 500,000 rows (quick test)."""
import pandas as pd, sys, time, os
from config import CRSP_RAW, CRSP_CLEAN

COLS = ['PERMNO', 'PERMCO', 'YYYYMM', 'PrimaryExch', 'SecurityType', 'SecuritySubType', 'ShareType', 'USIncFlg',
        'SICCD', 'MthRet', 'MthRetx', 'MthCap', 'MthPrc', 'ShrOut', 'MthVol']
KEEP = ['PERMNO', 'PERMCO', 'YYYYMM', 'PrimaryExch', 'SICCD', 'MthRet', 'MthRetx', 'MthCap', 'MthPrc', 'ShrOut', 'MthVol']
STR = {c: 'str' for c in ('PrimaryExch', 'SecurityType', 'SecuritySubType', 'ShareType', 'USIncFlg', 'SICCD')}

nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
t0 = time.time(); parts = []; n = 0
for ch in pd.read_csv(CRSP_RAW, usecols=COLS, dtype=STR, chunksize=200_000, nrows=nrows, low_memory=False):
    n += len(ch)
    m = ((ch.SecurityType == 'EQTY') & (ch.SecuritySubType == 'COM') & (ch.ShareType == 'NS')
         & (ch.USIncFlg == 'Y') & ch.PrimaryExch.isin(['N', 'A', 'Q']))
    parts.append(ch.loc[m, KEEP].drop_duplicates(['PERMNO', 'YYYYMM']))
    print(f'\r{n:,} rows read, {time.time() - t0:.0f}s', end='', flush=True)
d = pd.concat(parts).drop_duplicates(['PERMNO', 'YYYYMM'])
print(f'\nkept {len(d):,} stock-months, {d.PERMNO.nunique():,} stocks, {d.YYYYMM.min()}-{d.YYYYMM.max()}')
d.to_csv(CRSP_CLEAN, index=False, compression={'method': 'gzip', 'compresslevel': 3}, float_format='%.8g')
done = os.path.join(os.path.dirname(CRSP_CLEAN), 'crsp_clean.done')
if nrows is None:
    open(done, 'w').write(f'full run: {len(d)} stock-months\n')       # lets main.py skip this slow step next time
elif os.path.exists(done):
    os.remove(done)                                                   # a test run overwrote the full file
print('written', CRSP_CLEAN)
