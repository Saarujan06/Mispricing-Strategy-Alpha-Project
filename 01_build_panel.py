"""Step 1: build the firm-level (PERMCO) monthly panel as wide matrices (months x firms) + all signals.
Needs ~5 GB RAM; takes about a minute. Output: work/panel.npz
Signals: momentum, volatility, net share issuance, composite equity issuance (CRSP);
book-to-market, intangible-adjusted B/M, gross profitability, asset growth, accruals, NOA, investment-to-assets,
ROA, Ohlson O-score (Compustat).
Timing convention: signals are known at the end of formation month f and predict the return in f+1.
Accounting data for fiscal years ending in calendar year t-1 are used from end-June t to end-May t+1
(i.e., for returns July t ... June t+1), following Fama-French / the lecture's 6-month lag.
"""
import numpy as np, pandas as pd, os
from config import CRSP_CLEAN, COMP_RAW, PANEL

c = pd.read_csv(CRSP_CLEAN)
c['ME'] = c.MthCap / 1000.0                     # $ millions
c['PRC'] = c.MthPrc.abs()
c['DV'] = c.MthVol * c.PRC / 1e6                # $ millions traded
# firm level (PERMCO): ME summed over share classes, returns of the largest class
c = c.sort_values(['PERMCO', 'YYYYMM', 'ME'], ascending=[True, True, False])
agg = c.groupby(['PERMCO', 'YYYYMM']).agg(ME=('ME', 'sum'), DV=('DV', 'sum')).reset_index()
top = c.drop_duplicates(['PERMCO', 'YYYYMM'])[['PERMCO', 'YYYYMM', 'PERMNO', 'MthRet', 'MthRetx', 'PRC', 'PrimaryExch', 'SICCD']]
f = agg.merge(top, on=['PERMCO', 'YYYYMM'])
del c
months = np.sort(f.YYYYMM.unique()); firms = np.sort(f.PERMCO.unique())
mi = pd.Index(months); fi = pd.Index(firms)
T, N = len(months), len(firms)
r_idx = mi.get_indexer(f.YYYYMM); c_idx = fi.get_indexer(f.PERMCO)
def wide(col, dtype=np.float64):
    a = np.full((T, N), np.nan, dtype=dtype); a[r_idx, c_idx] = f[col].values; return a
RET, RETX, ME, PRC, DV = wide('MthRet'), wide('MthRetx'), wide('ME'), wide('PRC'), wide('DV')
NYSE = np.zeros((T, N), bool); NYSE[r_idx, c_idx] = (f.PrimaryExch.values == 'N')
SIC = wide('SICCD')

# ---- return-based signals (known at end of month t) ----
lr = np.log1p(RET)
cs = np.vstack([np.zeros((1, N)), np.nancumsum(np.nan_to_num(lr), axis=0)])
cnt = np.vstack([np.zeros((1, N)), np.cumsum(~np.isnan(lr), axis=0)])
MOM = np.full((T, N), np.nan); VOL = np.full((T, N), np.nan); NSI = np.full((T, N), np.nan)
for t in range(T):
    # 12-1 momentum: months t-11..t-1 (skip month t)
    if t >= 11:
        s = cs[t] - cs[t - 11]; n = cnt[t] - cnt[t - 11]
        MOM[t] = np.where(n == 11, np.expm1(s), np.nan)
    if t >= 35:
        blk = RET[t - 35:t + 1]; n = (~np.isnan(blk)).sum(0)
        with np.errstate(all='ignore'): sd = np.nanstd(blk, axis=0, ddof=1)
        VOL[t] = np.where(n >= 24, sd, np.nan)
lrx = np.log1p(RETX); csx = np.vstack([np.zeros((1, N)), np.nancumsum(np.nan_to_num(lrx), axis=0)])
cntx = np.vstack([np.zeros((1, N)), np.cumsum(~np.isnan(lrx), axis=0)])
for t in range(12, T):
    n = cntx[t + 1] - cntx[t + 1 - 12]
    NSI[t] = np.where(n == 12, np.log(ME[t] / ME[t - 12]) - (csx[t + 1] - csx[t + 1 - 12]), np.nan)
NSI[~np.isfinite(NSI)] = np.nan
# composite equity issuance (Daniel & Titman 2006): 5-year log ME growth minus 5-year log total return
CEI = np.full((T, N), np.nan)
for t in range(60, T):
    n = cnt[t + 1] - cnt[t + 1 - 60]
    with np.errstate(all='ignore'):
        CEI[t] = np.where(n == 60, np.log(ME[t] / ME[t - 60]) - (cs[t + 1] - cs[t + 1 - 60]), np.nan)
CEI[~np.isfinite(CEI)] = np.nan

for _a in ('MOM','VOL','NSI','CEI'): globals()[_a] = globals()[_a].astype(np.float32)
del lr, lrx, cs, cnt, csx, cntx
# ---- Compustat ----
k = pd.read_csv(COMP_RAW, low_memory=False)
k['datadate'] = pd.to_datetime(k.datadate)
k = k.sort_values(['GVKEY', 'datadate']).drop_duplicates(['GVKEY', 'fyear'], keep='last')
g = k.drop_duplicates(['GVKEY', 'datadate']).copy()   # firm-level accounting (one row per gvkey-year)
g = g.sort_values(['GVKEY', 'fyear'])
ps = g.pstkrv.fillna(g.pstkl).fillna(g.pstk).fillna(0)
se = g.seq.fillna(g.ceq + g.pstk.fillna(0)).fillna(g['at'] - g['lt'])
g['BE'] = se + g.txditc.fillna(0) - ps
g.loc[g.BE <= 0, 'BE'] = np.nan
# intangible capital (Peters & Taylor 2017 perpetual inventory; Eisfeldt, Kim & Papanikolaou 2022)
xrd = g.xrd.fillna(0).clip(lower=0)
sga = np.where(g.xsga.notna() & (g.xsga > xrd), g.xsga - xrd, g.xsga.fillna(0)).clip(min=0)
g['_rd'] = xrd.values; g['_sga'] = 0.3 * sga
KRD, KORG, AGE = [], [], []
for gv, grp in g.groupby('GVKEY', sort=False):
    krd = korg = None; prev_fy = None; age = 0
    for fy, rd, og in zip(grp.fyear.values, grp._rd.values, grp._sga.values):
        if krd is None or (prev_fy is not None and fy - prev_fy > 1):
            krd = rd / (0.10 + 0.15); korg = og / (0.10 + 0.20); age = 0   # steady-state initialisation
        else:
            krd = 0.85 * krd + rd; korg = 0.80 * korg + og; age += 1
        KRD.append(krd); KORG.append(korg); AGE.append(age); prev_fy = fy
g['KRD'] = KRD; g['KORG'] = KORG; g['KAGE'] = AGE
g['BEINT'] = g.BE + g.KRD + g.KORG - g.gdwl.fillna(0)
g.loc[g.BEINT <= 0, 'BEINT'] = np.nan
g['GP'] = (g.revt - g.cogs) / g['at']
g['OP'] = (g.revt - g.cogs - g.xsga.fillna(0) - g.xint.fillna(0)) / g.BE
lag = g.groupby('GVKEY').shift(1)
cons = (g.fyear - lag.fyear) == 1
g['AG'] = np.where(cons & (lag['at'] > 0), g['at'] / lag['at'] - 1, np.nan)
dp = (g.oibdp - g.oiadp).fillna(g.dpc)
dca = (g.act - lag.act) - (g.che - lag.che); dcl = (g.lct - lag.lct) - (g.dlc.fillna(0) - lag.dlc.fillna(0))
g['ACC'] = np.where(cons, (dca - dcl - dp) / ((g['at'] + lag['at']) / 2), np.nan)
g['ROE'] = np.where(cons, g.ib / lag.BE, np.nan)
# --- additional Stambaugh-Yuan (2017) anomalies ---
opa = g['at'] - g.che.fillna(0)
opl = g['at'] - g.dlc.fillna(0) - g.dltt.fillna(0) - g.mib.fillna(0) - g.pstk.fillna(0) - g.ceq
g['NOA'] = np.where(cons & (lag['at'] > 0), (opa - opl) / lag['at'], np.nan)            # Hirshleifer et al. (2004)
g['IVA'] = np.where(cons & (lag['at'] > 0), ((g.ppent - lag.ppent) + (g.invt.fillna(0) - lag.invt.fillna(0))) / lag['at'], np.nan)  # Lyandres et al.
g['ROA'] = np.where(cons & (lag['at'] > 0), g.ib / lag['at'], np.nan)
ffo = g.ib + dp.fillna(0)
ni_l = lag.ni
g['OSC'] = (-1.32 - 0.407 * np.log(g['at'].where(g['at'] > 0)) + 6.03 * g['lt'] / g['at'] - 1.43 * (g.act - g.lct) / g['at']
            + 0.0757 * g.lct / g.act.where(g.act > 0) - 1.72 * (g['lt'] > g['at']) - 2.37 * g.ni / g['at'] - 1.83 * ffo / g['lt'].where(g['lt'] > 0)
            + 0.285 * ((g.ni < 0) & (ni_l < 0)) - 0.521 * (g.ni - ni_l) / (g.ni.abs() + ni_l.abs()).where(lambda x: x > 0))   # Ohlson (1980)
g.loc[~cons, 'OSC'] = np.nan
for v in ['NOA', 'IVA', 'ROA', 'OSC']:
    g[v] = g[v].replace([np.inf, -np.inf], np.nan)
g.loc[g['at'] <= 0, ['GP', 'AG', 'ACC', 'NOA', 'IVA', 'ROA', 'OSC']] = np.nan
g['cal_year'] = g.datadate.dt.year
# map to PERMCO: if several gvkeys per permco-year, keep largest assets
g = g.rename(columns={'LPERMCO': 'PERMCO'})
g = g.sort_values(['PERMCO', 'cal_year', 'at']).drop_duplicates(['PERMCO', 'cal_year'], keep='last')
acc_vars = ['BE', 'BEINT', 'GP', 'OP', 'AG', 'ACC', 'ROE', 'at', 'KAGE', 'NOA', 'IVA', 'ROA', 'OSC']
# fill wide matrices: accounting from cal_year y-1 applies to formation months June y .. May y+1
#   (formation month f = end-June y -> return July y)
ACC = {v: np.full((T, N), np.nan, dtype=np.float32) for v in acc_vars}
years = months // 100; mon = months % 100
form_year = np.where(mon >= 6, years, years - 1)        # 'June-year' of each formation month
gi = g[g.PERMCO.isin(firms)]
for y in np.unique(form_year):
    rows = np.where(form_year == y)[0]
    sub = gi[gi.cal_year == y - 1]
    cols = fi.get_indexer(sub.PERMCO)
    for v in acc_vars:
        ACC[v][np.ix_(rows, cols)] = np.broadcast_to(sub[v].values, (len(rows), len(cols)))
# December ME of year y-1 (firm level) for B/M
MEDEC = np.full((T, N), np.nan, dtype=np.float32)
for y in np.unique(form_year):
    rows = np.where(form_year == y)[0]
    d = np.where(months == (y - 1) * 100 + 12)[0]
    if len(d): MEDEC[rows] = ME[d[0]]
BM = ACC['BE'] / MEDEC; BMINT = ACC['BEINT'] / MEDEC
np.savez_compressed(PANEL, months=months, firms=firms, RET=RET, RETX=RETX, ME=ME, PRC=PRC, DV=DV,
                    NYSE=NYSE, SIC=SIC, MOM=MOM, VOL=VOL, NSI=NSI, BM=BM, BMINT=BMINT, GP=ACC['GP'], OP=ACC['OP'],
                    AG=ACC['AG'], ACC=ACC['ACC'], ROE=ACC['ROE'], AT=ACC['at'], KAGE=ACC['KAGE'], BE=ACC['BE'], MEDEC=MEDEC,
                    NOA=ACC['NOA'], IVA=ACC['IVA'], ROA=ACC['ROA'], OSC=ACC['OSC'], CEI=CEI)
print('T,N', T, N)
for v, a in [('MOM', MOM), ('VOL', VOL), ('NSI', NSI), ('BM', BM), ('BMINT', BMINT), ('GP', ACC['GP']), ('AG', ACC['AG']), ('ACC', ACC['ACC'])]:
    ok = ~np.isnan(ME)
    cov = [np.mean(~np.isnan(a[t][ok[t]])) for t in [24, 150, 300, 450, 600, T - 1] if t < T]
    print(v, np.round(cov, 2))
