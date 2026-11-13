"""Step 2: validate the pipeline (lecture asset-growth deciles; own HML/MOM vs official Fama-French)."""
import numpy as np, pandas as pd, json, os
from lib import *
P = load_panel(); ff = load_ff()
months, RET, ME, NYSE = P['months'], P['RET'], P['ME'], P['NYSE']
T, N = RET.shape

def annual_sort_returns(sig, nq=10, start=196906, end=202506, nyse_bp=True):
    """Sort each June on sig (NYSE breakpoints), hold Jul..Jun. Returns DataFrame of EW/VW decile returns by YYYYMM."""
    rows = []
    for t in range(T - 1):
        ym = months[t]
        if ym % 100 != 6 or ym < start or ym > end: continue
        s = sig[t]; me = ME[t]
        ok = ~np.isnan(s) & ~np.isnan(me) & (me > 0)
        bp_set = ok & NYSE[t] if nyse_bp else ok
        bps = np.quantile(s[bp_set], np.linspace(0, 1, nq + 1)[1:-1])
        q = np.full(N, -1); q[ok] = np.searchsorted(bps, s[ok], side='right')
        for h in range(1, 13):
            tt = t + h
            if tt >= T: break
            r = RET[tt]; w = ME[tt - 1]
            for d in range(nq):
                m = (q == d) & ~np.isnan(r)
                mw = m & ~np.isnan(w)
                rows.append((months[tt], d + 1, r[m].mean(), np.sum(r[mw] * w[mw]) / np.sum(w[mw]), m.sum()))
    df = pd.DataFrame(rows, columns=['YYYYMM', 'dec', 'ew', 'vw', 'n'])
    return df

ag = annual_sort_returns(-(-P['AG']))  # sort on AG itself: decile 1 = low growth
ew = ag.pivot(index='YYYYMM', columns='dec', values='ew'); vw = ag.pivot(index='YYYYMM', columns='dec', values='vw')
res = {'sample': f'{ew.index.min()}-{ew.index.max()}', 'months': len(ew),
       'ew': (ew.mean() * 100).round(2).tolist(), 'vw': (vw.mean() * 100).round(2).tolist(),
       'ew_10m1': round((ew[10] - ew[1]).mean() * 100, 2), 'vw_10m1': round((vw[10] - vw[1]).mean() * 100, 2),
       'ew_10m1_t': round(nw_ols((ew[10] - ew[1]).values, None)[1][0], 2), 'vw_10m1_t': round(nw_ols((vw[10] - vw[1]).values, None)[1][0], 2),
       'avg_n': ag.groupby('dec').n.mean().round(0).tolist()}
res['lecture_ew'] = [1.50, 1.46, 1.37, 1.28, 1.26, 1.23, 1.23, 1.17, 1.11, 0.66]
res['lecture_vw'] = [1.14, 1.14, 1.08, 0.97, 0.98, 0.95, 0.97, 1.00, 0.99, 0.88]
res['corr_ew_with_lecture'] = round(float(np.corrcoef(res['ew'], res['lecture_ew'])[0, 1]), 3)
res['corr_vw_with_lecture'] = round(float(np.corrcoef(res['vw'], res['lecture_vw'])[0, 1]), 3)

# ---- own HML (2x3, FF June) and momentum (2x3 monthly) vs official ----
def two_by_three(sig, t_form, hold, size_t):
    me = ME[size_t]; s = sig[t_form]
    ok = ~np.isnan(s) & ~np.isnan(me) & (me > 0)
    smed = np.median(me[ok & NYSE[size_t]]); b30, b70 = np.quantile(s[ok & NYSE[t_form]], [0.3, 0.7])
    big = me > smed; grp = np.where(s <= b30, 0, np.where(s <= b70, 1, 2))
    out = []
    for tt in hold:
        r = RET[tt]; w = ME[tt - 1]; m0 = ok & ~np.isnan(r) & ~np.isnan(w)
        pr = {}
        for sz in (0, 1):
            for g in (0, 1, 2):
                m = m0 & (big == sz) & (grp == g); pr[(sz, g)] = np.sum(r[m] * w[m]) / np.sum(w[m])
        out.append((months[tt], 0.5 * (pr[(0, 2)] + pr[(1, 2)]) - 0.5 * (pr[(0, 0)] + pr[(1, 0)])))
    return out
hml, umd = [], []
for t in range(T - 1):
    if months[t] % 100 == 6 and months[t] >= 196906:
        hml += two_by_three(P['BM'], t, range(t + 1, min(t + 13, T)), t)
    if months[t] >= 196901:
        umd += two_by_three(P['MOM'], t, [t + 1], t)
hml = pd.Series(dict(hml)); umd = pd.Series(dict(umd))
j = pd.concat([hml.rename('HML_own'), umd.rename('UMD_own'), ff[['HML', 'MOM']]], axis=1).dropna()
res['corr_HML'] = round(j.HML_own.corr(j.HML), 3); res['corr_MOM'] = round(j.UMD_own.corr(j.MOM), 3)
res['mean_HML_own_vs_ff'] = [round(j.HML_own.mean() * 100, 2), round(j.HML.mean() * 100, 2)]
res['mean_MOM_own_vs_ff'] = [round(j.UMD_own.mean() * 100, 2), round(j.MOM.mean() * 100, 2)]
# market: own VW all-stock vs FF MKT+RF
mk = []
for t in range(1, T):
    r = RET[t]; w = ME[t - 1]; m = ~np.isnan(r) & ~np.isnan(w); mk.append((months[t], np.sum(r[m] * w[m]) / np.sum(w[m])))
mk = pd.Series(dict(mk)); jj = pd.concat([mk.rename('own'), (ff.MKT + ff.RF).rename('ff')], axis=1).dropna()
res['corr_MKT'] = round(jj.own.corr(jj.ff), 4)
# CRSP-Compustat link: is every accounting row matched through a link that was valid on its fiscal year-end date?
LINK = os.path.join(RAW_DIR, 'CRSP_Compustat link data.csv')
if os.path.exists(LINK) and os.path.exists(COMP_RAW):
    l = pd.read_csv(LINK, low_memory=False); l['d0'] = pd.to_datetime(l.LINKDT)
    l['d1'] = pd.to_datetime(l.LINKENDDT.replace('E', '2099-12-31'), errors='coerce')
    k = pd.read_csv(COMP_RAW, usecols=['GVKEY', 'LPERMNO', 'datadate'], low_memory=False); k['datadate'] = pd.to_datetime(k.datadate)
    mm = k.reset_index().merge(l[['gvkey', 'LPERMNO', 'd0', 'd1']], left_on=['GVKEY', 'LPERMNO'], right_on=['gvkey', 'LPERMNO'], how='left')
    v = ((mm.d0 <= mm.datadate) & (mm.datadate <= mm.d1)).groupby(mm['index']).any()
    res['link_rows'] = int(len(k)); res['link_valid_share'] = float(v.mean())
json.dump(res, open(f'{OUT}/validation.json', 'w'), indent=1)
ew.to_csv(f'{OUT}/asset_growth_deciles_ew.csv'); vw.to_csv(f'{OUT}/asset_growth_deciles_vw.csv')
print(json.dumps(res, indent=1))
