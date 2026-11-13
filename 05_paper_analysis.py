"""Step 5: every additional table and figure input for the paper -> output/paper_results.json (+ CSVs).
Run after 04_realtime_strategy.py (it reuses work/realtime.pkl and the functions of step 4)."""
import importlib, json, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from lib import *
rt = importlib.import_module('04_realtime_strategy')           # precomputes ranks and slopes (~20 s)
P, months, RET, ME, NYSE, PRC, K, PUB, SIGN = rt.P, rt.months, rt.RET, rt.ME, rt.NYSE, rt.PRC, rt.K, rt.PUB, rt.SIGN
T, N = RET.shape
ff, MODELS, QSRC = model_set(P)          # all Lecture-2 models: CAPM, FF3, FF3+MOM, FF5, FF5+MOM, q4, q5
F6 = MODELS['FF5+MOM']; QKEY = 'q5' if 'q5' in MODELS else ('q4' if 'q4' in MODELS else None)
OUTR = {}
def reg(y, cols, X=None):
    X = ff if X is None else X
    j = pd.concat([y.rename('y'), X[cols]], axis=1, join='inner').dropna(); b, t, e, r2 = nw_ols(j.y.values, j[cols].values)
    return {'a': b[0] * 100, 't': t[0], 'b': dict(zip(cols, [float(v) for v in b[1:]])), 'r2': r2}
def sm(y):  # mean (monthly %) and NW t
    y = y.dropna(); return {'mean': y.mean() * 100, 't': nw_ols(y.values, None)[1][0], 'sr': y.mean() / y.std() * 12 ** .5}
O = pickle.load(open(f'{WORK}/realtime.pkl', 'rb'))
main, maindec, mainw = O[('ew_pub', 'buf')]; RT0, RT1 = 199701, 202512

# ---------------- q-factor diagnostics ----------------
OUTR['q_source'] = QSRC; OUTR['models'] = list(MODELS)
if QKEY: OUTR['qproxy_corr'] = {'ME_SMB': ff.R_ME.corr(ff.SMB), 'IA_CMA': ff.R_IA.corr(ff.CMA), 'ROE_RMW': ff.R_ROE.corr(ff.RMW)}

# ---------------- 1. descriptives ----------------
desc = []
for y in range(1970, 2026, 5):
    ts = [t for t in rt.U if months[t] // 100 == y and months[t] % 100 == 6]
    if not ts: continue
    t = ts[0]; u = rt.U[t]
    desc.append({'year': y, 'n': int(u.sum()), 'nyse_share': float(NYSE[t][u].mean()), 'tot_me_bn': float(np.nansum(ME[t][u]) / 1e3),
                 'med_me_m': float(np.nanmedian(ME[t][u])), **{f'cov_{k}': float(np.mean(~np.isnan(rt.RANK[k][t][u]))) for k in K}})
OUTR['desc'] = desc
# anomaly summary: coverage over 1970-2025 & average cross-sectional Spearman correlation of ranks
cov = {k: float(np.mean([np.mean(~np.isnan(rt.RANK[k][t][rt.U[t]])) for t in rt.U])) for k in K}
C = np.zeros((len(K), len(K))); cnt = 0
for t in list(rt.U)[::6]:
    if months[t] < 197506: continue
    M = np.vstack([rt.RANK[k][t] for k in K]).T; m = rt.U[t] & ~np.isnan(M).any(1)
    if m.sum() < 200: continue
    C += np.corrcoef(M[m].T); cnt += 1
OUTR['coverage'] = cov; OUTR['rank_corr'] = (C / cnt).round(2).tolist()
Cm = C / cnt; _within = lambda idx: float(np.mean([Cm[i, j] for i in idx for j in idx if j < i]))   # K: first six = management, last four = performance
OUTR['rank_corr_groups'] = {'mgmt': _within(range(6)), 'perf': _within(range(6, 10)), 'cross': float(Cm[6:, :6].mean())}
_un = [int(rt.U[t].sum()) for t in rt.U if RT0 <= months[t + 1] <= RT1]
OUTR['universe_n'] = {'min': min(_un), 'max': max(_un), 'mean': float(np.mean(_un))}

# ---------------- 2. single anomalies: pre vs post publication (VW deciles, NYSE bp, price>=5) ----------------
single = {}
for k in K:
    rows = []
    for t in sorted(rt.U):
        s = rt.RANK[k][t]; u = rt.U[t]; ok = u & ~np.isnan(s)
        if (ok & NYSE[t]).sum() < 100: continue
        b1, b9 = np.quantile(s[ok & NYSE[t]], [0.1, 0.9]); r = np.nan_to_num(RET[t + 1]); me = np.nan_to_num(ME[t])
        hi, lo = ok & (s >= b9), ok & (s <= b1)
        rows.append((months[t + 1], np.sum(r[hi] * me[hi]) / me[hi].sum() - np.sum(r[lo] * me[lo]) / me[lo].sum()))
    ls = pd.Series(dict(rows)); cut = (PUB[k] + 1) * 100 + 1
    single[k] = {'pub': PUB[k], 'pre': {**sm(ls[ls.index < cut]), 'ff6': reg(ls[ls.index < cut], F6)},
                 'post': {**sm(ls[ls.index >= cut]), 'ff6': reg(ls[ls.index >= cut], F6)}, 'n_pre': int((ls.index < cut).sum()), 'n_post': int((ls.index >= cut).sum())}
OUTR['single'] = single

# ---------------- 3. real-time score: decile characteristics and returns ----------------
dchar = {d: {'n': [], 'me': [], 'bm': [], 'score': [], 'ew': [], 'vw': []} for d in range(10)}
dec_ew, dec_vw, scores = [], [], {}
for t in sorted(rt.U):
    ym = months[t + 1]
    if ym < RT0: continue
    s, w = rt.score(t, 'ew_pub'); u = rt.U[t]; ok = u & ~np.isnan(s); scores[t] = s
    bp = np.quantile(s[ok & NYSE[t]], np.linspace(0, 1, 11)[1:-1]); q = np.full(N, -1); q[ok] = np.searchsorted(bp, s[ok], side='right')
    r = np.nan_to_num(RET[t + 1]); me = np.nan_to_num(ME[t]); bm = P['BE'][t] / P['MEDEC'][t]
    rowe, rowv = [ym], [ym]
    for d in range(10):
        m = q == d; rowe.append(r[m].mean()); rowv.append(np.sum(r[m] * me[m]) / me[m].sum())
        dchar[d]['n'].append(m.sum()); dchar[d]['me'].append(np.nanmedian(ME[t][m])); dchar[d]['bm'].append(np.nanmedian(bm[m])); dchar[d]['score'].append(np.nanmean(s[m]))
    dec_ew.append(rowe); dec_vw.append(rowv)
cols = ['YYYYMM'] + [f'D{i}' for i in range(1, 11)]
DEW = pd.DataFrame(dec_ew, columns=cols).set_index('YYYYMM'); DVW = pd.DataFrame(dec_vw, columns=cols).set_index('YYYYMM')
rf = ff.RF
deciles = []
for d in range(10):
    c = f'D{d + 1}'; ex = DVW[c] - rf.reindex(DVW.index)
    deciles.append({'dec': d + 1, 'n': float(np.mean(dchar[d]['n'])), 'me_med': float(np.mean(dchar[d]['me'])), 'bm_med': float(np.nanmean(dchar[d]['bm'])),
                    'score': float(np.mean(dchar[d]['score'])), 'vw': DVW[c].mean() * 100, 'ew': DEW[c].mean() * 100,
                    'ff6_vw': reg(ex, F6)['a'], 't_ff6_vw': reg(ex, F6)['t'], 'capm_vw': reg(ex, ['MKT'])['a']})
OUTR['deciles'] = deciles
for nm, D in (('vw', DVW), ('ew', DEW)):
    ls = D['D10'] - D['D1']; OUTR[f'dec_ls_{nm}'] = {**sm(ls), **{m: reg(ls, c) for m, c in MODELS.items()}, 'q': reg(ls, MODELS[QKEY]) if QKEY else None}
OUTR['grs'] = {'FF5+MOM': rt.grs(DVW, ff, F6), 'FF5': rt.grs(DVW, ff, F6[:5]), 'FF3': rt.grs(DVW, ff, F6[:3])}

# ---------------- 4. size x score (NYSE size terciles x score quintiles, independent, VW) ----------------
ss = {}
for t, s in scores.items():
    u = rt.U[t]; ok = u & ~np.isnan(s); me = np.nan_to_num(ME[t]); r = np.nan_to_num(RET[t + 1])
    sb = np.quantile(me[ok & NYSE[t]], [1 / 3, 2 / 3]); qb = np.quantile(s[ok & NYSE[t]], [.2, .4, .6, .8])
    gs = np.searchsorted(sb, me); gq = np.searchsorted(qb, s)
    for a in range(3):
        for b in range(5):
            m = ok & (gs == a) & (gq == b); ss.setdefault((a, b), []).append((months[t + 1], np.sum(r[m] * me[m]) / me[m].sum() if m.any() else np.nan))
SS = {k: pd.Series(dict(v)) for k, v in ss.items()}
size_tab = []
for a, nm in enumerate(['Small', 'Mid', 'Large']):
    row = {'size': nm, 'q': [SS[(a, b)].mean() * 100 for b in range(5)]}
    ls = SS[(a, 4)] - SS[(a, 0)]; row['ls'] = sm(ls); row['ff6'] = reg(ls, F6); size_tab.append(row)
OUTR['size_sort'] = size_tab

# ---------------- 5. Fama-MacBeth (stocks above NYSE 20th pct, price >= 5) ----------------
fm = []
for t, s in scores.items():
    ux = rt.UX[t]; y = RET[t + 1]; lme = np.log(ME[t]); lbm = np.log(P['BE'][t] / P['MEDEC'][t]); mom = P['MOM'][t]
    def cs(xs):
        X = np.column_stack(xs); m = ux & ~np.isnan(X).any(1) & ~np.isnan(y)
        Xm = X[m].copy()
        for j in range(Xm.shape[1]):
            if j > 0: Xm[:, j] = (Xm[:, j] - Xm[:, j].mean()) / Xm[:, j].std()   # controls standardised; score kept in rank units
        return np.linalg.lstsq(np.column_stack([np.ones(m.sum()), Xm]), y[m], rcond=None)[0][1:], int(m.sum())
    sc = s.copy()
    b1, n1 = cs([sc]); b2, _ = cs([sc, lme, lbm]); b3, _ = cs([sc, lme, lbm, mom.astype(float)])
    fm.append([months[t + 1], n1] + list(b1) + list(b2) + list(b3))
FM = pd.DataFrame(fm, columns=['YYYYMM', 'n', 's1', 's2', 'me2', 'bm2', 's3', 'me3', 'bm3', 'mom3']).set_index('YYYYMM')
OUTR['fm'] = {c: sm(FM[c]) for c in FM.columns if c != 'n'}; OUTR['fm_n'] = float(FM.n.mean())

# ---------------- 6. main strategy: gross/net, legs, alphas by window ----------------
WIN = {'1997-2025': (199701, 202512), '1997-2004': (199701, 200412), '2005-2025': (200501, 202512), '2014-2025': (201401, 202512)}
def block(df):
    o = {}
    for wn, (a, b) in WIN.items():
        x = df.loc[a:b]; r = {}
        for kind, y in (('gross', x.ls), ('net', x.ls_net)):
            r[kind] = {**sm(y), **{m: reg(y, c) for m, c in MODELS.items()}, 'q': reg(y, MODELS[QKEY]) if QKEY else None}
        r['turn'] = (x.turnL.mean() + x.turnS.mean()) * 100; r['cost'] = 12 * (x.costL + x.costS).mean() * 100   # turnover: % per month, buys+sells (Lecture 3)
        r['cost_spread'] = 12 * (x.hsL + x.hsS).mean() * 100; r['cost_impact'] = 12 * (x.impL + x.impS).mean() * 100
        r['turnL'] = x.turnL.mean() * 100; r['turnS'] = x.turnS.mean() * 100; r['nL'] = x.nL.mean(); r['nS'] = x.nS.mean()
        o[wn] = r
    return o
OUTR['versions'] = {f'{m}|{tag}': block(O[(m, tag)][0]) for (m, tag) in O}
x = main.loc[RT0:RT1]; rfx = rf.reindex(x.index); mkt = ff.MKT.reindex(x.index)
OUTR['legs'] = {'long_ff6': reg(x.rL - rfx, F6), 'short_ff6': reg(rfx - x.rS, F6), 'long_capm': reg(x.rL - rfx, ['MKT']),
                'short_capm': reg(rfx - x.rS, ['MKT']), 'long_excess': sm(x.rL - rfx), 'short_minus_rf': sm(rfx - x.rS)}
OUTR['main_betas'] = reg(x.ls_net, F6)['b']
# break-even proportional cost multiplier: net = gross - m*cost -> m* = mean(gross)/mean(cost)
OUTR['breakeven_mult'] = float(x.ls.mean() / (x.costL + x.costS).mean())

# short-selling (borrowing) fees, not in the trading costs: a fee f per year on the short leg lowers the return by f/12
fees = {}
for f_ in (0.0025, 0.005, 0.01, 0.02, 0.03):
    rr = reg(x.ls_net - f_ / 12, F6); fees[str(f_)] = {'a': rr['a'], 't': rr['t']}
a6 = reg(x.ls_net, F6)
OUTR['fees'] = fees
OUTR['fee_breakeven_pa'] = float(12 * a6['a'])                         # fee (in % p.a.) that sets the FF5+MOM alpha to zero
OUTR['fee_t196_pa'] = float(12 * (a6['a'] - 1.96 * a6['a'] / a6['t']))  # fee (in % p.a.) at which t falls to 1.96

# ---------------- 7. robustness runs ----------------
rob = {}
for nm, kw in {'cost_flat': {'costs': 'flat'}, 'spread_half': {'spread_mult': 0.5}, 'spread_x2': {'spread_mult': 2.0}, 'cost_stock': {'costs': 'stock'},
               'aum_1bn': {'aum': 1e9}, 'aum_10bn': {'aum': 1e10},
               'publag_minus1': {'pub_lag': -1}, 'publag_plus2': {'pub_lag': 3}, 'exmicro': {'exmicro': True}}.items():
    df, _, _ = rt.run('ew_pub', buffer=True, **kw); rob[nm] = block(df)['1997-2025']; rob[nm]['test'] = block(df)['2005-2025']
    print('robustness', nm, round(rob[nm]['net']['FF5+MOM']['a'], 2), round(rob[nm]['net']['FF5+MOM']['t'], 2))
OUTR['robust'] = {k: {'net_a6': v['net']['FF5+MOM']['a'], 'net_t6': v['net']['FF5+MOM']['t'], 'gross_a6': v['gross']['FF5+MOM']['a'], 'gross_t6': v['gross']['FF5+MOM']['t'],
                      'cost': v['cost'], 'cost_spread': v['cost_spread'], 'cost_impact': v['cost_impact'], 'turn': v['turn'], 'test_net_a6': v['test']['net']['FF5+MOM']['a'], 'test_net_t6': v['test']['net']['FF5+MOM']['t']} for k, v in rob.items()}
OUTR['robust']['ew_deciles_gross'] = {'a6': OUTR['dec_ls_ew']['FF5+MOM']['a'], 't6': OUTR['dec_ls_ew']['FF5+MOM']['t']}

# ---------------- 8. fund-manager view: overlay on market, crises ----------------
net = x.ls_net; mk = mkt + rfx
ov = {}
for lam in (0.0, 0.25, 0.5, 1.0):
    p = mkt + lam * net; ov[str(lam)] = {'sr': p.mean() / p.std() * 12 ** .5, 'ret': p.mean() * 1200, 'vol': p.std() * 12 ** .5 * 100,
                                         'maxdd': float((((1 + p + rfx).cumprod()) / ((1 + p + rfx).cumprod()).cummax() - 1).min() * 100)}
OUTR['overlay'] = ov
OUTR['corr_mkt'] = float(net.corr(mkt))
wealth = (1 + net).cumprod(); OUTR['maxdd'] = float((wealth / wealth.cummax() - 1).min() * 100); OUTR['vol'] = float(net.std() * 12 ** .5 * 100)
dd = wealth / wealth.cummax() - 1; trough = dd.idxmin(); peak = wealth.loc[:trough].idxmax()
rec = wealth.loc[trough:][wealth.loc[trough:] >= wealth.loc[peak]]
OUTR['dd_dates'] = {'peak': int(peak), 'trough': int(trough), 'recovered': int(rec.index[0]) if len(rec) else None}
OUTR['capm_beta'] = float(reg(net, ['MKT'])['b']['MKT'])
OUTR['missing_ret'] = {'held_missing': int(x[['missL', 'missS']].sum().sum()) if 'missL' in x else None, 'held_total': int((x.nL + x.nS).sum())}
OUTR['ann_ret'] = float((wealth.iloc[-1] ** (12 / len(net)) - 1) * 100)
crises = {'Dot-com bust (2000-03 to 2002-09)': (200003, 200209), 'Global financial crisis (2007-11 to 2009-02)': (200711, 200902),
          'Momentum crash (2009-03 to 2009-12)': (200903, 200912), 'Covid crash (2020-02 to 2020-03)': (202002, 202003), 'Rate shock (2022)': (202201, 202212)}
OUTR['crises'] = {k: {'strategy': float(((1 + net.loc[a:b]).prod() - 1) * 100), 'market': float(((1 + mk.loc[a:b]).prod() - 1) * 100)} for k, (a, b) in crises.items()}
OUTR['n_months'] = len(net)
# hindsight vs real time (look-ahead bias)
OUTR['lookahead'] = {k: OUTR['versions'][k]['1997-2025']['net']['FF5+MOM'] for k in ('hindsight|buf', 'ew_pub|buf', 'comb_pub|buf')}
# combination weights snapshot (what the learned weights do)
cw = O[('comb_pub', 'buf')][2]; OUTR['comb_weights_avg'] = cw.mean().round(3).to_dict()
# in-sample (hindsight, 1970-2025) for context
ins = json.load(open(f'{OUT}/insample_sy10.json'))
OUTR['insample'] = {'vw_full': {k: ins['all']['vw']['full'][k] for k in ('mean', 't', 'ff6', 't_ff6', 'ff5', 't_ff5', 'capm')}, 'grs_ff6': ins['all']['vw']['grs_ff6']}
OUTR['validation'] = json.load(open(f'{OUT}/validation.json'))
json.dump(OUTR, open(f'{OUT}/paper_results.json', 'w'), indent=1, default=float)
pd.concat([main.loc[RT0:RT1], DVW.add_prefix('VW_'), DEW.add_prefix('EW_')], axis=1).to_csv(f'{OUT}/paper_main_series.csv', float_format='%.6f')
FM.to_csv(f'{OUT}/fama_macbeth_monthly.csv', float_format='%.6f')
print('done; main net FF5+MOM alpha', OUTR['versions']['ew_pub|buf']['1997-2025']['net']['FF5+MOM'])
