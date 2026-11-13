"""Step 3 (in-sample benchmark, uses hindsight): Stambaugh-Yuan-style mispricing score with all anomalies from 1970.
Usage: python3 03_insample_mispricing.py [variant]   variant in {sy10, sy10_bmint}"""
import numpy as np, pandas as pd, json, sys, pickle
from scipy.stats import rankdata
from lib import *
P = load_panel(); ff = load_ff(); months = P['months']; RET, ME, NYSE, PRC = P['RET'], P['ME'], P['NYSE'], P['PRC']; T, N = RET.shape
SY10 = {'NSI': -1, 'CEI': -1, 'ACC': -1, 'NOA': -1, 'AG': -1, 'IVA': -1, 'MOM': +1, 'GP': +1, 'ROA': +1, 'OSC': -1}
VARIANTS = {'sy10': SY10, 'sy10_bmint': {**SY10, 'BMINT': +1}}
variant = sys.argv[1] if len(sys.argv) > 1 else 'sy10'; SIG = VARIANTS[variant]
MIN_SIG = 3      # a stock needs at least three anomalies, as in the main strategy

def score_at(t, univ):
    R = []
    for k, sg in SIG.items():
        x = sg * P[k][t].astype(float); m = univ & ~np.isnan(x)
        r = np.full(N, np.nan)
        if m.sum() > 50: r[m] = (rankdata(x[m]) - 0.5) / m.sum()
        R.append(r)
    R = np.vstack(R); n = (~np.isnan(R)).sum(0)
    with np.errstate(all='ignore'): s = np.nanmean(R, 0)
    s[n < MIN_SIG] = np.nan
    return s

def universe(t, exmicro=False):
    me = ME[t]; u = ~np.isnan(me) & (me > 0) & (PRC[t] >= 5) & ~np.isnan(RET[t])
    if exmicro: u &= me > np.quantile(me[u & NYSE[t]], 0.2)
    return u

def run_sorts(exmicro=False, start=197007, nq=10, buffer=False):
    """monthly decile sorts on score; returns dict of DataFrames (ew, vw, n) and turnover for legs"""
    ew, vw, nn, turn = [], [], [], []
    prev = {}; held = {'L': None, 'S': None}; chars = []
    for t in range(T - 1):
        if months[t + 1] < start: continue
        u = universe(t, exmicro); s = score_at(t, u); ok = u & ~np.isnan(s)
        bp = np.quantile(s[ok & NYSE[t]], np.linspace(0, 1, nq + 1)[1:-1])
        q = np.full(N, -1); q[ok] = np.searchsorted(bp, s[ok], side='right')
        r = np.nan_to_num(RET[t + 1], nan=0.0); me = ME[t]
        rowe, rowv, rown = [], [], []
        for d in range(nq):
            m = q == d; rowe.append(r[m].mean()); rowv.append(np.sum(r[m] * me[m]) / me[m].sum()); rown.append(m.sum())
        # legs with optional 10/20 buffer (Lecture 3)
        legw = {}
        for leg, (enter, stay) in {'L': (q == nq - 1, ok & (s >= np.quantile(s[ok & NYSE[t]], 0.8))),
                                   'S': (q == 0, ok & (s <= np.quantile(s[ok & NYSE[t]], 0.2)))}.items():
            mem = enter.copy()
            if buffer and held[leg] is not None: mem |= (held[leg] & stay)
            held[leg] = mem
            w = np.where(mem, np.nan_to_num(me), 0.0); w /= w.sum(); legw[leg] = w
        for leg in ('L', 'S'):
            w = legw[leg]
            tr = 0.5 * np.abs(w - prev[leg]).sum() if leg in prev else np.nan
            rl = float(w @ r); prev[leg] = w * (1 + r) / (1 + rl)
            turn.append((months[t + 1], leg, tr, rl, (w > 0).sum()))
        ew.append([months[t + 1]] + rowe); vw.append([months[t + 1]] + rowv); nn.append([months[t + 1]] + rown)
        if months[t] % 100 == 6:
            for d in range(nq):
                m = q == d
                chars.append((months[t], d + 1, m.sum(), np.nanmedian(me[m]), np.nanmedian((P['BE'][t] / P['MEDEC'][t])[m]), np.nanmean(s[m])))
    cols = ['YYYYMM'] + [f'D{d + 1}' for d in range(nq)]
    out = {k: pd.DataFrame(v, columns=cols).set_index('YYYYMM') for k, v in (('ew', ew), ('vw', vw), ('n', nn))}
    tdf = pd.DataFrame(turn, columns=['YYYYMM', 'leg', 'turn', 'ret', 'nhold'])
    out['legs'] = tdf.pivot(index='YYYYMM', columns='leg', values=['turn', 'ret', 'nhold'])
    out['chars'] = pd.DataFrame(chars, columns=['YYYYMM', 'dec', 'n', 'me_med', 'bm_med', 'score'])
    return out

FF3 = ['MKT', 'SMB', 'HML']; FF5 = FF3 + ['RMW', 'CMA']; FF6 = FF5 + ['MOM']
def alphas(y):
    o = {'mean': y.mean() * 100, 't': nw_ols(y.values, None)[1][0], 'sr': y.mean() / y.std() * 12 ** .5}
    for nm, cols in (('capm', ['MKT']), ('ff3', FF3), ('ff5', FF5), ('ff6', FF6)):
        j = pd.concat([y.rename('y'), ff[cols]], axis=1, join='inner').dropna()
        b, t, e, r2 = nw_ols(j.y.values, j[cols].values); o[nm] = b[0] * 100; o['t_' + nm] = t[0]
        if nm == 'ff6': o['b6'] = dict(zip(cols, np.round(b[1:], 2))); o['ir6'] = b[0] / e.std() * 12 ** .5
    return o

def grs(port, cols):
    j = pd.concat([port, ff[cols]], axis=1, join='inner').dropna(); Y = j[port.columns].values - j[['RF']].values if False else j[port.columns].values
    X = np.column_stack([np.ones(len(j)), j[cols].values]); Tn, Nn, L = len(j), Y.shape[1], len(cols)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]; E = Y - X @ B; a = B[0]
    S = E.T @ E / (Tn - L - 1); F = j[cols].values; mu = F.mean(0); Om = np.cov(F.T, ddof=1).reshape(L, L)
    stat = (Tn / Nn) * ((Tn - Nn - L) / (Tn - L - 1)) * (a @ np.linalg.solve(S, a)) / (1 + mu @ np.linalg.solve(Om, mu))
    from scipy.stats import f as fdist
    return float(stat), float(1 - fdist.cdf(stat, Nn, Tn - Nn - L))

if __name__ == '__main__':
    res = {'variant': variant, 'signals': list(SIG)}
    for tag, kw in (('all', {}), ('exmicro', {'exmicro': True}), ('all_buf', {'buffer': True}), ('exmicro_buf', {'exmicro': True, 'buffer': True})):
        o = run_sorts(**kw); pickle.dump(o, open(f'{WORK}/mp_{variant}_{tag}.pkl', 'wb'))
        r = {}
        for wt in ('ew', 'vw'):
            df = o[wt]; ls = df['D10'] - df['D1']
            r[wt] = {'dec_mean': (df.mean() * 100).round(3).tolist()}
            for wn, (a, b) in {'full': (197007, 202512), 'pre05': (197007, 200412), 'post05': (200501, 202512), 'post14': (201401, 202512)}.items():
                r[wt][wn] = alphas(ls.loc[a:b])
            ex = df.sub(ff.RF.reindex(df.index), axis=0)
            r[wt]['grs_ff6'] = grs(ex, FF6); r[wt]['grs_ff5'] = grs(ex, FF5)
        lg = o['legs']; vwls = lg['ret']['L'] - lg['ret']['S']
        r['legs'] = {'turn_L': float(lg['turn']['L'].mean() * 1200), 'turn_S': float(lg['turn']['S'].mean() * 1200),
                     'n_L': float(lg['nhold']['L'].mean()), 'n_S': float(lg['nhold']['S'].mean()), 'vw_ls': alphas(vwls.loc[197007:202512]),
                     'vw_ls_post05': alphas(vwls.loc[200501:202512])}
        # long / short leg alphas vs FF5+MOM (excess over RF)
        rf = ff.RF.reindex(lg.index)
        r['legs']['long_ff6'] = alphas((lg['ret']['L'] - rf).dropna()); r['legs']['short_ff6'] = alphas((rf - lg['ret']['S']).dropna())
        res[tag] = r
        print(tag, 'VW 10-1 full: mean %.2f t %.2f | FF5+MOM a %.2f t %.2f | post05 FF5+MOM a %.2f t %.2f | EW FF5+MOM a %.2f t %.2f | turnL %.0f%% turnS %.0f%%' % (
            r['vw']['full']['mean'], r['vw']['full']['t'], r['vw']['full']['ff6'], r['vw']['full']['t_ff6'], r['vw']['post05']['ff6'], r['vw']['post05']['t_ff6'],
            r['ew']['full']['ff6'], r['ew']['full']['t_ff6'], r['legs']['turn_L'], r['legs']['turn_S']))
    json.dump(res, open(f'{OUT}/insample_{variant}.json', 'w'), indent=1, default=float)
