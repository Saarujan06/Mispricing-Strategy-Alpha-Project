"""Shared helpers: data loading, Fama-French factors, Newey-West OLS, cross-sectional z-scores."""
import numpy as np, pandas as pd, os, io
from config import *

def load_panel():
    z = np.load(PANEL)
    return {k: z[k] for k in z.files}

def _read_french_csv(path):
    """Parse an official Ken French CSV (monthly block only). Returns DataFrame in decimals indexed by YYYYMM."""
    lines = open(path, encoding='latin-1').read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(',') and any(c.isalpha() for c in l))
    rows = []
    for l in lines[start + 1:]:
        p = [x.strip() for x in l.split(',')]
        if len(p[0]) == 6 and p[0].isdigit(): rows.append(p)
        elif rows: break                                   # end of monthly block (annual block follows)
    cols = [c.strip() for c in lines[start].split(',')][1:]
    df = pd.DataFrame([r[1:] for r in rows], index=[int(r[0]) for r in rows], columns=cols).astype(float) / 100
    df.index.name = 'YYYYMM'
    return df

def load_ff():
    """FF5 + MOM (momentum, Ken French's UMD) + RF, monthly, decimals, indexed by YYYYMM.
    Uses the official Ken French files if you put them in factors/ (recommended for the final paper):
      F-F_Research_Data_5_Factors_2x3.csv  and  F-F_Momentum_Factor.csv
    otherwise falls back to factors/us_ff5_mom.csv.gz (a mirror of the same library, data to Aug 2026)."""
    f5, fm = os.path.join(FACTORS, 'F-F_Research_Data_5_Factors_2x3.csv'), os.path.join(FACTORS, 'F-F_Momentum_Factor.csv')
    if os.path.exists(f5) and os.path.exists(fm):
        a = _read_french_csv(f5).rename(columns={'Mkt-RF': 'MKT'}); b = _read_french_csv(fm)
        a['MOM'] = b.iloc[:, 0]
        return a[['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']]
    f = pd.read_csv(os.path.join(FACTORS, 'us_ff5_mom.csv.gz'))
    f['YYYYMM'] = pd.to_datetime(f.iloc[:, 0]).dt.strftime('%Y%m').astype(int)
    f = f.set_index('YYYYMM').iloc[:, 1:] / 100.0
    return f.rename(columns={'MKT_RF': 'MKT', 'Mom': 'MOM'})

def nw_ols(y, X, lags=None):
    """OLS with Newey-West HAC t-stats (constant added). Returns coef, t, resid, r2.
    Default lag = floor(4 (T/100)^(2/9)) (Newey & West 1994)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)]) if X is not None and np.size(X) else np.ones((len(y), 1))
    ok = ~np.isnan(y) & ~np.isnan(X).any(1); y, X = y[ok], X[ok]; T = len(y)
    if lags is None: lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); u = X * e[:, None]; S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1); G = u[l:].T @ u[:-l]; S += w * (G + G.T)
    V = XtXi @ S @ XtXi; t = b / np.sqrt(np.diag(V))
    return b, t, e, 1 - e.var() / y.var()

def winsor_z(x, mask, lo=0.01, hi=0.99):
    """Cross-sectional winsorise (1/99%) + z-score among mask; NaN outside mask or if missing."""
    z = np.full(x.shape, np.nan); v = x[mask & ~np.isnan(x)]
    if len(v) < 20: return z
    a, b = np.quantile(v, [lo, hi]); xx = np.clip(x, a, b)
    m = mask & ~np.isnan(x); mu = xx[m].mean(); sd = xx[m].std()
    z[m] = (xx[m] - mu) / sd if sd > 0 else 0.0
    return z


# ---------------------------------------------------------------------------------------------
# Factor models from Lecture 2: CAPM, FF3, FF3+momentum (Carhart), FF5, FF5+momentum, q4, q5
# ---------------------------------------------------------------------------------------------
def load_q():
    """Official Hou-Xue-Zhang q-factors. Put the monthly file from https://global-q.org/factors.html
    (e.g. q5_factors_monthly_2025.csv) in factors/. Returns DataFrame (decimals, index YYYYMM) or None."""
    import glob
    fs = sorted(glob.glob(os.path.join(FACTORS, 'q5_factors_monthly*.csv')))
    if not fs: return None
    d = pd.read_csv(fs[-1]); d.columns = [c.strip().lower() for c in d.columns]
    if 'year' in d.columns and 'month' in d.columns:
        idx = d['year'].astype(int) * 100 + d['month'].astype(int)
    else:
        idx = pd.to_datetime(d.iloc[:, 0].astype(str)).dt.strftime('%Y%m').astype(int)
    out = pd.DataFrame({c.upper(): d[c].values / 100 for c in ('r_mkt', 'r_me', 'r_ia', 'r_roe', 'r_eg') if c in d.columns}, index=idx.values)
    out.index.name = 'YYYYMM'
    return out

def build_q_proxy(P):
    """Fallback if the official q-factors are missing: size, investment (asset growth) and annual-ROE factors
    from independent 2x3x3 NYSE-breakpoint sorts each June (value-weighted). No expected-growth factor."""
    months, RET, ME, NYSE = P['months'], P['RET'], P['ME'], P['NYSE']; T = len(months); rows = []
    for t in range(T - 1):
        if months[t] % 100 != 6 or months[t] < 197006: continue
        me = ME[t]; ia = P['AG'][t]; roe = P['ROE'][t]; ok = ~np.isnan(me) & ~np.isnan(ia) & ~np.isnan(roe) & (me > 0)
        sm_ = np.median(me[ok & NYSE[t]]); i3 = np.quantile(ia[ok & NYSE[t]], [.3, .7]); r3 = np.quantile(roe[ok & NYSE[t]], [.3, .7])
        big = me > sm_; gi = np.searchsorted(i3, ia); gr = np.searchsorted(r3, roe)
        for h in range(1, 13):
            tt = t + h
            if tt >= T: break
            r = RET[tt]; w = ME[tt - 1]; m0 = ok & ~np.isnan(r) & ~np.isnan(w); pr = {}
            for a in (0, 1):
                for b in range(3):
                    for c in range(3):
                        m = m0 & (big == a) & (gi == b) & (gr == c); pr[(a, b, c)] = np.sum(r[m] * w[m]) / np.sum(w[m]) if m.any() else np.nan
            mean = lambda keys: np.nanmean([pr[k] for k in keys]); KK = list(pr)
            rows.append((months[tt], mean([k for k in KK if k[0] == 0]) - mean([k for k in KK if k[0] == 1]),
                         mean([k for k in KK if k[1] == 0]) - mean([k for k in KK if k[1] == 2]), mean([k for k in KK if k[2] == 2]) - mean([k for k in KK if k[2] == 0])))
    return pd.DataFrame(rows, columns=['YYYYMM', 'R_ME', 'R_IA', 'R_ROE']).set_index('YYYYMM')

def model_set(P=None, verbose=True):
    """Returns (X, MODELS, q_source). X: all factor returns by YYYYMM; MODELS: name -> list of columns.
    q4/q5 use the official file when present; otherwise q4 uses the in-house proxy (needs P) and q5 is skipped."""
    ff = load_ff(); q = load_q(); src = 'official'
    if q is None:
        src = 'proxy'
        if P is None: q = None
        else: q = build_q_proxy(P)
        if verbose: print('NOTE: official q-factors not found in factors/ -> q4 uses the in-house proxy, q5 skipped.')
    X = ff.copy()
    MODELS = {'CAPM': ['MKT'], 'FF3': ['MKT', 'SMB', 'HML'], 'FF3+MOM': ['MKT', 'SMB', 'HML', 'MOM'],
              'FF5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA'], 'FF5+MOM': ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']}
    if q is not None:
        X = X.join(q[[c for c in ('R_MKT', 'R_ME', 'R_IA', 'R_ROE', 'R_EG') if c in q.columns]], how='left')
        qm = 'R_MKT' if 'R_MKT' in q.columns else 'MKT'       # q models use their own market factor (as in the lecture)
        MODELS['q4'] = [qm, 'R_ME', 'R_IA', 'R_ROE']
        if 'R_EG' in q.columns: MODELS['q5'] = [qm, 'R_ME', 'R_IA', 'R_ROE', 'R_EG']
    return X, MODELS, src

MODEL_ORDER = ['CAPM', 'FF3', 'FF3+MOM', 'FF5', 'FF5+MOM', 'q4', 'q5']


def lecture_table(y, X, MODELS, cost_rows=None, label=''):
    """Regression table laid out like the lecture: one row per model with annualised alpha (cnst, % per year)
    and factor loadings, t-statistics (Newey-West) in the row below, adjusted R^2 in the last column.
    The first row ('-') is the raw average return. cost_rows: dict of extra rows (e.g. turnover, spread, impact)."""
    COLS = ['cnst', 'MktRf', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'MktRf (q)', 'ME', 'IA', 'ROE', 'EG', 'AR2']
    NAME = {'MKT': 'MktRf', 'SMB': 'SMB', 'HML': 'HML', 'RMW': 'RMW', 'CMA': 'CMA', 'MOM': 'MOM',
            'R_MKT': 'MktRf (q)', 'R_ME': 'ME', 'R_IA': 'IA', 'R_ROE': 'ROE', 'R_EG': 'EG'}
    rows = []
    def add(name, coef, tstat, ar2):
        r1 = {'Model': name, **{c: '' for c in COLS}}; r2 = {'Model': '', **{c: '' for c in COLS}}
        for k, v in coef.items(): r1[k] = f'{v:.2f}'
        for k, v in tstat.items(): r2[k] = f'({v:.2f})'
        r1['AR2'] = f'{ar2 * 100:.0f}%'; rows.extend([r1, r2])
    b, t, e, r2 = nw_ols(y.dropna().values, None); add('-', {'cnst': 12 * 100 * b[0]}, {'cnst': t[0]}, 0.0)
    for nm in [m for m in MODEL_ORDER if m in MODELS]:
        cols = MODELS[nm]; j = pd.concat([y.rename('y'), X[cols]], axis=1, join='inner').dropna()
        b, t, e, r2 = nw_ols(j.y.values, j[cols].values); n, k = len(j), len(cols)
        ar2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        coef = {'cnst': 12 * 100 * b[0], **{NAME[c]: b[i + 1] for i, c in enumerate(cols)}}
        tst = {'cnst': t[0], **{NAME[c]: t[i + 1] for i, c in enumerate(cols)}}
        add(nm, coef, tst, ar2)
    out = pd.DataFrame(rows, columns=['Model'] + COLS)
    if cost_rows:
        out = pd.concat([out, pd.DataFrame([{'Model': '', **{c: '' for c in COLS}}] +
                                           [{'Model': k, 'cnst': v, **{c: '' for c in COLS[1:]}} for k, v in cost_rows.items()])], ignore_index=True)
    out = out.rename(columns={'Model': label or 'Model'})
    return out
