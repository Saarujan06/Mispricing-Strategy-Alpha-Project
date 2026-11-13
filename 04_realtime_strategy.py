"""Step 4 (MAIN STRATEGY): real-time, out-of-sample mispricing strategy.
- Anomaly k enters only from January of the year after its publication (what an investor could know).
- Weights: (a) equal rank average of published anomalies; (b) Lecture-4 style combination forecast:
  each anomaly's forecast = its past average Fama-MacBeth slope x its demeaned rank (expanding window, data up to t-1),
  combined by simple averaging over published anomalies (Rapach, Strauss & Zhou 2010 logic, applied cross-sectionally).
- Benchmark: full-hindsight SY10 (all 10 anomalies from 1970) to measure the cost of being honest.
- Costs: the course's method (Lecture 3, slides 12 and 22-25; see LAMBDA / HALF_SPREAD below).
"""
import numpy as np, pandas as pd, pickle, json
from scipy.stats import rankdata
from lib import *
P = load_panel(); months = P['months']; RET, ME, NYSE, PRC = P['RET'], P['ME'], P['NYSE'], P['PRC']; T, N = RET.shape
SIGN = {'NSI': -1, 'CEI': -1, 'ACC': -1, 'NOA': -1, 'AG': -1, 'IVA': -1, 'MOM': +1, 'GP': +1, 'ROA': +1, 'OSC': -1}
PUB = {'MOM': 1993, 'NSI': 1995, 'ACC': 1996, 'OSC': 1998, 'NOA': 2004, 'IVA': 2004, 'CEI': 2006, 'ROA': 2006, 'AG': 2008, 'GP': 2013}
REF = {'MOM': 'Jegadeesh & Titman (1993)', 'NSI': 'Loughran & Ritter (1995)', 'ACC': 'Sloan (1996)', 'OSC': 'Dichev (1998)',
       'NOA': 'Hirshleifer et al. (2004)', 'IVA': 'Titman, Wei & Xie (2004)', 'CEI': 'Daniel & Titman (2006)', 'ROA': 'Fama & French (2006)',
       'AG': 'Cooper, Gulen & Schill (2008)', 'GP': 'Novy-Marx (2013)'}
K = list(SIGN)

def universe(t, exmicro=False):
    me = ME[t]; u = ~np.isnan(me) & (me > 0) & (PRC[t] >= 5) & ~np.isnan(RET[t])
    if exmicro: u &= me > np.quantile(me[u & NYSE[t]], 0.2)
    return u

# ---- precompute ranks (all-stock universe, price >= 5) and FM slopes (ex-microcap, EW OLS) ----
start_t = int(np.where(months == 197006)[0][0])
RANK = {k: {} for k in K}; U = {}; UX = {}; slopes = []
for t in range(start_t, T - 1):
    u = universe(t); ux = universe(t, True); U[t] = u; UX[t] = ux
    r1 = RET[t + 1]; row = [months[t + 1]]
    for k in K:
        x = SIGN[k] * P[k][t].astype(float); m = u & ~np.isnan(x); rk = np.full(N, np.nan)
        if m.sum() > 50: rk[m] = (rankdata(x[m]) - 0.5) / m.sum() - 0.5     # demeaned rank in [-0.5, 0.5]
        RANK[k][t] = rk.astype(np.float32)
        mm = ux & ~np.isnan(rk) & ~np.isnan(r1)
        if mm.sum() > 100:
            xx = rk[mm] - rk[mm].mean(); row.append(float(xx @ (r1[mm] - r1[mm].mean()) / (xx @ xx)))
        else: row.append(np.nan)
    slopes.append(row)
SL = pd.DataFrame(slopes, columns=['YYYYMM'] + K).set_index('YYYYMM')   # slope realised in return month YYYYMM
SL.to_csv(f'{OUT}/fm_slopes_by_anomaly.csv')

# ---- trading costs: the method of the course (Lecture 3, slides 12 and 22-25) ----
# Turnover: sum_i |dw_i|, where dw_i = new weight - old weight drifted with market cap (slides 22-23).
# Price impact (slide 24): sum_i lambda_i * sqrt(V0 * |dw_i| / DolVol_i) * |dw_i|, DolVol = the stock's dollar volume in the
#   month, V0 = fund size ($100m in the lecture's tables); lambda = 2.5 / 1.5 / 1.0 (in %) for micro / small / large stocks
#   (slide 25: "for 1% of ADV, price impact is 25, 15 and 10 bps"). Size groups: NYSE 20th and 50th percentiles.
# Spread (slide 12): sum_i |dw_i| * s_i / 2. Our CRSP extract has no bid/ask quotes, so we use one half-spread per size group
#   (the lecture also fills missing quotes with size-group means, slide 25), CALIBRATED so that our replication of the
#   lecture's momentum tables (slides 27 and 30: all stocks and non-micro-caps, EW and VW) reproduces its six spread numbers:
#   micro 2.08%, small 0.68%, large 0.44% (06_extra_checks.py, section A, reports the replication). These are averages over
#   the lecture's full sample, so they are conservative for 1997-2025 (spreads fell sharply after 2001).
LAMBDA = np.array([2.5, 1.5, 1.0]) / 100                     # micro, small, large
HALF_SPREAD = np.array([0.02075, 0.00675, 0.00441])          # micro, small, large (calibrated, see above)
# Alternatives (robustness): 'flat' = 0.72% per dollar traded (Lecture 3 slide 30 momentum 10/20, VW: (0.47 + 0.09) / 0.78);
# 'stock' = our own earlier model (5/15/40 bp half-spreads by NYSE 30/70 size tier, x2 before April 2001, impact
# sigma_daily * sqrt(trade / daily dollar volume)).
FLAT = 0.0072

def size_group(me, nyse):
    """0 = micro (< NYSE 20th pct), 1 = small (20th-50th), 2 = large (> NYSE median)."""
    q20, q50 = np.nanquantile(me[nyse & (me > 0)], [0.2, 0.5])
    return np.where(me < q20, 0, np.where(me < q50, 1, 2))

def published(k, ym, pub_lag=1, pub=None): return ym // 100 >= (pub or PUB)[k] + pub_lag
def score(t, method, pub_lag=1, pub=None):
    ym = months[t + 1]
    if method == 'hindsight': ks = K
    else: ks = [k for k in K if published(k, ym, pub_lag, pub)]
    if not ks: return np.full(N, np.nan), {}
    if method == 'hindsight' or method == 'ew_pub':
        R = np.vstack([RANK[k][t] for k in ks]); n = (~np.isnan(R)).sum(0)
        with np.errstate(all='ignore'): s = np.nanmean(R, 0)
        s[n < min(3, len(ks))] = np.nan; return s, {k: 1 / len(ks) for k in ks}
    if method == 'comb_pub':
        past = SL.loc[SL.index < ym]                                  # slopes realised up to month t
        b = past[ks].mean()                                           # expanding-window average slope
        F = np.vstack([b[k] * np.nan_to_num(RANK[k][t]) for k in ks])  # individual forecasts (missing rank -> 0 = average)
        avail = np.vstack([~np.isnan(RANK[k][t]) for k in ks]).sum(0)
        s = F.mean(0); s[avail < min(3, len(ks))] = np.nan
        return s, (b / b.abs().sum()).to_dict()

def run(method, exmicro=False, buffer=False, aum=100e6, start=199701, spread_mult=1.0, pub_lag=1, min_signals=3, costs='course', exit_q=0.2, rebal=1, pub=None,
        drift='mcap', miss_ret=0.0):
    """pub_lag: anomaly usable from Jan of (publication year + pub_lag); spread_mult scales the spread cost;
    min_signals: number of published anomalies needed before the strategy starts (3 for the main strategy);
    costs: 'course' (the course's method, main), 'flat' (0.72% per dollar traded) or 'stock' (our earlier stock-level model).
    exit_q: buffer exit point (0.2 = the 10%/20% rule); rebal: rebalance every `rebal` months (1 = monthly);
    pub: optional dict of publication years replacing PUB (robustness).
    drift: 'mcap' = weights drift with market capitalisation (the course's rule, main); 'retx' = with the price return
      (robustness: a holding does not grow when the firm issues shares);
    miss_ret: return assumed when a held stock has no return next month (0 = main; -0.30 as a robustness check).
    Turnover turn{L,S} = sum of |weight changes| in the leg (buys + sells), the Lecture 3 definition."""
    rows, W_ = [], []; prev = {}; held = {'L': None, 'S': None}; dec = []
    for t in range(start_t, T - 1):
        ym = months[t + 1]
        if ym < start: continue
        u = UX[t] if exmicro else U[t]; s, wts = score(t, method, pub_lag, pub); ok = u & ~np.isnan(s)
        if (ok & NYSE[t]).sum() < 100 or len(wts) < min_signals: continue
        q10, q20, q80, q90 = np.quantile(s[ok & NYSE[t]], [0.1, exit_q, 1 - exit_q, 0.9])
        hold_only = rebal > 1 and len(rows) % rebal != 0 and held['L'] is not None   # between rebalancing dates: no trades
        r = np.nan_to_num(RET[t + 1], nan=miss_ret); me = np.nan_to_num(ME[t])
        # deciles (for GRS / characteristics)
        bp = np.quantile(s[ok & NYSE[t]], np.linspace(0, 1, 11)[1:-1]); q = np.full(N, -1); q[ok] = np.searchsorted(bp, s[ok], side='right')
        dec.append([ym] + [np.sum(r[q == d] * me[q == d]) / me[q == d].sum() for d in range(10)])
        # legs
        out = {'YYYYMM': ym}
        for leg, enter, stay in (('L', ok & (s >= q90), ok & (s >= q80)), ('S', ok & (s <= q10), ok & (s <= q20))):
            if hold_only:
                mem = held[leg]; w = prev[leg]; dw = np.zeros(N)
            else:
                mem = enter.copy()
                if buffer and held[leg] is not None: mem |= held[leg] & stay
                held[leg] = mem; w = np.where(mem, me, 0.0); w /= w.sum()
                dw = np.abs(w - prev[leg]) if leg in prev else w.copy()                # first month: the initial purchase is charged too
            if costs == 'stock':
                # our earlier stock-level model (robustness)
                p30, p70 = np.quantile(me[ok & NYSE[t]], [0.3, 0.7])
                hs = np.where(me >= p70, 0.0005, np.where(me >= p30, 0.0015, 0.0040)) * (2.0 if ym < 200104 else 1.0) * spread_mult
                adv = np.nan_to_num(P['DV'][t]) * 1e6 / 21.0; sig_d = np.nan_to_num(P['VOL'][t], nan=0.12) / np.sqrt(21)
                with np.errstate(all='ignore'): imp = np.minimum(np.where(adv > 0, sig_d * np.sqrt(dw * aum / adv), 0.0), 0.10)
                c_hs, c_imp = float(np.sum(dw * hs)), float(np.sum(dw * imp))
            elif costs == 'flat':
                c_hs, c_imp = float(dw.sum() * 0.47 / 0.78 / 100 * spread_mult), float(dw.sum() * 0.09 / 0.78 / 100 * np.sqrt(aum / 100e6))
            else:
                # the course's method (main)
                g_ = size_group(np.where(me > 0, me, np.nan), NYSE[t] & (me > 0))
                dv = np.nan_to_num(P['DV'][t]) * 1e6                                   # dollar volume in month t
                tr = dw > 0
                if tr.any():
                    med = np.array([np.median(dv[(g_ == k_) & (dv > 0)]) if ((g_ == k_) & (dv > 0)).any() else 1e9 for k_ in range(3)])
                    dv_ = np.where(dv > 0, dv, med[g_])                                  # missing volume: size-group median
                    c_imp = float(np.sum(LAMBDA[g_[tr]] * np.sqrt(aum * dw[tr] / dv_[tr]) * dw[tr]))
                    c_hs = float(np.sum(dw[tr] * HALF_SPREAD[g_[tr]])) * spread_mult
                else: c_hs = c_imp = 0.0
            rl = float(w @ r)
            # weights drift with market capitalisation (slide 22); if next month's market cap is missing (delisting), with the return
            with np.errstate(all='ignore'): g = np.where(w > 0, ME[t + 1] / np.where(me > 0, me, np.nan), 0.0)
            g = np.where(np.isnan(g), 1 + r, g)
            if drift == 'retx':                                                    # robustness: drift with the price return
                gx = 1 + P['RETX'][t + 1]; g = np.where(w > 0, np.where(np.isnan(gx), g, gx), 0.0)
            wd = w * g; prev[leg] = wd / wd.sum() if wd.sum() > 0 else w
            out[f'miss{leg}'] = int((mem & np.isnan(RET[t + 1])).sum())          # held, next-month return missing -> treated as 0 (cash)
            out.update({f'r{leg}': rl, f'turn{leg}': float(dw.sum()), f'cost{leg}': c_hs + c_imp, f'n{leg}': int(mem.sum()),
                        f'hs{leg}': c_hs, f'imp{leg}': c_imp})
        rows.append(out); W_.append({'YYYYMM': ym, **wts})
    df = pd.DataFrame(rows).set_index('YYYYMM')
    df['ls'] = df.rL - df.rS; df['ls_net'] = df.ls - df.costL - df.costS
    dcols = ['YYYYMM'] + [f'D{i + 1}' for i in range(10)]
    return df, pd.DataFrame(dec, columns=dcols).set_index('YYYYMM'), pd.DataFrame(W_).set_index('YYYYMM')

def grs(port, ff, cols):
    """Gibbons-Ross-Shanken test that all alphas are zero (Lecture 2)."""
    from scipy.stats import f as fdist
    j = pd.concat([port, ff[cols + ['RF']]], axis=1, join='inner').dropna()
    Y = j[port.columns].values - j[['RF']].values; F = j[cols].values
    X = np.column_stack([np.ones(len(j)), F]); Tn, Nn, L = len(j), Y.shape[1], len(cols)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]; E = Y - X @ B; a = B[0]
    S = E.T @ E / (Tn - L - 1); mu = F.mean(0); Om = np.atleast_2d(np.cov(F.T, ddof=1))
    stat = (Tn / Nn) * ((Tn - Nn - L) / (Tn - L - 1)) * (a @ np.linalg.solve(S, a)) / (1 + mu @ np.linalg.solve(Om, mu))
    return float(stat), float(1 - fdist.cdf(stat, Nn, Tn - Nn - L))

if __name__ == '__main__':
    out = {}
    for method in ('hindsight', 'ew_pub', 'comb_pub'):
        for tag, kw in (('base', {}), ('buf', {'buffer': True}), ('exmicro_buf', {'exmicro': True, 'buffer': True})):
            print('running', method, tag); out[(method, tag)] = run(method, **kw)
    pickle.dump(out, open(f'{WORK}/realtime.pkl', 'wb'))
    ff, MODELS, QSRC = model_set(P)   # CAPM, FF3, FF3+MOM, FF5, FF5+MOM, q4, q5 (Lecture 2)
    def alpha(y, cols):
        j = pd.concat([y.rename('y'), ff[cols]], axis=1, join='inner').dropna(); b, t, e, r2 = nw_ols(j.y.values, j[cols].values)
        return b[0] * 100, t[0]
    WINDOWS = {'1997-2025': (199701, 202512), '2005-2025': (200501, 202512), '2014-2025': (201401, 202512)}
    rows = []
    for (m, tag), (df, dec, wts) in out.items():
        for wn, (a, b) in WINDOWS.items():
            x = df.loc[a:b]
            for kind, y in (('gross', x.ls), ('net', x.ls_net)):
                r = {'method': m, 'version': tag, 'window': wn, 'returns': kind, 'mean_%pm': y.mean() * 100,
                     't_mean': nw_ols(y.values, None)[1][0], 'sharpe': y.mean() / y.std() * 12 ** .5}
                for nm, cols in MODELS.items():
                    r[f'alpha_{nm}_%pm'], r[f't_{nm}'] = alpha(y, cols)
                r['turnover_%pm (both legs, buys+sells)'] = (x.turnL.mean() + x.turnS.mean()) * 100
                r['cost_%pa'] = 12 * (x.costL + x.costS).mean() * 100
                rows.append(r)
    summ = pd.DataFrame(rows); summ.to_csv(f'{OUT}/realtime_summary.csv', index=False, float_format='%.3f')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    show = summ[summ['returns'] == 'net'][['method', 'version', 'window', 'mean_%pm', 'alpha_FF5+MOM_%pm', 't_FF5+MOM', 'sharpe', 'turnover_%pm (both legs, buys+sells)', 'cost_%pa']]
    print('\nNET-OF-COST results (monthly %, Newey-West t):'); print(show.round(2).to_string(index=False))
    # ---- main strategy details: real-time, equal weights, 10/20 buffer ----
    df, dec, wts = out[('ew_pub', 'buf')]; x = df.loc[199701:202512]; rf = ff.RF.reindex(x.index); mkt = ff.MKT.reindex(x.index)
    x.assign(MKT=mkt, RF=rf).to_csv(f'{OUT}/main_strategy_monthly.csv', float_format='%.6f')
    dec.to_csv(f'{OUT}/main_strategy_deciles_vw.csv', float_format='%.6f'); wts.to_csv(f'{OUT}/main_strategy_anomaly_weights.csv', float_format='%.3f')
    net = x.ls_net; wealth = (1 + net).cumprod()
    print('\nMAIN STRATEGY (real-time, equal weights, 10/20 buffer), 1997-2025, net of costs:')
    print(f'  mean {net.mean()*100:.2f}%/month, vol {net.std()*12**.5*100:.1f}% p.a., Sharpe {net.mean()/net.std()*12**.5:.2f}, '
          f'max drawdown {(wealth/wealth.cummax()-1).min()*100:.0f}%, correlation with market {net.corr(mkt):.2f}')
    aL, tL = alpha(x.rL - rf, MODELS['FF5+MOM']); aS, tS = alpha(rf - x.rS, MODELS['FF5+MOM'])
    print(f'  FF5+MOM alpha (monthly %): long leg {aL:.2f} (t={tL:.1f}), short leg {aS:.2f} (t={tS:.1f})')
    print(f'  avg stocks long/short {x.nL.mean():.0f}/{x.nS.mean():.0f}; turnover long/short {x.turnL.mean()*100:.0f}%/{x.turnS.mean()*100:.0f}% per month; '
          f'costs: spread {(x.hsL+x.hsS).mean()*1200:.2f}% + impact {(x.impL+x.impS).mean()*1200:.2f}% p.a.')
    print('  decile VW mean returns (%/month):', (dec.loc[199701:202512].mean() * 100).round(2).tolist())
    print('  GRS test vs FF5+MOM across 10 deciles: F = %.2f, p = %.4f' % grs(dec.loc[199701:202512], ff, MODELS['FF5+MOM']))
    # lecture-style tables (cnst = alpha in % per year; t-stats below; adjusted R^2)
    costs_m = {'Turnover (long + short, % per month)': f"{100 * (x.turnL + x.turnS).mean():.0f}%",
               'Spread (% per month)': f"{100 * (x.hsL + x.hsS).mean():.2f}%",
               'Price impact ($100m, % per month)': f"{100 * (x.impL + x.impS).mean():.2f}%"}
    for kind, y in (('gross', x.ls), ('net', x.ls_net)):
        tab = lecture_table(y, ff, MODELS, cost_rows=costs_m if kind == 'gross' else None, label=f'{x.index[0] // 100}:{x.index[0] % 100:02d} - {x.index[-1] // 100}:{x.index[-1] % 100:02d} ({kind})')
        tab.to_csv(f'{OUT}/lecture_table_main_{kind}.csv', index=False)
    print('\n  Lecture-style table (gross), also saved to output/lecture_table_main_gross.csv:')
    print(lecture_table(x.ls, ff, MODELS, cost_rows=costs_m).to_string(index=False))
    for lam in (0.0, 0.5, 1.0):
        p_ = mkt + lam * net; print(f'  market + {lam:.1f} x strategy: Sharpe {p_.mean()/p_.std()*12**.5:.2f}')
