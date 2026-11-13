"""Step 6: extra checks that back up statements we make in the report and the presentation.
Run after 04_realtime_strategy.py. Writes output/extra_checks.json and a readable output/extra_checks.txt.

A. Trading costs. The main strategy uses the course method (see 04_realtime_strategy.py, LAMBDA and HALF_SPREAD).
   Here we replicate the lecture's momentum tables with the course's method (turnover and price impact exactly as on
   slides 22-24; spreads by size group), re-derive the size-group half-spreads used in step 4, and show the main
   strategy under alternative cost assumptions.

B. Alternative strategies built with exactly the same rules as the main strategy (real time, 10/20 rule, costs):
   momentum only, asset growth only (Lecture 1), share issuance only (FIN idea, Lecture 2), the two Stambaugh-Yuan
   clusters separately, and all ten anomalies (= main strategy).

C. Volatility scaling of the main strategy (hold less after volatile months), compared at equal average risk.

D. Design choices: buffer width, rebalancing frequency and publication dates, all reported whatever the outcome,
   plus an example of an 'improvement' that only works with hindsight.
"""
import importlib, json, numpy as np, pandas as pd
from lib import *
rt = importlib.import_module('04_realtime_strategy')            # precomputes ranks (~20 s)
O = __import__('pickle').load(open(f'{WORK}/realtime.pkl', 'rb'))
X, MODELS, _ = model_set(rt.P, verbose=False)
A, B = 199701, 202512
RES, TXT = {}, []
def say(s=''): print(s); TXT.append(s)
def alphas(y):
    out = {}
    for m in MODEL_ORDER:
        if m not in MODELS: continue
        j = pd.concat([y.rename('y'), X[MODELS[m]]], axis=1, join='inner').dropna()
        b, t, _, _ = nw_ols(j.y.values, j[MODELS[m]].values); out[m] = (round(100 * b[0], 3), round(t[0], 2))
    return out
def fmt(al, keys=('CAPM', 'FF5+MOM', 'q5')): return '  '.join(f'{k} {al[k][0]:.2f} ({al[k][1]:.1f})' for k in keys if k in al)

main = O[('ew_pub', 'buf')][0].loc[A:B]
cpd_main = (main.costL + main.costS).sum() / (main.turnL + main.turnS).sum()   # cost per dollar traded (main)

# ---------------- A. costs: replicate the lecture's cost tables with the course's method ----------------
say('A. TRADING COSTS: REPLICATION OF THE LECTURE 3 TABLES (slides 26, 27, 30) WITH THE COURSE METHOD (slides 12, 22-25)')
P_ = rt.P; M_ = rt.months; ME_, RET_, NY_, DV_ = P_['ME'], P_['RET'], P_['NYSE'], P_['DV']
def course_replica(sig, ew=False, buf=False, exmicro=False, annual=False, start=196912):
    """Decile 1 and 10 of `sig` (NYSE breakpoints, all stocks), as in the lecture. Returns monthly averages (% per month) of
    turnover, price impact (slide 24, $100m) and traded dollars by size group (micro, small, large) for the spread calibration."""
    prev, held = {}, {}; acc = {k: [] for k in ('turn', 'imp', 'tr0', 'tr1', 'tr2')}
    for t in range(len(M_) - 1):
        if M_[t] < start: continue
        me = ME_[t]; s_ = sig[t]; ok = ~np.isnan(s_) & (me > 0) & ~np.isnan(RET_[t])
        if (ok & NY_[t]).sum() < 50: continue
        grp = rt.size_group(np.where(me > 0, me, np.nan), NY_[t] & (me > 0))
        if exmicro: ok &= grp > 0
        q10, q20, q80, q90 = np.quantile(s_[ok & NY_[t]], [.1, .2, .8, .9]); rebalance = (not annual) or M_[t] % 100 == 6 or not prev
        row = {k: 0.0 for k in acc}
        for leg, enter, stay in (('L', ok & (s_ >= q90), ok & (s_ >= q80)), ('S', ok & (s_ <= q10), ok & (s_ <= q20))):
            mem = ((enter | (held.get(leg, np.zeros_like(ok)) & stay)) if buf else enter) if rebalance else held[leg] & ok
            held[leg] = mem; w = np.where(mem, 1.0 if ew else np.nan_to_num(me), 0.0); w = w / w.sum()
            if leg in prev:
                dw = np.abs(w - prev[leg]); dv = np.nan_to_num(DV_[t]) * 1e6
                with np.errstate(all='ignore'): imp = np.where(dv > 0, rt.LAMBDA[grp] * np.sqrt(1e8 * dw / dv), 0.0)
                row['turn'] += dw.sum(); row['imp'] += float(np.sum(imp * dw))
                for k_ in range(3): row[f'tr{k_}'] += float(dw[grp == k_].sum())
            with np.errstate(all='ignore'): g = np.where(w > 0, ME_[t + 1] / me, 0.0)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0); wd = w * g; prev[leg] = wd / wd.sum() if wd.sum() > 0 else w
        if len(prev) == 2 and row['turn'] > 0:
            for k in acc: acc[k].append(row[k])
    return {k: 100 * np.mean(v) for k, v in acc.items()}
TARGETS = [  # (label, kwargs, lecture: turnover, impact, spread) -- value-weighted (vw) and equal-weighted (ew) 10-1
    ('Momentum, VW', {'sig': 'MOM'}, (142, 0.20, 0.88)), ('Momentum, EW', {'sig': 'MOM', 'ew': True}, (121, 0.45, 1.91)),
    ('Momentum 10/20, all, VW', {'sig': 'MOM', 'buf': True}, (78, 0.09, 0.47)), ('Momentum 10/20, all, EW', {'sig': 'MOM', 'buf': True, 'ew': True}, (82, 0.28, 1.30)),
    ('Momentum 10/20, non-micro, VW', {'sig': 'MOM', 'buf': True, 'exmicro': True}, (82, 0.09, 0.39)),
    ('Momentum 10/20, non-micro, EW', {'sig': 'MOM', 'buf': True, 'exmicro': True, 'ew': True}, (86, 0.15, 0.57))]
rep = []
for lab, kw, lec in TARGETS:
    kw = dict(kw); sig = P_[kw.pop('sig')]; rep.append((lab, course_replica(sig, **kw), lec))
from scipy.optimize import nnls
Amat = np.array([[r['tr0'], r['tr1'], r['tr2']] for _, r, _ in rep]) / 100; bvec = np.array([lec[2] for _, _, lec in rep]) / 100
hs_fit, _ = nnls(Amat, bvec)
say(f"  Half-spreads that reproduce the lecture's spread costs (micro, small, large): {', '.join(f'{100*h:.2f}%' for h in hs_fit)}"
    f"   [used in 04: {', '.join(f'{100*h:.2f}%' for h in rt.HALF_SPREAD)}]")
say(f"  {'Lecture strategy (10-1)':<32s} {'turnover ours/lecture':>22s} {'impact ours/lecture':>20s} {'spread ours/lecture':>20s}")
RES['replication'] = []
for (lab, r, lec), sp in zip(rep, 100 * Amat @ rt.HALF_SPREAD):
    say(f"  {lab:<32s} {r['turn']:>12.0f} / {lec[0]:<8d} {r['imp']:>10.2f} / {lec[1]:<8.2f} {sp:>10.2f} / {lec[2]:.2f}")
    RES['replication'].append({'label': lab, 'turn': r['turn'], 'turn_lec': lec[0], 'imp': r['imp'], 'imp_lec': lec[1], 'spread': float(sp), 'spread_lec': lec[2]})
say(f"  Main strategy: turnover long {100*main.turnL.mean():.0f}%, short {100*main.turnS.mean():.0f}% per month; cost per dollar traded "
    f"{100*(main.costL+main.costS).sum()/(main.turnL+main.turnS).sum():.2f}%")
RES['costs'] = {}
for lab, kw in (('Course method (MAIN)', {}), ('Spreads x 0.5', {'spread_mult': 0.5}), ('Spreads x 2', {'spread_mult': 2.0}),
                ('Flat 0.72% per dollar traded', {'costs': 'flat'}), ('Our earlier stock-level model', {'costs': 'stock'})):
    df = main if not kw else rt.run('ew_pub', buffer=True, **kw)[0].loc[A:B]
    cpd = 100 * (df.costL + df.costS).sum() / (df.turnL + df.turnS).sum(); al = alphas(df.ls_net)
    RES['costs'][lab] = {'cost_per_dollar_%': cpd, 'cost_%pa': 1200 * (df.costL + df.costS).mean(), 'alphas_net': al}
    say(f'  {lab:<34s} ({cpd:.2f}% per $; {1200*(df.costL+df.costS).mean():.2f}% p.a.)  ' + fmt(al, ('CAPM', 'FF3', 'FF5', 'FF5+MOM', 'q4', 'q5')))

# ---------------- B. alternative strategies, same rules ----------------
say('\nB. ALTERNATIVE STRATEGIES, SAME RULES AS THE MAIN STRATEGY (real time, 10/20 rule, Lecture 3 costs), 1997-2025')
SETS = {'Momentum only': ['MOM'], 'Asset growth only (Lecture 1, from 2009)': ['AG'],
        'Share issuance only (NSI + CEI, FIN idea)': ['NSI', 'CEI'],
        'SY management cluster (NSI, CEI, ACC, NOA, AG, IVA)': ['NSI', 'CEI', 'ACC', 'NOA', 'AG', 'IVA'],
        'SY performance cluster (MOM, OSC, ROA, GP)': ['MOM', 'OSC', 'ROA', 'GP'], 'All ten anomalies (main strategy)': list(rt.SIGN)}
K0 = rt.K; RES['alternatives'] = {}
for lab, ks in SETS.items():
    rt.K = ks
    ms = 3 if len(ks) == len(K0) else min(len(ks), 2)          # main strategy needs 3 published anomalies; subsets 2 (or 1)
    df = rt.run('ew_pub', buffer=True, min_signals=ms)[0].loc[A:B]
    g, n = alphas(df.ls), alphas(df.ls_net)
    RES['alternatives'][lab] = {'start': int(df.index[0]), 'gross': g, 'net': n, 'turnover_%pm': 100 * (df.turnL + df.turnS).mean()}
    say(f'  {lab:<52s} from {df.index[0]}  gross: {fmt(g)}   net: {fmt(n)}')
rt.K = K0

# ---------------- C. volatility scaling ----------------
say('\nC. VOLATILITY SCALING OF THE MAIN STRATEGY (position = 12% / volatility of the previous 6 months; same average risk)')
full = O[('ew_pub', 'buf')][0]
for lab, col in (('gross', 'ls'), ('net', 'ls_net')):
    x = full[col]; lev = 0.12 / (x.rolling(6).std().shift(1) * np.sqrt(12)); y0 = x.loc[A:B]; m = lev.loc[A:B].notna()
    y1 = (x * lev).loc[A:B][m]; c = y0[m].std() / y1.std(); y1 = y1 * c; L = (lev.loc[A:B][m] * c)
    if col == 'ls_net':                                        # extra trading from changing the position size
        y1 = y1 - cpd_main * 2 * L.diff().abs().fillna(0)
    r = {'sharpe_orig': y0.mean() / y0.std() * 12 ** .5, 'sharpe_scaled': y1.mean() / y1.std() * 12 ** .5,
         'alpha_orig': alphas(y0), 'alpha_scaled': alphas(y1), 'lev_mean': L.mean(), 'lev_max': L.max(),
         'worst_orig_%': 100 * y0.min(), 'worst_scaled_%': 100 * y1.min(), 'worst_scaled_month': int(y1.idxmin())}
    RES[f'volscale_{lab}'] = r
    say(f"  {lab}: Sharpe {r['sharpe_orig']:.2f} -> {r['sharpe_scaled']:.2f}; FF5+MOM alpha {r['alpha_orig']['FF5+MOM'][0]:.2f} ({r['alpha_orig']['FF5+MOM'][1]:.1f})"
        f" -> {r['alpha_scaled']['FF5+MOM'][0]:.2f} ({r['alpha_scaled']['FF5+MOM'][1]:.1f}); leverage mean {r['lev_mean']:.2f}, max {r['lev_max']:.1f};"
        f" worst month {r['worst_orig_%']:.1f}% -> {r['worst_scaled_%']:.1f}% ({r['worst_scaled_month']})")

# ---------------- D. design choices: every variant we tried, reported whatever the outcome ----------------
say('\nD. DESIGN CHOICES (all variants tried; main = 10/20 rule, monthly, publication years as in Table 1), net of Lecture 3 costs')
PUBSY = dict(rt.PUB); PUBSY.update({'NSI': 1991, 'OSC': 1980})   # earliest studies cited by Stambaugh-Yuan (2017)
RES['design'] = {}
for lab, kw in (('Main: 10%/20% rule, monthly', {}), ('Buffer 10%/15%', {'exit_q': 0.15}), ('Buffer 10%/30%', {'exit_q': 0.30}),
                ('Buffer 10%/40%', {'exit_q': 0.40}), ('Quarterly rebalancing', {'rebal': 3}),
                ('Earliest publication years cited by SY (2017)', {'pub': PUBSY}),
                ('Weights drift with the price return', {'drift': 'retx'}),
                ('Missing returns of held stocks set to -30%', {'miss_ret': -0.30}),
                ('Weights learned from past returns (not equal)', {'method': 'comb_pub'})):
    kw = dict(kw); meth = kw.pop('method', 'ew_pub')
    df = rt.run(meth, buffer=True, **kw)[0].loc[A:B]
    g, n, nl = alphas(df.ls), alphas(df.ls_net), alphas(df.ls_net.loc[200501:])
    RES['design'][lab] = {'turnover_%pm': 100 * (df.turnL + df.turnS).mean(), 'gross': g, 'net': n, 'net_2005': nl}
    say(f"  {lab:<46s} turnover {100*(df.turnL+df.turnS).mean():4.0f}%  gross FF5+MOM {g['FF5+MOM'][0]:.2f} ({g['FF5+MOM'][1]:.1f})  net: {fmt(n)}"
        f"  net 2005-25 FF5+MOM {nl['FF5+MOM'][0]:.2f} ({nl['FF5+MOM'][1]:.1f})")
# a tempting 'improvement' that uses hindsight: drop the long leg (no alpha) and hedge the short leg with the market
mk = (X.MKT + X.RF).reindex(main.index); hedged = mk - main.rS - main.costS
switch = main.ls_net.where(main.index < 201801, hedged)       # real-time version: switch only after SY (2017) is published
RES['hedged'] = {'all_years': alphas(hedged), 'from_2018': alphas(switch)}
say(f"  Short leg + market, all years (USES HINDSIGHT: we only know the long leg has no alpha ex post): {fmt(alphas(hedged))}")
say(f"  Same, switched on only from 2018 (after Stambaugh-Yuan 2017 was published):               {fmt(alphas(switch))}")

json.dump(RES, open(f'{OUT}/extra_checks.json', 'w'), indent=1, default=float)
open(f'{OUT}/extra_checks.txt', 'w').write('\n'.join(TXT) + '\n')
print('\nsaved output/extra_checks.json and output/extra_checks.txt')
