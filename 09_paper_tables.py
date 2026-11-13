"""Step 9: LaTeX tables + number macros for the paper (paper/tab/*.tex, paper/macros.tex), all from output/*.json."""
import json, os, re
from config import OUT, HERE
R = json.load(open(f'{OUT}/paper_results.json')); V = R['versions']
T = os.path.join(HERE, 'paper', 'tab'); os.makedirs(T, exist_ok=True)
def f(x, d=2):
    if x is None or x != x: return '--'
    return f'{x:.{d}f}'.replace('-', '\\ensuremath{-}')
def tt(x): return f'({f(x)})'
def w(n, s): open(f'{T}/{n}.tex', 'w').write(s.strip() + '\n')
K = ['MOM', 'NSI', 'ACC', 'OSC', 'NOA', 'IVA', 'CEI', 'ROA', 'AG', 'GP']
NAME = {'MOM': 'Momentum (12--1)', 'NSI': 'Net stock issues', 'ACC': 'Total accruals', 'OSC': 'O-score (distress)', 'NOA': 'Net operating assets',
        'IVA': 'Investment-to-assets', 'CEI': 'Composite equity issues', 'ROA': 'Return on assets', 'AG': 'Asset growth', 'GP': 'Gross profitability'}
DEF = {'MOM': 'Cumulative return, months $t-11$ to $t-1$', 'NSI': 'Split-adjusted growth in shares outstanding, past 12 months',
       'ACC': '$[(\\Delta$ACT$-\\Delta$CHE$)-(\\Delta$LCT$-\\Delta$DLC$)-$DP$]/\\overline{\\text{AT}}$', 'OSC': 'Ohlson (1980) bankruptcy score',
       'NOA': '(Operating assets $-$ operating liabilities)/AT$_{y-1}$', 'IVA': '($\\Delta$PPENT + $\\Delta$INVT)/AT$_{y-1}$',
       'CEI': '5-year log ME growth minus 5-year log return', 'ROA': 'IB/AT$_{y-1}$', 'AG': 'AT$_y$/AT$_{y-1}-1$', 'GP': '(REVT $-$ COGS)/AT'}
SIDE = {'MOM': 'High', 'NSI': 'Low', 'ACC': 'Low', 'OSC': 'Low', 'NOA': 'Low', 'IVA': 'Low', 'CEI': 'Low', 'ROA': 'High', 'AG': 'Low', 'GP': 'High'}
SRC = {'MOM': 'Jegadeesh \\& Titman', 'NSI': 'Loughran \\& Ritter', 'ACC': 'Sloan', 'OSC': 'Dichev', 'NOA': 'Hirshleifer et al.', 'IVA': 'Titman, Wei \\& Xie',
       'CEI': 'Daniel \\& Titman', 'ROA': 'Fama \\& French', 'AG': 'Cooper, Gulen \\& Schill', 'GP': 'Novy-Marx'}
IDEA = {'MOM': 'Past winners keep outperforming past losers for several months', 'NSI': 'Firms that issue new shares underperform in the following years',
        'ACC': 'Earnings driven by accruals are less persistent than cash earnings, which investors do not fully see',
        'OSC': 'Firms with high bankruptcy risk earn lower, not higher, returns', 'NOA': 'Balance sheets ``bloated\'\' by past accounting earnings not backed by cash predict low returns',
        'IVA': 'Firms that raise capital investment strongly underperform', 'CEI': 'Equity issued net of repurchases over five years predicts low returns',
        'ROA': 'More profitable firms earn higher returns', 'AG': 'Firms whose total assets grow fast earn low returns', 'GP': 'Firms with high gross profits relative to assets earn higher returns'}
# Table 1: anomalies
w('anomalies', '\n'.join(f"{NAME[k]} & {DEF[k]} & {IDEA[k]} & {SIDE[k]} & {SRC[k]} ({R['single'][k]['pub']}) & {R['single'][k]['pub'] + 1}\\\\" for k in K))
# Table 2: sample
w('sample', '\n'.join(f"{d['year']} & {d['n']:,} & {f(100 * d['nyse_share'], 0)} & {d['tot_me_bn']:,.0f} & {d['med_me_m']:,.0f}\\\\" for d in R['desc']))
# Table 3: single anomalies pre/post publication
rows = []
for k in K:
    s = R['single'][k]; a, b = s['pre'], s['post']
    rows.append(f"{NAME[k]} & {s['pub']} & {f(a['mean'])} & {tt(a['t'])} & {f(a['ff6']['a'])} & {tt(a['ff6']['t'])} & {f(b['mean'])} & {tt(b['t'])} & {f(b['ff6']['a'])} & {tt(b['ff6']['t'])}\\\\")
w('single', '\n'.join(rows))
# Table 4: deciles
rows = []
for d in R['deciles']:
    rows.append(f"{d['dec']} & {f(d['n'], 0)} & {d['me_med']:,.0f} & {f(d['bm_med'])} & {f(d['score'])} & {f(d['vw'])} & {f(d['ew'])} & {f(d['capm_vw'])} & {f(d['ff6_vw'])} & {tt(d['t_ff6_vw'])}\\\\")
lv, le = R['dec_ls_vw'], R['dec_ls_ew']
rows.append('\\midrule')
rows.append(f"10$-$1 & & & & & {f(lv['mean'])} & {f(le['mean'])} & {f(lv['CAPM']['a'])} & {f(lv['FF5+MOM']['a'])} & {tt(lv['FF5+MOM']['t'])}\\\\")
w('deciles', '\n'.join(rows))
# Table 5: alphas of decile spread under every Lecture-2 model
MODS = [m for m in ('CAPM', 'FF3', 'FF3+MOM', 'FF5', 'FF5+MOM', 'q4', 'q5') if m in R.get('models', [])]
LAB = {'CAPM': 'CAPM', 'FF3': 'FF3', 'FF3+MOM': 'FF3+MOM', 'FF5': 'FF5', 'FF5+MOM': 'FF5+MOM', 'q4': 'q4', 'q5': 'q5'}
rows = []
for nm, d in (('Value-weighted', lv), ('Equal-weighted', le)):
    rows.append(f"{nm} & {f(d['mean'])} & " + ' & '.join(f(d[m]['a']) for m in MODS) + '\\\\')
    rows.append(f" & {tt(d['t'])} & " + ' & '.join(tt(d[m]['t']) for m in MODS) + '\\\\')
w('alphas', '\n'.join(rows))
open(f'{T}/alphas_header.tex', 'w').write('l' + 'r' * (len(MODS) + 1) + '|' + ' & Mean & ' + ' & '.join(LAB[m] for m in MODS) + '\\\\\n')
# Table 5b: main strategy (buffer, $100m) under every model: gross and net alphas
rows = []
for m in MODS:
    v = V['ew_pub|buf']
    rows.append(f"{LAB[m]} & {f(v['1997-2025']['gross'][m]['a'])} & {tt(v['1997-2025']['gross'][m]['t'])} & {f(v['1997-2025']['net'][m]['a'])} & {tt(v['1997-2025']['net'][m]['t'])} & "
                f"{f(v['2005-2025']['net'][m]['a'])} & {tt(v['2005-2025']['net'][m]['t'])} & {f(v['2014-2025']['net'][m]['a'])} & {tt(v['2014-2025']['net'][m]['t'])}\\\\")
w('allmodels', '\n'.join(rows))

# Lecture-format table: main strategy net of costs, all models with loadings (from step 4 output)
import csv as _csv
def lecture_tex(path, cost_path=None):
    rows = list(_csv.reader(open(path)))[1:]
    out = []
    for r in rows:
        if not any(r): continue
        name = r[0].replace('-', '---') if r[0] == '-' else r[0]
        cells = [c.replace('%', '\\%').replace('-', '\\ensuremath{-}') if c else '' for c in r[1:]]
        out.append(name + ' & ' + ' & '.join(cells) + '\\\\')
    return '\n'.join(out)
w('lecture_net', lecture_tex(f'{OUT}/lecture_table_main_net.csv'))
g = list(_csv.reader(open(f'{OUT}/lecture_table_main_gross.csv')))
costs = [r for r in g if r[0].startswith(('Turnover', 'Spread', 'Price impact'))]
SHORT = {'Turnover': 'Turnover (\\%/mo)', 'Spread': 'Spread (\\%/mo)', 'Price impact': 'Impact (\\%/mo)'}
def _short(lbl): return next(v for k_, v in SHORT.items() if lbl.startswith(k_))
def _pct(x): return x.replace('%', '\\%')   # (no backslashes inside f-strings: Python 3.11 compatible)
w('lecture_costs', '\n'.join(f"{_short(r[0])} & {_pct(r[1])} & \\multicolumn{{12}}{{l}}{{}}\\\\" for r in costs))

# Table 6: size x score
rows = []
for r in R['size_sort']:
    rows.append(f"{r['size']} & " + ' & '.join(f(v) for v in r['q']) + f" & {f(r['ls']['mean'])} & {tt(r['ls']['t'])} & {f(r['ff6']['a'])} & {tt(r['ff6']['t'])}\\\\")
w('size', '\n'.join(rows))
# Table 7: FM
fm = R['fm']
def c(k): return f"{f(fm[k]['mean'])}" if k in fm else ''
def ct(k): return f"{tt(fm[k]['t'])}" if k in fm else ''
w('fm', f"""Mispricing score & {c('s1')} & {c('s2')} & {c('s3')}\\\\
 & {ct('s1')} & {ct('s2')} & {ct('s3')}\\\\
Log market equity & & {c('me2')} & {c('me3')}\\\\
 & & {ct('me2')} & {ct('me3')}\\\\
Log book-to-market & & {c('bm2')} & {c('bm3')}\\\\
 & & {ct('bm2')} & {ct('bm3')}\\\\
Momentum & & & {c('mom3')}\\\\
 & & & {ct('mom3')}\\\\""")
# Table 8: main strategy gross/net by window
rows = []
for wn in ('1997-2025', '1997-2004', '2005-2025', '2014-2025'):
    v = V['ew_pub|buf'][wn]
    for kind in ('gross', 'net'):
        x = v[kind]
        rows.append(f"{wn.replace('-', '--') if kind == 'gross' else ''} & {kind.capitalize()} & {f(x['mean'])} & {tt(x['t'])} & {f(x['sr'])} & {f(x['CAPM']['a'])} & {f(x['FF5+MOM']['a'])} & {tt(x['FF5+MOM']['t'])} & {f(x['q']['a'])} & {tt(x['q']['t'])}\\\\")
    rows.append(f" & \\multicolumn{{9}}{{l}}{{\\footnotesize Turnover {f(v['turn'], 0)}\\% per month; costs {f(v['cost'])}\\% p.a. (spread {f(v['cost_spread'])}, impact {f(v['cost_impact'])})}}\\\\[2pt]")
w('main', '\n'.join(rows))
# Table 9: implementation
rows = []
for lab, key in (('No buffer (deciles)', 'ew_pub|base'), ('10\\%/20\\% buffer (main)', 'ew_pub|buf'), ('Buffer, excluding microcaps', 'ew_pub|exmicro_buf')):
    v = V[key]['1997-2025']
    rows.append(f"{lab} & {f(v['turn'], 0)} & {f(v['cost'])} & {f(v['gross']['FF5+MOM']['a'])} & {tt(v['gross']['FF5+MOM']['t'])} & {f(v['net']['FF5+MOM']['a'])} & {tt(v['net']['FF5+MOM']['t'])} & {f(V[key]['2005-2025']['net']['FF5+MOM']['a'])} & {tt(V[key]['2005-2025']['net']['FF5+MOM']['t'])}\\\\")
rows.append('\\midrule')
RB = R['robust']
for lab, key in (('Spreads $\\times 0.5$', 'spread_half'), ('Spreads $\\times 2$', 'spread_x2'), ('Flat 0.72\\% per \\$ traded', 'cost_flat'),
                 ('Our earlier stock-level model', 'cost_stock'), ('Fund size \\$1bn', 'aum_1bn'), ('Fund size \\$10bn', 'aum_10bn')):
    v = RB[key]
    rows.append(f"{lab} & {f(v['turn'], 0)} & {f(v['cost'])} & {f(v['gross_a6'])} & {tt(v['gross_t6'])} & {f(v['net_a6'])} & {tt(v['net_t6'])} & {f(v['test_net_a6'])} & {tt(v['test_net_t6'])}\\\\")
rows.append('\\midrule')
vmain = V['ew_pub|buf']['1997-2025']
for f_, lab in (('0.01', 'Borrowing fee 1\\% p.a.'), ('0.02', 'Borrowing fee 2\\% p.a.')):
    fe = R['fees'][f_]
    rows.append(f"{lab} & {f(vmain['turn'], 0)} & {f(vmain['cost'] + 100 * float(f_))} & & & {f(fe['a'])} & {tt(fe['t'])} & & \\\\")
w('implementation', '\n'.join(rows))
# Table 10: look-ahead / learned weights / publication timing
rows = []
for lab, key in (('Real time (main)', 'ew_pub|buf'), ('Hindsight: all ten anomalies from the start', 'hindsight|buf')):
    v = V[key]
    rows.append(f"{lab} & {f(v['1997-2025']['gross']['FF5+MOM']['a'])} & {tt(v['1997-2025']['gross']['FF5+MOM']['t'])} & {f(v['1997-2025']['net']['FF5+MOM']['a'])} & {tt(v['1997-2025']['net']['FF5+MOM']['t'])} & {f(v['2005-2025']['net']['FF5+MOM']['a'])} & {tt(v['2005-2025']['net']['FF5+MOM']['t'])} & {f(v['1997-2025']['turn'], 0)}\\\\")
rows.append('\\midrule')
for lab, key in (('Entry two years earlier (working paper)', 'publag_minus1'), ('Entry two years later', 'publag_plus2')):
    v = RB[key]
    rows.append(f"{lab} & {f(v['gross_a6'])} & {tt(v['gross_t6'])} & {f(v['net_a6'])} & {tt(v['net_t6'])} & {f(v['test_net_a6'])} & {tt(v['test_net_t6'])} & {f(v['turn'], 0)}\\\\")
w('versions', '\n'.join(rows))
# Table: alternative strategies built with the same rules (from step 6)
try:
    XC = json.load(open(f'{OUT}/extra_checks.json'))
    rows = []
    for lab, d in XC['alternatives'].items():
        st = str(d['start']); g, n = d['gross']['FF5+MOM'], d['net']['FF5+MOM']; nc, nq = d['net']['CAPM'], d['net']['q5']
        SHORTLAB = {'Asset growth only': 'Asset growth only', 'Share issuance only': 'Share issuance only (NSI, CEI)', 'SY management': 'Management signals only (6)',
                    'SY performance': 'Performance signals only (4)', 'All ten': 'All ten anomalies (main strategy)', 'Momentum only': 'Momentum only'}
        lab = next(v for k_, v in SHORTLAB.items() if lab.startswith(k_))
        rows.append(f"{lab} & {st[:4]} & {f(g[0])} & {tt(g[1])} & {f(nc[0])} & {tt(nc[1])} & {f(n[0])} & {tt(n[1])} & {f(nq[0])} & {tt(nq[1])}\\\\")
    w('alternatives', '\n'.join(rows))
    RP = {d['label']: d for d in XC['replication']}
    m_extra = {'repmomturn': f(RP['Momentum, VW']['turn'], 0), 'repmomimp': f(RP['Momentum, VW']['imp']),
               'repbufturn': f(RP['Momentum 10/20, all, VW']['turn'], 0), 'repbufimp': f(RP['Momentum 10/20, all, VW']['imp']),
               'repnmturn': f(RP['Momentum 10/20, non-micro, VW']['turn'], 0), 'repnmimp': f(RP['Momentum 10/20, non-micro, VW']['imp'])}
    vs = XC['volscale_net']; m_extra.update({'vsrorig': f(vs['sharpe_orig']), 'vsrscaled': f(vs['sharpe_scaled']), 'vlevmax': f(vs['lev_max'], 1)})
except FileNotFoundError:
    m_extra = {}
# Table: replication of the course's cost tables (from step 6)
try:
    rows = []
    for d in XC['replication']:
        rows.append(f"{d['label']} & {f(d['turn'], 0)} & {d['turn_lec']} & {f(d['imp'])} & {f(d['imp_lec'])} & {f(d['spread'])} & {f(d['spread_lec'])}\\\\")
    w('replication', '\n'.join(rows))
except (NameError, KeyError):
    pass
# Table: design choices (from step 6) and look-ahead test (from step 7)
try:
    rows = []
    for lab, d in XC['design'].items():
        g, n, nl = d['gross']['FF5+MOM'], d['net'], d['net_2005']['FF5+MOM']
        lab_ = lab.replace('-30%', '$-$30%').replace('%', '\\%')
        rows.append(f"{lab_} & {f(d['turnover_%pm'], 0)} & {f(g[0])} & {tt(g[1])} & {f(n['CAPM'][0])} & {tt(n['CAPM'][1])} & {f(n['FF5+MOM'][0])} & {tt(n['FF5+MOM'][1])} & {f(nl[0])} & {tt(nl[1])}\\\\")
    w('design', '\n'.join(rows))
    _dn = [d['net']['FF5+MOM'][0] for lab_, d in XC['design'].items() if not lab_.startswith('Weights learned')]
    m_extra.update({'designmin': f(min(_dn)), 'designmax': f(max(_dn))})
    h = XC['hedged']; m_extra.update({'hedgeall': f(h['all_years']['FF5+MOM'][0]), 'hedgeallt': f(h['all_years']['FF5+MOM'][1], 1),
                                      'hedgert': f(h['from_2018']['FF5+MOM'][0]), 'hedgertt': f(h['from_2018']['FF5+MOM'][1], 1)})
except (NameError, KeyError):
    pass
try:
    LA = json.load(open(f'{OUT}/lookahead_test.json'))
    m_extra.update({'lacut': str(LA['last'] // 100), 'lamonths': str(LA['n_months']), 'ladiff': f"{LA['max_diff']:.0e}".replace('e-', '\\times 10^{-').replace('e+', '\\times 10^{') + '}',
                    'lapass': 'identical' if LA['pass'] else 'NOT identical'})
except FileNotFoundError:
    m_extra.update({'lacut': '2010', 'lamonths': '--', 'ladiff': '--', 'lapass': '(test not run)'})
# Table 11: fund manager view
ov = R['overlay']
rows = [f"Market + {lam} $\\times$ strategy & {f(ov[lam]['ret'], 1)} & {f(ov[lam]['vol'], 1)} & {f(ov[lam]['sr'])} & {f(ov[lam]['maxdd'], 0)}\\\\" for lam in ('0.0', '0.25', '0.5', '1.0')]
w('overlay', '\n'.join(rows))
w('crises', '\n'.join(f"{k} & {f(v['strategy'], 1)} & {f(v['market'], 1)}\\\\" for k, v in R['crises'].items()))
# Appendix: rank correlations
RC = R['rank_corr']
w('corr', '\n'.join(f"{k} & " + ' & '.join(f(RC[i][j]) if j <= i else '' for j in range(len(K))) + '\\\\' for i, k in enumerate(['NSI', 'CEI', 'ACC', 'NOA', 'AG', 'IVA', 'MOM', 'GP', 'ROA', 'OSC'])))
# ---- macros ----
m = {}
v = V['ew_pub|buf']
for wn, tag in (('1997-2025', 'full'), ('1997-2004', 'early'), ('2005-2025', 'late'), ('2014-2025', 'recent')):
    for kind in ('gross', 'net'):
        x = v[wn][kind]; p = f'{kind}{tag}'
        m[p + 'mean'] = f(x['mean']); m[p + 'meant'] = f(x['t'], 1); m[p + 'sr'] = f(x['sr']); m[p + 'a'] = f(x['FF5+MOM']['a']); m[p + 'at'] = f(x['FF5+MOM']['t'], 1)
        m[p + 'q'] = f(x['q']['a']); m[p + 'qt'] = f(x['q']['t'], 1); m[p + 'capm'] = f(x['CAPM']['a']); m[p + 'ann'] = f(12 * x['FF5+MOM']['a'], 1); m[p + 'capmt'] = f(x['CAPM']['t'], 1)
    m['turn' + tag] = f(v[wn]['turn'], 0); m['cost' + tag] = f(v[wn]['cost'])
m['costspread'] = f(v['1997-2025']['cost_spread']); m['costimpact'] = f(v['1997-2025']['cost_impact'])
m['turnL'] = f(v['1997-2025']['turnL'], 0); m['turnS'] = f(v['1997-2025']['turnS'], 0); m['nL'] = f(v['1997-2025']['nL'], 0); m['nS'] = f(v['1997-2025']['nS'], 0)
b = V['ew_pub|base']['1997-2025']; m['baseturn'] = f(b['turn'], 0); m['basecost'] = f(b['cost']); m['basenet'] = f(b['net']['FF5+MOM']['a']); m['basenett'] = f(b['net']['FF5+MOM']['t'], 1)
m['basegross'] = f(b['gross']['FF5+MOM']['a']); m['basegrosst'] = f(b['gross']['FF5+MOM']['t'], 1)
h = V['hindsight|buf']['1997-2025']['net']['FF5+MOM']; m['hindnet'] = f(h['a']); m['hindnett'] = f(h['t'], 1)
c_ = V['comb_pub|buf']['1997-2025']; m['combnet'] = f(c_['net']['FF5+MOM']['a']); m['combnett'] = f(c_['net']['FF5+MOM']['t'], 1); m['combturn'] = f(c_['turn'], 0)
m['combgross'] = f(c_['gross']['FF5+MOM']['a']); m['combgrosst'] = f(c_['gross']['FF5+MOM']['t'], 1); m['combmom'] = f(100 * R['comb_weights_avg']['MOM'], 0)
m['decvwmean'] = f(lv['mean']); m['decvwmeant'] = f(lv['t'], 1); m['decvwa'] = f(lv['FF5+MOM']['a']); m['decvwat'] = f(lv['FF5+MOM']['t'], 1)
m['decvwq'] = f(lv['q']['a']); m['decvwqt'] = f(lv['q']['t'], 1); m['decvwcapm'] = f(lv['CAPM']['a']); m['decvwcapmt'] = f(lv['CAPM']['t'], 1)
m['decvwfivea'] = f(lv['FF5']['a']); m['decvwfiveat'] = f(lv['FF5']['t'], 1)
m['dececwa'] = f(le['FF5+MOM']['a']); m['dececwat'] = f(le['FF5+MOM']['t'], 1)
d1 = R['deciles'][0]; m['donevw'] = f(d1['vw']); m['donea'] = f(d1['ff6_vw']); m['doneat'] = f(d1['t_ff6_vw'], 1); m['dtenvw'] = f(R['deciles'][9]['vw'])
m['grsF'] = f(R['grs']['FF5+MOM'][0]); m['grsp'] = f(R['grs']['FF5+MOM'][1], 4); m['grsfiveF'] = f(R['grs']['FF5'][0]); m['grsfivep'] = f(R['grs']['FF5'][1], 4)
L = R['legs']; m['longa'] = f(L['long_ff6']['a']); m['longat'] = f(L['long_ff6']['t'], 1); m['shorta'] = f(L['short_ff6']['a']); m['shortat'] = f(L['short_ff6']['t'], 1)
m['fmscore'] = f(fm['s3']['mean']); m['fmscoret'] = f(fm['s3']['t'], 1); m['fmn'] = f(R['fm_n'], 0); m['fmscoreone'] = f(fm['s1']['mean']); m['fmscoreonet'] = f(fm['s1']['t'], 1)
for r in R['size_sort']: m['size' + r['size'].lower() + 'a'] = f(r['ff6']['a']); m['size' + r['size'].lower() + 'at'] = f(r['ff6']['t'], 1)
m['corrmkt'] = f(R['corr_mkt']); m['maxdd'] = f(R['maxdd'], 0); m['vol'] = f(R['vol'], 1); m['annret'] = f(R['ann_ret'], 1)
for lam, tag in (('0.0', 'zero'), ('0.5', 'half'), ('1.0', 'one')): m['ovsr' + tag] = f(ov[lam]['sr']); m['ovdd' + tag] = f(ov[lam]['maxdd'], 0)
cr = R['crises']; ks = list(cr)
m['crdot'] = f(cr[ks[0]]['strategy'], 0); m['crdotm'] = f(cr[ks[0]]['market'], 0); m['crgfc'] = f(cr[ks[1]]['strategy'], 0); m['crgfcm'] = f(cr[ks[1]]['market'], 0)
m['crmom'] = f(cr[ks[2]]['strategy'], 0); m['crmomm'] = f(cr[ks[2]]['market'], 0); m['crcovid'] = f(cr[ks[3]]['strategy'], 1); m['crcovidm'] = f(cr[ks[3]]['market'], 0)
m['crrate'] = f(cr[ks[4]]['strategy'], 0); m['cratem'] = f(cr[ks[4]]['market'], 0)
m['bemult'] = f(R['breakeven_mult'], 1)
for key, tag in (('aum_1bn', 'onebn'), ('aum_10bn', 'tenbn'), ('cost_flat', 'flat'), ('spread_half', 'sphalf'), ('spread_x2', 'sptwo'), ('cost_stock', 'stock'), ('publag_minus1', 'early'), ('publag_plus2', 'late'), ('exmicro', 'exmicro')):
    m['rob' + tag] = f(RB[key]['net_a6']); m['rob' + tag + 't'] = f(RB[key]['net_t6'], 1)
m['robonebncost'] = f(RB['aum_1bn']['cost']); m['robtenbncost'] = f(RB['aum_10bn']['cost']); m['robstockcost'] = f(RB['cost_stock']['cost'])
m['robstockgross'] = f(RB['cost_stock']['gross_a6']); m['robflatcost'] = f(RB['cost_flat']['cost'])
FE = R['fees']; m['feeone'] = f(FE['0.01']['a']); m['feeonet'] = f(FE['0.01']['t'], 1); m['feetwo'] = f(FE['0.02']['a']); m['feetwot'] = f(FE['0.02']['t'], 1)
m['feequarter'] = f(FE['0.0025']['a']); m['feequartert'] = f(FE['0.0025']['t'], 1)
m['feebe'] = f(R['fee_breakeven_pa'], 1); m['feesig'] = f(R['fee_t196_pa'], 2)
MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
_md = lambda ym: f"{MON[ym % 100 - 1]} {ym // 100}" if ym else 'the end of the sample (not yet recovered)'
DD = R['dd_dates']; m['ddstart'] = _md(DD['peak']); m['ddtrough'] = _md(DD['trough']); m['ddrec'] = _md(DD['recovered'])
m['capmbeta'] = f(R['capm_beta'])
MR = R['missing_ret']; m['impfill'] = f"{MR['held_missing']:,}" if MR['held_missing'] is not None else '--'; m['nstockmonths'] = f"{MR['held_total']:,}"

import re as _re
_src = open(os.path.join(HERE, '04_realtime_strategy.py')).read()
_hs = [float(v) for v in _re.search(r'HALF_SPREAD = np.array\(\[([^\]]+)\]', _src).group(1).split(',')]
m['hsmicro'], m['hssmall'], m['hslarge'] = (f(100 * v) for v in _hs)
ins = R['insample']['vw_full']; m['insa'] = f(ins['ff6']); m['insat'] = f(ins['t_ff6'], 1); m['insmean'] = f(ins['mean']); m['insmeant'] = f(ins['t'], 1)
va = R['validation']; m['valew'] = f(va['corr_ew_with_lecture'], 3); m['valvw'] = f(va['corr_vw_with_lecture'], 3); m['valhml'] = f(va['corr_HML'], 3)
m['valumd'] = f(va['corr_MOM'], 3); m['valmkt'] = f(va['corr_MKT'], 4)
q = R.get('qproxy_corr') or {'ME_SMB': float('nan'), 'IA_CMA': float('nan'), 'ROE_RMW': float('nan')}; m['qme'] = f(q['ME_SMB']); m['qia'] = f(q['IA_CMA']); m['qroe'] = f(q['ROE_RMW'])
m['qsource'] = R.get('q_source', 'proxy')
if 'q5' in R.get('models', []):
    m['egload'] = f(V['ew_pub|buf']['1997-2025']['net']['q5']['b']['R_EG']); m['roeload'] = f(V['ew_pub|buf']['1997-2025']['net']['q5']['b']['R_ROE'])
    m['decvwqfour'] = f(lv['q4']['a']); m['decvwqfourt'] = f(lv['q4']['t'], 1); m['decewqfive'] = f(le['q5']['a']); m['decewqfivet'] = f(le['q5']['t'], 1)
    m['grossqfivelate'] = f(V['ew_pub|buf']['2005-2025']['gross']['q5']['a']); m['grossqfivelatet'] = f(V['ew_pub|buf']['2005-2025']['gross']['q5']['t'], 1)
MACRO = {'FF3+MOM': 'ffthreemom', 'q4': 'qfour', 'q5': 'qfive'}
for mod in ('FF3+MOM', 'q4', 'q5'):
    if mod in R.get('models', []):
        k_ = MACRO[mod]
        x = V['ew_pub|buf']['1997-2025']['net'][mod]; m['net' + k_ + 'a'] = f(x['a']); m['net' + k_ + 't'] = f(x['t'], 1)
        x = V['ew_pub|buf']['2005-2025']['net'][mod]; m['late' + k_ + 'a'] = f(x['a']); m['late' + k_ + 't'] = f(x['t'], 1)
for mod, k_ in (('FF3', 'ffthree'), ('FF5', 'fffive')):
    x = V['ew_pub|buf']['1997-2025']['net'][mod]; m['net' + k_ + 'a'] = f(x['a']); m['net' + k_ + 't'] = f(x['t'], 1)
x = V['ew_pub|buf']['1997-2025']['gross']; m['grossfullcapmt'] = f(x['CAPM']['t'], 1)
m['betalong'] = f(L['long_capm']['b']['MKT']); m['betashort'] = f(-L['short_capm']['b']['MKT'])
try:
    for lab_, d_ in XC['alternatives'].items():
        tag = 'altmgmt' if lab_.startswith('SY management') else 'altperf' if lab_.startswith('SY performance') else None
        if tag: m[tag] = f(d_['net']['FF5+MOM'][0]); m[tag + 't'] = f(d_['net']['FF5+MOM'][1], 1)
except (NameError, KeyError):
    pass
ar = lecture_ar2 = [r for r in _csv.reader(open(f'{OUT}/lecture_table_main_net.csv')) if r and r[0] == 'FF5+MOM']
if ar: m['arsq'] = ar[0][-1].replace('%', '')
RG = R.get('rank_corr_groups')
if RG: m['corrmgmt'] = f(RG['mgmt']); m['corrperf'] = f(RG['perf']); m['corrcross'] = f(RG['cross'])
UN = R.get('universe_n')
if UN: m['universemin'] = f"{100 * round(UN['min'] / 100):,}"; m['universemax'] = f"{100 * round(UN['max'] / 100):,}"
if 'link_valid_share' in va: m['linkrows'] = f"{va['link_rows']:,}"; m['linkvalid'] = f(100 * va['link_valid_share'], 1)
_dv = [d['vw'] for d in R['deciles'][1:]]; m['dresmin'] = f(min(_dv)); m['dresmax'] = f(max(_dv))
try:
    for lab_, tag in (('Weights drift with the price return', 'drift'), ('Missing returns of held stocks set to -30%', 'miss'), ('Quarterly rebalancing', 'quart')):
        d_ = XC['design'][lab_]; m[tag + 'net'] = f(d_['net']['FF5+MOM'][0]); m[tag + 'nett'] = f(d_['net']['FF5+MOM'][1], 1); m[tag + 'turn'] = f(d_['turnover_%pm'], 0)
except (NameError, KeyError):
    pass
m['nmonths'] = str(R['n_months']); m.update(m_extra)
be = R['main_betas']; m['bsmb'] = f(be['SMB']); m['brmw'] = f(be['RMW']); m['bcma'] = f(be['CMA']); m['bmom'] = f(be['MOM']); m['bmkt'] = f(be['MKT']); m['bhml'] = f(be['HML'])
DIG = dict(zip('0123456789', ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']))
open(os.path.join(HERE, 'paper', 'macros.tex'), 'w').write('\n'.join(f"\\newcommand{{\\{re.sub(r'[0-9]', lambda g: DIG[g.group()], k)}}}{{{v_}}}" for k, v_ in m.items()) + '\n')
print(len(m), 'macros; tables in', T)
