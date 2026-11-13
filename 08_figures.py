"""Step 8: figures for the paper -> output/fig/*.pdf"""
import json, pickle, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lib import *
R = json.load(open(f'{OUT}/paper_results.json')); O = pickle.load(open(f'{WORK}/realtime.pkl', 'rb')); ff = load_ff()
FIG = f'{OUT}/fig'; os.makedirs(FIG, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.6, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
def dt(ix): return pd.to_datetime([f'{i // 100}-{i % 100:02d}-28' for i in ix])
main = O[('ew_pub', 'buf')][0].loc[199701:202512]; wts = O[('ew_pub', 'buf')][2].loc[199701:202512]

# Figure 1: anomalies in the score over time
PUB = {'MOM': 1993, 'NSI': 1995, 'ACC': 1996, 'OSC': 1998, 'NOA': 2004, 'IVA': 2004, 'CEI': 2006, 'ROA': 2006, 'AG': 2008, 'GP': 2013}
n = wts.notna().sum(1); fig, a = plt.subplots(figsize=(6.5, 2.4))
a.step(dt(n.index), n.values, where='post', color=C[0]); a.set_ylim(0, 12.5); a.set_ylabel('Anomalies in the score')
groups = {}
for k, y in PUB.items(): groups.setdefault(max(y + 1, 1997), []).append(k)
for y, ks in sorted(groups.items()):
    lvl = sum(1 for v in PUB.values() if max(v + 1, 1997) <= y)
    a.annotate(', '.join(sorted(ks)) + (' (start)' if y == 1997 else ''), (pd.Timestamp(f'{y}-01-15'), lvl), xytext=(3, -11), textcoords='offset points', fontsize=7.5, color=INK2)
fig.savefig(f'{FIG}/fig1_entry.pdf'); plt.close()

# Figure 2: decile returns and FF5+MOM alphas
d = pd.DataFrame(R['deciles']); x = np.arange(1, 11)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
ax[0].bar(x, d.vw, color=C[0], width=0.7); ax[0].set_title('Mean return, VW (% per month)', loc='left', fontsize=8.5, color=INK)
ax[1].bar(x, d.ff6_vw, color=[C[7] if v < 0 else C[2] for v in d.ff6_vw], width=0.7); ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].set_title('FF5+MOM alpha (% per month)', loc='left', fontsize=8.5, color=INK)
for a in ax: a.set_xticks(x); a.set_xlabel('Decile (1 = most overpriced)')
fig.savefig(f'{FIG}/fig2_deciles.pdf'); plt.close()

# Figure 3: cumulative net returns
rf = ff.RF.reindex(main.index); mkt = ff.MKT.reindex(main.index)
series = {'Strategy, long-short net of costs': main.ls_net, 'Market excess return': mkt, 'Market + 0.5 x strategy': mkt + 0.5 * main.ls_net}
fig, a = plt.subplots(figsize=(6.5, 2.8))
for i, (k, s) in enumerate(series.items()):
    a.plot(dt(s.index), np.log((1 + s).cumprod()), color=C[i], label=k)
for (lo, hi) in [(200003, 200209), (200711, 200902), (202002, 202003)]:
    a.axvspan(pd.Timestamp(f'{lo // 100}-{lo % 100:02d}-01'), pd.Timestamp(f'{hi // 100}-{hi % 100:02d}-28'), color='#cfcfcf', alpha=0.4, lw=0)
a.set_ylabel('Log cumulative return'); a.legend(loc='upper left', fontsize=7.5)
fig.savefig(f'{FIG}/fig3_cumulative.pdf'); plt.close()

# Figure 4: rolling 60-month FF5+MOM alpha (net)
roll = []
for i in range(59, len(main)):
    sub = main.index[i - 59:i + 1]; y = main.ls_net.loc[sub]; X = ff.loc[sub, ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']].values
    b, t, e, r2 = nw_ols(y.values, X); se = b[0] / t[0]; roll.append((sub[-1], b[0] * 100, (b[0] - 1.96 * se) * 100, (b[0] + 1.96 * se) * 100))
roll = pd.DataFrame(roll, columns=['ym', 'a', 'lo', 'hi']).set_index('ym')
fig, a = plt.subplots(figsize=(6.5, 2.4))
a.fill_between(dt(roll.index), roll.lo, roll.hi, color=C[0], alpha=0.15, lw=0); a.plot(dt(roll.index), roll.a, color=C[0])
a.axhline(0, color=INK2, lw=0.8); a.set_ylabel('Net FF5+MOM alpha, % / month')
fig.savefig(f'{FIG}/fig4_rolling.pdf'); plt.close()

# Figure 5: our two ideas -- real time vs hindsight, before vs after costs (cumulative long-short return)
fig, a = plt.subplots(figsize=(6.5, 2.6))
for i, (k, col, lab) in enumerate([(('hindsight', 'buf'), 'ls_net', 'Hindsight (all 10 anomalies from the start), after costs'),
                                   (('ew_pub', 'buf'), 'ls', 'Real time (main), before costs'),
                                   (('ew_pub', 'buf'), 'ls_net', 'Real time (main), after costs')]):
    s = O[k][0].loc[199701:202512][col]; a.plot(dt(s.index), np.log((1 + s).cumprod()), color=C[[1, 3, 0][i]], label=lab, lw=[1.3, 1.3, 1.9][i])
a.set_ylabel('Log cumulative return'); a.legend(loc='upper left', fontsize=7.5)
fig.savefig(f'{FIG}/fig5_versions.pdf'); plt.close()
print('figures written to', FIG)
