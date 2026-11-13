<<<<<<< HEAD
# Mispricing-Strategy-Alpha-Project
=======
# Real-time mispricing strategy — code

Applied Investments (Prof. Goyal), HEC Lausanne — portfolio strategy project.

## What it does

Each month we rank all US common stocks (price ≥ $5) on the 10 Stambaugh–Yuan anomalies:

| Anomaly | Good side | Published | Enters the strategy |
|---|---|---|---|
| Momentum (12–1) | high | Jegadeesh & Titman (1993) | Jan 1994 |
| Net share issuance | low | Loughran & Ritter (1995) | Jan 1996 |
| Accruals | low | Sloan (1996) | Jan 1997 |
| Ohlson O-score (distress) | low | Dichev (1998) | Jan 1999 |
| Net operating assets | low | Hirshleifer et al. (2004) | Jan 2005 |
| Investment-to-assets | low | Titman, Wei & Xie (2004) | Jan 2005 |
| Composite equity issuance (5y) | low | Daniel & Titman (2006) | Jan 2007 |
| Return on assets | high | Fama & French (2006) | Jan 2007 |
| Asset growth | low | Cooper, Gulen & Schill (2008) | Jan 2009 |
| Gross profitability | high | Novy-Marx (2013) | Jan 2014 |

**Out-of-sample rule:** an anomaly is used only from the January after its publication, so the strategy uses
only what an investor could have known at the time. It starts in January 1997, once three anomalies are public.

The score is the average percentile rank across the published anomalies. The strategy goes long the top 10%
and short the bottom 10% (NYSE breakpoints, value-weighted, rebalanced monthly), with the 10%/20% buffer rule
from Lecture 3: a stock is bought when it enters the top (bottom) 10% and kept until it leaves the top (bottom) 20%.

**Trading costs** (main specification) follow the course's method (Lecture 3, slides 12 and 22-25):
- turnover = sum of |weight changes| per leg, with weights drifting with market capitalisation between rebalancings;
- price impact = sum_i lambda_i * sqrt(V0 * |dw_i| / DolVol_i) * |dw_i|, V0 = $100m, DolVol = the stock's dollar volume
  in the month, lambda = 2.5% / 1.5% / 1.0% for micro / small / large stocks (NYSE 20th and 50th percentiles);
- spread = sum_i |dw_i| * s_i / 2. The CRSP extract has no bid/ask quotes, so s/2 is one value per size group
  (2.08% / 0.68% / 0.44%), chosen so that our replication of the lecture's momentum tables reproduces its spread costs
  (`06_extra_checks.py`, section A, shows the replication and re-derives these numbers).
Alternatives in the robustness table: spreads x0.5 and x2, a flat 0.72% per dollar traded, and our earlier stock-level model.
Short-selling (borrowing) fees are reported separately as a grid (1%, 2% p.a.) with the break-even fee.

**Comparison versions:** `hindsight` uses all 10 anomalies from the start (look-ahead benchmark). `comb_pub`
(weights learned from past Fama–MacBeth slopes) is still computed but not used in the report.

## How to run

Requires Python 3.9+ and `pip install -r requirements.txt`. Steps 1–4 need about 5 GB of RAM.

1. Put the three WRDS files in one folder. The default is
   `~/Downloads/Compustat_Fundamental_Annuals_1968-2025`; otherwise set `RAW_DIR` in `config.py`
   or as an environment variable:
   - `CRSP_MonthlyDataStock_1968-2025.csv`
   - `Compustat_Fundamental_Annuals_1968-2025.csv`
   - `CRSP_Compustat link data.csv` (optional: the Compustat extract already contains LPERMCO; step 2 uses this file to
     check that every firm-year is matched through a link that was valid at the fiscal year-end -- 100% are)
2. Put the official q-factor file from https://global-q.org/factors.html (monthly q5 factors, e.g.
   `q5_factors_monthly_2025.csv`) in `factors/`. It is used for the q4 and q5 models; without it, q4 uses an
   in-house approximation and q5 is skipped. All Lecture-2 models are estimated: CAPM, FF3, FF3+MOM, FF5,
   FF5+MOM, q4, q5 (named exactly like that in the output files; the momentum factor is called MOM).
3. (Recommended for the final paper) put the official Ken French files in `factors/`:
   `F-F_Research_Data_5_Factors_2x3.csv` and `F-F_Momentum_Factor.csv` (unzipped, from the Data Library).
   Without them the code uses `factors/us_ff5_mom.csv.gz`, a mirror of the same data up to August 2026.
4. Run everything with one command (from inside this folder, in the VS Code terminal):

```
python3 main.py
```

`main.py` checks the setup, then runs steps 00-09 below in order and stops with a clear message if a step fails.
Step 00 (the slow one) is skipped once it has been run in full. Options: `--from 4` (start at step 4),
`--only 5` (one step), `--redo-prep` (rebuild step 00), `--no-paper` (don't compile the PDF).

Or run the steps one at a time, in order:

```
python3 00_prep_crsp.py            # ~3-5 min: filter the 2.4 GB CRSP file -> work/crsp_clean.csv.gz
python3 01_build_panel.py          # ~1 min: firm-level panel + all signals -> work/panel.npz
python3 02_validate.py             # checks vs. lecture asset-growth deciles and official FF factors
python3 03_insample_mispricing.py  # in-sample (hindsight) mispricing sorts, 1970-2025
python3 04_realtime_strategy.py    # MAIN: real-time strategy, costs, alphas, GRS, market overlay
python3 05_paper_analysis.py       # extra analyses for the paper -> output/paper_results.json
python3 06_extra_checks.py         # costs vs Lecture 3, alternatives, design choices -> output/extra_checks.txt
python3 07_lookahead_test.py       # rebuild from data ending 2010, compare with full run -> output/lookahead_test.txt
python3 08_figures.py              # figures -> output/fig/
python3 09_paper_tables.py         # LaTeX tables + number macros -> paper/tab/, paper/macros.tex
cd paper && python3 build.py       # compiles paper/paper.pdf (needs a LaTeX installation with pdflatex)
```

The paper text is in `paper/paper_src.tex`; every number in it comes from `paper/macros.tex` or `paper/tab/`,
which step 9 regenerates. To edit the paper in Overleaf, upload the `paper/` folder together with `output/fig/`
(or change the figure paths). Replace "[Group member ...]" on the title page with your names.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `validation.json` | Asset-growth deciles vs. lecture; own HML/momentum/market vs. official FF |
| `insample_sy10.json` | Hindsight mispricing sorts: decile returns, CAPM/FF3/FF5/FF5+MOM alphas, GRS, legs |
| `realtime_summary.csv` | Every version × window × gross/net: mean, Sharpe, alphas and t-stats for CAPM, FF3, FF3+MOM, FF5, FF5+MOM, q4, q5, turnover (% per month), costs (% per year) |
| `main_strategy_monthly.csv` | Main strategy month by month: leg returns, long–short gross/net, turnover, costs, market, RF |
| `main_strategy_deciles_vw.csv` | Value-weighted decile returns of the real-time score |
| `main_strategy_anomaly_weights.csv` | Which anomalies were in the score each month |
| `fm_slopes_by_anomaly.csv` | Monthly Fama–MacBeth slopes of each anomaly (used by `comb_pub`) |
| `paper_results.json` | Every number used in the paper (step 5) |
| `paper_main_series.csv` | Main strategy plus value- and equal-weighted decile returns, monthly |
| `fama_macbeth_monthly.csv` | Monthly Fama–MacBeth coefficients |
| `extra_checks.txt` / `.json` | Costs vs. Lecture 3 (slides 26-30), alternative strategies with the same rules, volatility scaling |
| `fig/*.pdf` | The five figures |

## Conventions

- Firm level = CRSP `PERMCO`: market equity summed over share classes, returns of the largest class.
- CIZ `MthRet` already includes delisting returns; a missing next-month return is treated as 0 (cash).
- Accounting data for fiscal years ending in calendar year y−1 are used from end-June y (the six-month lag).
- t-statistics are Newey–West with lag floor(4(T/100)^(2/9)).
- Alphas are monthly % in realtime_summary.csv; FF5+MOM = Fama–French five factors + momentum (MOM = Ken French's UMD).
- `lecture_table_main_gross.csv` / `lecture_table_main_net.csv`: the main strategy in the lecture format (cnst = alpha in % per year, loadings, t-stats, adjusted R², turnover/spread/impact).

## No look-ahead: how it is guaranteed and tested

- Every signal at the end of month t uses only data up to month t: returns up to t (momentum skips month t), accounting
  data with the six-month lag (fiscal years ending in year y-1 used from end-June y), ranks, NYSE breakpoints and the
  price filter computed from month-t data only. Weights drift with realised returns before the next rebalancing.
- An anomaly enters only from January after its publication year; equal weights (nothing is estimated).
- Factor models, the Lecture 3 cost levels and the hindsight version are used only to evaluate, never to choose trades.
- `07_lookahead_test.py` rebuilds the whole panel from CRSP/Compustat data that end in December 2010 and re-runs the
  strategy: results for 1997-2010 must be identical to the full run (they are; max difference ~1e-16).
- What the test cannot rule out: the anomaly list itself was chosen by Stambaugh-Yuan using data up to about 2010; the
  Compustat file may contain restated or back-filled data. 2014-2025 is the cleanest out-of-sample period.

## Known limitations

- Return on assets uses annual data (Stambaugh–Yuan use quarterly); failure probability (the 11th anomaly) needs daily data and is omitted.
- The CRSP extract has no bid/ask quotes, so costs use the Lecture 3 levels rather than measured spreads.
- Turnover is the Lecture 3 definition: sum of |weight changes| per month (buys + sells), with weights drifting with
  market capitalisation. `extra_checks.txt` replicates the lecture's momentum cost tables with this method, and reports
  the main strategy with weights drifting with the price return instead (net FF5+MOM alpha 0.36 vs 0.37).
- Short-selling (borrowing) fees are not in the main cost estimate; step 5 reports a fee grid and the break-even fee.
- Missing next-month returns of held stocks are set to 0 (held as cash; using -30% instead raises the alpha, see extra_checks.txt); a few accounting components (txditc, pstk, xrd,
  dlc, dltt, mib, che, invt, dp) are set to 0 when blank inside formulas. Missing signals are never set to 0.
>>>>>>> 41d92bf (Initial commit)
