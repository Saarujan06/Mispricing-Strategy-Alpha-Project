"""Run the whole project with one command:

    python3 main.py

What it does
  1. Checks that the Python packages, the WRDS data files and the factor files are in place.
  2. Runs every step in order (00 -> 09) with the same Python you used to start it.
     Step 00 (the slow CRSP filter, ~3-5 min) is skipped if it has already been run in full.
  3. Builds the paper PDF if LaTeX (pdflatex) is installed; otherwise it skips that part.

Options
  python3 main.py --from 4      start at step 4 (re-use everything before it)
  python3 main.py --only 5      run a single step
  python3 main.py --redo-prep   force step 00 to run again
  python3 main.py --no-paper    do not try to compile the PDF
"""
import argparse, importlib, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402

STEPS = [
    (0, '00_prep_crsp.py', 'Filter the raw CRSP file to US common stocks'),
    (1, '01_build_panel.py', 'Build the firm-month panel and all signals'),
    (2, '02_validate.py', 'Validate against the lecture and the official factors'),
    (3, '03_insample_mispricing.py', 'Hindsight benchmark (all anomalies from 1970)'),
    (4, '04_realtime_strategy.py', 'MAIN STRATEGY: real-time, costs, alphas'),
    (5, '05_paper_analysis.py', 'Extra analyses for the paper'),
    (6, '06_extra_checks.py', 'Extra checks: course-level costs, alternative strategies, design choices'),
    (7, '07_lookahead_test.py', 'Look-ahead test: rebuild from data ending 2010 and compare'),
    (8, '08_figures.py', 'Figures'),
    (9, '09_paper_tables.py', 'LaTeX tables and numbers for the paper'),
]
PREP_DONE = os.path.join(config.WORK, 'crsp_clean.done')   # written by step 00 after a full (not test) run


def line(char='-'):
    print(char * 78)


def preflight():
    ok = True
    print('Checking setup ...')
    print(f'  Python: {sys.version.split()[0]} ({sys.executable})')
    for pkg in ('numpy', 'pandas', 'scipy', 'matplotlib'):
        try:
            mod = importlib.import_module(pkg); print(f'  [ok] {pkg} {mod.__version__}')
        except ImportError:
            print(f'  [MISSING] {pkg}  ->  run:  {os.path.basename(sys.executable)} -m pip install {pkg}'); ok = False
    for label, path in (('CRSP monthly file', config.CRSP_RAW), ('Compustat file', config.COMP_RAW)):
        if os.path.exists(path): print(f'  [ok] {label}')
        elif path == config.CRSP_RAW and os.path.exists(PREP_DONE) and os.path.exists(config.CRSP_CLEAN):
            print(f'  [ok] {label} not found, but work/crsp_clean.csv.gz was already built from it (step 0 will be skipped)')
        else: print(f'  [MISSING] {label}: {path}\n            -> put the WRDS files there, or change RAW_DIR in config.py'); ok = False
    ff_official = all(os.path.exists(os.path.join(config.FACTORS, f)) for f in ('F-F_Research_Data_5_Factors_2x3.csv', 'F-F_Momentum_Factor.csv'))
    ff_mirror = os.path.exists(os.path.join(config.FACTORS, 'us_ff5_mom.csv.gz'))
    if ff_official: print('  [ok] Fama-French factors (official Ken French files)')
    elif ff_mirror: print('  [ok] Fama-French factors (included copy; official files optional)')
    else: print('  [MISSING] Fama-French factors in factors/'); ok = False
    import glob
    if glob.glob(os.path.join(config.FACTORS, 'q5_factors_monthly*.csv')): print('  [ok] q-factors (official, global-q.org) -> q4 and q5 will be estimated')
    else: print('  [note] no official q-factor file in factors/ -> q4 uses an approximation, q5 is skipped')
    return ok


def run_step(num, script, desc):
    line(); print(f'STEP {num}: {desc}   [{script}]'); line()
    t0 = time.time()
    res = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, script)], cwd=HERE)
    dt = time.time() - t0
    if res.returncode != 0:
        line('!'); print(f'STEP {num} FAILED after {dt:.0f}s ({script}).')
        print('Copy the error message printed just above and send it, then re-run from this step with:')
        print(f'    {os.path.basename(sys.executable)} main.py --from {num}'); line('!')
        sys.exit(res.returncode)
    print(f'--> step {num} done in {dt:.0f}s\n')


def build_paper():
    line(); print('PAPER: compiling paper/paper.pdf'); line()
    if not shutil.which('pdflatex'):
        print('pdflatex is not installed, so the PDF is not compiled here (that is fine).')
        print('All tables/figures are updated: upload the paper/ folder and output/fig/ to Overleaf to compile.')
        return
    res = subprocess.run([sys.executable, 'build.py'], cwd=os.path.join(HERE, 'paper'))
    print('paper/paper.pdf updated' if res.returncode == 0 else 'paper build failed (see messages above)')


def main():
    ap = argparse.ArgumentParser(description='Run the mispricing-strategy project end to end.')
    ap.add_argument('--from', dest='start', type=int, default=0, help='first step to run (0-9)')
    ap.add_argument('--only', type=int, default=None, help='run only this step')
    ap.add_argument('--redo-prep', action='store_true', help='force step 00 even if already done')
    ap.add_argument('--no-paper', action='store_true', help='skip compiling the PDF')
    a = ap.parse_args()

    line('='); print('MISPRICING STRATEGY - full run'); line('=')
    if not preflight():
        print('\nFix the [MISSING] items above, then run main.py again.'); sys.exit(1)
    print()
    steps = [s for s in STEPS if (s[0] == a.only if a.only is not None else s[0] >= a.start)]
    t0 = time.time()
    for num, script, desc in steps:
        if num == 0 and not a.redo_prep and a.only is None and os.path.exists(PREP_DONE) and os.path.exists(config.CRSP_CLEAN):
            print(f'STEP 0 skipped: {config.CRSP_CLEAN} already built from the full CRSP file (use --redo-prep to rebuild)\n')
            continue
        if num >= 1 and not os.path.exists(config.CRSP_CLEAN):
            print('work/crsp_clean.csv.gz is missing: run step 0 first (python3 main.py).'); sys.exit(1)
        run_step(num, script, desc)
    if not a.no_paper and a.only is None and 9 in [s[0] for s in steps]:
        build_paper()
    line('='); print(f'ALL DONE in {(time.time() - t0) / 60:.1f} minutes.')
    print('Results: output/realtime_summary.csv (main table), output/main_strategy_monthly.csv, output/extra_checks.txt, output/lookahead_test.txt, output/fig/, output/paper_results.json')
    line('=')


if __name__ == '__main__':
    main()
