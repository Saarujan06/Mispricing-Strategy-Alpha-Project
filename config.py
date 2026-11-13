"""Paths used by every script. Edit RAW_DIR if your data folder is somewhere else
(or set the environment variable RAW_DIR before running)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Folder that holds the three raw WRDS files:
#   CRSP_MonthlyDataStock_1968-2025.csv, Compustat_Fundamental_Annuals_1968-2025.csv, CRSP_Compustat link data.csv
RAW_DIR = os.environ.get('RAW_DIR', os.path.expanduser('~/Downloads/Compustat_Fundamental_Annuals_1968-2025'))
CRSP_RAW = os.path.join(RAW_DIR, 'CRSP_MonthlyDataStock_1968-2025.csv')
COMP_RAW = os.path.join(RAW_DIR, 'Compustat_Fundamental_Annuals_1968-2025.csv')

WORK = os.environ.get('WORK_DIR', os.path.join(HERE, 'work'))   # intermediate files (cleaned CRSP, panel)
OUT = os.environ.get('OUT_DIR', os.path.join(HERE, 'output'))    # tables and monthly return series
FACTORS = os.path.join(HERE, 'factors')  # Fama-French factors
CRSP_CLEAN = os.path.join(WORK, 'crsp_clean.csv.gz')
PANEL = os.path.join(WORK, 'panel.npz')
for d in (WORK, OUT, FACTORS):
    os.makedirs(d, exist_ok=True)
