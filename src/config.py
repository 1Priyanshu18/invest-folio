from pathlib import Path

# PATHS
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT_DIR / "data" / "raw"
DATA_PROC     = ROOT_DIR / "data" / "processed"
OUTPUTS       = ROOT_DIR / "outputs"
FIGURES       = OUTPUTS / "figures"
MODELS        = OUTPUTS / "models"

# Special files inside data/raw
METADATA_FILE = DATA_RAW / "stock_metadata.csv"
COMBINED_FILE = DATA_RAW / "NIFTY50_all.csv"

# Ensure output dirs exist at import time
for _d in (DATA_PROC, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.065
PRICE_COL   = "Close"       # primary price for returns
VWAP_COL    = "VWAP"        # volume-weighted price (cleaner intraday avg)
VOLUME_COL  = "Volume"
DELIV_COL   = "%Deliverble" # delivery fraction — conviction signal

# Files in data/raw that are NOT individual stocks
NON_STOCK_FILES = {"stock_metadata.csv", "NIFTY50_all.csv"}

RANDOM_SEED = 42
TEST_SIZE   = 0.2