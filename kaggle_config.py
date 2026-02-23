# config.py
# Configuration for Kaggle deployment

from pathlib import Path
from datetime import date

# --- Core Project Paths ---

# The root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the directory where parquet files will be stored.
PARQUET_DIR = PROJECT_ROOT / "parquet_files"

# Path to the directory where raw bhavcopy files will be stored.
DATA_DIR = PROJECT_ROOT / "data"

# --- Logging Configuration ---

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

# --- Kaggle Configuration ---

# Kaggle dataset name (replace with your username and dataset name)
KAGGLE_DATASET_NAME = "your-username/nse-fno-data"

# NSE API endpoints
NSE_BASE_URL = "https://www1.nseindia.com"

# F&O bhavcopy URL format
NSE_FO_URL = (
    "https://www1.nseindia.com/archives/fo/bhavcopy/prcs/fo{date_str}bhav.csv.zip"
)

# --- Data Processing Configuration ---

# Number of years to keep in parquet (for cleanup if needed)
MAX_YEARS_TO_KEEP = 10

# Compression for parquet files
PARQUET_COMPRESSION = "snappy"


# --- A function to print the configuration for verification ---
def print_config():
    """Prints the current configuration settings."""
    print("--- Kaggle Configuration ---")
    print(f"Project Root:      {PROJECT_ROOT}")
    print(f"Parquet Directory: {PARQUET_DIR}")
    print(f"Data Directory:    {DATA_DIR}")
    print(f"Log Directory:     {LOG_DIR}")
    print(f"Kaggle Dataset:    {KAGGLE_DATASET_NAME}")
    print("---------------------------")


if __name__ == "__main__":
    # If you run this file directly (e.g., `python config.py`), it will print the settings.
    print_config()
