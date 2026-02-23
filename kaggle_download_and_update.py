# kaggle_download_and_update.py
# Daily update script for Kaggle - downloads new data and updates parquet

import os
import sys
import json
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
import requests
from zipfile import ZipFile
import logging
from logging.handlers import RotatingFileHandler

# Kaggle API (if available)
try:
    import kaggle

    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "parquet_files"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "kaggle_update.log"

# NSE API endpoints
NSE_BASE_URL = "https://www1.nseindia.com"
BSE_BASE_URL = "https://www.bseindia.com"

# NSE F&O bhavcopy URL format
NSE_FO_URL = (
    "https://www1.nseindia.com/archives/fo/bhavcopy/prcs/fo{date_str}bhav.csv.zip"
)

# BSE F&O bhavcopy URL format
BSE_FO_URL = "https://www.bseindia.com/download/BhavCopy/Equity/{date_str}_Equity.zip"


class NSERecentDownloader:
    """Class to download recent NSE F&O bhavcopy data."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def download_bhavcopy(self, trade_date: date, dest_dir: Path) -> bool:
        """
        Download F&O bhavcopy for a specific date.

        Args:
            trade_date: Date for which to download data
            dest_dir: Directory where file should be saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Format date string
            date_str = trade_date.strftime("%d%m%Y")

            # Create URL
            url = NSE_FO_URL.format(date_str=date_str)

            # Download zip file
            response = self.session.get(url, stream=True, timeout=30)

            # Check if file exists (NSE returns 200 even for non-existent files)
            if "file not found" in response.text.lower():
                print(f"  No data for {trade_date}")
                return False

            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Save zip file
            zip_path = (
                dest_dir / f"fo{trade_date.strftime('%d%b%Y').upper()}bhav.csv.zip"
            )
            with open(zip_path, "wb") as f:
                f.write(response.content)

            print(f"  Downloaded: {zip_path.name}")
            return True

        except Exception as e:
            print(f"  Error downloading {trade_date}: {e}")
            return False


def create_empty_dataframe():
    """Create empty DataFrame with correct schema."""
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="str"),
            "expiry_dt": pd.Series(dtype="datetime64[ns]"),
            "strike_pr": pd.Series(dtype="float64"),
            "option_typ": pd.Series(dtype="str"),
            "instrument_type": pd.Series(dtype="str"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "last_price": pd.Series(dtype="float64"),
            "settle_pr": pd.Series(dtype="float64"),
            "prev_close": pd.Series(dtype="float64"),
            "underlying_price": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "turnover": pd.Series(dtype="float64"),
            "num_trades": pd.Series(dtype="int64"),
            "open_int": pd.Series(dtype="int64"),
            "chg_in_oi": pd.Series(dtype="int64"),
            "lot_size": pd.Series(dtype="int64"),
            "isin": pd.Series(dtype="str"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "business_date": pd.Series(dtype="datetime64[ns]"),
        }
    )


def get_current_year_file(dataset_files: list, current_year: int) -> str:
    """Get current year parquet file, creating if needed."""
    year_filename = f"fno_data_{current_year}.parquet"

    if year_filename not in dataset_files:
        print(f"\n✨ New year {current_year}! Creating new parquet file...")
        # Create empty DataFrame
        df = create_empty_dataframe()
        # Save to parquet
        df.to_parquet(
            year_filename, index=False, compression="snappy", engine="pyarrow"
        )
        return year_filename

    return year_filename


def download_from_kaggle(dataset_name: str, filename: str, dest_path: Path) -> bool:
    """Download a file from Kaggle dataset."""
    if not KAGGLE_AVAILABLE:
        print("Kaggle API not available")
        return False

    try:
        print(f"  Downloading {filename} from Kaggle...")
        kaggle.api.dataset_download_file(
                            'ravithejathaduri/nse-fno-database',
                            current_year_file,
                            path=str(PARQUET_DIR),
                            quiet=False
                        )
                return True
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return False


def upload_to_kaggle(dataset_name: str, file_path: Path) -> bool:
    """Upload file to Kaggle dataset."""
    if not KAGGLE_AVAILABLE:
        print("Kaggle API not available")
        return False

    try:
        print(f"  Uploading {file_path.name} to Kaggle...")
        kaggle.api.dataset_create_version(
            str(file_path), version_notes=f"Update {file_path.stem}", quiet=False
        )
        return True
    except Exception as e:
        print(f"  Error uploading {file_path.name}: {e}")
        return False


def extract_bhavcopy_data(zip_path: Path, trade_date: date) -> pd.DataFrame:
    """Extract data from bhavcopy zip file."""
    try:
        with ZipFile(zip_path, "r") as zip_ref:
            # Find CSV file
            csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
            if not csv_files:
                print(f"  No CSV file found in {zip_path.name}")
                return pd.DataFrame()

            csv_file = csv_files[0]

            # Read CSV into DataFrame
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f)

            # Rename columns to match our schema
            column_mapping = {
                "SYMBOL": "symbol",
                "EXPIRY_DT": "expiry_dt",
                "STRIKE_PR": "strike_pr",
                "OPTION_TYP": "option_typ",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "SETTLE_PR": "settle_pr",
                "CONTRACTS": "volume",
                "VAL_INLAKH": "turnover",
                "OPEN_INT": "open_int",
                "CHG_IN_OI": "chg_in_oi",
                "TIMESTAMP": "timestamp",
                "INSTRUMENT": "instrument_type",
                "LAST": "last_price",
                "PREVCLOSE": "prev_close",
                "NO_TRADES": "num_trades",
                "UNDERLYING": "underlying_price",
                "LOTSIZE": "lot_size",
                "ISIN": "isin",
                "DATE": "business_date",
            }

            df.rename(columns=column_mapping, inplace=True)

            # Keep only relevant columns
            df = df[list(column_mapping.values())]

            # Convert data types
            df["expiry_dt"] = pd.to_datetime(df["expiry_dt"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["business_date"] = pd.to_datetime(df["business_date"], errors="coerce")
            df["strike_pr"] = pd.to_numeric(df["strike_pr"], errors="coerce")
            df["volume"] = (
                pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
            )
            df["open_int"] = (
                pd.to_numeric(df["open_int"], errors="coerce").fillna(0).astype("int64")
            )
            df["chg_in_oi"] = (
                pd.to_numeric(df["chg_in_oi"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
            df["lot_size"] = (
                pd.to_numeric(df["lot_size"], errors="coerce").fillna(0).astype("int64")
            )
            df["num_trades"] = (
                pd.to_numeric(df["num_trades"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

            return df

    except Exception as e:
        print(f"  Error extracting data from {zip_path.name}: {e}")
        return pd.DataFrame()


def setup_logging():
    """Setup logging configuration."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Remove existing handlers
    logger.handlers.clear()

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_last_update_date(current_year_file: str) -> date:
    """Return the latest timestamp in the parquet file.

    If the parquet file does not exist **or** is empty (which happens on
    Jan 1 when we create a brand‑new empty file), the function returns ``None``.
    The caller will then treat this as "no existing data" and start the
    back‑fill from the first day of the year.
    """
    parquet_path = PARQUET_DIR / current_year_file
    # File may not exist yet (first run) or may be empty after creation on Jan 1.
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        # Empty DataFrame indicates a newly‑created parquet with no rows.
        if df.empty:
            return None
        latest_ts = df["timestamp"].max()
        if pd.isna(latest_ts):
            return None
        return latest_ts.date()
    except Exception as e:
        print(f"  Error reading parquet for last date: {e}")
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        if df.empty:
            return None
        latest_ts = df["timestamp"].max()
        if pd.isna(latest_ts):
            return None
        return latest_ts.date()
    except Exception as e:
        print(f"  Error reading parquet for last date: {e}")
        return None


def get_missing_dates(start_date: date, end_date: date) -> list:
    """Return a list of weekdays between start_date and end_date inclusive."""
    missing = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday
            missing.append(current)
        current += timedelta(days=1)
    return missing


def download_missing_data(missing_dates: list) -> pd.DataFrame:
    """Download bhavcopy for each missing date and concatenate into a DataFrame."""
    downloader = NSERecentDownloader()
    all_frames = []
    for trade_date in missing_dates:
        print(f"  Downloading missing data for {trade_date}...")
        success = downloader.download_bhavcopy(trade_date, DATA_DIR)
        if not success:
            print(f"    No data for {trade_date} (holiday/weekend?)")
            continue
        zip_path = DATA_DIR / f"fo{trade_date.strftime('%d%b%Y').upper()}bhav.csv.zip"
        if not zip_path.exists():
            print(f"    Zip not found after download: {zip_path}")
            continue
        df = extract_bhavcopy_data(zip_path, trade_date)
        if not df.empty:
            all_frames.append(df)
        # Clean up zip file to save space
        try:
            zip_path.unlink()
        except Exception:
            pass
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """Main function for daily update with incremental missing‑date handling."""
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("KAGGLE DAILY UPDATE STARTED")
    logger.info("=" * 70)

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    current_year = today.year
    current_year_file = f"fno_data_{current_year}.parquet"

    logger.info(f"Today: {today}")
    logger.info(f"Current year: {current_year}")

    # On Jan 1 create an empty parquet (if not already present) and process today directly
    if today.month == 1 and today.day == 1:
        # Ensure parquet file exists (empty if new)
        if not (PARQUET_DIR / current_year_file).exists():
            logger.info("✨ Happy New Year! Creating empty parquet for the new year...")
            create_empty_dataframe().to_parquet(
                PARQUET_DIR / current_year_file,
                index=False,
                compression="snappy",
                engine="pyarrow",
            )
            logger.info(f"  Created: {current_year_file}")
        # Directly download today's bhavcopy (no back‑fill needed)
        logger.info("[STEP 1] Downloading today's bhavcopy (Jan 1 special case)...")
        downloader = NSERecentDownloader()
        if downloader.download_bhavcopy(today, DATA_DIR):
            zip_path = DATA_DIR / f"fo{today.strftime('%d%b%Y').upper()}bhav.csv.zip"
            new_df = extract_bhavcopy_data(zip_path, today)
            if not new_df.empty:
                # Overwrite (or append) the parquet for the new year
                new_df.to_parquet(
                    PARQUET_DIR / current_year_file,
                    index=False,
                    compression="snappy",
                    engine="pyarrow",
                )
                logger.info("  Saved today's data to parquet.")
                # Upload to Kaggle if possible
                if KAGGLE_AVAILABLE:
                    logger.info("Uploading updated parquet to Kaggle...")
                    try:
                        kaggle.api.dataset_create_version(
                            str(PARQUET_DIR / current_year_file),
                            version_notes=f"Jan 1 update {today}",
                            quiet=False,
                        )
                        logger.info("  Upload successful.")
                    except Exception as e:
                        logger.error(f"  Error uploading to Kaggle: {e}")
                else:
                    logger.info("Kaggle API not available – skipping upload.")
            else:
                logger.warning("  No data extracted for Jan 1.")
        else:
            logger.warning("  Failed to download bhavcopy for Jan 1.")
        logger.info("\n" + "=" * 70)
        logger.info("DAILY UPDATE COMPLETED (Jan 1 special case)")
        logger.info("=" * 70)
        return

    # Determine last date present in the parquet (if any)
    last_date = get_last_update_date(current_year_file)
    if last_date:
        logger.info(f"Last date in parquet: {last_date}")
    else:
        logger.info(
            "No existing data in parquet (or file missing). Starting from Jan 1."
        )
        # If there's no data, start from Jan 1 of current year
        last_date = date(current_year, 1, 1) - timedelta(days=1)

    # Compute missing dates (from day after last_date up to today)
    start_missing = last_date + timedelta(days=1)
    if start_missing > today:
        logger.info("Parquet already up‑to‑date. No download needed.")
        return
    missing_dates = get_missing_dates(start_missing, today)
    logger.info(f"Missing dates to download: {len(missing_dates)}")

    # Download missing data
    new_data = download_missing_data(missing_dates)
    if new_data.empty:
        logger.info("No new data downloaded – likely weekend/holiday only.")
        logger.info("DAILY UPDATE COMPLETED – No changes applied")
        return

    # Load existing parquet (if present) and append new data
    if (PARQUET_DIR / current_year_file).exists():
        existing_data = pd.read_parquet(PARQUET_DIR / current_year_file)
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        logger.info(
            f"Existing rows: {len(existing_data):,}, new rows: {len(new_data):,}, combined rows: {len(combined):,}"
        )
    else:
        combined = new_data
        logger.info(f"Created new parquet with {len(new_data):,} rows")

    # Save updated parquet
    combined.to_parquet(
        PARQUET_DIR / current_year_file,
        index=False,
        compression="snappy",
        engine="pyarrow",
    )
    logger.info(f"Saved updated parquet: {current_year_file}")

    # Upload to Kaggle if possible
    if KAGGLE_AVAILABLE:
        logger.info("Uploading updated parquet to Kaggle...")
        try:
            kaggle.api.dataset_create_version(
                str(PARQUET_DIR / current_year_file),
                version_notes=f"Update for {today}",
                quiet=False,
            )
            logger.info("Upload successful.")
        except Exception as e:
            logger.error(f"Error uploading to Kaggle: {e}")
    else:
        logger.info("Kaggle API not available – skipping upload.")

    logger.info("\n" + "=" * 70)
    logger.info("DAILY UPDATE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Updated file: {current_year_file}")
    logger.info(f"Date processed: {today}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
