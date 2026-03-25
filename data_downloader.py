"""
data_downloader.py
──────────────────
Professional local-data layer for NSE Sentinel.

• Downloads 6-month daily OHLCV from yfinance → saves as CSV per ticker
  in /data/ folder relative to this file.
• update_data_if_old(tickers)  — refreshes CSVs older than 1 day.
• load_csv(ticker)            — load a single ticker from CSV.
• bulk_download(tickers)      — parallel download with progress reporting.

Strategy Layer Integration
──────────────────────────
`strategy_engines/_engine_utils.download_history()` calls `load_csv()`
first; falls back to live yfinance only when CSV is absent or corrupt.
"""

from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

_FILE_LOCK = threading.Lock()

import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ──────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_DIR   = _HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Concurrency controls ───────────────────────────────────────────────
_DOWNLOAD_WORKERS = 10          # max parallel yfinance threads
_MAX_STALENESS_H  = 24          # re-download if CSV older than N hours
_MIN_ROWS         = 30          # minimum acceptable rows in a CSV

# ── Progress callback type ─────────────────────────────────────────────
_ProgressCB = None              # set externally to a callable(done, total, found)


# ═══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _csv_path(ticker_ns: str) -> Path:
    """Return /data/<TICKER_NS>.csv path."""
    safe = ticker_ns.replace(":", "_").replace("/", "_")
    return DATA_DIR / f"{safe}.csv"


def _csv_age_hours(ticker_ns: str) -> float:
    """Return age of CSV in hours, or ∞ if missing."""
    p = _csv_path(ticker_ns)
    if not p.exists():
        return float("inf")
    mtime = p.stat().st_mtime
    return (time.time() - mtime) / 3600.0


def load_csv(ticker_ns: str) -> pd.DataFrame | None:
    """
    Load ticker CSV from /data/.
    Returns cleaned DataFrame (≥ MIN_ROWS rows) or None.
    Handles MultiIndex columns if somehow stored.
    """
    p = _csv_path(ticker_ns)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Normalise column names — yfinance sometimes changes capitalisation
        df.columns = [c.strip().title() for c in df.columns]
        # Ensure required columns exist
        required = {"Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return None
        df = df.dropna(subset=["Close", "Volume"])
        return df if len(df) >= _MIN_ROWS else None
    except Exception:
        return None


def _download_one(ticker_ns: str, period: str = "6mo", force: bool = False) -> tuple[pd.DataFrame | None, str]:
    """
    Download one ticker from yfinance incrementally and save to CSV.
    Returns (cleaned DataFrame or None, status string ["updated", "skipped", "failed"]).
    """
    try:
        old_df = load_csv(ticker_ns)
        today = datetime.now()
        
        if old_df is not None and not old_df.empty:
            last_date = pd.to_datetime(old_df.index.max())
            
            # 1. Market-Aware Skip Logic
            if not force:
                # If last recorded date is on or after today's date
                if last_date.date() >= today.date():
                    return old_df, "skipped"
                # If today is weekend (Sat/Sun) and last_date is Friday, it's fresh
                if today.weekday() >= 5 and last_date.weekday() == 4 and (today.date() - last_date.date()).days <= 3:
                    return old_df, "skipped"

            # 4. Use period="5d" (Optional Optimization implemented)
            new_df = yf.download(
                ticker_ns,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
                threads=False,
            )
            time.sleep(0.1) # 8. Small delay to avoid API block
        else:
            new_df = yf.download(
                ticker_ns,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
                threads=False,
            )
            time.sleep(0.1)

        if new_df is None or new_df.empty:
            return old_df, "failed"

        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)
            
        new_df.columns = [c.strip().title() for c in new_df.columns]
        
        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        avail_cols = [c for c in req_cols if c in new_df.columns]
        new_df = new_df[avail_cols]
        new_df = new_df.dropna(how="all")
        new_df = new_df.dropna(subset=[c for c in ["Close", "Volume"] if c in avail_cols])
        
        if old_df is not None and not old_df.empty:
            avail_old = [c for c in req_cols if c in old_df.columns]
            old_df = old_df[avail_old]
            df = pd.concat([old_df, new_df])
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = new_df

        if len(df) < _MIN_ROWS:
            return None, "failed"

        df = df.sort_index(ascending=True)
        df = df[[c for c in req_cols if c in df.columns]]
        
        # 3. File Lock added
        with _FILE_LOCK:
            df.to_csv(_csv_path(ticker_ns))
            
        return df, "updated"

    except Exception as e:
        # 7. Do NOT crash, return old if failed
        return load_csv(ticker_ns), "failed"


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def bulk_download(
    tickers: list[str],
    period: str = "6mo",
    force: bool = False,
    print_progress: bool = True,
    progress_callback = None
) -> dict[str, int]:
    """
    Download `tickers` in parallel (up to _DOWNLOAD_WORKERS threads).
    Returns a dict with tracked states: updated, skipped, failed.
    """
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    
    # Track stats
    stats = {"updated": 0, "skipped": 0, "failed": 0}

    if print_progress:
        print(f"[DataDownloader] Updating {len(tickers_ns)} tickers "
              f"(workers={_DOWNLOAD_WORKERS}) …")

    done = 0
    total = len(tickers_ns)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(_download_one, t, period, force): t for t in tickers_ns}
        for fut in as_completed(futs):
            ticker = futs[fut]
            done += 1
            try:
                df, status = fut.result()
                stats[status] += 1
            except Exception:
                stats["failed"] += 1
                
            if progress_callback:
                progress_callback(done, total)
                
            if print_progress and done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 1
                eta = (total - done) / rate
                print(f"  [{done:4d}/{total}] Upd:{stats['updated']} Skip:{stats['skipped']} Fail:{stats['failed']}  ETA {eta:.0f}s")

    if stats["failed"] > 0 and print_progress:
        print(f"⚠️ Partial update — {stats['failed']} tickers failed. Data may be inconsistent.")
    
    return stats


def update_all_data(tickers: list[str], period: str = "6mo") -> dict:
    """
    Download/refresh CSVs for all tickers in parallel.
    Returns {updated, skipped, failed} counts.
    No UI interaction — safe to call from any context.
    """
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    stats = {"updated": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_one, t, period): t for t in tickers_ns}
        for future in as_completed(futures):
            try:
                _, status = future.result()
            except Exception:
                status = "failed"
            stats[status] = stats.get(status, 0) + 1

    return stats


def update_data_if_old(
    tickers: list[str],
    max_age_hours: float = _MAX_STALENESS_H,
    period: str = "6mo",
    print_progress: bool = True,
) -> int:
    """
    Legacy convenience wrapper — returns count of updated tickers.
    """
    results = update_all_data(tickers, period=period)
    return results["updated"]


def data_status_summary(tickers: list[str]) -> dict:
    """
    Return a summary dict with counts of fresh / stale / missing CSVs.
    Useful for sidebar display in Streamlit.
    """
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    fresh  = sum(1 for t in tickers_ns if _csv_age_hours(t) <= _MAX_STALENESS_H)
    stale  = sum(1 for t in tickers_ns if _MAX_STALENESS_H < _csv_age_hours(t) < float("inf"))
    missing = sum(1 for t in tickers_ns if _csv_age_hours(t) == float("inf"))
    oldest = max((_csv_age_hours(t) for t in tickers_ns if _csv_age_hours(t) < float("inf")),
                 default=None)
    return {
        "total":   len(tickers_ns),
        "fresh":   fresh,
        "stale":   stale,
        "missing": missing,
        "oldest_h": round(oldest, 1) if oldest is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT  (python data_downloader.py)
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Default top-50 NSE universe for quick bootstrap
    UNIVERSE = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "AXISBANK.NS",
        "TATAMOTORS.NS", "MARUTI.NS", "LT.NS", "NTPC.NS", "ADANIPORTS.NS",
        "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "ONGC.NS", "POWERGRID.NS",
        "BHARTIARTL.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "HEROMOTOCO.NS",
        "BAJAJ-AUTO.NS", "EICHERMOT.NS", "M&M.NS", "TATACONSUM.NS", "BRITANNIA.NS",
        "TECHM.NS", "INDUSINDBK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS", "GRASIM.NS",
        "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "SUNPHARMA.NS", "APOLLOHOSP.NS",
        "ITC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "VEDL.NS",
        "ZOMATO.NS", "NAUKRI.NS", "IRCTC.NS", "DMART.NS", "TRENT.NS",
    ]

    force_flag = "--force" in sys.argv
    print(f"NSE Data Downloader — {len(UNIVERSE)} tickers, force={force_flag}")
    bulk_download(UNIVERSE, period="6mo", force=force_flag, print_progress=True)