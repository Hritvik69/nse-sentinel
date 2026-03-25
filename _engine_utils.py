"""
strategy_engines/_engine_utils.py
──────────────────────────────────
Shared low-level helpers: EMA, RSI (vectorised), yfinance download, safe cast.
Also owns the central ALL_DATA store and preload helpers for zero-API scanning.
Imported by every mode engine — kept minimal and side-effect-free.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

_MAX_CONC = 6                               # conservative per-engine concurrency
_SEM      = threading.BoundedSemaphore(_MAX_CONC)

# ── Central data store (zero-API scan) ───────────────────────────────
ALL_DATA: dict[str, pd.DataFrame | None] = {}
_ALL_DATA_LOCK = threading.Lock()

# ── optional sklearn ──────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def safe(v: object, default: float = 0.0) -> float:
    """Return float(v) if finite, else default."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi_vec(close: pd.Series, period: int = 14) -> pd.Series:
    """Fully vectorised RSI series — no per-row Python loop."""
    d = close.diff()
    g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def download_history(ticker_ns: str, period: str = "6mo") -> pd.DataFrame | None:
    """Download daily OHLCV; returns None on failure or if < 30 rows."""
    try:
        # Check ALL_DATA first — zero-API
        df_cached = ALL_DATA.get(ticker_ns)
        if df_cached is not None and len(df_cached) >= 30:
            return df_cached

        with _SEM:
            df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12, threads=False,
            )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "Volume"])
        return df if len(df) >= 30 else None
    except Exception:
        return None


def _fetch_one(ticker_ns: str, period: str) -> tuple[str, pd.DataFrame | None]:
    """Download one ticker; return (ticker, df_or_None)."""
    # Try local CSV first (via data_downloader if available)
    try:
        from data_downloader import load_csv
        df = load_csv(ticker_ns)
        if df is not None and len(df) >= 30:
            return ticker_ns, df
    except Exception:
        pass

    # Fall back to yfinance
    try:
        with _SEM:
            df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12, threads=False,
            )
        if df is None or df.empty:
            return ticker_ns, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        df = df.dropna(subset=["Close", "Volume"])
        return ticker_ns, (df if len(df) >= 30 else None)
    except Exception:
        return ticker_ns, None


def preload_all(
    tickers: list[str],
    period: str = "6mo",
    workers: int = 12,
) -> None:
    """
    Fill ALL_DATA with OHLCV DataFrames for every ticker in parallel.
    Called once before run_scan() so analyse() can use get_df_for_ticker().
    """
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, t, period): t for t in tickers_ns}
        for fut in as_completed(futs):
            try:
                ticker_ns, df = fut.result()
                with _ALL_DATA_LOCK:
                    ALL_DATA[ticker_ns] = df
            except Exception:
                pass


def preload_history_batch(
    tickers: list[str],
    period: str = "6mo",
    workers: int = 12,
) -> None:
    """Alias for preload_all — kept for backward compatibility."""
    preload_all(tickers, period=period, workers=workers)


def get_df_for_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Return the pre-loaded DataFrame for a ticker from ALL_DATA.
    If not found, attempts a live download as fallback.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    df = ALL_DATA.get(ticker_ns)
    if df is not None:
        return df
    # fallback: live fetch (only if not preloaded)
    return download_history(ticker_ns)