"""
strategy_engines/_engine_utils.py
──────────────────────────────────
Shared low-level helpers: EMA, RSI (vectorised), yfinance download, safe cast.
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
    """Load one ticker for preloading; prefer CSV if available."""
    try:
        from data_downloader import load_csv
        df = load_csv(ticker_ns)
        if df is not None and len(df) >= 30:
            return ticker_ns, df
    except Exception:
        pass
    return ticker_ns, download_history(ticker_ns, period=period)


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
    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, t, period): t for t in tickers_ns}
        for fut in as_completed(futs):
            try:
                ticker_ns, df = fut.result()
                with _ALL_DATA_LOCK:
                    ALL_DATA[ticker_ns] = df
            except Exception:
                continue


def preload_history_batch(
    tickers: list[str],
    period: str = "6mo",
    workers: int = 12,
) -> None:
    """Back-compat alias for preload_all()."""
    preload_all(tickers, period=period, workers=workers)


def get_df_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Return preloaded DF for a ticker, with fallback to live download."""
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    with _ALL_DATA_LOCK:
        df = ALL_DATA.get(ticker_ns)
    if df is not None:
        return df
    return download_history(ticker_ns, period="6mo")


def add_rank_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Top-Ranked High-Probability scoring columns (trend/momentum/volume/near-high/rsi).
    Fail-safe: on any unexpected error, returns the original df unchanged.
    """
    try:
        if df is None or df.empty:
            return df

        out = df.copy()

        # Requested columns (new-add-on layer)
        out["trend_score"] = 0.0
        out["momentum_score"] = 0.0
        out["volume_score"] = 0.0
        out["near_high_score"] = 0.0
        out["rsi_score"] = 0.0
        out["rank_score"] = 0.0

        # Cache per symbol to avoid duplicate downloads.
        _df_cache: dict[str, pd.DataFrame | None] = {}

        def _get_df_cached(sym: str) -> pd.DataFrame | None:
            if sym in _df_cache:
                return _df_cache[sym]
            _df_cache[sym] = get_df_for_ticker(sym)
            return _df_cache[sym]

        for i, row in out.iterrows():
            sym = row.get("Symbol", None) or row.get("Ticker", None) or ""
            sym = str(sym).strip()
            if not sym:
                continue

            price = safe(row.get("Price (₹)", 0.0), 0.0)
            rsi_v = safe(row.get("RSI", 50.0), 50.0)
            vol_r = safe(row.get("Vol / Avg", 1.0), 1.0)
            d_ema20 = safe(row.get("Δ vs EMA20 (%)", 0.0), 0.0)
            r5d = safe(row.get("5D Return (%)", 0.0), 0.0)
            d20h = safe(row.get("Δ vs 20D High (%)", -5.0), -5.0)

            # Momentum (5D) score: based on 5D return (%)
            momentum_score = float(np.clip(50.0 + r5d * 3.0, 0.0, 100.0))

            # Volume score: normalized volume ratio (cap at 3.5x)
            vol_clip = float(np.clip(vol_r, 0.0, 3.5))
            volume_score = float((vol_clip / 3.5) * 100.0) if 3.5 > 0 else 0.0

            # RSI score: peak near ~60, penalize distance.
            rsi_score = float(np.clip(100.0 - abs(rsi_v - 60.0) * 4.0, 0.0, 100.0))

            trend_score = np.nan
            near_high_score = np.nan

            df_h = None
            try:
                df_h = _get_df_cached(sym)
            except Exception:
                df_h = None

            # Compute 60d trend + rolling-high proximity when OHLCV is available.
            try:
                if df_h is not None and isinstance(df_h, pd.DataFrame) and len(df_h) >= 30:
                    close_s = df_h["Close"].dropna() if "Close" in df_h.columns else None
                    high_s = df_h["High"].dropna() if "High" in df_h.columns else None

                    # trend_score (60d): fraction of last 60 days trading above EMA20 + 60d return component
                    if close_s is not None and len(close_s) >= 60:
                        tail = close_s.tail(60)
                        e20s = ema(close_s, 20)
                        e20_tail = e20s.reindex(tail.index)
                        above_ratio = float((tail > e20_tail).mean()) if len(tail) > 0 else 0.0
                        close60 = safe(close_s.iloc[-60], 0.0)
                        ret60 = (float(close_s.iloc[-1]) / close60 - 1.0) * 100.0 if close60 > 0 else 0.0
                        ret_comp = float(np.clip(50.0 + ret60 * 2.0, 0.0, 100.0))
                        trend_score = float(np.clip(0.6 * (above_ratio * 100.0) + 0.4 * ret_comp, 0.0, 100.0))

                    # near_high_score: proximity to rolling 20d high using High series
                    if high_s is not None and len(high_s) >= 20 and price > 0:
                        roll_high = safe(high_s.tail(20).max(), 0.0)
                        if roll_high > 0:
                            near_ratio = float(price) / float(roll_high)
                            near_high_score = float(np.clip((near_ratio - 0.95) / 0.10 * 100.0, 0.0, 100.0))
            except Exception:
                pass

            # Fail-safe fallbacks (use already-computed row fields)
            try:
                if not np.isfinite(trend_score):
                    trend_score = float(np.clip(50.0 + d_ema20 * 2.5, 0.0, 100.0))
            except Exception:
                trend_score = float(0.0)

            try:
                if not np.isfinite(near_high_score):
                    near_high_score = float(np.clip(50.0 + d20h * 4.0, 0.0, 100.0))
            except Exception:
                near_high_score = float(0.0)

            rank_score = float(
                np.clip(
                    0.25 * float(trend_score)
                    + 0.25 * float(momentum_score)
                    + 0.20 * float(volume_score)
                    + 0.15 * float(near_high_score)
                    + 0.15 * float(rsi_score),
                    0.0,
                    100.0,
                )
            )

            out.at[i, "trend_score"] = round(float(trend_score), 2)
            out.at[i, "momentum_score"] = round(float(momentum_score), 2)
            out.at[i, "volume_score"] = round(float(volume_score), 2)
            out.at[i, "near_high_score"] = round(float(near_high_score), 2)
            out.at[i, "rsi_score"] = round(float(rsi_score), 2)
            out.at[i, "rank_score"] = round(float(rank_score), 2)

        return out
    except Exception:
        pass

    return df
