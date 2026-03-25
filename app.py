"""
NSE Sentinel — Production-Ready Streamlit App  (Enhanced Edition)
Dark terminal aesthetic | Multi-strategy scanner | 1000+ NSE stocks
Run: streamlit run app.py

CHANGES vs original:
  • Scoring layer  (compute_score)         — added AFTER scan, never touches filters
  • Backtest prob  (compute_backtest_prob)  — added AFTER scan
  • ML probability (train_model_once /
                    predict_ml_probability) — added AFTER scan
  • Final rank     (enhance_results)        — combines the three
  • Bull-trap warning                       — display only
  • Top Picks section                       — display only
  • All existing analyse() / run_scan()
    logic is 100 % untouched.
"""

from __future__ import annotations

import io
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from strategy_engines import (
    get_engine_functions,
    preload_all,
    backtest_with_preloaded,
    get_df_for_ticker,
)
from strategy_engines._engine_utils import preload_history_batch

warnings.filterwarnings("ignore")

# ── TradingView symbol helper ─────────────────────────────────────────
def tv_symbol(ticker: str) -> str:
    """
    Convert yfinance NSE ticker to TradingView symbol.
    e.g. RELIANCE.NS → NSE:RELIANCE
    """
    return "NSE:" + ticker.replace(".NS", "")


def tv_chart_url(symbol_no_ns: str) -> str:
    """Return TradingView chart URL for a bare symbol (no .NS suffix)."""
    return f"https://www.tradingview.com/chart/?symbol=NSE:{symbol_no_ns}"


# ── Data downloader (optional, graceful if missing) ───────────────────
try:
    from data_downloader import (
        update_data_if_old,
        update_all_data,
        data_status_summary,
        bulk_download,
        load_csv,
    )
    _DATA_DOWNLOADER_OK = True
except ImportError:
    _DATA_DOWNLOADER_OK = False

    def update_data_if_old(tickers, **kw):  # type: ignore[misc]
        return 0

    def update_all_data(tickers, **kw):  # type: ignore[misc]
        return {"updated": 0, "skipped": 0, "failed": 0}

    def data_status_summary(tickers):  # type: ignore[misc]
        return {}

    def bulk_download(tickers, **kw):  # type: ignore[misc]
        return {}

    def load_csv(ticker):  # type: ignore[misc]
        return None

# ── optional sklearn (graceful fallback if missing) ───────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# NOTE:
# External module imports (scoring_engine/backtest_engine/ml_engine/ui_components)
# intentionally removed. Mode-specific strategy engines are used directly.

# ─────────────────────────────────────────────────────────────────────
# YFINANCE THROTTLING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────
MAX_YF_CONCURRENCY = 12
_YF_SEM = threading.BoundedSemaphore(MAX_YF_CONCURRENCY)

_MKT_LOCK   = threading.Lock()
_MKT_CACHE: dict[str, float] = {}
_NIFTY_LOCK = threading.Lock()
_NIFTY_20D_RET: float | None = None


def get_mktcap_cr(ticker: str) -> float:
    """DO NOT MODIFY — strategy rule"""
    with _MKT_LOCK:
        if ticker in _MKT_CACHE:
            return _MKT_CACHE[ticker]
    try:
        with _YF_SEM:
            info = yf.Ticker(ticker).fast_info
            raw  = getattr(info, "market_cap", 0) or 0
    except Exception:
        raw = 0
    mc_cr = float(raw) / 1e7 if raw else 0.0
    with _MKT_LOCK:
        _MKT_CACHE[ticker] = mc_cr
    return mc_cr


def get_nifty_20d_return() -> float | None:
    """20-day return for Nifty (^NSEI), shared across all stocks."""
    global _NIFTY_20D_RET
    with _NIFTY_LOCK:
        if _NIFTY_20D_RET is not None:
            return _NIFTY_20D_RET
    try:
        with _YF_SEM:
            df_n = yf.download(
                "^NSEI", period="2mo", interval="1d",
                auto_adjust=True, progress=False, timeout=10, threads=False,
            )
        if df_n is None or len(df_n) < 21:
            return None
        close_n = df_n["Close"].dropna()
        if len(close_n) < 21:
            return None
        n_today = float(close_n.iloc[-1])
        n_ago20 = float(close_n.iloc[-21])
        if n_ago20 <= 0:
            return None
        ret = (n_today - n_ago20) / n_ago20
    except Exception:
        return None
    with _NIFTY_LOCK:
        _NIFTY_20D_RET = ret
    return ret


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Sentinel — Stock Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  (Space Mono + Syne, terminal/Bloomberg aesthetic)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #060a0f; --bg2: #0b1017; --bg3: #0f1823;
  --border: #1a2840; --border2: #243550;
  --accent: #00d4a8; --accent2: #0094ff; --accent3: #f0b429;
  --red: #ff4d6d; --text: #ccd9e8; --muted: #4a6480;
  --mono: 'Space Mono', monospace; --sans: 'Syne', sans-serif;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--mono) !important; }
.stApp::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px); }
.stApp::after { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(0,212,168,0.018) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,168,0.018) 1px,transparent 1px);
  background-size:40px 40px; }
[data-testid="stSidebar"] { background-color:var(--bg2) !important; border-right:1px solid var(--border) !important; font-family:var(--mono) !important; }
[data-testid="stSidebar"] * { color:var(--text) !important; }
section[data-testid="stSidebar"] > div { padding-top:20px !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; font-family:var(--mono) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:var(--accent) !important; box-shadow:0 0 8px var(--accent) !important; }
[data-testid="stMetric"] { background:var(--bg2) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:18px 20px !important; }
[data-testid="stMetricValue"] { font-family:var(--sans) !important; font-weight:800 !important; font-size:2rem !important; color:var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family:var(--sans) !important; font-size:10px !important; letter-spacing:2px !important; text-transform:uppercase !important; color:var(--muted) !important; }
[data-testid="stMetricDelta"] { color:var(--accent3) !important; }
.stButton > button { background:transparent !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:var(--mono) !important; font-size:13px !important; font-weight:700 !important; letter-spacing:1px !important; border-radius:10px !important; padding:14px 28px !important; width:100% !important; transition:all 0.2s !important; }
.stButton > button:hover { background:var(--accent) !important; color:var(--bg) !important; }
.stButton > button:disabled { border-color:var(--muted) !important; color:var(--muted) !important; }
.stDownloadButton > button { background:rgba(0,148,255,0.1) !important; border:1px solid var(--accent2) !important; color:var(--accent2) !important; font-family:var(--mono) !important; font-weight:700 !important; border-radius:8px !important; width:100% !important; }
.stDownloadButton > button:hover { background:var(--accent2) !important; color:var(--bg) !important; }
.stProgress > div > div { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; box-shadow:0 0 10px var(--accent) !important; }
.stProgress > div { background:var(--border) !important; border-radius:3px !important; height:6px !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:12px !important; overflow:hidden !important; font-family:var(--mono) !important; }
.stDataFrame thead tr th { background:var(--bg3) !important; color:var(--muted) !important; font-family:var(--sans) !important; font-size:10px !important; letter-spacing:1.5px !important; text-transform:uppercase !important; }
.stDataFrame tbody tr:hover td { background:rgba(0,212,168,0.04) !important; }
.stAlert { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:10px !important; font-family:var(--mono) !important; }
h1,h2,h3,h4 { font-family:var(--sans) !important; }
h1 { color:var(--accent) !important; font-weight:800 !important; }
h2,h3 { color:#79c0ff !important; font-weight:700 !important; }
hr { border-color:var(--border) !important; }

@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
.live-dot { width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);animation:pdot 2s ease infinite;display:inline-block;margin-right:8px; }
.section-lbl { font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border); }
.mode-pill { display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid currentColor;font-family:var(--mono); }
.pill-m1 { color:#00d4a8;background:rgba(0,212,168,0.08); }
.pill-m2 { color:#0094ff;background:rgba(0,148,255,0.08); }
.pill-m3 { color:#f0b429;background:rgba(240,180,41,0.08); }
.pill-m5 { color:#ff8c00;background:rgba(255,140,0,0.08); }
.pill-m6 { color:#ff4d6d;background:rgba(255,77,109,0.08); }
.top-banner { display:flex;align-items:center;gap:16px;padding:20px 0 8px 0; }
.banner-logo { font-family:var(--sans);font-weight:800;font-size:26px;color:var(--accent);letter-spacing:-0.5px; }
.count-pill { background:rgba(0,212,168,0.1);border:1px solid var(--accent);color:var(--accent);border-radius:20px;padding:2px 12px;font-size:13px;font-weight:700;font-family:var(--mono); }
.result-hdr { display:flex;align-items:center;gap:12px;padding:14px 0; }
.result-hdr h3 { margin:0 !important; }
.status-line { display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;font-size:12px;color:var(--muted); }
.sdot { width:7px;height:7px;border-radius:50%;display:inline-block; }
.sdot-green { background:var(--accent);box-shadow:0 0 6px var(--accent); }

/* ── NEW: score badge + top-pick cards ─────────────────────────── */
.score-green  { color:#00d4a8;font-weight:700; }
.score-blue   { color:#0094ff;font-weight:700; }
.score-yellow { color:#f0b429;font-weight:700; }
.score-red    { color:#ff4d6d;font-weight:700; }
.pick-card {
  background:#0b1017;border:1px solid #1a2840;border-radius:14px;
  padding:18px 20px;transition:border 0.2s;
}
.pick-card:hover { border-color:#243550; }
.pick-rank  { font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#00d4a8;line-height:1; }
.pick-sym   { font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#ccd9e8; }
.pick-score { font-size:11px;color:#4a6480;margin-top:4px; }
.trap-badge { color:#ff4d6d;font-weight:700;font-size:12px; }
.breakdown-box {
  background:#060a0f;border:1px solid #1a2840;border-radius:8px;
  padding:10px 14px;font-size:11px;line-height:1.9;color:#4a6480;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# NSE TICKER LOADER  (unchanged)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nse_tickers() -> list:
    tickers = set()
    headers = {"User-Agent": "Mozilla/5.0"}

    for days_back in range(0, 7):
        try:
            dt = datetime.now() - timedelta(days=days_back)
            if dt.weekday() >= 5:
                continue
            date_str = dt.strftime("%d%b%Y").upper()
            url = (
                f"https://archives.nseindia.com/content/historical/EQUITIES/"
                f"{dt.year}/{dt.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
            )
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                z  = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                for s in df["SYMBOL"].dropna().unique():
                    tickers.add(f"{s.strip()}.NS")
                break
        except Exception:
            continue

    if len(tickers) < 100:
        try:
            r = requests.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv", headers=headers, timeout=12)
            if r.status_code == 200:
                df  = pd.read_csv(io.StringIO(r.text))
                col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
                for s in df[col].dropna().unique():
                    tickers.add(f"{str(s).strip()}.NS")
        except Exception:
            pass

    if len(tickers) < 100:
        try:
            r = requests.get("https://archives.nseindia.com/content/indices/ind_nifty500list.csv", headers=headers, timeout=12)
            if r.status_code == 200:
                df  = pd.read_csv(io.StringIO(r.text))
                col = "Symbol" if "Symbol" in df.columns else df.columns[1]
                for s in df[col].dropna().unique():
                    tickers.add(f"{str(s).strip()}.NS")
        except Exception:
            pass

    if len(tickers) < 200:
        fallback = [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
            "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
            "BAJFINANCE","HCLTECH","SUNPHARMA","TITAN","ULTRACEMCO","ONGC",
            "NESTLEIND","WIPRO","POWERGRID","NTPC","TECHM","INDUSINDBK","ADANIPORTS",
            "TATAMOTORS","JSWSTEEL","BAJAJFINSV","HINDALCO","GRASIM","DIVISLAB",
            "CIPLA","DRREDDY","BPCL","EICHERMOT","APOLLOHOSP","TATACONSUM","BRITANNIA",
            "COALINDIA","HEROMOTOCO","SHREECEM","SBILIFE","HDFCLIFE","ADANIENT",
            "BAJAJ-AUTO","TATASTEEL","UPL","M&M","VEDL","BHEL","NMDC","SAIL",
            "GAIL","IOC","HPCL","RECLTD","PFC","IRFC","NHPC","SJVN",
            "PIDILITIND","BERGEPAINT","HAVELLS","VOLTAS","POLYCAB","FINOLEX",
            "MOTHERSON","BOSCHLTD","EXIDEIND","BALKRISHIND","MRF","APOLLOTYRE",
            "TVSMOTORS","PAGEIND","MUTHOOTFIN","CHOLAFIN","MANAPPURAM","AUBANK",
            "FEDERALBNK","IDFCFIRSTB","BANDHANBNK","RBLBANK","YESBANK",
            "PNB","CANBK","UNIONBANK","KARURVYSYA","DCBBANK","CITYUNIONB",
            "LTTS","PERSISTENT","COFORGE","MPHASIS","TATAELXSI","KPITTECH",
            "ZENSAR","RATEGAIN","NAUKRI","IRCTC","ZOMATO","DELHIVERY",
            "AUROPHARMA","LUPIN","TORNTPHARM","ALKEM","IPCALAB","NATCOPHARM",
            "GRANULES","GLAND","LAURUS","BIOCON","ZYDUSLIFE","MANKIND",
            "ASTRAL","SUPREMEIND","ALKYLAMINE","AAVAS","CANFINHOME","LICHSGFIN",
            "PNBHOUSING","APTUS","TATAPOWER","CESC","TORNTPOWER","ADANIGREEN",
            "GUJGASLTD","IGL","MGL","PETRONET","GREENPLY","GREENLAM",
            "TRENT","DMART","WESTLIFE","JUBLFOOD","DEVYANI","DIXON","AMBER",
            "PRESTIGE","DLF","OBEROIRLTY","GODREJPROP","PHOENIXLTD","BRIGADE",
            "SOBHA","KOLTEPATIL","SUNTECK","SHRIRAMFIN","MFSL","ICICIGI",
            "HDFCGI","STARHEALTH","NIACL","GICRE","IIFL","MOTILALOFS",
            "ANGELONE","PAYTM","POLICYBZR","NYKAA","CARTRADE",
        ]
        for s in fallback:
            tickers.add(f"{s}.NS")

    return sorted(list(tickers))


# ─────────────────────────────────────────────────────────────────────
# TECHNICAL HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    r     = 100 - (100 / (1 + rs))
    return float(r.iloc[-1]) if not r.empty else np.nan


# ─────────────────────────────────────────────────────────────────────
# MARKET BIAS ENGINE (add-on; does not affect scan/mode engines)
# ─────────────────────────────────────────────────────────────────────
def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def compute_market_bias(include_bank: bool = True) -> dict:
    """
    Compute probabilistic market bias for next day using only free yfinance data.
    Fail-safe: returns conservative "Sideways / no edge" output if index data missing.
    """
    try:
        def _fetch_index(symbol: str) -> pd.DataFrame | None:
            try:
                df_i = yf.download(
                    symbol, period="3mo", interval="1d",
                    auto_adjust=True, progress=False, timeout=12, threads=False,
                )
                if df_i is None or df_i.empty:
                    return None
                if isinstance(df_i.columns, pd.MultiIndex):
                    df_i.columns = df_i.columns.get_level_values(0)
                return df_i
            except Exception:
                return None

        def _index_features(df_i: pd.DataFrame) -> dict:
            close = df_i["Close"].dropna() if "Close" in df_i.columns else pd.Series(dtype=float)
            if len(close) < 60:
                return {}

            ema20_s = ema(close, 20)
            ema50_s = ema(close, 50)
            rsi14_v = rsi(close, 14)
            c_last  = _safe_float(close.iloc[-1], 0.0)
            e20_last = _safe_float(ema20_s.iloc[-1], 0.0)
            e50_last = _safe_float(ema50_s.iloc[-1], 0.0)

            ret5d = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 and float(close.iloc[-6]) != 0 else np.nan
            ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 and float(close.iloc[-21]) != 0 else np.nan

            # Vol relative to 20D average (if volume exists; indices may lack Volume)
            vol_ratio = None
            if "Volume" in df_i.columns:
                vol = df_i["Volume"].dropna()
                if len(vol) >= 21:
                    avg20 = float(vol.iloc[-21:-1].mean()) if len(vol.iloc[-21:-1]) >= 20 else float(vol.mean())
                    lastv = float(vol.iloc[-1])
                    vol_ratio = (lastv / avg20) if avg20 > 0 else None

            # Realized volatility proxy for expected move range
            ret_1d = close.pct_change().dropna().tail(20) * 100.0
            sigma_pct = float(ret_1d.std()) if not ret_1d.empty else 0.0

            features = {
                "close": c_last,
                "ema20": e20_last,
                "ema50": e50_last,
                "rsi14": _safe_float(rsi14_v, 50.0),
                "ret5d": _safe_float(ret5d, 0.0),
                "ret20d": _safe_float(ret20d, 0.0),
                "vol_ratio": (float(vol_ratio) if vol_ratio is not None else None),
                "sigma_pct": max(0.0, sigma_pct),
            }
            return features

        df_nifty = _fetch_index("^NSEI")
        if df_nifty is None:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "±0.30% to ±0.70%",
                "breakdown": ["Nifty data unavailable (fallback)."],
            }

        nifty_feat = _index_features(df_nifty)
        if not nifty_feat:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "±0.30% to ±0.70%",
                "breakdown": ["Nifty indicators insufficient (fallback)."],
            }

        bank_feat = None
        if include_bank:
            df_bn = _fetch_index("^NSEBANK")
            if df_bn is not None:
                bf = _index_features(df_bn)
                bank_feat = bf if bf else None

        # Score / interpret (nifty dominates; bank is confirmation)
        return interpret_market_bias(nifty_feat, bank_feat)
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "±0.30% to ±0.70%",
            "breakdown": ["Market bias computation failed (fallback)."],
        }


def interpret_market_bias(nifty_feat: dict, bank_feat: dict | None = None) -> dict:
    """
    Convert index features into conservative bias/confidence/expected-range.
    """
    try:
        def _signal(feat: dict) -> tuple[float, dict]:
            close = _safe_float(feat.get("close", 0.0), 0.0)
            ema20_v = _safe_float(feat.get("ema20", 0.0), 0.0)
            ema50_v = _safe_float(feat.get("ema50", 0.0), 0.0)
            rsi14_v = _safe_float(feat.get("rsi14", 50.0), 50.0)
            ret5d_v = _safe_float(feat.get("ret5d", 0.0), 0.0)
            ret20d_v = _safe_float(feat.get("ret20d", 0.0), 0.0)
            vol_ratio = feat.get("vol_ratio", None)

            bull_trend = close > ema20_v > ema50_v
            bear_trend = close < ema20_v < ema50_v
            trend_sig = 1.0 if bull_trend else (-1.0 if bear_trend else 0.0)

            momentum_sig = 1.0 if ret5d_v > 0.30 else (-1.0 if ret5d_v < -0.30 else 0.0)
            rsi_sig = 1.0 if rsi14_v >= 55.0 else (-1.0 if rsi14_v <= 45.0 else 0.0)

            if vol_ratio is None:
                volume_sig = 0.0
            else:
                volume_sig = 1.0 if vol_ratio >= 1.10 else (-1.0 if vol_ratio <= 0.90 else 0.0)

            # Breakdown/support context for extra conservatism
            breakdown = (close < ema20_v and ret20d_v < 0.0)
            support = (close > ema20_v and ret20d_v > 0.0)

            base_score = 0.35 * trend_sig + 0.25 * momentum_sig + 0.20 * rsi_sig + 0.20 * volume_sig

            details = {
                "trend_sig": trend_sig,
                "momentum_sig": momentum_sig,
                "rsi_sig": rsi_sig,
                "volume_sig": volume_sig,
                "breakdown": breakdown,
                "support": support,
                "close": close,
                "ema20": ema20_v,
                "ema50": ema50_v,
                "rsi14": rsi14_v,
                "ret5d": ret5d_v,
                "ret20d": ret20d_v,
                "vol_ratio": vol_ratio,
            }
            return base_score, details

        nifty_score, nf = _signal(nifty_feat)

        bank_score = 0.0
        bf = None
        if bank_feat:
            bank_score, bf = _signal(bank_feat)

        # Conservative combination: bank only provides confirmation/penalty.
        combined = nifty_score
        bank_used = bf is not None
        if bank_used:
            combined = 0.80 * nifty_score + 0.20 * bank_score

        # Strict (high-edge) bias selection
        trend_pos = nf["trend_sig"] > 0
        trend_neg = nf["trend_sig"] < 0
        mom_pos = nf["momentum_sig"] > 0
        mom_neg = nf["momentum_sig"] < 0
        rsi_pos = nf["rsi_sig"] > 0
        rsi_neg = nf["rsi_sig"] < 0
        vol_pos = nf["volume_sig"] > 0 or nf["vol_ratio"] is None
        vol_neg = nf["volume_sig"] < 0 or nf["vol_ratio"] is None

        bullish_strict = trend_pos and mom_pos and vol_pos and rsi_pos
        bearish_strict = trend_neg and mom_neg and vol_neg and rsi_neg

        bullish_relaxed = (bullish_strict or ((trend_pos or nf["support"]) and mom_pos and rsi_pos and vol_pos))
        bearish_relaxed = (bearish_strict or ((trend_neg or nf["breakdown"]) and mom_neg and rsi_neg and vol_neg))

        # Bank confirmation / conflict adjustment
        if bank_used and bf is not None:
            bank_trend_pos = bf["trend_sig"] > 0
            bank_trend_neg = bf["trend_sig"] < 0
            if bullish_relaxed and bank_trend_neg and bf["momentum_sig"] < 0:
                bullish_relaxed = False
            if bearish_relaxed and bank_trend_pos and bf["momentum_sig"] > 0:
                bearish_relaxed = False

        if bullish_strict:
            bias = "Bullish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bearish_strict:
            bias = "Bearish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bullish_relaxed and not bearish_relaxed:
            bias = "Bullish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        elif bearish_relaxed and not bullish_relaxed:
            bias = "Bearish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        else:
            bias = "Sideways / no edge"
            conf = min(58, int(round(45 - abs(combined) * 18)))
            conf = max(conf, 25)

        # Expected move range from sigma (conservative scaling)
        sigma_pct = _safe_float(nifty_feat.get("sigma_pct", 0.0), 0.0)
        sigma_pct = max(sigma_pct, 0.10)  # keep non-zero
        if bias == "Bullish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"+{low_mag:.2f}% to +{high_mag:.2f}%"
        elif bias == "Bearish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"-{low_mag:.2f}% to -{high_mag:.2f}%"
        else:
            side = sigma_pct * 0.55 * (0.80 + conf / 220.0)
            expected_range = f"±{side:.2f}% to ±{(side * 1.2):.2f}%"

        # Breakdown text (short)
        breakdown = []
        trend_txt = "Bullish trend (Close > EMA20 > EMA50)" if nf["trend_sig"] > 0 else (
            "Bearish trend (Close < EMA20 < EMA50)" if nf["trend_sig"] < 0 else "Neutral trend (EMA alignment mixed)"
        )
        breakdown.append(f"Trend: {trend_txt}.")
        breakdown.append(f"Momentum: 5D return {nf['ret5d']:+.2f}%.")
        breakdown.append(f"RSI(14): {nf['rsi14']:.1f}.")
        if nf["vol_ratio"] is None:
            breakdown.append("Volume: not available for index (confidence kept conservative).")
        else:
            breakdown.append(f"Volume: Vol/20D avg {nf['vol_ratio']:.2f}x.")
        breakdown.append(f"20D return: {nf['ret20d']:+.2f}%.")

        if bank_used and bf is not None:
            bn_txt = "BankNifty confirms bullishness" if bf["trend_sig"] > 0 and bf["momentum_sig"] > 0 else (
                "BankNifty confirms bearishness" if bf["trend_sig"] < 0 and bf["momentum_sig"] < 0 else
                "BankNifty mixed / neutral."
            )
            breakdown.append(bn_txt)

        return {
            "bias": bias,
            "confidence": conf,
            "expected_range": expected_range,
            "breakdown": breakdown[:6],
        }
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "±0.30% to ±0.70%",
            "breakdown": ["Interpretation failed (fallback)."],
        }


# ── FRESH ISOLATED MARKET BIAS ENGINE (Task 4.3 — UI Version) ─────────
@st.cache_data(ttl=600, show_spinner=False)
def compute_market_bias_ui() -> dict:
    """
    Independent function for the 'Market Bias Engine' UI button.
    Does not touch scanner/strategy mode logic.
    """
    try:
        # 1. Fetch data
        df = yf.download("^NSEI", period="4mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return {
                "bias": "Sideways / No Edge",
                "confidence": 50,
                "expected_move": "±0.5% (fallback)",
                "reasons": ["Insufficient data from yfinance for Nifty (^NSEI)."]
            }
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()

        # 2. Indicators
        e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        
        # RSI manual calc (pure dependency-free)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50.0

        c_last = float(close.iloc[-1])
        ret5d  = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
        ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else 0.0
        
        avg_vol = vol.iloc[-21:-1].mean() if len(vol) >= 21 else 1.0
        vol_r   = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        # 3. Bias Logic (Strictly Isolated)
        bullish = (c_last > e20 > e50) and (rsi_val > 52) and (ret5d > 0.1)
        bearish = (c_last < e20 < e50) and (rsi_val < 48) and (ret5d < -0.1)
        
        score = 50
        if bullish:
            score += 15
            if ret20d > 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bullish / Strong Bias" if score > 70 else "Bullish"
        elif bearish:
            score += 15
            if ret20d < 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bearish / Negative Bias" if score > 70 else "Bearish"
        else:
            bias = "Sideways / No Clear Edge"
        
        score = min(95, max(5, score))

        # 4. Volatility based move range
        daily_returns = close.pct_change().tail(20)
        volatility = daily_returns.std() * 100.0 if not daily_returns.empty else 0.5
        move_pct = round(volatility * 0.8, 2)
        expected_move = f"±{move_pct}% to ±{round(move_pct*1.5, 2)}%"

        # 5. Reason Breakdown
        reasons = []
        trend_txt = "Bullish stack" if c_last > e20 > e50 else ("Bearish stack" if c_last < e20 < e50 else "Neutral trend")
        reasons.append(f"Trend: {trend_txt} (Close={c_last:.0f}, EMA20={e20:.0f}).")
        reasons.append(f"RSI(14): {rsi_val:.1f} ({'Overbought' if rsi_val > 70 else ('Oversold' if rsi_val < 30 else 'Neutral')}).")
        reasons.append(f"Momentum: 5-day return {ret5d:+.2f}%.")
        reasons.append(f"Volume: Recent volume is {vol_r:.2f}x of 20-day average.")
        reasons.append(f"Structure: 20-day return is {ret20d:+.2f}%.")

        return {
            "bias": bias,
            "confidence": int(score),
            "expected_move": expected_move,
            "reasons": reasons
        }
    except Exception as e:
        return {
            "bias": "Sideways / Unknown",
            "confidence": 50,
            "expected_move": "±0.5% (error)",
            "reasons": [f"Market analysis error: {str(e)}"]
        }


# ─────────────────────────────────────────────────────────────────────
# STOCK ANALYSER  (Zero-API Refactored)
# ─────────────────────────────────────────────────────────────────────
def analyse(ticker, mode, retries=2):
    try:
        ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
        df = get_df_for_ticker(ticker_ns)
        
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "Close", "Volume"])
        if df.empty or len(df) < 30:
            return None
        try:
            last_idx = df.index[-1]
            last_dt  = pd.to_datetime(last_idx).to_pydatetime()
        except Exception:
            return None
        if (datetime.now() - last_dt).days > 7:
            return None

        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        open_p = df["Open"].dropna()
        if len(close) < 25:
            return None

        lc  = float(close.iloc[-1])
        lo  = float(open_p.iloc[-1])
        lv  = float(volume.iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])
        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ri  = rsi(close)

        if not (1 < lc <= 100000):
            return None
        if lv <= 0:
            return None
        if any(np.isnan(v) for v in (ri, e20, e50)):
            return None

        mktcap_cr = get_mktcap_cr(ticker)

        ok = False
        if mode == 1:
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            ok  = (lc > e20 and e20 > e50 and lv > 1.7 * avg_vol
                   and 55 <= ri <= 72 and lc > h10 and lc > lo
                   and lc > 70 and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 2:
            h15 = float(close.iloc[-16:-1].max()) if len(close) >= 16 else float(close.max())
            ok  = (50 < lc < 1500 and lc > e20 and e20 > e50
                   and lv > 1.5 * avg_vol and 52 <= ri <= 70
                   and lc >= 0.97 * h15 and lc > lo
                   and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 3:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            ok  = (lc > e20 and e20 > e50 and lv > 1.3 * avg_vol
                   and 50 <= ri <= 72 and lc >= 0.95 * h20 and lc > 50)
        elif mode == 4:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            if len(close) < 21:
                return None
            base_20 = float(close.iloc[-21])
            if base_20 <= 0:
                return None
            stock_ret_20d = (lc - base_20) / base_20
            nifty_ret_20d = get_nifty_20d_return()
            if nifty_ret_20d is None:
                return None
            ok = (
                lc > e20 and e20 > e50 and
                lv > 1.5 * avg_vol and 55 <= ri <= 70 and
                lc >= 0.98 * h20 and stock_ret_20d > nifty_ret_20d and lc > lo
            )
        elif mode == 5:
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and lv > 1.1 * avg_vol_sma and
                lc > h10 and 52 < ri < 60 and lc > lo and lc > 20
            )
        elif mode == 6:
            if len(close) < 2:
                return None
            prev_e20     = float(ema(close, 20).iloc[-2])
            prev_vol     = float(volume.iloc[-2])
            h10          = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma  = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and e20 > prev_e20 and
                lv > 1.3 * avg_vol_sma and lv > prev_vol and
                lc > h10 and 53 < ri < 59 and lc > lo and lc > 40
            )

        if ok:
            h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
            dist_ema20    = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
            ret_5d  = (lc / float(close.iloc[-6]) - 1.0) * 100.0  if len(close) >= 6  else np.nan
            ret_20d = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else np.nan

            return {
                "Symbol":            ticker.replace(".NS", ""),
                "Price (₹)":         round(lc, 2),
                "Volume":            int(lv),
                "RSI":               round(ri, 2),
                "EMA 20":            round(e20, 2),
                "EMA 50":            round(e50, 2),
                "Vol / Avg":         round(lv / avg_vol, 2) if avg_vol > 0 else 0,
                "Mode":              {1:"🟢 Momentum",2:"🔵 Balanced",3:"🟡 Relaxed",
                                      4:"🟣 Institutional",5:"🟠 Intraday",6:"🔴 Swing"}[mode],
                "Δ vs 20D High (%)": round(dist_20d_high, 2),
                "Δ vs EMA20 (%)":    round(dist_ema20, 2),
                "5D Return (%)":     round(ret_5d, 2)  if not np.isnan(ret_5d)  else np.nan,
                "20D Return (%)":    round(ret_20d, 2) if not np.isnan(ret_20d) else np.nan,
            }
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# PARALLEL SCANNER  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def run_scan(tickers, mode, workers=20):
    results = []
    total   = len(tickers)
    done    = 0

    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a: status  = st.empty()
    with col_b: eta_box = st.empty()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyse, t, mode): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                results.append(r)
            pct     = done / total
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 1
            remaining = (total - done) / rate if rate > 0 else 0
            progress_bar.progress(pct)
            status.markdown(
                f'<div class="status-line"><span class="sdot sdot-green"></span>'
                f'&nbsp;Scanned <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
                f' &nbsp;·&nbsp; Found <b style="color:#00d4a8">{len(results)}</b> candidates</div>',
                unsafe_allow_html=True)
            eta_box.markdown(
                f'<div class="status-line" style="justify-content:center">'
                f'⏱ ETA <b style="color:#f0b429">{remaining:.0f}s</b></div>',
                unsafe_allow_html=True)

    progress_bar.progress(1.0)
    elapsed_total = time.time() - t0
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} stocks in'
        f' <b style="color:#f0b429">{elapsed_total:.1f}s</b>'
        f' &nbsp;·&nbsp; <b style="color:#00d4a8">{len(results)}</b> found</div>',
        unsafe_allow_html=True)
    eta_box.empty()
    return results, elapsed_total


# ═════════════════════════════════════════════════════════════════════
# ▼▼▼  NEW LAYER — SCORING / BACKTEST / ML  (added AFTER scan) ▼▼▼
# ═════════════════════════════════════════════════════════════════════

# ── mode-specific weight configs ─────────────────────────────────────
_MODE_WEIGHTS = {
    # (vol_bonus, breakout_bonus, ema_bonus, rsi_bonus, penalty_scale)
    1: dict(vol=1.4,  breakout=1.5, ema=1.0, rsi=1.0, pen=1.0),   # Momentum
    2: dict(vol=1.0,  breakout=1.0, ema=1.0, rsi=1.0, pen=1.0),   # Balanced
    3: dict(vol=0.8,  breakout=0.8, ema=0.8, rsi=0.8, pen=0.5),   # Relaxed
    4: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Institutional
    5: dict(vol=1.5,  breakout=1.2, ema=0.7, rsi=0.8, pen=0.9),   # Intraday
    6: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Swing
}


def _safe(v, default=0.0):
    """Return v if finite, else default."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def compute_score(row: dict, mode: int = 2) -> tuple[float, dict]:
    """
    Returns (score_0_100, breakdown_dict).
    breakdown_dict is used for the tooltip.
    """
    w   = _MODE_WEIGHTS.get(mode, _MODE_WEIGHTS[2])
    pts = {}

    ri       = _safe(row.get("RSI",            50))
    vol_r    = _safe(row.get("Vol / Avg",       1))
    d20h     = _safe(row.get("Δ vs 20D High (%)", -5))
    d_ema20  = _safe(row.get("Δ vs EMA20 (%)",   0))
    r5d      = _safe(row.get("5D Return (%)",    0))
    price    = _safe(row.get("Price (₹)",        0))
    e20      = _safe(row.get("EMA 20",           0))
    e50      = _safe(row.get("EMA 50",           0))

    # ── RSI zone ─────────────────────────────────────────────────────
    if 55 <= ri <= 65:
        pts["RSI 55-65"] = round(15 * w["rsi"])
    elif 65 < ri <= 70:
        pts["RSI 65-70"] = round(10 * w["rsi"])

    # ── Volume ratio ──────────────────────────────────────────────────
    if vol_r > 2.0:
        pts["Vol >2×"] = round(25 * w["vol"])
    elif vol_r > 1.5:
        pts["Vol >1.5×"] = round(20 * w["vol"])

    # ── Near 20D breakout ─────────────────────────────────────────────
    if -2.0 <= d20h <= 0.0:
        pts["Near 20D High"] = round(15 * w["breakout"])

    # ── Above EMA20 ───────────────────────────────────────────────────
    if price > e20 > 0:
        pts["Price > EMA20"] = round(10 * w["ema"])

    # ── EMA stack ─────────────────────────────────────────────────────
    if e20 > e50 > 0:
        pts["EMA20 > EMA50"] = round(10 * w["ema"])

    # ── 5D return zone ────────────────────────────────────────────────
    if 1.0 <= r5d <= 5.0:
        pts["5D Return 1-5%"] = round(10 * w["rsi"])

    # ── PENALTIES ─────────────────────────────────────────────────────
    if ri > 72:
        pts["RSI Overbought"] = round(-20 * w["pen"])
    if d_ema20 > 6.0:
        pts["Overextended EMA"] = round(-15 * w["pen"])
    if r5d > 8.0:
        pts["5D Return >8%"] = round(-10 * w["pen"])
    if vol_r < 1.2:
        pts["Low Volume"] = round(-15 * w["pen"])

    raw = sum(pts.values())
    score = float(np.clip(raw, 0, 100))
    return score, pts


# ── Backtest cache: ticker → float ───────────────────────────────────
_BT_CACHE: dict[str, float] = {}
_BT_LOCK = threading.Lock()


def _download_history(ticker_ns: str, period: str = "6mo") -> pd.DataFrame | None:
    """Download history; returns None on failure."""
    try:
        with _YF_SEM:
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


def compute_backtest_probability(row: dict, ticker: str, mode: int = 2) -> float:
    """
    Mode-aware backtest probability (Task 7).
    Each mode uses different RSI/vol tolerances + extra conditions that
    mirror what the live scanner filter actually checks.
    Cache key is ticker+mode so each strategy gets its own probability.
    Returns 50 (neutral) if fewer than 20 matching samples found.
    """
    # per-mode matching config: rsi_tol, vol_tol, ema_trend required,
    # near-high required, rolling-high window, near-high threshold
    _MBTCFG: dict[int, dict] = {
        1: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=10, hp=0.02),
        2: dict(rsi_tol=4,  vol_tol=0.25, ema_trend=True,  near_high=False, hw=15, hp=0.03),
        3: dict(rsi_tol=5,  vol_tol=0.30, ema_trend=False, near_high=False, hw=20, hp=0.05),
        4: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=20, hp=0.02),
        5: dict(rsi_tol=2,  vol_tol=0.15, ema_trend=True,  near_high=True,  hw=10, hp=0.01),
        6: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=False, hw=10, hp=0.02),
    }
    cfg = _MBTCFG.get(mode, _MBTCFG[2])
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    cache_key = f"{ticker_ns}|m{mode}"

    with _BT_LOCK:
        if cache_key in _BT_CACHE:
            return _BT_CACHE[cache_key]

    result = 50.0
    try:
        df = _download_history(ticker_ns)
        if df is None or len(df) < 40:
            raise ValueError("insufficient data")

        close  = df["Close"].copy()
        volume = df["Volume"].copy()

        e20s = ema(close, 20)
        e50s = ema(close, 50)
        # vectorised RSI — no per-row loop, avoids pandas 2.x FutureWarning
        _d  = close.diff()
        _g  = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l  = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_series = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))

        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        roll_high = close.rolling(cfg["hw"], min_periods=max(1, cfg["hw"] // 2)).max().shift(1)

        target_rsi  = _safe(row.get("RSI",      50))
        target_volr = _safe(row.get("Vol / Avg", 1))

        mask = (
            rsi_series.notna() &
            (rsi_series >= target_rsi  - cfg["rsi_tol"]) &
            (rsi_series <= target_rsi  + cfg["rsi_tol"]) &
            (vol_ratio  >= target_volr * (1 - cfg["vol_tol"])) &
            (vol_ratio  <= target_volr * (1 + cfg["vol_tol"]))
        )
        if cfg["ema_trend"]:
            mask &= (e20s > e50s)
        if cfg["near_high"]:
            mask &= (roll_high.notna() & (close >= roll_high * (1 - cfg["hp"])))

        # Mode-specific extra matching conditions (Task 7)
        if mode == 4:
            mask &= (close.pct_change(20) > 0)          # positive 20D return (institutional)
        elif mode == 5:
            mask &= (vol_ratio > 1.5)                    # strong vol spike (intraday)
        elif mode == 6:
            mask &= (e20s > e20s.shift(1))               # rising EMA20 slope (swing)

        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]         # exclude last row

        if len(idx) < 20:
            raise ValueError(f"too few samples: {len(idx)}")

        close_vals  = close.values
        green_count = int(sum(close_vals[i + 1] > close_vals[i] for i in idx))
        result = round((green_count / len(idx)) * 100, 1)

    except Exception:
        result = 50.0

    with _BT_LOCK:
        _BT_CACHE[cache_key] = result
    return result


# ── ML model cache ────────────────────────────────────────────────────
_ML_MODEL: "LogisticRegression | None" = None
_ML_SCALER: "StandardScaler | None"   = None
_ML_LOCK = threading.Lock()
_ML_TICKERS_TRAINED: list[str] = []


def _build_ml_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Build training feature matrix for one ticker.
    All computations vectorised — no per-row Python loop (BUG 3 fix).
    """
    try:
        if len(close) < 30:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        ema_dist  = (close / e20s.replace(0, np.nan) - 1.0) * 100
        # vectorised RSI — no per-row loop
        _d        = close.diff()
        _g        = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l        = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_col   = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))
        ret5      = close.pct_change(5)  * 100
        ret20     = close.pct_change(20) * 100
        target    = (close.shift(-1) > close).astype(int)
        ema_trend = (e20s > e50s).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_col,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ret_5d":    ret5,
            "ret_20d":   ret20,
            "ema_trend": ema_trend,
            "target":    target,
        }).dropna()
        return df if len(df) >= 10 else None
    except Exception:
        return None


# 50-stock training universe (Task 4 — was only 15 stocks)
_ML_TRAIN_UNIVERSE: list[str] = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","AXISBANK.NS",
    "TATAMOTORS.NS","MARUTI.NS","LT.NS","NTPC.NS","ADANIPORTS.NS",
    "HINDALCO.NS","JSWSTEEL.NS","COALINDIA.NS","ONGC.NS","POWERGRID.NS",
    "BHARTIARTL.NS","TITAN.NS","NESTLEIND.NS","ULTRACEMCO.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","EICHERMOT.NS","M&M.NS","TATACONSUM.NS","BRITANNIA.NS",
    "TECHM.NS","INDUSINDBK.NS","KOTAKBANK.NS","ASIANPAINT.NS","GRASIM.NS",
    "DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","SUNPHARMA.NS","APOLLOHOSP.NS",
    "ITC.NS","BPCL.NS","IOC.NS","GAIL.NS","VEDL.NS",
    "ZOMATO.NS","NAUKRI.NS","IRCTC.NS","DMART.NS","TRENT.NS",
]


def train_model_once(tickers_sample: list[str] | None = None) -> bool:
    """
    Train LogisticRegression on up to 50 NSE stocks (1-year history).
    Task 4 upgrades:
      • 80/20 stratified train/test split
      • Prints test accuracy to stdout
      • class_weight='balanced' to reduce overfitting
      • C=0.5 (moderate regularisation)
      • 50-stock training universe (was 15)
    Model + scaler cached in module globals; re-entrant safe.
    """
    global _ML_MODEL, _ML_SCALER, _ML_TICKERS_TRAINED

    if not _SKLEARN_OK:
        return False

    with _ML_LOCK:
        if _ML_MODEL is not None:
            return True

    sample = (list(tickers_sample)[:50]
              if (tickers_sample and len(tickers_sample) >= 5)
              else _ML_TRAIN_UNIVERSE[:50])

    all_rows: list[pd.DataFrame] = []
    for t in sample:
        df_h = _download_history(t, period="1y")
        if df_h is None:
            continue
        rows = _build_ml_features(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False

    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 100:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ret_5d", "ret_20d", "ema_trend"]
    X = data[FEAT].values
    y = data["target"].values

    try:
        # 80/20 stratified split (Task 4)
        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
        except Exception:
            split = int(len(data) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        scaler    = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_tr)
        X_te_sc   = scaler.transform(X_te)

        model = LogisticRegression(
            max_iter=500, C=0.5, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )
        model.fit(X_tr_sc, y_tr)

        acc = model.score(X_te_sc, y_te)
        print(
            f"[ML] Model trained on {len(data)} samples "
            f"({len(sample)} tickers) — test accuracy: {acc:.3f}"
        )

        with _ML_LOCK:
            _ML_MODEL            = model
            _ML_SCALER           = scaler
            _ML_TICKERS_TRAINED  = sample
        return True
    except Exception as exc:
        print(f"[ML] Training failed: {exc}")
        return False


def predict_ml_probability(row: dict, mode: int = 2) -> float:
    """
    Returns next-day-green probability (0-100) from the trained LR model.
    Task 7: base probability is adjusted by mode-specific signal weights
    so each strategy context influences the final ML score differently:
      Mode 1 (Momentum) → boosts high-vol + near-breakout
      Mode 3 (Relaxed)  → slight confidence haircut
      Mode 4 (Institutional) → rewards RSI + 20D return
      Mode 5 (Intraday)  → rewards vol spike, penalises high RSI
      Mode 6 (Swing)     → rewards EMA distance + 5D return
    Falls back to 50 if model not ready or features invalid.
    """
    if not _SKLEARN_OK:
        return 50.0

    with _ML_LOCK:
        model  = _ML_MODEL
        scaler = _ML_SCALER

    if model is None or scaler is None:
        return 50.0

    try:
        ri    = _safe(row.get("RSI",               50))
        vol_r = _safe(row.get("Vol / Avg",           1))
        de20  = _safe(row.get("Δ vs EMA20 (%)",      0))
        r5d   = _safe(row.get("5D Return (%)",        0))
        r20d  = _safe(row.get("20D Return (%)",       0))
        d20h  = _safe(row.get("Δ vs 20D High (%)", -10))

        feat    = np.array([[ri, vol_r, de20, r5d, r20d, 1.0]])
        feat_sc = scaler.transform(feat)
        base_p  = float(model.predict_proba(feat_sc)[0][1])

        # ── mode-specific probability adjustment (Task 7) ─────────────
        adj = 0.0
        if mode == 1:                              # Momentum
            if vol_r > 1.7:              adj += 0.05
            if -2.0 <= d20h <= 0.0:      adj += 0.03
        elif mode == 2:                            # Balanced — no adjustment
            pass
        elif mode == 3:                            # Relaxed — confidence haircut
            adj -= 0.03
        elif mode == 4:                            # Institutional
            if ri > 58 and r20d > 3.0:   adj += 0.04
            if vol_r > 1.5:              adj += 0.02
        elif mode == 5:                            # Intraday
            if vol_r > 1.5:              adj += 0.06
            if ri > 60:                  adj -= 0.03
        elif mode == 6:                            # Swing
            if r5d > 1.5:               adj += 0.04
            if de20 < 3.0:              adj += 0.02   # not overextended

        final_p = float(np.clip(base_p + adj, 0.01, 0.99))
        return round(final_p * 100, 1)
    except Exception:
        return 50.0


def compute_next_day_signal(row: dict, df: pd.DataFrame | None) -> str:
    """Compute short-term confirmational signal using last 10 days geometry."""
    if df is None or len(df) < 10:
        return "❌ No Data"
    try:
        last_10 = df.tail(10)
        closes = last_10["Close"].values
        vols = last_10["Volume"].values
        highs = last_10["High"].values if "High" in last_10.columns else closes

        if len(closes) < 10:
            return "❌ No Data"

        last_3 = closes[-3:]
        vol_last = vols[-1]
        vol_avg10 = vols.mean()
        high_10 = highs.max()
        close_today = closes[-1]

        momentum = (last_3[0] < last_3[1] < last_3[2])
        vol_spike = vol_last > 1.3 * vol_avg10
        near_breakout = close_today >= high_10 * 0.98
        overextended = _safe(row.get("Δ vs EMA20 (%)", 0)) > 5.0

        if momentum and vol_spike and near_breakout and not overextended:
            return "🔥 Strong Green"
        elif momentum and near_breakout:
            return "🟢 Possible Up"
        elif overextended:
            return "⚠️ Risky (Late Entry)"
        else:
            return "❌ Weak Setup"
    except Exception:
        return "❌ Error"


def check_bull_trap(row: dict) -> str:
    """Return warning string or empty string."""
    ri    = _safe(row.get("RSI",         50))
    vol_r = _safe(row.get("Vol / Avg",    1))
    de20  = _safe(row.get("Δ vs EMA20 (%)", 0))

    traps = []
    if ri > 75:
        traps.append("RSI overbought")
    if vol_r < 1.0:
        traps.append("vol declining")
    if de20 > 8.0:
        traps.append("far from EMA20")

    return "⚠️ Bull Trap" if len(traps) >= 2 else ""


def enhance_results(results: list[dict], mode: int) -> pd.DataFrame:
    """
    Given raw scan results, attach Score / Backtest% / ML% / FinalRank.
    Uses the central ALL_DATA store for zero-API backtest computation.
    Returns a DataFrame sorted by FinalScore DESC.
    """
    if not results:
        return pd.DataFrame()

    _eng_score, _eng_bt, _eng_ml, _eng_trap = get_engine_functions(mode)
    max_workers = 10

    tickers = [
        (r.get("Ticker") or r.get("Symbol") or "") + ".NS"
        if not (r.get("Ticker") or r.get("Symbol") or "").endswith(".NS")
        else (r.get("Ticker") or r.get("Symbol") or "")
        for r in results if isinstance(r, dict)
    ]
    # Data is already preloaded before run_scan, so we remove duplicate preload_all

    # ── Step 1: score all rows (fast, no I/O) ─────────────────────────
    pre_rows: list[dict] = []
    for r in results:
        sym = r.get("Ticker") or r.get("Symbol") or ""
        try:
            score, breakdown = _eng_score(r)
        except Exception:
            score, breakdown = 0.0, {}
        score = _safe(score, 0.0)
        pre_rows.append({
            "row":       r,
            "sym":       sym,
            "score":     round(score, 2),
            "breakdown": breakdown if isinstance(breakdown, dict) else {},
        })

    # ── Step 2: backtest top 50 only (zero-API via ALL_DATA) ──────────
    top_bt = {
        id(x["row"])
        for x in sorted(pre_rows, key=lambda x: x["score"], reverse=True)[:50]
    }

    def _process_enriched(pr: dict) -> dict:
        r         = pr["row"]
        sym       = pr["sym"]
        score     = pr["score"]
        breakdown = pr["breakdown"]

        # Backtest: zero-API — reads from ALL_DATA via backtest_with_preloaded
        if id(r) in top_bt:
            try:
                bt_prob = float(backtest_with_preloaded(mode, r, sym))
            except Exception:
                bt_prob = 50.0
        else:
            bt_prob = 50.0

        try:
            ml_prob = float(_eng_ml(r))
        except Exception:
            ml_prob = 50.0
        try:
            trap = _eng_trap(r)
        except Exception:
            trap = ""

        try:
            df_for_signal = get_df_for_ticker(sym)
            nd_signal = compute_next_day_signal(r, df_for_signal)
        except Exception:
            nd_signal = "❌ Error"

        bt_prob = _safe(bt_prob, 50.0)
        ml_prob = _safe(ml_prob, 50.0)
        final   = round(0.5 * score + 0.3 * bt_prob + 0.2 * ml_prob, 2)
        return {
            **r,
            "Score":       score,
            "_breakdown":  breakdown,
            "Backtest %":  round(bt_prob, 2),
            "ML %":        round(ml_prob, 2),
            "Final Score": final,
            "Trap":        trap,
            "Next-Day Signal": nd_signal,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_enriched, pr): pr for pr in pre_rows}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                pr    = futs[fut]
                r     = pr["row"]
                score = _safe(pr.get("score", 0.0), 0.0)
                rows.append({
                    **r,
                    "Score":       round(score, 2),
                    "_breakdown":  pr.get("breakdown", {}),
                    "Backtest %":  50.0,
                    "ML %":        50.0,
                    "Final Score": round(0.5 * score + 0.3 * 50.0 + 0.2 * 50.0, 2),
                    "Trap":        "",
                    "Next-Day Signal": "❌ Error",
                })

    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def _score_color(v: float) -> str:
    if v > 75:   return "#00d4a8"
    if v > 60:   return "#0094ff"
    if v > 40:   return "#f0b429"
    return "#ff4d6d"


def _score_label(v: float) -> str:
    if v > 75:   return "score-green"
    if v > 60:   return "score-blue"
    if v > 40:   return "score-yellow"
    return "score-red"


def render_top_picks(df: pd.DataFrame, n: int = 5) -> None:
    """Render the Top N pick cards in a horizontal strip."""
    st.markdown('<div class="section-lbl">🏅 Top Picks</div>', unsafe_allow_html=True)
    cols = st.columns(min(n, len(df)))
    for i, (col, (_, row)) in enumerate(zip(cols, df.head(n).iterrows())):
        sc   = row.get("Score",      0)
        bt   = row.get("Backtest %", 50)
        ml   = row.get("ML %",       50)
        fin  = row.get("Final Score", 0)
        trap = row.get("Trap",        "")
        sym  = row.get("Symbol",     "—")
        nd_sig= row.get("Next-Day Signal", "❌ No Data")
        c    = _score_color(sc)
        bd   = row.get("_breakdown", {})
        bd_html = "".join(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span>{k}</span>'
            f'<span style="color:{"#00d4a8" if v>=0 else "#ff4d6d"}">{v:+d}</span></div>'
            for k, v in bd.items()
        )
        with col:
            st.markdown(
                f'<div class="pick-card">'
                f'<div class="pick-rank">#{i+1}</div>'
                f'<div class="pick-sym">{sym}</div>'
                f'<div class="pick-score">'
                f'Score <span style="color:{c};font-weight:700">{sc:.0f}</span> &nbsp;|&nbsp; '
                f'BT <span style="color:#0094ff">{bt:.0f}%</span> &nbsp;|&nbsp; '
                f'ML <span style="color:#b08cff">{ml:.0f}%</span><br>'
                f'<b style="color:{c}">Final {fin:.1f}</b>'
                f'<br><span style="font-size:12px;color:#ccd9e8;font-weight:bold;">{nd_sig}</span>'
                f'{"&nbsp;&nbsp;<span class=trap-badge>" + trap + "</span>" if trap else ""}'
                f'</div>'
                f'<details style="margin-top:8px">'
                f'<summary style="font-size:11px;color:#4a6480;cursor:pointer">Score breakdown ▾</summary>'
                f'<div class="breakdown-box" style="margin-top:6px">{bd_html}</div>'
                f'</details>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR  (unchanged logic, unchanged options)
# ═════════════════════════════════════════════════════════════════════
mode_names  = {1:"Momentum", 2:"Balanced", 3:"Relaxed", 4:"Institutional", 5:"Intraday", 6:"Swing"}
mode_colors = {1:"#00d4a8",  2:"#0094ff",  3:"#f0b429", 4:"#b08cff", 5:"#ff8c00", 6:"#ff4d6d"}
pill_cls    = {1:"pill-m1",  2:"pill-m2",  3:"pill-m3", 4:"pill-m3", 5:"pill-m5", 6:"pill-m6"}

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:20px;'
        'color:#00d4a8;letter-spacing:-0.5px;padding:4px 0 16px 0;">'
        '<span class="live-dot"></span>NSE SENTINEL</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Strategy Mode</div>', unsafe_allow_html=True)

    mode_map = {
        "🟢  Mode 1 — Momentum (Breakout)":       1,
        "🔵  Mode 2 — Balanced (Swing)":          2,
        "🟡  Mode 3 — Relaxed (Early Setup)":     3,
        "🟣  Mode 4 — Institutional Strength":    4,
        "🟠  Mode 5 — Intraday":                  5,
        "🔴  Mode 6 — Swing":                     6,
    }
    mode_label = st.selectbox(
        "Strategy mode",
        list(mode_map.keys()),
        label_visibility="collapsed",
        key="strategy_mode",
    )
    mode = mode_map[mode_label]

    filter_data = {
        1: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.7× avg"),
            ("RSI","55 – 72"),("Price Floor","₹70"),
            ("10D High","Break above"),("Mkt Cap","> ₹500 Cr")],
        2: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.5× avg"),
            ("RSI","52 – 70"),("Price","₹50 – 1500"),
            ("15D High","Within 3%"),("Mkt Cap","> ₹500 Cr")],
        3: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.3× avg"),
            ("RSI","50 – 72"),("Price Floor","₹50"),
            ("20D High","Within 5%"),("Mkt Cap","—")],
        4: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.5× avg"),
            ("RSI","55 – 70"),("20D High","Within 2%"),
            ("Rel Strength","Stock > Nifty 20D"),("Price Action","Close > Open")],
        5: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.1× avg"),
            ("RSI","52 – 60"),("Price Floor","₹20"),
            ("10D High","Break above"),("Price Action","Close > Open")],
        6: [("EMA Trend","Close > EMA20 > EMA50"),("EMA20 Slope","Rising"),
            ("Volume","> 1.3× avg & > prev"),("RSI","53 – 59"),
            ("Price Floor","₹40"),("10D High","Break above")],
    }
    colors = {1:"#00d4a8",2:"#0094ff",3:"#f0b429",4:"#b08cff",5:"#ff8c00",6:"#ff4d6d"}
    mc = colors[mode]

    params_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480">{k}</span>'
        f'<span style="font-size:11px;color:{mc};font-weight:700">{v}</span></div>'
        for k, v in filter_data[mode]
    ])
    st.markdown(
        f'<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;'
        f'padding:12px 14px;margin-bottom:16px;">{params_html}</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Parallel Workers</div>', unsafe_allow_html=True)
    workers = st.slider(
        "Parallel workers",
        min_value=4,
        max_value=MAX_YF_CONCURRENCY,
        value=MAX_YF_CONCURRENCY,
        step=2,
        label_visibility="collapsed",
        key="workers",
    )
    st.markdown(
        f'<div style="text-align:center;font-size:11px;color:#4a6480;margin-top:-8px;">'
        f'<b style="color:{mc};font-size:18px">{workers}</b> worker threads '
        f'(Yahoo requests internally capped at {MAX_YF_CONCURRENCY})</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    scan_clicked = st.button("▶  SCAN MARKET NOW")
    market_bias_clicked = st.button("📊 Market Bias Engine", key="market_bias_engine_btn")
    csv_scan_clicked = st.button("📂 CSV Next-Day Potential", key="csv_next_day_btn")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Data Management Panel ─────────────────────────────────────
    st.markdown('<div class="section-lbl">📦 Local Data Cache</div>', unsafe_allow_html=True)
    if _DATA_DOWNLOADER_OK:
        if st.button("🔄 Download / Refresh Data Now", key="refresh_data_btn"):
            with st.spinner("Updating data..."):
                try:
                    _tickers_for_dl = fetch_nse_tickers()
                    update_all_data(_tickers_for_dl)
                    st.success("Data updated")
                except Exception as _e:
                    st.error(f"Data update failed: {_e}")
        # Show cache status
        try:
            _status = data_status_summary(fetch_nse_tickers())
            st.markdown(
                f'<div style="font-size:11px;color:#4a6480;line-height:1.9;">'
                f'Fresh: <b style="color:#00d4a8">{_status.get("fresh", "?")}</b> &nbsp;'
                f'Stale: <b style="color:#f0b429">{_status.get("stale", "?")}</b> &nbsp;'
                f'Missing: <b style="color:#ff4d6d">{_status.get("missing", "?")}</b></div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass
    else:
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;">'
            'data_downloader.py not found — using live yfinance.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#4a6480;line-height:1.7;">'
        'Data: Yahoo Finance (NSE)<br>Indicators: EMA · RSI · Volume<br>'
        'Universe: NSE equity archive<br><br>'
        '⚠️ Educational use only.<br>Not financial advice.</div>',
        unsafe_allow_html=True)

    # NEW: Isolated Market Bias Engine Button (EXACTLY below Download section)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">📊 PRO MARKET BIAS</div>', unsafe_allow_html=True)
    if st.button("📊 Market Bias Engine", key="market_bias_engine_btn_v4"):
        st.session_state["show_bias_engine"] = True


# ─────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────
mc = mode_colors[mode]

st.markdown(
    f'<div class="top-banner">'
    f'<div class="banner-logo"><span class="live-dot"></span>NSE SENTINEL</div>'
    f'<div style="margin-left:auto">'
    f'<span class="mode-pill {pill_cls[mode]}">MODE {mode} · {mode_names[mode].upper()}</span>'
    f'</div></div>',
    unsafe_allow_html=True)
st.markdown(
    f'<p style="color:#4a6480;font-size:12px;font-family:\'Space Mono\',monospace;'
    f'margin-top:-8px;margin-bottom:20px;">'
    f'Automated multi-strategy scanner for NSE equities · '
    f'{datetime.now().strftime("%d %b %Y, %H:%M")}</p>',
    unsafe_allow_html=True)

with st.spinner("Loading NSE ticker list..."):
    all_tickers = fetch_nse_tickers()
n = len(all_tickers)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("📋 NSE Tickers Loaded", f"{n:,}")
with c2: st.metric("🎯 Active Mode", f"M{mode} · {mode_names[mode]}")
with c3: st.metric("⚡ Workers", workers)
with c4:
    found_val  = len(st.session_state.get("results", [])) if "results" in st.session_state else None
    elapsed_v  = st.session_state.get("elapsed", None)
    if found_val is not None:
        st.metric("✅ Last Scan Found", found_val,
                  delta=f"{elapsed_v:.1f}s" if elapsed_v else None)
    else:
        st.metric("✅ Last Scan Found", "—")

st.markdown("<hr>", unsafe_allow_html=True)

# ── SCAN ──────────────────────────────────────────────────────────────
if scan_clicked:
    st.markdown(
        f'<div class="section-lbl">⏳ Scanning {n:,} NSE Equities — Mode {mode}: {mode_names[mode]}</div>',
        unsafe_allow_html=True)

    # Background ML pre-train (non-blocking daemon thread)
    if _SKLEARN_OK and _ML_MODEL is None:
        def _bg_train():
            try:
                from strategy_engines import get_train_function
                get_train_function(mode)()
            except Exception:
                train_model_once(all_tickers[:50])

        threading.Thread(target=_bg_train, daemon=True).start()

    results, elapsed = run_scan(all_tickers, mode, workers=workers)
    st.session_state.update({
        "results":   results,
        "scan_time": datetime.now().strftime("%H:%M:%S"),
        "elapsed":   elapsed,
        "mode":      mode,
    })
    st.rerun()

# ── MARKET BIAS ENGINE ─────────────────────────────────────────────
if market_bias_clicked:
    st.markdown('<div class="section-lbl">📊 Market Bias Engine</div>', unsafe_allow_html=True)
    with st.spinner("Analyzing Nifty (and Bank Nifty confirmation if available) …"):
        _mb = compute_market_bias(include_bank=True)

    st.markdown(f"**Market Bias:** {_mb.get('bias', 'Sideways / no edge')}")
    st.markdown(f"**Confidence:** {_mb.get('confidence', 50)}%")
    st.markdown(f"**Expected Next-Day Move:** {_mb.get('expected_range', '±0.30% to ±0.70%')}")
    st.markdown("**Breakdown (why):**")
    for _line in _mb.get("breakdown", []):
        st.write(f"- {_line}")

# ── NEW: MARKET BIAS UI PANEL (Isolated) ──────────────────────────────
if st.session_state.get("show_bias_engine"):
    st.markdown('<div class="section-lbl">📊 Market Bias Engine (Analytics)</div>', unsafe_allow_html=True)
    with st.spinner("Crunching latest Nifty (^NSEI) indicators..."):
        _bias_data = compute_market_bias_ui()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Bias", _bias_data["bias"])
    with col2:
        st.metric("Confidence", f"{_bias_data['confidence']}%")
    with col3:
        st.metric("Expected Move", _bias_data["expected_move"])
    
    with st.expander("Short Reason Breakdown", expanded=True):
        for r in _bias_data["reasons"]:
            st.markdown(f"- {r}")
    
    if st.button("Close Bias Panel"):
        st.session_state["show_bias_engine"] = False
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────────────
if "results" in st.session_state:
    results     = st.session_state["results"]
    stored_mode = st.session_state.get("mode", mode)
    elapsed_d   = st.session_state.get("elapsed", 0)
    scan_time_d = st.session_state.get("scan_time", "—")
    mc2         = mode_colors.get(stored_mode, "#00d4a8")

    st.markdown(
        f'<div class="result-hdr">'
        f'<h3>🏆 Bullish Candidates</h3>'
        f'<span class="mode-pill {pill_cls[stored_mode]}">M{stored_mode} · {mode_names[stored_mode].upper()}</span>'
        f'<span class="count-pill">{len(results)}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:#4a6480;">'
        f'Scanned at {scan_time_d} · {elapsed_d:.1f}s</span></div>',
        unsafe_allow_html=True)

    if results:
        # ── Summary metrics (unchanged) ──────────────────────────────
        df_raw = pd.DataFrame(results)
        i1, i2, i3 = st.columns(3)
        with i1: st.metric("📊 Avg RSI",      f"{df_raw['RSI'].mean():.1f}")
        with i2: st.metric("💰 Avg Price",    f"₹{df_raw['Price (₹)'].mean():,.0f}")
        with i3: st.metric("⚡ Avg Vol / Avg", f"{df_raw['Vol / Avg'].mean():.2f}×")

        # ── Enhance with scoring / backtest / ML ─────────────────────
        with st.spinner("🔢 Computing Smart Scores, Backtest & ML probabilities …"):
            df = enhance_results(results, stored_mode)

        if "Next-Day Signal" in df.columns:
            with st.sidebar:
                counts = df["Next-Day Signal"].value_counts().to_dict()
                sg = counts.get("🔥 Strong Green", 0)
                pu = counts.get("🟢 Possible Up", 0)
                ri = counts.get("⚠️ Risky (Late Entry)", 0)
                we = counts.get("❌ Weak Setup", 0)
                
                st.markdown('<hr><div class="section-lbl">📊 Next-Day Signals Summary</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;padding:12px 14px;margin-bottom:16px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    f'<span style="font-size:12px;color:#ccd9e8;">🔥 Strong Green</span>'
                    f'<span style="font-size:13px;color:#00d4a8;font-weight:700">{sg}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    f'<span style="font-size:12px;color:#ccd9e8;">🟢 Possible Up</span>'
                    f'<span style="font-size:13px;color:#0094ff;font-weight:700">{pu}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    f'<span style="font-size:12px;color:#ccd9e8;">⚠️ Risky (Late Entry)</span>'
                    f'<span style="font-size:13px;color:#f0b429;font-weight:700">{ri}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    f'<span style="font-size:12px;color:#ccd9e8;">❌ Weak Setup</span>'
                    f'<span style="font-size:13px;color:#ff4d6d;font-weight:700">{we}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


        # ── TASK 4: Section Headers & Spacing ─────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2>🔥 Top Picks</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 1: Top Picks Cards ───────────────────────────────────
        top_n = min(5, len(df))
        cols = st.columns(top_n)
        for i, (col, (_, row)) in enumerate(zip(cols, df.head(top_n).iterrows())):
            sym = row.get("Symbol", "—")
            fin   = row.get("Final Score", 0)
            bt    = row.get("Backtest %", 50)
            ml    = row.get("ML %", 50)
            rsi_v = float(row.get("RSI", 0))
            vol   = row.get("Vol / Avg", 0)
            trap  = row.get("Trap", "")
            tv_link = tv_chart_url(sym)
            nd_sig  = row.get("Next-Day Signal", "❌ No Data")

            # Score colour
            if fin > 75:     color = "#00d4a8"
            elif fin >= 60:  color = "#0094ff"
            elif fin >= 40:  color = "#f0b429"
            else:            color = "#ff4d6d"

            with col:
                trap_html = (
                    f'<div style="margin-top:8px;color:#ff4d6d;font-size:12px;'
                    f'font-weight:bold;background:rgba(255,77,109,0.1);'
                    f'padding:4px 8px;border-radius:4px;display:inline-block;">⚠️ {trap}</div>'
                    if trap else ""
                )
                st.markdown(
                    f'<div style="border:1px solid #1a2840;padding:16px;border-radius:8px;'
                    f'background:#0f1823;position:relative;">'
                    f'<div style="font-size:20px;font-weight:bold;margin-bottom:8px;">{sym}</div>'
                    f'<div style="font-size:32px;font-weight:bold;color:{color};margin-bottom:12px;">{fin:.1f}</div>'
                    f'<div style="font-size:14px;font-weight:bold;color:#ccd9e8;margin-bottom:8px;">{nd_sig}</div>'
                    f'<div style="font-size:14px;color:#ccd9e8;line-height:1.6;">'
                    f'<b>BT:</b> {bt:.1f}%&nbsp; '
                    f'<b>ML:</b> {ml:.1f}%&nbsp; '
                    f'<b>RSI:</b> {rsi_v:.1f}&nbsp; '
                    f'<b>Vol:</b> {vol:.1f}x'
                    f'</div>{trap_html}</div>',
                    unsafe_allow_html=True,
                )
                st.link_button("📈 TradingView", tv_link, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2>📊 Full Rankings</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 2 & 5: Clean Table with TradingView Link ─────────────
        table_df = df.copy()
        try:
            from strategy_engines._engine_utils import add_rank_score_columns
            table_df = add_rank_score_columns(table_df)
            if "rank_score" in table_df.columns:
                table_df = table_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
                table_df["Rank Score"] = table_df["rank_score"]
        except Exception:
            pass
        if "Rank Score" not in table_df.columns:
            table_df["Rank Score"] = 0.0
        table_df.insert(0, "Rank", range(1, len(table_df) + 1))
        table_df["Ticker"] = table_df["Symbol"]
        table_df["TradingView"] = table_df["Symbol"].apply(lambda s: tv_chart_url(s))

        display_cols = ["Rank", "Rank Score", "Ticker", "Score", "Backtest %", "ML %", "Final Score", "Trap", "Next-Day Signal", "TradingView"]
        display_cols = [c for c in display_cols if c in table_df.columns]
        
        st.dataframe(
            table_df[display_cols],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank"),
                "Rank Score": st.column_config.NumberColumn("Rank Score", format="%.2f"),
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Backtest %": st.column_config.NumberColumn("Backtest %", format="%.1f%%"),
                "ML %": st.column_config.NumberColumn("ML %", format="%.1f%%"),
                "Final Score": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "Trap": st.column_config.TextColumn("Trap"),
                "Next-Day Signal": st.column_config.TextColumn("Signal"),
                "TradingView": st.column_config.LinkColumn("TradingView Link", display_text="📈 Open Chart"),
            },
            use_container_width=True,
            hide_index=True
        )

        # ── TASK 3: Expandable Details ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            sym = row.get("Symbol", "—")
            fin = row.get("Final Score", 0)
            with st.expander(f"🔍 {sym} Details (Final Score: {fin:.1f})"):
                brk_col, ind_col = st.columns([1, 2])
                with brk_col:
                    st.markdown("**Score Breakdown**")
                    st.json(row.get("_breakdown", {}))
                with ind_col:
                    st.markdown("**Key Indicators**")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("RSI", f"{row.get('RSI', 0):.1f}")
                    ic2.metric("EMA 20", f"₹{row.get('EMA 20', 0):.2f}")
                    ic3.metric("EMA 50", f"₹{row.get('EMA 50', 0):.2f}")
                    ic4.metric("Vol / Avg", f"{row.get('Vol / Avg', 0):.2f}x")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("5D Return", f"{row.get('5D Return (%)', 0):+.2f}%")
                    rc2.metric("20D Return", f"{row.get('20D Return (%)', 0):+.2f}%")
                    
                    # Handle possible missing columns gracefully
                    h_val = row.get('Δ vs 20D High (%)', 0)
                    rc3.metric("Δ vs 20D Hi", f"{h_val:+.2f}%" if pd.notna(h_val) else "—")

        # ── CSV download ──────────────────────────────────────────────
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv_buf = io.StringIO()
        # Drop internal fields for clean CSV
        csv_export = table_df.drop(columns=["_breakdown", "TradingView", "Rank", "📈 Chart"], errors="ignore")
        csv_export.to_csv(csv_buf, index=False)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv_buf.getvalue().encode(),
                file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}_mode{stored_mode}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── ML status note ───────────────────────────────────────────
        if not _SKLEARN_OK:
            st.info("ℹ️  scikit-learn not installed — ML % column shows neutral 50. "
                    "Run `pip install scikit-learn` to enable ML probability.")

    else:
        if not market_bias_clicked:
            st.markdown(
                f'<div style="text-align:center;padding:60px 24px;background:#0f1823;'
                f'border:1px solid #1a2840;border-radius:12px;">'
                f'<div style="font-size:48px;opacity:0.3;margin-bottom:16px;">📡</div>'
                f'<div style="color:#4a6480;font-size:13px;line-height:1.7;">'
                f'No stocks matched Mode {stored_mode} criteria.<br>'
                f'Try <b style="color:#ccd9e8">Mode 3 (Relaxed)</b> for a broader scan.</div></div>',
                unsafe_allow_html=True)
else:
    if not market_bias_clicked:
        st.markdown(
            '<div style="text-align:center;padding:64px 24px;background:#0f1823;'
            'border:1px solid #1a2840;border-radius:12px;">'
            '<div style="font-size:52px;opacity:0.25;margin-bottom:18px;">📡</div>'
            '<div style="color:#4a6480;font-size:14px;line-height:1.8;">'
            'Select a <b style="color:#ccd9e8">strategy mode</b> in the sidebar<br>'
            'then click <b style="color:#00d4a8">▶ SCAN MARKET NOW</b> to begin.'
            '</div></div>',
            unsafe_allow_html=True)


# ── CSV NEXT-DAY POTENTIAL (Completely standalone — no analyse/run_scan) ──────
if csv_scan_clicked:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h2>📂 CSV Next-Day Potential</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
        'Fast local scan using cached CSV data · Independent of main screener</div>',
        unsafe_allow_html=True
    )

    if not _DATA_DOWNLOADER_OK:
        st.warning("data_downloader.py not found — CSV scan unavailable.")
    else:
        def _compute_rsi(close: pd.Series, period: int = 14) -> float:
            try:
                d = close.diff()
                g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
                l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
                rs = g / l.replace(0, float("nan"))
                rsi_s = 100 - (100 / (1 + rs))
                return float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
            except Exception:
                return 50.0

        def _scan_csv_ticker(ticker: str) -> dict | None:
            """Return a result dict if ticker passes next-day criteria, else None."""
            try:
                df = load_csv(ticker)
                if df is None or len(df) < 25:
                    return None
                df = df.tail(60).copy()
                close  = df["Close"].astype(float)
                volume = df["Volume"].astype(float)
                if len(close) < 21:
                    return None
                ema20     = close.ewm(span=20, adjust=False).mean()
                ema20_val = float(ema20.iloc[-1])
                close_val = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                avg_vol   = float(volume.rolling(20, min_periods=10).mean().iloc[-2])
                last_vol  = float(volume.iloc[-1])
                vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0
                rsi_val   = _compute_rsi(close)
                high_10   = float(close.tail(10).max())
                near_10d_high = close_val >= high_10 * 0.98

                cond_rsi   = 50.0 <= rsi_val <= 65.0
                cond_vol   = vol_ratio >= 1.3
                cond_ema   = close_val > ema20_val > 0
                cond_close = close_val > prev_close
                cond_high  = near_10d_high

                if cond_rsi and cond_vol and cond_ema and cond_close and cond_high:
                    return {
                        "Ticker":      ticker.replace(".NS", ""),
                        "Close (₹)":   round(close_val, 2),
                        "RSI":         round(rsi_val, 1),
                        "Vol / Avg":   round(vol_ratio, 2),
                        "EMA20":       round(ema20_val, 2),
                        "Near 10D Hi": "✅ Yes",
                        "TradingView": tv_chart_url(ticker.replace(".NS", "")),
                    }
                return None
            except Exception:
                return None   # fail-safe: never crash

        csv_tickers = [t if t.endswith(".NS") else t + ".NS" for t in all_tickers]
        csv_results: list[dict] = []

        with st.spinner("📂 Scanning local CSVs …"):
            with ThreadPoolExecutor(max_workers=16) as _ex:
                _futs = {_ex.submit(_scan_csv_ticker, t): t for t in csv_tickers}
                for _fut in as_completed(_futs):
                    try:
                        res = _fut.result()
                        if res is not None:
                            csv_results.append(res)
                    except Exception:
                        pass   # fail-safe: never crash

        if csv_results:
            csv_df = pd.DataFrame(csv_results).sort_values("RSI", ascending=True).reset_index(drop=True)
            st.success(f"✅ {len(csv_df)} stocks matched next-day CSV criteria")
            st.dataframe(
                csv_df,
                column_config={
                    "Ticker":      st.column_config.TextColumn("Ticker"),
                    "Close (₹)":   st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
                    "RSI":         st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Vol / Avg":   st.column_config.NumberColumn("Vol/Avg", format="%.2fx"),
                    "EMA20":       st.column_config.NumberColumn("EMA20", format="₹%.2f"),
                    "Near 10D Hi": st.column_config.TextColumn("Near 10D Hi"),
                    "TradingView": st.column_config.LinkColumn("Chart", display_text="📈 Open"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No stocks matched CSV next-day criteria. Run '🔄 Download / Refresh Data Now' first if CSVs are missing.")


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:\'Space Mono\',monospace;'
    'font-size:11px;color:#2a3f58;padding:8px 0 16px;">'
    'NSE SENTINEL · Python + Streamlit + yFinance &nbsp;|&nbsp; '
    'For educational purposes only — not financial advice</div>',
    unsafe_allow_html=True)