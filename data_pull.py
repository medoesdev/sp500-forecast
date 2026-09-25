"""Keyless data fetching with a tiny file cache for the SP500 Forecast prototype.

Sources (verified working WITHOUT API keys):
- Yahoo Finance chart JSON API:  ^GSPC (S&P 500), ^VIX, ^TNX (10Y yield). Keyless,
  fast, reliable -> the primary host for price-like data.
- Alpha Vantage demo key:  CPI monthly series (keyless `apikey=demo`).
- FRED fredgraph.csv:  Fed funds (DFF) as a graceful-best-effort source.
  NOTE: FRED drops persistent connections hard, so every FRED call uses a
  FRESH requests connection with a short timeout - single calls work fine.
- stooq CSV: fallback for S&P 500 daily closes.
"""
from __future__ import annotations

import io
import json
import os
import time
from typing import Callable
from urllib.parse import quote

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 30
MAX_ATTEMPTS = 2

# Session reused ONLY for Yahoo/stooq (they tolerate keep-alive).
SESSION = requests.Session()
SESSION.headers.update(UA)


def _get(url: str, **kwargs) -> requests.Response:
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return SESSION.get(url, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last = exc
            time.sleep(1 + attempt)
    assert last is not None
    raise last


def _get_fresh(url: str, timeout: int = 15, attempts: int = 2) -> requests.Response:
    """Fresh connection per call - FRED resets keep-alive connections."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return requests.get(url, headers=UA, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
            time.sleep(1 + attempt)
    assert last is not None
    raise last

# --------------------------------------------------------------------------- #
# Tiny file cache: stores a CSV plus a JSON sidecar with the fetch timestamp. #
# --------------------------------------------------------------------------- #


def _cache_paths(name: str) -> tuple[str, str]:
    return (
        os.path.join(DATA_DIR, f"{name}.csv"),
        os.path.join(DATA_DIR, f"{name}.json"),
    )


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def _cached(name: str, ttl_seconds: float, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    csv_path, meta_path = _cache_paths(name)
    now = time.time()
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if now - float(meta.get("fetched", 0.0)) < ttl_seconds:
                df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
                return _normalize_index(df)
        except Exception:  # pragma: no cover - corrupt cache, refetch
            pass
    df = _normalize_index(fetch())
    df.reset_index().to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"fetched": now}, fh)
    return df


def _refresh(name: str, ttl: float, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    df = _normalize_index(fetch())
    csv_path, meta_path = _cache_paths(name)
    df.reset_index().to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"fetched": time.time()}, fh)
    return df

# --------------------------------------------------------------------------- #
# Yahoo Finance chart JSON (keyless, the endpoint yfinance wraps)             #
# --------------------------------------------------------------------------- #


def _yahoo_chart(symbol: str, range_: str, interval: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?range={range_}&interval={interval}"
    resp = _get(url)
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    idx = (
        pd.to_datetime(ts, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
    )
    df = pd.DataFrame({"close": closes}, index=idx)
    return df.dropna()

# --------------------------------------------------------------------------- #
# FRED CSV via a FRESH connection (no session reuse - server resets them)     #
# --------------------------------------------------------------------------- #


def _fred_single(series_id: str, timeout: int = 15, attempts: int = 2) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = _get_fresh(url, timeout=timeout, attempts=attempts)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    df.columns = [str(c).strip().lower() for c in df.columns]  # "DATE"/"observation_date"
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=[series_id]).set_index("date")
    return df[[series_id]]

# --------------------------------------------------------------------------- #
# Alpha Vantage CPI (demo key is keyless)                                     #
# --------------------------------------------------------------------------- #


def _av_cpi() -> pd.DataFrame:
    url = "https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=demo"
    resp = _get_fresh(url)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not data:
        raise RuntimeError(f"Alpha Vantage CPI: unexpected payload {str(payload)[:120]}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"].str.slice(0, 7))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")
    return df[["value"]].sort_index()

# --------------------------------------------------------------------------- #
# stooq CSV fallback for S&P 500                                              #
# --------------------------------------------------------------------------- #


def _stooq_spx() -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=%5Espx&i=d"
    resp = _get(url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], na_values="")
    df = df.set_index("Date").rename(columns=str.lower)
    return df["close"].dropna().to_frame()

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

SPX_RANGE = "2y"
VIX_RANGE = "6mo"
TNX_RANGE = "1y"


def get_spx(ttl_seconds: float = 900) -> pd.DataFrame:
    """Daily S&P 500 closes (column: close). Yahoo primary, stooq fallback."""
    try:
        return _cached("spx", ttl_seconds, lambda: _yahoo_chart("^GSPC", SPX_RANGE))
    except Exception:
        return _cached("spx", ttl_seconds, _stooq_spx)


def get_vix(ttl_seconds: float = 900) -> pd.DataFrame:
    """Daily VIX closes (column: close), last 6 months."""
    return _cached("vix", ttl_seconds, lambda: _yahoo_chart("^VIX", VIX_RANGE))


def get_dgs10(ttl_seconds: float = 86400) -> pd.DataFrame:
    """10Y Treasury yield in percent (column: value). Yahoo ^TNX primary,
    FRED DGS10 fallback."""
    def primary():
        df = _yahoo_chart("^TNX", TNX_RANGE)
        return df.rename(columns={"close": "value"})

    try:
        return _cached("yield10y", ttl_seconds, primary)
    except Exception:
        try:
            return _cached("yield10y", ttl_seconds, lambda: _fred_single("DGS10").rename(columns={"DGS10": "value"}))
        except Exception:
            return pd.DataFrame(columns=["value"]).astype({"value": float})


def get_cpi(ttl_seconds: float = 86400) -> pd.DataFrame:
    """Monthly CPI index (column: value). Alpha Vantage demo primary,
    FRED fallback. Returns an empty frame when both fail."""
    try:
        return _cached("cpi", ttl_seconds, _av_cpi)
    except Exception:
        try:
            return _cached("cpi", ttl_seconds, lambda: _fred_single("CPIAUCSL").rename(columns={"CPIAUCSL": "value"}))
        except Exception:
            return pd.DataFrame(columns=["value"]).astype({"value": float})


_DFF_FAIL_MARKER = os.path.join(DATA_DIR, "shortrate_dff.json")


def _dff_failed_recently(within: float = 3600.0) -> bool:
    """Remember FRED DFF failures so we don't pay the timeout tax every call."""
    if not os.path.exists(_DFF_FAIL_MARKER):
        return False
    try:
        with open(_DFF_FAIL_MARKER, encoding="utf-8") as fh:
            checked = float(json.load(fh).get("checked", 0.0))
        return time.time() - checked < within
    except Exception:
        return False


def _mark_dff_failure() -> None:
    with open(_DFF_FAIL_MARKER, "w", encoding="utf-8") as fh:
        json.dump({"checked": time.time()}, fh)


def get_shortrate(ttl_seconds: float = 86400) -> pd.DataFrame:
    """Short-term rate (column: value). FRED DFF primary, Yahoo ^IRX fallback.
    Best effort either way - never raises."""

    def fallback() -> pd.DataFrame:
        return _cached(
            "shortrate_irx", ttl_seconds,
            lambda: _yahoo_chart("^IRX", "1y").rename(columns={"close": "value"}),
        )

    if not _dff_failed_recently():
        try:
            return _cached(
                "shortrate_dff", ttl_seconds,
                lambda: _fred_single("DFF", timeout=10, attempts=1).rename(
                    columns={"DFF": "value"}
                ),
            )
        except Exception:
            _mark_dff_failure()
    try:
        return fallback()
    except Exception:
        return pd.DataFrame(columns=["value"]).astype({"value": float})


def shortrate_source() -> str:
    """Which short-rate source the cache actually used."""
    dff = os.path.join(DATA_DIR, "shortrate_dff.csv")
    irx = os.path.join(DATA_DIR, "shortrate_irx.csv")
    if os.path.exists(dff):
        return "effective Fed funds rate (FRED DFF)"
    if os.path.exists(irx):
        return "3-month Treasury bill (Yahoo ^IRX)"
    return "no short-rate data available"


def get_all_market(force: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience bundle used by the app. Canonical column names:
    spx/vix -> close ; yield10y/cpi/fedfunds -> value."""
    if force:
        spx = _refresh("spx", 900, lambda: _yahoo_chart("^GSPC", SPX_RANGE))
        vix = _refresh("vix", 900, lambda: _yahoo_chart("^VIX", VIX_RANGE))
    else:
        spx = get_spx()
        vix = get_vix()
    return {
        "spx": spx,
        "vix": vix,
        "yield10y": get_dgs10(),
        "cpi": get_cpi(),
        "shortrate": get_shortrate(),
    }


def refresh_all() -> dict[str, pd.DataFrame]:
    return get_all_market(force=True)


if __name__ == "__main__":
    probes = [
        ("spx", get_spx),
        ("vix", get_vix),
        ("dgs10", get_dgs10),
        ("cpi", get_cpi),
        ("shortrate", get_shortrate),
    ]
    for name, fetcher in probes:
        t0 = time.time()
        try:
            df = fetcher()
            print(f"{name}: {len(df)} rows, last {df.index.max().date()} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED {type(exc).__name__} ({time.time() - t0:.1f}s)", flush=True)
    market = get_all_market()
    print(f"\nLatest S&P 500 close: {market['spx']['close'].iloc[-1]:,.2f}", flush=True)