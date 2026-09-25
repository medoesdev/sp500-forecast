"""Feature engineering: turns raw market data into a snapshot of market state.

All features are plain numbers so the forecast engine and report writer can
consume them without touching data frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_pull

TRADING_DAYS_PER_YEAR = 252


def _log_return(prices: pd.Series, days: int) -> float | None:
    if len(prices) <= days:
        return None
    return float(np.log(prices.iloc[-1] / prices.iloc[-1 - days]))


def _realized_vol(prices: pd.Series, window: int) -> float | None:
    rets = np.log(prices).diff().dropna()
    if len(rets) < window:
        return None
    return float(rets.iloc[-window:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _rsi(prices: pd.Series, window: int = 14) -> float | None:
    delta = prices.diff().dropna()
    if len(delta) < window + 1:
        return None
    up = delta.clip(lower=0.0).rolling(window).mean()
    down = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = up / down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return float(out.iloc[-1]) if not np.isnan(out.iloc[-1]) else None


def compute_features(market: dict[str, pd.DataFrame]) -> dict:
    spx = market["spx"]["close"].dropna()
    vix = market["vix"]["close"].dropna()
    dgs10 = market["yield10y"]["value"].dropna()
    cpi = market["cpi"]["value"].dropna()
    dff = market["shortrate"]["value"].dropna()

    last_close = float(spx.iloc[-1])
    prev_close = float(spx.iloc[-2]) if len(spx) > 1 else last_close

    # --- S&P 500: momentum, trend, technicals ----------------------------- #
    f: dict = {}
    f["last_close"] = last_close
    f["prev_close"] = prev_close
    f["daily_return_pct"] = (last_close / prev_close - 1.0) * 100.0
    f["ret_1d"] = _log_return(spx, 1)
    f["ret_5d"] = _log_return(spx, 5)
    f["ret_21d"] = _log_return(spx, 21)
    f["ret_63d"] = _log_return(spx, 63)

    vol20 = _realized_vol(spx, 20)
    vol60 = _realized_vol(spx, 60)
    f["vol20_ann"] = vol20
    f["vol60_ann"] = vol60

    f["sma50"] = float(spx.rolling(50).mean().iloc[-1]) if len(spx) >= 50 else None
    f["sma200"] = float(spx.rolling(200).mean().iloc[-1]) if len(spx) >= 200 else None
    if f["sma200"]:
        f["dist_from_sma200_pct"] = (last_close / f["sma200"] - 1.0) * 100.0
    else:
        f["dist_from_sma200_pct"] = None

    hi_52w = float(spx.iloc[-252:].max()) if len(spx) >= 252 else float(spx.max())
    lo_52w = float(spx.iloc[-252:].min()) if len(spx) >= 252 else float(spx.min())
    f["high_52w_pct"] = (last_close / hi_52w - 1.0) * 100.0
    f["low_52w_pct"] = (last_close / lo_52w - 1.0) * 100.0
    f["rsi14"] = _rsi(spx)

    # --- VIX: level, percentile, and the volatility risk premium ----------- #
    vix_last = float(vix.iloc[-1])
    f["vix"] = vix_last
    f["vix_20d_avg"] = float(vix.iloc[-20:].mean()) if len(vix) >= 20 else vix_last
    if len(vix) >= 60:
        window = vix.iloc[-126:]
        f["vix_pctile_6mo"] = float((window < vix_last).mean() * 100.0)
    else:
        f["vix_pctile_6mo"] = 50.0

    if vol20:
        # Ratio > 1 means options imply more fear than recent realized moves.
        f["vix_vs_realized_ratio"] = vix_last / (vol20 * 100.0) if vol20 > 0 else None
    else:
        f["vix_vs_realized_ratio"] = None

    # --- Macro: rates and inflation ---------------------------------------- #
    f["yield10y"] = float(dgs10.iloc[-1]) if len(dgs10) >= 1 else None
    f["yield10y_chg_20d_pts"] = (
        float(dgs10.iloc[-1] - dgs10.iloc[-21]) if len(dgs10) > 20 else None
    )

    f["fed_funds"] = float(dff.iloc[-1]) if len(dff) >= 1 else None
    f["fed_funds_chg_1m_pts"] = (
        float(dff.iloc[-1] - dff.iloc[-6]) if len(dff) > 5 else None
    )
    f["fed_source"] = data_pull.shortrate_source()

    if len(cpi) >= 2:
        f["cpi_mom_pct"] = float((cpi.iloc[-1] / cpi.iloc[-2] - 1.0) * 100.0)
        f["cpi_ymom"] = cpi.index[-1].strftime("%Y-%m")
    else:
        f["cpi_mom_pct"] = None
        f["cpi_ymom"] = None

    f["as_of"] = spx.index.max().strftime("%Y-%m-%d")
    return f