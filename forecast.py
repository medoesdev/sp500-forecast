"""Forecast engine: honest probability cones via a calibrated random walk.

Methodology (deliberately transparent for a v0.1 prototype):
- The base assumption is a random walk: expected future returns are ~0
  (market efficiency), plus a small momentum drift capped at +/-0.05%/day.
- Daily volatility comes from a variance blend of 20-day realized vol and
  the VIX implied vol (both annualized), floored at 8% to avoid absurdly
  narrow bands in dead-calm markets.
- Horizon-vol scaling uses sigma * sqrt(h), the standard i.i.d. model.
- Quantiles come from the normal CDF (stdlib NormalDist), so results are
  deterministic and reproducible.

This is intentionally NOT a "point prediction". The output is a range that
is wide when uncertainty is high and narrow when it isn't.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_ANNUAL_VOL = 0.08
REALIZED_WEIGHT = 0.7
VIX_WEIGHT = 0.3
MAX_DAILY_DRIFT = 0.0005  # ~ +/-12% annualized drift cap
HORIZONS = (1, 5, 21, 63)  # days: 1 day, 1 week, 1 month, 1 quarter
HORIZON_LABELS = {"1": "Daily", "5": "Weekly", "21": "Monthly", "63": "Quarterly"}


def _blended_daily_vol(features: dict) -> float:
    realized_ann = features.get("vol20_ann")
    if realized_ann is None or not math.isfinite(realized_ann):
        realized_ann = 0.15
    vix = features.get("vix")
    vix_ann = vix / 100.0 if vix else 0.15
    realized_ann = max(realized_ann, MIN_ANNUAL_VOL)
    vix_ann = max(vix_ann, MIN_ANNUAL_VOL)
    var = REALIZED_WEIGHT * realized_ann**2 + VIX_WEIGHT * vix_ann**2
    vol_ann = math.sqrt(var)
    return vol_ann / math.sqrt(TRADING_DAYS_PER_YEAR)


def _daily_drift(features: dict) -> float:
    ret_63d = features.get("ret_63d")
    if ret_63d is None:
        return 0.0
    drift = ret_63d / 63.0
    return max(-MAX_DAILY_DRIFT, min(MAX_DAILY_DRIFT, drift))


def _price_quantiles(log_vol_day: float, day: int, drift_day: float, last_close: float) -> dict:
    mu = drift_day * day
    sigma = log_vol_day * math.sqrt(day)
    z = NormalDist()
    out = {}
    for q in QUANTILES:
        out[f"p{int(q * 100)}"] = last_close * math.exp(mu + z.inv_cdf(q) * sigma)
    out["prob_up_pct"] = z.cdf(mu / sigma) * 100.0 if sigma > 0 else 50.0
    return out


def forecast_cones(
    prices: pd.DataFrame, features: dict, horizons: tuple[int, ...] = HORIZONS
) -> dict:
    closes = prices["close"].dropna()
    last_close = float(closes.iloc[-1])
    log_vol_day = _blended_daily_vol(features)
    drift_day = _daily_drift(features)
    vol_annual_pct = log_vol_day * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

    cones = {}
    for h in horizons:
        q = _price_quantiles(log_vol_day, h, drift_day, last_close)
        cones[str(h)] = {
            "horizon_days": h,
            "label": HORIZON_LABELS.get(str(h), f"{h}d"),
            "last_close": round(last_close, 2),
            "p10": round(q["p10"], 2),
            "p25": round(q["p25"], 2),
            "p50": round(q["p50"], 2),
            "p75": round(q["p75"], 2),
            "p90": round(q["p90"], 2),
            "prob_up_pct": round(q["prob_up_pct"], 1),
            "median_pct_change": round((q["p50"] / last_close - 1.0) * 100.0, 2),
        }

    return {
        "as_of": features.get("as_of"),
        "last_close": round(last_close, 2),
        "vol_annual_pct": round(vol_annual_pct, 1),
        "vol_daily_pct": round(log_vol_day * 100.0, 2),
        "drift_daily_pct": round(drift_day * 100.0, 3),
        "model": "random walk + VIX/realized vol blend",
        "cones": cones,
    }


def projected_path(
    prices: pd.DataFrame,
    features: dict,
    horizon_days: int = 63,
) -> pd.DataFrame:
    """Daily quantile path for the fan chart, for the next `horizon_days` days."""
    closes = prices["close"].dropna()
    last_close = float(closes.iloc[-1])
    last_date = closes.index.max()
    log_vol_day = _blended_daily_vol(features)
    drift_day = _daily_drift(features)

    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
    rows = []
    for i, d in enumerate(future_dates, start=1):
        q = _price_quantiles(log_vol_day, i, drift_day, last_close)
        rows.append(
            {
                "date": d,
                "p10": q["p10"],
                "p25": q["p25"],
                "p50": q["p50"],
                "p75": q["p75"],
                "p90": q["p90"],
            }
        )
    return pd.DataFrame(rows).set_index("date")