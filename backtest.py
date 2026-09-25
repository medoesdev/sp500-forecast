"""Walk-forward calibration backtest for the v0.1 cone model.

For every historical date (trading day) with enough lookback, the model is
re-run using ONLY data available up to that date (no lookahead), producing
the same cones the live app would have shown that morning. We then check,
for each horizon, whether the price that actually materialized h days later
fell inside the 50% band (p25-p75) and the 80% band (p10-p90).

Interpretation:
- Well calibrated  ->  ~50% of outcomes inside the 50% band, ~80% inside 80%.
- Systematically lower hit rates  ->  cones too narrow; model understates risk.
- Median gap (realized vs median) ~0  ->  no systematic drift bias.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

import data_pull
import features
import forecast

MIN_HISTORY = 200       # trading days of lookback required before a test date
HORIZONS = (1, 5, 21, 63)   # 1 day, 1 week, 1 month, 1 quarter
CACHE_PATH = os.path.join(data_pull.DATA_DIR, "backtest.json")
CACHE_TTL = 86400       # re-run at most daily unless forced


def _market_snapshot(spx: pd.DataFrame, vix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A market dict with macro feeds empty (the cone model only needs px+VIX)."""
    empty = pd.DataFrame(columns=["value"]).astype({"value": float})
    return {"spx": spx, "vix": vix, "yield10y": empty, "cpi": empty, "shortrate": empty}


def run_backtest() -> dict:
    spx = data_pull.get_spx()["close"].dropna()
    vix = data_pull.get_vix_long()["close"].dropna()

    n = len(spx)
    start = MIN_HISTORY
    end = n - max(HORIZONS) - 1
    dates_index = spx.index

    rows = []
    for i in range(start, end + 1):
        asof = dates_index[i]
        prices_hist = spx.iloc[: i + 1].to_frame("close")
        vix_hist = vix[vix.index <= asof].to_frame("close")

        feats = features.compute_features(_market_snapshot(prices_hist, vix_hist))
        cones = forecast.forecast_cones(prices_hist, feats, horizons=HORIZONS)

        for hkey in ("1", "5", "21", "63"):
            c = cones["cones"][hkey]
            h = int(hkey)
            realized = float(spx.iloc[i + h])  # price h trading days later
            rows.append(
                {
                    "date": asof,
                    "horizon": hkey,
                    "last_close": c["last_close"],
                    "p10": c["p10"],
                    "p25": c["p25"],
                    "p50": c["p50"],
                    "p75": c["p75"],
                    "p90": c["p90"],
                    "realized": realized,
                    "in50": c["p25"] <= realized <= c["p75"],
                    "in80": c["p10"] <= realized <= c["p90"],
                    "prob_up": c["prob_up_pct"],
                }
            )

    btdf = pd.DataFrame(rows)

    per_horizon = {}
    for hkey in ("1", "5", "21", "63"):
        g = btdf[btdf.horizon == hkey]
        if g.empty:
            continue
        hit50 = float(g.in50.mean() * 100.0)
        hit80 = float(g.in80.mean() * 100.0)
        med_gap = float((g.realized / g.p50 - 1.0).median() * 100.0)
        avg_gap = float((g.realized - g.p50).mean() / g.last_close.mean() * 100.0)
        up_pred = g.p50 > g.last_close
        up_actual = g.realized > g.last_close
        dir_acc = float((up_pred == up_actual).mean() * 100.0)
        per_horizon[hkey] = {
            "label": forecast.HORIZON_LABELS[hkey],
            "n": int(len(g)),
            "hit50_pct": round(hit50, 1),
            "expected50": 50.0,
            "hit80_pct": round(hit80, 1),
            "expected80": 80.0,
            "median_gap_pct": round(med_gap, 2),
            "avg_gap_pct": round(avg_gap, 2),
            "directional_acc_pct": round(dir_acc, 1),
        }

    # Directional (P(up)) bucket calibration across all horizons pooled.
    buckets = {}
    for label, mask in (
        ("< 40%", btdf.prob_up < 40),
        ("40-60%", (btdf.prob_up >= 40) & (btdf.prob_up <= 60)),
        ("> 60%", btdf.prob_up > 60),
    ):
        g = btdf[mask]
        if len(g) == 0:
            continue
        buckets[label] = {
            "n": int(len(g)),
            "avg_prob_up": round(float(g.prob_up.mean()), 1),
            "actual_up_pct": round(float((g.realized > g.last_close).mean() * 100.0), 1),
        }

    return {
        "as_of": dates_index[-1].strftime("%Y-%m-%d"),
        "test_dates": int(len(btdf) / 4),
        "window": f"{dates_index[start].date()} .. {dates_index[end].date()}",
        "per_horizon": per_horizon,
        "prob_up_buckets": buckets,
        "model": forecast.forecast_cones(
            spx.to_frame("close"),
            features.compute_features(_market_snapshot(spx.to_frame("close"), vix.to_frame("close"))),
        ).get("model"),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_backtest(force: bool = False, ttl_seconds: float = CACHE_TTL) -> dict:
    if not force and os.path.exists(CACHE_PATH):
        try:
            if time.time() - os.path.getmtime(CACHE_PATH) < ttl_seconds:
                with open(CACHE_PATH, encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception:  # corrupt cache -> recompute
            pass
    result = run_backtest()
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return result


if __name__ == "__main__":
    bt = get_backtest(force=True)
    print(f"Backtest window: {bt['window']}  ({bt['test_dates']} test days)")
    for hkey in ("1", "5", "21", "63"):
        s = bt["per_horizon"][hkey]
        print(
            f"  {s['label']:<9} n={s['n']:>3}  hit50={s['hit50_pct']:>5}% (exp 50)  "
            f"hit80={s['hit80_pct']:>5}% (exp 80)  dirAcc={s['directional_acc_pct']:>5}%  "
            f"medGap={s['median_gap_pct']:+.2f}%"
        )