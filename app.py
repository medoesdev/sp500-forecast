"""SP500 Forecast - FastAPI backend.

Run:  python -m uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

import data_pull
import features
import forecast
import report


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the cache once at startup so the first request is fast.
    data_pull.get_all_market()
    yield


app = FastAPI(
    title="S&P 500 Forecast",
    description="Keyless-data prototype: probability cones for the S&P 500.",
    version="0.1.0",
    lifespan=lifespan,
)


def _compute_payload() -> dict:
    market = data_pull.get_all_market(force=False)
    feats = features.compute_features(market)
    cones = forecast.forecast_cones(market["spx"], feats)
    path = forecast.projected_path(market["spx"], feats)
    report_payload = report.build_report(cones, feats, market["spx"])
    return {
        "features": feats,
        "forecast": cones,
        "cone_path": [
            {"date": d.strftime("%Y-%m-%d"), **row}
            for d, row in path.iterrows()
        ],
        "report": report_payload,
    }


@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "endpoints": ["/health", "/prices", "/features", "/forecast", "/report", "/all"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": app.title, "version": app.version}


@app.get("/prices")
def prices() -> dict:
    market = data_pull.get_all_market(force=False)
    spx = market["spx"]["close"].dropna()
    vix = market["vix"]["close"].dropna()
    return {
        "as_of": spx.index.max().strftime("%Y-%m-%d"),
        "spx": {
            "last_close": round(float(spx.iloc[-1]), 2),
            "1d_chg_pct": round(
                (float(spx.iloc[-1]) / float(spx.iloc[-2]) - 1.0) * 100.0, 2
            )
            if len(spx) > 1
            else None,
            "recent": [
                {"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 2)}
                for d, v in spx.tail(30).items()
            ],
        },
        "vix": {
            "last": round(float(vix.iloc[-1]), 2),
            "recent": [
                {"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 2)}
                for d, v in vix.tail(30).items()
            ],
        },
    }


@app.get("/features")
def get_features() -> dict:
    market = data_pull.get_all_market(force=False)
    return features.compute_features(market)


@app.get("/forecast")
def get_forecast() -> dict:
    market = data_pull.get_all_market(force=False)
    feats = features.compute_features(market)
    return forecast.forecast_cones(market["spx"], feats)


@app.get("/report")
def get_report() -> dict:
    market = data_pull.get_all_market(force=False)
    feats = features.compute_features(market)
    cones = forecast.forecast_cones(market["spx"], feats)
    return report.build_report(cones, feats, market["spx"])


@app.get("/all")
def get_all() -> dict:
    return _compute_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)