# S&P 500 Forecast — v0.1 prototype

Keyless-data web app that shows **probability cones** (not point predictions) for the
S&P 500 at 1-day / 1-week / 1-month / 1-quarter horizons, with a driver report
explaining what's moving the outlook.

> **Not investment advice.** Bands are model estimates. See the disclaimer in the app.

## Quick start

```bash
python -m pip install -r requirements.txt

# Terminal 1 - backend API (port 8000)
python -m uvicorn app:app --port 8000

# Terminal 2 - dashboard (port 8501)
python -m streamlit run dashboard.py
```

Then open http://127.0.0.1:8501 (the dashboard auto-starts the backend if it isn't
already running, so `streamlit run dashboard.py` alone also works).

## API endpoints

| Endpoint | Returns |
|---|---|
| `/health` | liveness |
| `/prices` | S&P 500 + VIX recent closes |
| `/features` | ~20 market-state features |
| `/forecast` | probability cones (10/25/50/75/90%) per horizon |
| `/report` | driver bullets + risk tilt + disclaimer |
| `/backtest` | walk-forward calibration audit (hit rates of the 50%/80% bands) |
| `/all` | everything in one payload (used by the dashboard) |

## Data sources (all keyless)

| Data | Source | Notes |
|---|---|---|
| S&P 500, VIX | Yahoo Finance chart API | primary, reliable |
| 10Y Treasury yield | Yahoo `^TNX` | value is % (verified vs FRED DGS10) |
| CPI (monthly) | Alpha Vantage demo key | `apikey=demo` |
| Short rate | FRED DFF → fallback Yahoo `^IRX` | FRED blocks persistent connections from some networks; app remembers failures for 1h and falls back to the 3-month T-bill |
| S&P 500 (backup) | stooq CSV | used only if Yahoo fails |

All fetched data is cached in `data/` (CSV + timestamp) with per-source TTLs, so the
app works offline-ish and doesn't hammer sources.

## Methodology (deliberately transparent)

- Base assumption: **random walk** with a small momentum drift (capped ±0.05%/day).
- Daily volatility = 70/30 variance blend of 20-day realized vol and VIX implied vol,
  floored at 8% annualized; horizon scaling via sqrt(time).
- Quantiles from the normal CDF (stdlib `NormalDist`) — deterministic, reproducible.
- The driver report is **template-based** for now; Phase 2 swaps in an LLM to write the
  narrative while keeping the same data contract (LLM reads and explains; models produce
  the numbers — never the other way around).

## Roadmap

- [x] v0.1: keyless data + cones + template drivers + Streamlit dashboard
- [ ] Phase 0 study: test whether a Google Trends / news-sentiment index adds
      out-of-sample value over the baseline → decides feature weights
- [ ] Phase 2: LLM narrative layer + GDELT/news sentiment
- [x] Phase 3: backtest UI (how often did bands contain the realized price?) — live on
      the dashboard: 237 walk-forward test days, hit50 58–71% (vs 50% expected),
      hit80 87–90% (vs 80%): bands overstate risk → next model iteration tightens vol
- [ ] Phase 4: basic token auth + daily scheduled run for a small group

## Files

```
app.py          FastAPI backend
dashboard.py    Streamlit dashboard
data_pull.py    keyless data fetching + file cache
features.py     feature engineering (snapshot of market state)
forecast.py     quantile cone engine
report.py       template driver report
backtest.py     walk-forward calibration audit (also /backtest endpoint)
data/           cache (auto-created)
```