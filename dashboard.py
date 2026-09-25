"""S&P 500 Forecast - Streamlit dashboard.

Run:  python -m streamlit run dashboard.py
(If the FastAPI backend isn't running, this auto-starts it on port 8000.)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("SP500_API_URL", "http://127.0.0.1:8000")
API_HEALTH_TIMEOUT = 3

st.set_page_config(
    page_title="S&P 500 Forecast",
    page_icon="📈",
    layout="wide",
)


def _ensure_api() -> None:
    """Start the FastAPI backend in a subprocess if it isn't reachable."""
    try:
        requests.get(f"{API_URL}/health", timeout=API_HEALTH_TIMEOUT)
        return
    except requests.RequestException:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--port", "8000"]
    subprocess.Popen(cmd, cwd=here, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{API_URL}/health", timeout=1)
            return
        except requests.RequestException:
            continue
    st.error("Could not start the backend API. Run `python -m uvicorn app:app --port 8000`.")


@st.cache_data(ttl=600, show_spinner="Refreshing data...")
def fetch_all() -> dict:
    resp = requests.get(f"{API_URL}/all", timeout=45)
    resp.raise_for_status()
    return resp.json()


_ensure_api()

st.title("📈 S&P 500 Forecast")
st.caption("Keyless-data prototype — probability cones for the S&P 500 (not investment advice).")

col_refresh, col_fresh, col_spacer = st.columns([1, 3, 5])
with col_refresh:
    if st.button("🔄 Refresh data", use_container_width=True):
        fetch_all.clear()
        st.rerun()

payload = fetch_all()
features = payload["features"]
cone = payload["forecast"]
report_data = payload["report"]

with col_fresh:
    st.caption(
        f"Data as of **{features.get('as_of')}** · model vol "
        f"{cone.get('vol_annual_pct')}% annualized · drift "
        f"{cone.get('drift_daily_pct')}%/day"
    )

# --------------------------------------------------------------------------- #
# Headline metrics                                                             #
# --------------------------------------------------------------------------- #

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "S&P 500",
    f"{features['last_close']:,.2f}",
    f"{features.get('daily_return_pct', 0):+.2f}% (day)",
)
m2.metric("VIX", f"{features.get('vix'):.1f}" if features.get("vix") is not None else "n/a")
tilt = report_data.get("tilt", "neutral")
m3.metric("Risk tilt", tilt.upper())
m4.metric(
    "200-day trend",
    f"{features['last_close']:,.0f}",
    f"{features.get('dist_from_sma200_pct', 0):+.1f}% vs SMA200",
)

# --------------------------------------------------------------------------- #
# Horizon cards                                                                #
# --------------------------------------------------------------------------- #

st.subheader("Projection cones")
cards = st.columns(4)
horizon_order = ["1", "5", "21", "63"]
for col, key in zip(cards, horizon_order):
    h = cone["cones"][key]
    with col:
        st.markdown(f"**{h['label']}**")
        st.markdown(
            f"<div style='font-size:26px;font-weight:700'>"
            f"${h['p50']:,.0f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:14px;color:#888'>"
            f"median · 50% band ${h['p25']:,.0f} – ${h['p75']:,.0f} · "
            f"80% band ${h['p10']:,.0f} – ${h['p90']:,.0f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"P(up): **{h['prob_up_pct']}%** · {h['median_pct_change']:+.1f}%")

# --------------------------------------------------------------------------- #
# Fan chart                                                                    #
# --------------------------------------------------------------------------- #

hbar = requests.get(f"{API_URL}/prices", timeout=30).json()
hist_prices = pd.DataFrame(
    hbar["spx"]["recent"] if "spx" in hbar and "recent" in hbar["spx"] else []
)
cone_path = pd.DataFrame(payload["cone_path"])
cone_path["date"] = pd.to_datetime(cone_path["date"])

fig = go.Figure()
if not hist_prices.empty:
    hist_prices["date"] = pd.to_datetime(hist_prices["date"])
    fig.add_trace(
        go.Scatter(
            x=hist_prices["date"],
            y=hist_prices["close"],
            name="History (30d)",
            line=dict(color="#1f77b4", width=2),
        )
    )

if not cone_path.empty:
    x = cone_path["date"]
    fig.add_trace(go.Scatter(x=x, y=cone_path["p10"], mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=cone_path["p90"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(31,119,180,0.12)", name="80% band",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=cone_path["p25"], mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=cone_path["p75"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(31,119,180,0.25)", name="50% band",
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=cone_path["p50"], mode="lines", name="Median",
                   line=dict(color="#ff7f0e", width=2, dash="dash"))
    )

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    title="Probability cone — next 63 trading days",
    yaxis_title="Index level",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Drivers + sources + disclaimer                                               #
# --------------------------------------------------------------------------- #

st.subheader("What's driving the outlook")
for b in report_data.get("bullets", []):
    st.markdown(f"• {b}")

with st.expander("Sources & methodology"):
    for s in report_data.get("sources", []):
        st.markdown(f"- {s}")
    st.markdown(
        "Forecast: random walk with 70/30 blend of 20-day realized vol and "
        "VIX implied vol (sqrt-of-time scaling, normal quantiles). "
        "See `forecast.py` for the full methodology."
    )

st.divider()
st.markdown(report_data.get("disclaimer", ""), unsafe_allow_html=True)