"""Report writer: template-driven drivers + disclaimer.

v0.1 deliberately has NO LLM dependency: this module produces the narrative
""why" bullet points from the feature snapshot. Phase 2 replaces the body of
build_report() with an LLM call while keeping the same return shape, so the
dashboard and API don't care.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _vix_regime(vix: float | None) -> tuple[str, str]:
    if vix is None:
        return "unknown", "neutral"
    if vix < 15:
        return "low (complacent)", "neutral"
    if vix < 20:
        return "normal", "neutral"
    if vix < 25:
        return "elevated", "cautious"
    if vix < 35:
        return "high (stressed)", "cautious"
    return "extreme (panic)", "bearish"


def _trend_line(features: dict) -> str:
    dist = features.get("dist_from_sma200_pct")
    if dist is None:
        return "Price data too short to assess the 200-day trend."
    if dist > 5:
        return f"Index is {dist:.1f}% above its 200-day average - firm uptrend."
    if dist > 0:
        return f"Index is {dist:.1f}% above its 200-day average - mild uptrend."
    if dist > -5:
        return f"Index is {-dist:.1f}% below its 200-day average - mild downtrend."
    return f"Index is {-dist:.1f}% below its 200-day average - defensive posture."


def _momentum_line(features: dict) -> str:
    ret_63d = features.get("ret_63d")
    ret_21d = features.get("ret_21d")
    if ret_63d is None or ret_21d is None:
        return "Momentum: insufficient history."
    mo = "strengthening" if ret_21d > ret_63d / 3 else "cooling"
    sign = "positive" if ret_63d > 0 else "negative"
    return (
        f"Momentum: {sign} ({ret_63d * 100:+.1f}% over 3 months) and "
        f"{mo} over the last month."
    )


def _macro_lines(features: dict) -> list[str]:
    lines = []
    cpi = features.get("cpi_mom_pct")
    if cpi is not None:
        band = "mild" if cpi < 0.3 else "elevated"
        lines.append(
            f"Inflation: latest CPI reading ({features.get('cpi_ymom')}) is "
            f"+{cpi:.2f}% month-over-month ({band})."
        )
    y10 = features.get("yield10y")
    if y10 is not None:
        chg = features.get("yield10y_chg_20d_pts")
        if chg is not None and abs(chg) >= 0.05:
            direction = "rising" if chg > 0 else "falling"
            lines.append(
                f"Rates: 10-year Treasury yield is {direction} "
                f"({chg:+.2f} pts over 20 trading days, now {y10:.2f}%) - "
                + ("headwind for equities." if chg > 0.1 else "mild tailwind.")
            )
        else:
            lines.append(f"Rates: 10-year Treasury yield stable at {y10:.2f}%.")
    fed = features.get("fed_funds")
    ff_chg = features.get("fed_funds_chg_1m_pts")
    if fed is not None:
        state = "tightening" if ff_chg and ff_chg > 0.01 else "holding/ easing" if ff_chg and ff_chg < -0.01 else "on hold"
        src = features.get("fed_source") or "policy rate"
        lines.append(f"Fed: {src} at {fed:.2f}%, effectively {state}.")
    return lines


def build_report(
    forecast: dict,
    features: dict,
    prices: object = None,
) -> dict:
    vix = features.get("vix")
    regime, tilt = _vix_regime(vix)
    high_52w = features.get("high_52w_pct")

    bullets: list[str] = []
    if vix is not None:
        bullets.append(
            f"Volatility: VIX at {vix:.1f} - {regime} regime. "
            f"Options are pricing {forecast.get('vol_annual_pct', '?')}% "
            "annualized volatility."
        )
    bullets.append(_trend_line(features))
    bullets.append(_momentum_line(features))
    if high_52w is not None:
        if high_52w > -2:
            bullets.append(
                f"Index sits {high_52w:+.2f}% from its 52-week high - "
                "near the top of its yearly range."
            )
        elif high_52w < -15:
            bullets.append(
                f"Index is {-high_52w:.1f}% below its 52-week high - "
                "deep in correction territory."
            )
    bullets.extend(_macro_lines(features))

    # Risk tilt: combine vol regime with trend.
    dist = features.get("dist_from_sma200_pct")
    if tilt == "bearish":
        tilt_overall = "bearish"
    elif tilt == "cautious":
        tilt_overall = "cautious" if (dist is not None and dist > 0) else "bearish"
    elif (dist is not None and dist < -5) and vix and vix > 22:
        tilt_overall = "cautious"
    else:
        tilt_overall = "bullish" if (dist is None or dist > 0) else "cautious"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tilt": tilt_overall,
        "vix_regime": regime,
        "bullets": bullets,
        "sources": [
            "S&P 500, VIX & 10Y yield: Yahoo Finance chart API (keyless)",
            "CPI: Alpha Vantage demo (keyless) · Fed funds: FRED CSV (keyless)",
        ],
        "disclaimer": (
            "For research/education only - not investment advice. "
            "Probability bands are model estimates, not guarantees."
        ),
    }