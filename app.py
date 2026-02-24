"""
BTC Options Surface & Matrix Dashboard
=======================================
Backend Flask server that fetches live BTC options data from Deribit API,
computes the implied volatility surface and serves an options matrix.
"""

import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_cors import CORS
from scipy.interpolate import griddata
from scipy.stats import norm

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ─── Deribit API Configuration ───────────────────────────────────────────────
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# Cache to avoid hammering the API
_cache = {}
CACHE_TTL = 30  # seconds


def _get(endpoint, params=None):
    """Make a GET request to Deribit public API with caching."""
    cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    now = time.time()
    if cache_key in _cache and now - _cache[cache_key]["ts"] < CACHE_TTL:
        return _cache[cache_key]["data"]

    url = f"{DERIBIT_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("result", {})
    _cache[cache_key] = {"data": data, "ts": now}
    return data


def get_btc_index_price():
    """Get the current BTC index price."""
    data = _get("get_index_price", {"index_name": "btc_usd"})
    return data.get("index_price", 0)


def get_all_instruments():
    """Fetch all BTC option instruments from Deribit."""
    instruments = _get("get_instruments", {
        "currency": "BTC",
        "kind": "option",
        "expired": "false"
    })
    return instruments


def get_book_summary(currency="BTC"):
    """Fetch book summary for all BTC options."""
    data = _get("get_book_summary_by_currency", {
        "currency": currency,
        "kind": "option"
    })
    return data


def parse_instrument_name(name):
    """Parse Deribit instrument name like BTC-28MAR25-100000-C
    Returns: (underlying, expiry_str, strike, option_type)
    """
    parts = name.split("-")
    if len(parts) != 4:
        return None
    underlying = parts[0]
    expiry_str = parts[1]
    strike = float(parts[2])
    option_type = parts[3]  # C or P
    return underlying, expiry_str, strike, option_type


def expiry_str_to_datetime(expiry_str):
    """Convert Deribit expiry string like '28MAR25' to datetime."""
    try:
        return datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute_days_to_expiry(expiry_dt):
    """Compute days to expiry from now."""
    now = datetime.now(timezone.utc)
    delta = expiry_dt - now
    return max(delta.total_seconds() / 86400, 0.001)


# ─── Probability Calculation Helpers ─────────────────────────────────────────

def _get_iv_surface_data():
    """Collect all IV data points from live options for interpolation."""
    spot = get_btc_index_price()
    book_data = get_book_summary(currency="BTC")
    if not book_data:
        return [], spot

    points = []
    for item in book_data:
        name = item.get("instrument_name", "")
        parsed = parse_instrument_name(name)
        if not parsed:
            continue
        _, expiry_str, strike, opt_type = parsed
        expiry_dt = expiry_str_to_datetime(expiry_str)
        if not expiry_dt:
            continue
        dte = compute_days_to_expiry(expiry_dt)
        iv = item.get("mark_iv", 0)
        if iv is None or iv <= 0:
            continue
        moneyness = strike / spot if spot > 0 else 0
        points.append({
            "strike": strike,
            "moneyness": moneyness,
            "dte": dte,
            "iv": iv,              # in percentage
            "type": opt_type,
            "expiry": expiry_str,
        })
    return points, spot


def _interpolate_iv(points, target_moneyness, target_dte):
    """
    Interpolate the implied volatility at a given moneyness and DTE
    from the raw options data using multiple methods for robustness.
    Returns: (interpolated_iv, method_used, nearby_options)
    """
    if not points:
        return None, "no_data", []

    # Use calls for the surface (more liquid typically)
    call_points = [p for p in points if p["type"] == "C"]
    if len(call_points) < 5:
        call_points = points

    moneyness_arr = np.array([p["moneyness"] for p in call_points])
    dte_arr = np.array([p["dte"] for p in call_points])
    iv_arr = np.array([p["iv"] for p in call_points])

    # Filter reasonable range
    mask = (moneyness_arr > 0.1) & (moneyness_arr < 5.0) & (dte_arr > 0)
    moneyness_arr = moneyness_arr[mask]
    dte_arr = dte_arr[mask]
    iv_arr = iv_arr[mask]

    if len(moneyness_arr) < 3:
        return None, "insufficient_data", []

    # Find nearby options for context
    distances = np.sqrt(
        ((moneyness_arr - target_moneyness) * 5) ** 2 +
        ((dte_arr - target_dte) / 30) ** 2
    )
    nearest_idx = np.argsort(distances)[:6]
    nearby = []
    for idx in nearest_idx:
        cp = call_points[int(idx)] if int(idx) < len(call_points) else None
        if cp:
            nearby.append({
                "strike": cp["strike"],
                "dte": round(cp["dte"], 1),
                "iv": round(cp["iv"], 2),
                "expiry": cp["expiry"],
                "moneyness": round(cp["moneyness"], 4),
                "distance": round(float(distances[idx]), 4),
            })

    # Method 1: Try cubic interpolation
    try:
        iv_interp = griddata(
            (moneyness_arr, dte_arr), iv_arr,
            (target_moneyness, target_dte),
            method="cubic"
        )
        if not np.isnan(iv_interp) and iv_interp > 0:
            return float(iv_interp), "cubic", nearby
    except Exception:
        pass

    # Method 2: Linear interpolation
    try:
        iv_interp = griddata(
            (moneyness_arr, dte_arr), iv_arr,
            (target_moneyness, target_dte),
            method="linear"
        )
        if not np.isnan(iv_interp) and iv_interp > 0:
            return float(iv_interp), "linear", nearby
    except Exception:
        pass

    # Method 3: Nearest neighbor
    try:
        iv_interp = griddata(
            (moneyness_arr, dte_arr), iv_arr,
            (target_moneyness, target_dte),
            method="nearest"
        )
        if not np.isnan(iv_interp) and iv_interp > 0:
            return float(iv_interp), "nearest", nearby
    except Exception:
        pass

    # Method 4: Weighted average of nearest points
    if len(nearby) > 0:
        weights = [1 / (n["distance"] + 0.001) for n in nearby]
        total_w = sum(weights)
        weighted_iv = sum(n["iv"] * w for n, w in zip(nearby, weights)) / total_w
        return weighted_iv, "weighted_nearest", nearby

    return None, "failed", nearby


def _compute_bs_probability(spot, target_price, dte_days, iv_pct, scenario="above"):
    """
    Compute the Black-Scholes risk-neutral probability.

    P(S_T > K) = N(d2)   where:
        d2 = [ln(S/K) + (r - σ²/2) * T] / (σ * √T)

    Parameters:
        spot: current BTC price
        target_price: target price (K)
        dte_days: days to target date
        iv_pct: implied volatility in percentage (e.g., 55 for 55%)
        scenario: "above" or "below"

    Returns: dict with probability and all intermediate values
    """
    r = 0.045  # risk-free rate (approximate)
    sigma = iv_pct / 100.0  # convert from percentage
    T = dte_days / 365.0  # time in years

    if T <= 0 or sigma <= 0:
        return {"error": "Invalid T or sigma"}

    S = spot
    K = target_price

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    prob_above = float(norm.cdf(d2))   # P(S_T > K) risk-neutral
    prob_below = 1.0 - prob_above      # P(S_T < K)

    # Real-world probability adjustment using a drift estimate
    # Use a modest real-world excess return for BTC
    mu = 0.10  # assumed annual drift for BTC (real-world)
    d2_real = (np.log(S / K) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    prob_above_real = float(norm.cdf(d2_real))
    prob_below_real = 1.0 - prob_above_real

    if scenario == "above":
        prob_rn = prob_above
        prob_rw = prob_above_real
    else:
        prob_rn = prob_below
        prob_rw = prob_below_real

    return {
        "scenario": scenario,
        "probability_risk_neutral": round(prob_rn * 100, 2),
        "probability_real_world": round(prob_rw * 100, 2),
        "d1": round(float(d1), 4),
        "d2": round(float(d2), 4),
        "d2_real": round(float(d2_real), 4),
        "sigma": round(sigma, 4),
        "sigma_pct": round(iv_pct, 2),
        "T_years": round(T, 4),
        "risk_free_rate": r,
        "real_world_drift": mu,
        "spot": spot,
        "target_price": target_price,
        "dte_days": round(dte_days, 2),
        "prob_above_rn": round(prob_above * 100, 2),
        "prob_below_rn": round(prob_below * 100, 2),
        "prob_above_rw": round(prob_above_real * 100, 2),
        "prob_below_rw": round(prob_below_real * 100, 2),
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/spot")
def api_spot():
    """Return current BTC spot price."""
    price = get_btc_index_price()
    return jsonify({"spot": price})


@app.route("/api/options_data")
def api_options_data():
    """
    Return comprehensive options data:
    - All options with IV, greeks, prices
    - Grouped by expiry
    - Surface data points for 3D vol surface
    """
    try:
        spot = get_btc_index_price()
        book_data = get_book_summary(currency="BTC")

        if not book_data:
            return jsonify({"error": "No data from Deribit"}), 500

        options = []
        surface_points = []
        expiries_set = set()

        for item in book_data:
            name = item.get("instrument_name", "")
            parsed = parse_instrument_name(name)
            if not parsed:
                continue

            underlying, expiry_str, strike, opt_type = parsed
            expiry_dt = expiry_str_to_datetime(expiry_str)
            if not expiry_dt:
                continue

            dte = compute_days_to_expiry(expiry_dt)
            iv = item.get("mark_iv", 0)
            if iv is None or iv <= 0:
                continue

            mark_price_btc = item.get("mark_price", 0) or 0
            mark_price_usd = mark_price_btc * spot
            bid_price_btc = item.get("bid_price", 0) or 0
            ask_price_btc = item.get("ask_price", 0) or 0
            bid_usd = bid_price_btc * spot
            ask_usd = ask_price_btc * spot
            volume = item.get("volume", 0) or 0
            open_interest = item.get("open_interest", 0) or 0
            delta = item.get("delta") or 0
            gamma = item.get("gamma") or 0
            vega = item.get("vega") or 0
            theta = item.get("theta") or 0

            moneyness = strike / spot if spot > 0 else 0

            option_record = {
                "instrument": name,
                "expiry": expiry_str,
                "expiry_ts": expiry_dt.isoformat(),
                "dte": round(dte, 2),
                "strike": strike,
                "type": opt_type,
                "iv": round(iv, 2),
                "mark_btc": round(mark_price_btc, 6),
                "mark_usd": round(mark_price_usd, 2),
                "bid_usd": round(bid_usd, 2),
                "ask_usd": round(ask_usd, 2),
                "volume": volume,
                "open_interest": open_interest,
                "delta": round(delta, 5),
                "gamma": round(gamma, 8),
                "vega": round(vega, 4),
                "theta": round(theta, 4),
                "moneyness": round(moneyness, 4),
            }
            options.append(option_record)
            expiries_set.add(expiry_str)

            # Surface point (use calls for the surface, or both)
            surface_points.append({
                "strike": strike,
                "dte": round(dte, 2),
                "iv": round(iv, 2),
                "type": opt_type,
                "moneyness": round(moneyness, 4),
            })

        # Sort options by expiry then strike
        options.sort(key=lambda x: (x["dte"], x["strike"], x["type"]))

        # Group by expiry for the matrix
        expiry_groups = {}
        for opt in options:
            exp = opt["expiry"]
            if exp not in expiry_groups:
                expiry_groups[exp] = {"calls": [], "puts": [], "dte": opt["dte"]}
            if opt["type"] == "C":
                expiry_groups[exp]["calls"].append(opt)
            else:
                expiry_groups[exp]["puts"].append(opt)

        # Sort expiry groups by DTE
        sorted_expiries = sorted(expiry_groups.keys(),
                                  key=lambda e: expiry_groups[e]["dte"])

        # Build interpolated surface for smooth 3D rendering
        surface_grid = _build_surface_grid(surface_points, spot)

        return jsonify({
            "spot": spot,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_options": len(options),
            "expiries": sorted_expiries,
            "expiry_groups": expiry_groups,
            "options": options,
            "surface": surface_grid,
            "surface_raw": surface_points,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_surface_grid(points, spot):
    """
    Build a smooth interpolated volatility surface grid
    for 3D visualization.
    """
    if len(points) < 10:
        return {"x": [], "y": [], "z": []}

    # Filter to calls only for a cleaner surface
    call_points = [p for p in points if p["type"] == "C"]
    if len(call_points) < 10:
        call_points = points

    strikes = np.array([p["moneyness"] for p in call_points])
    dtes = np.array([p["dte"] for p in call_points])
    ivs = np.array([p["iv"] for p in call_points])

    # Filter reasonable range
    mask = (strikes > 0.3) & (strikes < 3.0) & (dtes > 0) & (dtes < 400)
    strikes = strikes[mask]
    dtes = dtes[mask]
    ivs = ivs[mask]

    if len(strikes) < 10:
        return {"x": [], "y": [], "z": []}

    # Create grid
    n_strike = 50
    n_dte = 50
    strike_grid = np.linspace(max(0.5, strikes.min()), min(2.5, strikes.max()), n_strike)
    dte_grid = np.linspace(max(1, dtes.min()), min(365, dtes.max()), n_dte)
    X, Y = np.meshgrid(strike_grid, dte_grid)

    try:
        Z = griddata(
            (strikes, dtes), ivs,
            (X, Y),
            method="cubic",
            fill_value=np.nan
        )
        # Fill NaN with nearest neighbor
        mask_nan = np.isnan(Z)
        if mask_nan.any():
            Z_nearest = griddata(
                (strikes, dtes), ivs,
                (X, Y),
                method="nearest"
            )
            Z[mask_nan] = Z_nearest[mask_nan]
    except Exception:
        Z = griddata(
            (strikes, dtes), ivs,
            (X, Y),
            method="linear",
            fill_value=np.nanmean(ivs)
        )

    return {
        "x": strike_grid.tolist(),  # moneyness
        "y": dte_grid.tolist(),      # DTE
        "z": Z.tolist(),             # IV
        "spot": spot,
    }


@app.route("/api/smile/<expiry>")
def api_smile(expiry):
    """Get volatility smile for a specific expiry."""
    try:
        spot = get_btc_index_price()
        book_data = get_book_summary(currency="BTC")

        calls = []
        puts = []

        for item in book_data:
            name = item.get("instrument_name", "")
            parsed = parse_instrument_name(name)
            if not parsed:
                continue

            _, exp_str, strike, opt_type = parsed
            if exp_str != expiry:
                continue

            iv = item.get("mark_iv", 0)
            if iv is None or iv <= 0:
                continue

            entry = {
                "strike": strike,
                "iv": round(iv, 2),
                "moneyness": round(strike / spot, 4) if spot > 0 else 0,
            }

            if opt_type == "C":
                calls.append(entry)
            else:
                puts.append(entry)

        calls.sort(key=lambda x: x["strike"])
        puts.sort(key=lambda x: x["strike"])

        return jsonify({
            "expiry": expiry,
            "spot": spot,
            "calls": calls,
            "puts": puts,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Probability Calculator Endpoint ────────────────────────────────────────

@app.route("/api/probability", methods=["POST"])
def api_probability():
    """
    Compute the probability of BTC being above/below a target price
    at a specific date, using IV interpolated from the live options surface.

    JSON Body:
        target_price: float  (e.g. 100000)
        target_date:  string (e.g. "2026-06-30")
        scenario:     string ("above" or "below")
    """
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "Missing JSON body"}), 400

        target_price = float(body.get("target_price", 0))
        target_date_str = body.get("target_date", "")
        scenario = body.get("scenario", "above").lower()

        if target_price <= 0:
            return jsonify({"error": "target_price must be > 0"}), 400
        if scenario not in ("above", "below"):
            return jsonify({"error": "scenario must be 'above' or 'below'"}), 400

        # Parse target date
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return jsonify({"error": f"Invalid date format: '{target_date_str}'. Use YYYY-MM-DD"}), 400

        now = datetime.now(timezone.utc)
        dte_days = (target_date - now).total_seconds() / 86400.0
        if dte_days <= 0:
            return jsonify({"error": "Target date must be in the future"}), 400

        # Get live IV surface data
        iv_points, spot = _get_iv_surface_data()
        if not iv_points:
            return jsonify({"error": "Could not fetch options data from Deribit"}), 500

        # Moneyness for interpolation
        target_moneyness = target_price / spot if spot > 0 else 0

        # Interpolate IV
        iv_interp, method, nearby = _interpolate_iv(iv_points, target_moneyness, dte_days)

        if iv_interp is None or iv_interp <= 0:
            return jsonify({
                "error": "Could not interpolate IV at the given strike/date. "
                         "The target may be outside the range of available options.",
                "spot": spot,
                "target_moneyness": round(target_moneyness, 4),
                "target_dte": round(dte_days, 2),
                "nearby_options": nearby,
            }), 400

        # Compute probability
        result = _compute_bs_probability(spot, target_price, dte_days, iv_interp, scenario)

        # Add metadata
        result["interpolation_method"] = method
        result["interpolated_iv_pct"] = round(iv_interp, 2)
        result["target_moneyness"] = round(target_moneyness, 4)
        result["target_date"] = target_date_str
        result["nearby_options"] = nearby
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Available expiries for reference
        expiries_available = sorted(
            set(p["expiry"] for p in iv_points),
            key=lambda e: next((p["dte"] for p in iv_points if p["expiry"] == e), 0)
        )
        result["available_expiries"] = expiries_available

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Strategy Endpoint ──────────────────────────────────────────────────────

from strategy import analyze_opportunities

@app.route("/api/strategy")
def api_strategy():
    """
    Run the Polymarket-vs-Options arbitrage strategy.
    Fetches all BTC barrier markets, computes model probabilities,
    and returns ranked trading opportunities.
    """
    try:
        # Get IV surface data for the model
        iv_points, spot = _get_iv_surface_data()
        if not iv_points:
            return jsonify({"error": "Could not fetch options data from Deribit"}), 500

        # Run strategy analysis
        result = analyze_opportunities(spot, iv_points)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  BTC OPTIONS SURFACE & MATRIX DASHBOARD")
    print("  + Polymarket Arbitrage Strategy")
    print("  Fetching live data from Deribit & Polymarket...")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
