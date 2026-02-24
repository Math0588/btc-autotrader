"""
🤖 Polymarket Autotrader — Web Server + Bot Combined
====================================================
Flask web server with live PnL dashboard + autonomous trading bot.
Designed for FREE deployment on Render.com.

The web server keeps the Render service alive while the bot
scans markets in a background thread.
"""

import os
import sys
import json
import time
import math
import hashlib
import logging
import threading
import traceback
import io
import base64
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from scipy.stats import norm
from flask import Flask, jsonify, render_template_string

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
    ],
)
log = logging.getLogger("autotrader")

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "starting_capital": float(os.getenv("STARTING_CAPITAL", "100")),
    "scan_interval": int(os.getenv("SCAN_INTERVAL", "300")),  # 5 min — aggressive
    "min_edge_pct": float(os.getenv("MIN_EDGE", "2.5")),      # Lower threshold = more trades
    "kelly_fraction": float(os.getenv("KELLY_FRAC", "0.15")), # Conservative
    "max_position_pct": 0.12,
    "max_exposure_pct": 0.70,
    "polymarket_fee_pct": 2.0,
    "risk_free_rate": 0.045,
    "min_dte": 1,
    "max_dte": 30,
    "min_liquidity": 3000,
    "min_volume": 500,
    "min_win_prob": 0.20,
    "max_drawdown_pct": 25,
    "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat": os.getenv("TELEGRAM_CHAT", ""),
}

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
DERIBIT_API = "https://www.deribit.com/api/v2/public"
STATE_FILE = Path(os.getenv("STATE_FILE", "autotrader_state.json"))
MONTH_NAMES = {1:"january",2:"february",3:"march",4:"april",5:"may",6:"june",
               7:"july",8:"august",9:"september",10:"october",11:"november",12:"december"}

# ─── Global State ─────────────────────────────────────────────────────────────

BOT_STATE = None
PNL_HISTORY = []  # [(timestamp_iso, capital, total_value, pnl)]


def default_state():
    return {
        "capital": CONFIG["starting_capital"],
        "initial_capital": CONFIG["starting_capital"],
        "positions": [],
        "closed_trades": [],
        "total_pnl": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "peak_capital": CONFIG["starting_capital"],
        "max_drawdown": 0.0,
        "last_scan": None,
        "scans_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state():
    d = default_state()
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for k, v in d.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            pass
    return d


def save_state(state):
    state["last_saved"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


# ─── Telegram ─────────────────────────────────────────────────────────────────

def tg_send(message):
    t, c = CONFIG["telegram_token"], CONFIG["telegram_chat"]
    if not t or not c:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{t}/sendMessage",
                      json={"chat_id": c, "text": message, "parse_mode": "HTML"},
                      timeout=10)
    except Exception:
        pass


def tg_send_chart():
    """Send PnL chart image to Telegram."""
    t, c = CONFIG["telegram_token"], CONFIG["telegram_chat"]
    if not t or not c or len(PNL_HISTORY) < 2:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        times = [datetime.fromisoformat(p[0]) for p in PNL_HISTORY]
        values = [p[2] for p in PNL_HISTORY]  # total_value
        pnls = [p[3] for p in PNL_HISTORY]    # pnl

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0f172a')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Portfolio Value
        ax1.plot(times, values, color='#10b981', linewidth=2)
        ax1.fill_between(times, CONFIG["starting_capital"], values,
                         where=[v >= CONFIG["starting_capital"] for v in values],
                         alpha=0.2, color='#10b981')
        ax1.fill_between(times, CONFIG["starting_capital"], values,
                         where=[v < CONFIG["starting_capital"] for v in values],
                         alpha=0.2, color='#ef4444')
        ax1.axhline(y=CONFIG["starting_capital"], color='#f59e0b', linestyle='--',
                    alpha=0.5, linewidth=1)
        ax1.set_ylabel('Portfolio Value ($)', color='#e2e8f0', fontsize=10)
        ax1.set_title('🤖 Polymarket Autotrader — Live PnL', color='#f1f5f9',
                       fontsize=13, fontweight='bold', pad=10)

        # Cumulative PnL
        colors = ['#10b981' if p >= 0 else '#ef4444' for p in pnls]
        ax2.bar(times, pnls, color=colors, width=0.01, alpha=0.8)
        ax2.axhline(y=0, color='#64748b', linewidth=0.5)
        ax2.set_ylabel('Cumulative PnL ($)', color='#e2e8f0', fontsize=10)
        ax2.set_xlabel('Time', color='#e2e8f0', fontsize=10)

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#0f172a', edgecolor='none')
        buf.seek(0)
        plt.close()

        requests.post(
            f"https://api.telegram.org/bot{t}/sendPhoto",
            data={"chat_id": c, "caption": "📊 PnL Update"},
            files={"photo": ("pnl.png", buf, "image/png")},
            timeout=15,
        )
    except Exception as e:
        log.warning(f"Chart send error: {e}")


# ─── Deribit Data ─────────────────────────────────────────────────────────────

def fetch_spot():
    try:
        r = requests.get(f"{DERIBIT_API}/get_index_price",
                         params={"index_name": "btc_usd"}, timeout=10)
        return r.json()["result"]["index_price"]
    except Exception:
        return None


def fetch_iv_surface():
    try:
        spot = fetch_spot()
        if not spot:
            return [], None
        r = requests.get(f"{DERIBIT_API}/get_book_summary_by_currency",
                         params={"currency": "BTC", "kind": "option"}, timeout=20)
        summaries = r.json().get("result", [])
        pts = []
        now = datetime.now(timezone.utc)
        for s in summaries:
            iv = s.get("mark_iv", 0)
            if not iv or iv <= 0:
                continue
            parts = s.get("instrument_name", "").split("-")
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
                opt_type = parts[3]
                exp_dt = datetime.strptime(parts[1], "%d%b%y").replace(
                    tzinfo=timezone.utc, hour=8)
                dte = (exp_dt - now).total_seconds() / 86400
                if dte <= 0 or dte > 400:
                    continue
                pts.append({"moneyness": strike/spot, "dte": dte, "iv": iv,
                            "strike": strike, "type": opt_type})
            except Exception:
                continue
        log.info(f"IV surface: {len(pts)} points, BTC=${spot:,.0f}")
        return pts, spot
    except Exception as e:
        log.error(f"IV fetch error: {e}")
        return [], None


# ─── Polymarket Discovery ────────────────────────────────────────────────────

def pm_get(url, params=None):
    r = requests.get(url, params=params,
                     headers={"Accept": "application/json"}, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_markets():
    import re
    markets = []
    seen = set()
    now = datetime.now(timezone.utc)

    for delta in range(-1, 21):
        dt = now + timedelta(days=delta)
        slug = f"bitcoin-above-on-{MONTH_NAMES[dt.month]}-{dt.day}"
        try:
            events = pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": slug})
            if not events or not isinstance(events, list) or len(events) == 0:
                continue
            for sm in events[0].get("markets", []):
                if sm.get("closed", False):
                    continue
                p = parse_market(sm)
                if p and p["cid"] not in seen:
                    seen.add(p["cid"])
                    markets.append(p)
        except requests.exceptions.HTTPError:
            continue
        except Exception:
            continue

    # Also scan ETH markets
    for delta in range(-1, 14):
        dt = now + timedelta(days=delta)
        for token in ["ethereum-above", "solana-above"]:
            slug = f"{token}-on-{MONTH_NAMES[dt.month]}-{dt.day}"
            try:
                events = pm_get(f"{POLYMARKET_GAMMA_API}/events", params={"slug": slug})
                if events and isinstance(events, list) and len(events) > 0:
                    for sm in events[0].get("markets", []):
                        if sm.get("closed", False):
                            continue
                        p = parse_market(sm, token_type="ETH" if "ethereum" in token else "SOL")
                        if p and p["cid"] not in seen:
                            seen.add(p["cid"])
                            markets.append(p)
            except Exception:
                continue

    markets.sort(key=lambda m: (m["expiry_dt"], m["barrier"]))
    log.info(f"Found {len(markets)} barrier markets")
    return markets


def parse_market(data, token_type="BTC"):
    import re
    title = data.get("question", "") or data.get("title", "") or ""
    tl = title.lower()
    if not any(k in tl for k in ["bitcoin","btc","ethereum","eth","solana","sol"]):
        return None
    if not any(k in tl for k in ["above","below","hit","reach","price"]):
        return None

    barrier = None
    for pat in [r'\$([0-9]{1,3}(?:,[0-9]{3})+)', r'\$([0-9]+(?:\.[0-9]+)?)\s*[kK]',
                r'\$([0-9]+(?:,?[0-9]+)*)']:
        m = re.search(pat, title)
        if m:
            val = float(m.group(1).replace(",", ""))
            if val < 1000:
                val *= 1000
            if val > 100:
                barrier = val
                break
    if not barrier:
        return None

    scenario = "below" if ("below" in tl or "under" in tl) else "above"

    expiry_str = data.get("end_date_iso", "") or data.get("endDate", "")
    try:
        expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    dte = (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400
    if dte <= 0:
        return None

    yes = None
    op = data.get("outcomePrices", "")
    if op:
        try:
            prices = json.loads(op) if isinstance(op, str) else op
            if len(prices) >= 2:
                yes = float(prices[0])
        except Exception:
            pass
    if yes is None:
        bb, ba = data.get("bestBid"), data.get("bestAsk")
        if bb and ba:
            yes = (float(bb) + float(ba)) / 2
    if yes is None:
        return None

    return {
        "title": title, "cid": data.get("conditionId", ""),
        "barrier": barrier, "scenario": scenario, "token": token_type,
        "yes": round(yes, 4), "no": round(1-yes, 4),
        "pm_prob": round(yes*100, 2), "expiry_dt": expiry_dt,
        "dte": round(dte, 2),
        "vol": float(data.get("volume",0) or data.get("volumeNum",0) or 0),
        "liq": float(data.get("liquidity",0) or data.get("liquidityNum",0) or 0),
    }


# ─── Probability Models ──────────────────────────────────────────────────────

def bs_prob(spot, barrier, dte, sigma, r=0.045, scenario="above"):
    T = dte / 365.0
    d2 = (math.log(spot/barrier) + (r - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm.cdf(d2) if scenario == "above" else norm.cdf(-d2)


def interp_iv(pts, target_m, target_dte):
    from scipy.interpolate import griddata
    if not pts or len(pts) < 3:
        return None
    cp = [p for p in pts if p.get("type") == "C"]
    if len(cp) < 5:
        cp = pts
    m = np.array([p["moneyness"] for p in cp])
    d = np.array([p["dte"] for p in cp])
    iv = np.array([p["iv"] for p in cp])
    mask = (m > 0.1) & (m < 5.0) & (d > 0)
    m, d, iv = m[mask], d[mask], iv[mask]
    if len(m) < 3:
        return None
    for method in ["cubic", "linear", "nearest"]:
        try:
            v = griddata((m, d), iv, (target_m, target_dte), method=method)
            if not np.isnan(v) and v > 0:
                return float(v)
        except Exception:
            continue
    dists = np.sqrt(((m-target_m)*5)**2 + ((d-target_dte)/30)**2)
    nn = np.argsort(dists)[:5]
    w = 1.0/(dists[nn]+0.001)
    return float(np.average(iv[nn], weights=w))


# ─── Strategy Engine ─────────────────────────────────────────────────────────

def analyze(market, spot, iv_pts, capital, exposure):
    barrier = market["barrier"]
    dte = market["dte"]
    pm_prob = market["pm_prob"]

    if dte < CONFIG["min_dte"] or dte > CONFIG["max_dte"]:
        return None
    if market["liq"] < CONFIG["min_liquidity"]:
        return None

    # Only analyze BTC for now (we have Deribit IV for BTC only)
    if market["token"] != "BTC":
        return None

    moneyness = barrier / spot
    iv = interp_iv(iv_pts, moneyness, dte)
    if not iv or iv <= 0:
        return None
    sigma = iv / 100.0

    prob = bs_prob(spot, barrier, dte, sigma, CONFIG["risk_free_rate"], market["scenario"])
    model_pct = prob * 100
    edge = model_pct - pm_prob

    if abs(edge) < CONFIG["min_edge_pct"]:
        return None

    if edge > 0:
        direction, entry, win_p = "BUY_YES", market["yes"], prob
        desc = f"{market['token']} AU-DESSUS ${barrier:,.0f}"
    else:
        direction, entry, win_p = "BUY_NO", market["no"], 1-prob
        desc = f"{market['token']} SOUS ${barrier:,.0f}"

    if win_p < CONFIG["min_win_prob"]:
        return None

    fee = CONFIG["polymarket_fee_pct"] / 100
    payout = (1.0 - entry) * (1 - fee)
    cost = entry
    if cost <= 0 or payout <= 0:
        return None

    b = payout / cost
    kelly = max(0, (win_p*b - (1-win_p))/b) * CONFIG["kelly_fraction"]

    # Drawdown control
    dd = (1 - capital / CONFIG["starting_capital"]) * 100
    if dd > CONFIG["max_drawdown_pct"]:
        kelly *= 0.4
    elif dd > CONFIG["max_drawdown_pct"]/2:
        kelly *= 0.7

    # Liquidity adjustment
    liq_factor = min(market["liq"] / 15000, 1.0)
    kelly *= liq_factor

    max_pos = capital * CONFIG["max_position_pct"]
    remaining = capital * CONFIG["max_exposure_pct"] - exposure
    if remaining <= 0:
        return None

    pos = min(kelly * capital, max_pos, remaining)
    n = int(pos / entry)
    if n < 1:
        return None

    actual = n * entry
    e_pnl = win_p * payout * n - (1-win_p) * cost * n

    return {
        "title": market["title"], "cid": market["cid"],
        "barrier": barrier, "scenario": market["scenario"],
        "token": market["token"], "direction": direction,
        "desc": desc, "dte": dte, "expiry_dt": market["expiry_dt"],
        "pm_prob": pm_prob, "model_prob": round(model_pct, 1),
        "edge": round(abs(edge), 1), "entry": entry,
        "win_prob": round(win_p*100, 1),
        "profit_pct": round((payout/cost)*100, 0),
        "n": n, "cost": round(actual, 2), "e_pnl": round(e_pnl, 2),
        "liq": market["liq"],
    }


# ─── Trade Execution ─────────────────────────────────────────────────────────

def execute(state, opp):
    if opp["cost"] > state["capital"]:
        return False
    pos = {
        "id": hashlib.md5(f"{opp['cid']}:{time.time()}".encode()).hexdigest()[:10],
        "cid": opp["cid"], "title": opp["title"], "desc": opp["desc"],
        "direction": opp["direction"], "barrier": opp["barrier"],
        "token": opp["token"], "entry": opp["entry"], "n": opp["n"],
        "cost": opp["cost"], "win_prob": opp["win_prob"],
        "profit_pct": opp["profit_pct"], "e_pnl": opp["e_pnl"],
        "expiry_dt": opp["expiry_dt"].isoformat() if isinstance(opp["expiry_dt"], datetime) else opp["expiry_dt"],
        "opened_at": datetime.now(timezone.utc).isoformat(), "status": "open",
    }
    state["capital"] -= opp["cost"]
    state["positions"].append(pos)
    state["total_trades"] += 1
    log.info(f"TRADE: {opp['desc']} | ${opp['cost']:.2f} | Win={opp['win_prob']:.0f}% | "
             f"Edge={opp['edge']:.1f}% | +{opp['profit_pct']:.0f}% if win")
    tg_send(f"📈 <b>NEW TRADE</b>: {opp['desc']}\n"
            f"Edge: {opp['edge']:.1f}% | Win: {opp['win_prob']:.0f}% | "
            f"Profit: +{opp['profit_pct']:.0f}%\n"
            f"Cost: <code>${opp['cost']:.2f}</code> | "
            f"DTE: {opp['dte']:.0f}j")
    return True


def settle(state, spot):
    now = datetime.now(timezone.utc)
    still_open = []
    for pos in state["positions"]:
        exp = datetime.fromisoformat(pos["expiry_dt"])
        if now < exp:
            still_open.append(pos)
            continue
        won = (spot >= pos["barrier"]) if pos["direction"] == "BUY_YES" else (spot < pos["barrier"])
        if won:
            fee = CONFIG["polymarket_fee_pct"] / 100
            payout = pos["n"] * (1.0 - pos["entry"]) * (1 - fee)
            state["capital"] += pos["cost"] + payout
            state["wins"] += 1
            state["total_pnl"] += payout
            pos["pnl"] = round(payout, 2)
            emoji, result = "✅", "WIN"
        else:
            state["losses"] += 1
            state["total_pnl"] -= pos["cost"]
            pos["pnl"] = -pos["cost"]
            emoji, result = "❌", "LOSS"

        if state["capital"] > state["peak_capital"]:
            state["peak_capital"] = state["capital"]
        state["max_drawdown"] = max(state["max_drawdown"],
            (1 - state["capital"]/state["peak_capital"])*100)

        pos["status"], pos["result"] = "closed", result
        pos["settled_at"] = now.isoformat()
        state["closed_trades"].append(pos)

        log.info(f"{emoji} {result}: {pos['desc']} | PnL: ${pos['pnl']:+.2f} | Capital: ${state['capital']:.2f}")
        tg_send(f"{emoji} <b>{result}</b>: {pos['desc']}\n"
                f"PnL: <code>${pos['pnl']:+.2f}</code> | "
                f"Capital: <code>${state['capital']:.2f}</code>")

    state["positions"] = still_open


# ─── Main Scan Loop ──────────────────────────────────────────────────────────

def run_scan(state):
    log.info(f"{'='*50}")
    log.info(f"SCAN #{state['scans_count']+1} | Capital: ${state['capital']:.2f} | "
             f"Positions: {len(state['positions'])} | PnL: ${state['total_pnl']:+.2f}")

    iv_pts, spot = fetch_iv_surface()
    if not spot:
        return

    settle(state, spot)

    markets = fetch_markets()
    if not markets:
        return

    exposure = sum(p["cost"] for p in state["positions"])
    opps = []
    for m in markets:
        o = analyze(m, spot, iv_pts, state["capital"], exposure)
        if o:
            opps.append(o)

    opps.sort(key=lambda o: o["edge"] * o["win_prob"], reverse=True)  # Score = edge × win probability
    log.info(f"Opportunities: {len(opps)} tradeable")

    traded_barriers = {p["barrier"] for p in state["positions"]}
    trades_done = 0
    for opp in opps:
        if trades_done >= 3:
            break
        if opp["barrier"] in traded_barriers:
            continue
        if execute(state, opp):
            trades_done += 1
            traded_barriers.add(opp["barrier"])
            exposure += opp["cost"]

    state["scans_count"] += 1
    state["last_scan"] = datetime.now(timezone.utc).isoformat()

    total_val = state["capital"] + exposure
    ret = (total_val - state["initial_capital"]) / state["initial_capital"] * 100
    wr = state["wins"] / max(state["total_trades"], 1) * 100

    PNL_HISTORY.append((datetime.now(timezone.utc).isoformat(),
                        state["capital"], total_val, state["total_pnl"]))

    log.info(f"Portfolio: ${total_val:.2f} | Return: {ret:+.1f}% | "
             f"WR: {wr:.0f}% ({state['wins']}W/{state['losses']}L) | DD: {state['max_drawdown']:.1f}%")

    save_state(state)

    # Send chart every 12 scans (~1 hour)
    if state["scans_count"] % 12 == 0:
        tg_send_chart()


def bot_loop():
    global BOT_STATE
    BOT_STATE = load_state()
    log.info(f"🤖 Autotrader started | Capital: ${BOT_STATE['capital']:.2f}")
    tg_send(f"🤖 <b>Autotrader Started!</b>\nCapital: ${BOT_STATE['capital']:.2f}\n"
            f"Scan every {CONFIG['scan_interval']}s")

    while True:
        try:
            run_scan(BOT_STATE)
        except Exception as e:
            log.error(f"Scan error: {e}\n{traceback.format_exc()}")
        time.sleep(CONFIG["scan_interval"])


# ─── Flask Web App (keeps Render alive + dashboard) ──────────────────────────

app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Polymarket Autotrader</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        h1 { color: #10b981; font-size: 1.5rem; margin-bottom: 20px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; text-align: center; }
        .card .label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .card .value { font-size: 1.4rem; font-weight: 700; margin-top: 6px; }
        .green { color: #10b981; }
        .red { color: #ef4444; }
        .yellow { color: #f59e0b; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th { background: #1e293b; color: #94a3b8; padding: 10px; text-align: left; font-size: 0.75rem; text-transform: uppercase; }
        td { padding: 10px; border-bottom: 1px solid #1e293b; font-size: 0.85rem; }
        tr:hover { background: rgba(16,185,129,0.05); }
        .badge { padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
        .badge-open { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge-win { background: rgba(16,185,129,0.15); color: #10b981; }
        .badge-loss { background: rgba(239,68,68,0.15); color: #ef4444; }
        #chart { width: 100%; height: 250px; background: #1e293b; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }
        .section { margin-top: 20px; }
        .section h2 { color: #f59e0b; font-size: 1rem; margin-bottom: 10px; }
        .footer { margin-top: 30px; text-align: center; color: #475569; font-size: 0.7rem; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>🤖 Polymarket Autotrader — Live Dashboard</h1>

    <div class="cards">
        <div class="card">
            <div class="label">Capital</div>
            <div class="value green" id="capital">$--</div>
        </div>
        <div class="card">
            <div class="label">Total PnL</div>
            <div class="value" id="pnl">$--</div>
        </div>
        <div class="card">
            <div class="label">Return</div>
            <div class="value" id="return">--%</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value" id="winrate">--%</div>
        </div>
        <div class="card">
            <div class="label">Trades</div>
            <div class="value" id="trades">--</div>
        </div>
        <div class="card">
            <div class="label">Open Positions</div>
            <div class="value yellow" id="open">--</div>
        </div>
        <div class="card">
            <div class="label">Max Drawdown</div>
            <div class="value red" id="dd">--%</div>
        </div>
        <div class="card">
            <div class="label">Scans</div>
            <div class="value" id="scans">--</div>
        </div>
    </div>

    <canvas id="chart"></canvas>

    <div class="section">
        <h2>📊 Open Positions</h2>
        <table>
            <thead><tr><th>Trade</th><th>Barrier</th><th>Cost</th><th>Win%</th><th>Profit%</th><th>DTE</th></tr></thead>
            <tbody id="positions"></tbody>
        </table>
    </div>

    <div class="section">
        <h2>📜 Recent Trades</h2>
        <table>
            <thead><tr><th>Result</th><th>Trade</th><th>PnL</th><th>Date</th></tr></thead>
            <tbody id="history"></tbody>
        </table>
    </div>

    <div class="footer">Auto-refresh: 30s | Paper Trading Mode | Polymarket ↔ Deribit</div>

    <script>
        let chart;
        async function update() {
            const r = await fetch('/api/status');
            const d = await r.json();
            const s = d.state;
            const exp = s.capital + d.exposure;
            const ret = ((exp - s.initial_capital) / s.initial_capital * 100);
            const wr = s.total_trades > 0 ? (s.wins / s.total_trades * 100) : 0;

            document.getElementById('capital').textContent = '$' + exp.toFixed(2);
            document.getElementById('capital').className = 'value ' + (exp >= s.initial_capital ? 'green' : 'red');
            document.getElementById('pnl').textContent = '$' + s.total_pnl.toFixed(2);
            document.getElementById('pnl').className = 'value ' + (s.total_pnl >= 0 ? 'green' : 'red');
            document.getElementById('return').textContent = (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%';
            document.getElementById('return').className = 'value ' + (ret >= 0 ? 'green' : 'red');
            document.getElementById('winrate').textContent = wr.toFixed(0) + '%';
            document.getElementById('winrate').className = 'value ' + (wr >= 50 ? 'green' : 'yellow');
            document.getElementById('trades').textContent = s.wins + 'W / ' + s.losses + 'L';
            document.getElementById('open').textContent = s.positions.length;
            document.getElementById('dd').textContent = s.max_drawdown.toFixed(1) + '%';
            document.getElementById('scans').textContent = s.scans_count;

            // Positions
            let ph = '';
            s.positions.forEach(p => {
                const exp_dt = new Date(p.expiry_dt);
                const dte_h = Math.max(0, (exp_dt - new Date()) / 3600000).toFixed(0);
                ph += '<tr><td>' + p.desc + '</td><td>$' + p.barrier.toLocaleString() +
                    '</td><td>$' + p.cost.toFixed(2) + '</td><td>' + p.win_prob + '%</td><td>+' +
                    p.profit_pct + '%</td><td>' + dte_h + 'h</td></tr>';
            });
            document.getElementById('positions').innerHTML = ph || '<tr><td colspan="6" style="color:#475569">No open positions</td></tr>';

            // History
            let hh = '';
            s.closed_trades.slice(-10).reverse().forEach(t => {
                const cls = t.result === 'WIN' ? 'badge-win' : 'badge-loss';
                const dt = new Date(t.settled_at).toLocaleString();
                hh += '<tr><td><span class="badge ' + cls + '">' + (t.result === 'WIN' ? '✅' : '❌') + ' ' +
                    t.result + '</span></td><td>' + t.desc + '</td><td style="color:' +
                    (t.pnl >= 0 ? '#10b981' : '#ef4444') + '">$' + t.pnl.toFixed(2) +
                    '</td><td style="color:#64748b">' + dt + '</td></tr>';
            });
            document.getElementById('history').innerHTML = hh || '<tr><td colspan="4" style="color:#475569">No trades settled yet</td></tr>';

            // Chart
            const pnl = d.pnl_history || [];
            if (pnl.length > 1) {
                const labels = pnl.map(p => new Date(p[0]).toLocaleTimeString());
                const values = pnl.map(p => p[2]);
                if (chart) chart.destroy();
                chart = new Chart(document.getElementById('chart'), {
                    type: 'line',
                    data: { labels, datasets: [{
                        label: 'Portfolio Value ($)',
                        data: values,
                        borderColor: values[values.length-1] >= s.initial_capital ? '#10b981' : '#ef4444',
                        backgroundColor: (values[values.length-1] >= s.initial_capital ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)'),
                        fill: true, tension: 0.3, pointRadius: 0,
                    }]},
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: '#64748b', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
                            y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' },
                                 suggestedMin: s.initial_capital * 0.9, suggestedMax: s.initial_capital * 1.1 }
                        }
                    }
                });
            }
        }
        update();
        setInterval(update, 30000);
    </script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    s = BOT_STATE or default_state()
    exposure = sum(p["cost"] for p in s.get("positions", []))
    return jsonify({
        "state": s,
        "exposure": exposure,
        "pnl_history": PNL_HISTORY[-200:],  # Last 200 data points
        "config": {k: v for k, v in CONFIG.items() if "token" not in k},
    })


@app.route("/health")
def health():
    return "OK", 200


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start bot in background thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    # Start web server
    port = int(os.getenv("PORT", "5001"))
    log.info(f"🌐 Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
