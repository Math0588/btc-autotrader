"""
📊 Polymarket Alpha Strategy Backtest — Recorrelation & Early Exits 
===================================================================
Simulates trading the spread between Retail-biased Polymarket prices
and true Options/Black-Scholes pricing.

Key Strategy:
1. Trade High-Probability events only (Win Prob > 45%). Avoid cheap lottery tickets.
2. Continually mark-to-market open positions every day.
3. Exit early (Take Profit) when Polymarket price converges with the model.
4. Cut losses early (Stop Loss) to avoid capital turning to $0 at expiry.
"""

import json
import math
import random
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import norm

# ─── Configuration ────────────────────────────────────────────────────────────

BACKTEST_CONFIG = {
    "starting_capital": 100.0,
    "min_edge_pct": 2.0,          # Minimum edge to enter
    "kelly_fraction": 0.20,       # Moderate Kelly sizing
    "max_position_pct": 0.15,     # Max 15% capital per trade
    "max_exposure_pct": 0.80,     # Max 80% portfolio exposure
    "polymarket_fee_pct": 2.0,    # Fees on payout/sell
    "risk_free_rate": 0.045,
    "min_win_prob": 0.45,         # HIGH PROB ONLY! No cheap lotteries.
    "btc_annual_vol": 0.55,       # Baseline BTC annualized vol
    
    # Advanced Early Exit Parameters
    "take_profit_pct": 35.0,      # Exit if position gains 35%
    "stop_loss_pct": -40.0,       # Exit if position loses 40% (prevent 100% loss)
    "prevent_100_loss": True,     # Avoid trailing to $0
    
    # Polymarket behavior simulation
    "pm_bias_pct": 3.0,           # Retail bullish bias on Polymarket (YES is overpriced)
    "pm_noise_std": 2.5,          # Daily volatility in PM's mispricing
}


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_btc_history(days=730):
    """Fetch 2 years of daily BTC prices using Binance API."""
    print(f"Fetching {days} days of BTC price history...")
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1d", "limit": days}, timeout=30)
    data = resp.json()
    prices = []
    for row in data:
        ts = row[0]
        price = float(row[4])  # Close price
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        prices.append({"date": dt, "price": price})
    return prices


# ─── Probability Models ───────────────────────────────────────────────────────

def model_probability(spot, strike, dte, vol, scenario="above"):
    """Black-Scholes probability."""
    if dte <= 0: return 1.0 if (spot >= strike and scenario == "above") or (spot < strike and scenario == "below") else 0.0
    T = dte / 365.0
    d2 = (math.log(spot / strike) + (BACKTEST_CONFIG["risk_free_rate"] - 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    return norm.cdf(d2) if scenario == "above" else norm.cdf(-d2)

def pm_simulated_price(true_prob_above, days_to_expiry):
    """Simulates how Polymarket retail prices the YES share (often biased & noisy)."""
    # Noise reduces as expiry approaches
    noise_factor = min(1.0, math.sqrt(days_to_expiry / 14.0)) 
    bias = (BACKTEST_CONFIG["pm_bias_pct"] / 100) * noise_factor
    noise = random.gauss(0, (BACKTEST_CONFIG["pm_noise_std"] / 100) * noise_factor)
    return min(0.99, max(0.01, true_prob_above + bias + noise))


# ─── Main Tracker ─────────────────────────────────────────────────────────────

def run_advanced_backtest():
    prices = fetch_btc_history(730)
    if len(prices) < 30: return
    
    capital = BACKTEST_CONFIG["starting_capital"]
    initial_capital = capital
    positions = []
    trades_log = []
    equity_curve = []
    
    wins = 0; losses = 0; early_exits_win = 0; early_exits_loss = 0
    total_pnl = 0.0
    
    print(f"\n🚀 Recorrelation Strategy Bot Started")
    print(f"Starting capital: ${capital:.2f}")
    
    def calc_vol(prices_slice):
        if len(prices_slice) < 10: return BACKTEST_CONFIG["btc_annual_vol"]
        returns = [math.log(prices_slice[i]["price"] / prices_slice[i-1]["price"]) for i in range(1, len(prices_slice))]
        return np.std(returns) * math.sqrt(365)

    # Simulator Loop (Day by day)
    for i in range(30, len(prices)):
        today_data = prices[i]
        today_date = today_data["date"]
        spot = today_data["price"]
        
        # Volatility
        vol = calc_vol(prices[max(0, i-30):i])
        vol = max(0.30, min(vol, 1.20))
        
        still_open = []
        exposure = 0
        
        # ─── 1. Mark-To-Market & Early Exits ───
        for pos in positions:
            dte = (pos["expiry"] - today_date).days
            
            # Settlement at Expiry?
            if dte <= 0:
                won = (spot >= pos["strike"]) if pos["direction"] == "BUY_YES" else (spot < pos["strike"])
                if won:
                    payout = pos["n"] * (1.0 - pos["entry"]) * (1 - BACKTEST_CONFIG["polymarket_fee_pct"]/100)
                    capital += pos["cost"] + payout
                    total_pnl += payout
                    wins += 1
                    trades_log.append({"date": today_date.strftime("%Y-%m-%d"), "desc": pos["desc"], "result": "WIN(Exp)", "pnl": payout})
                else:
                    total_pnl -= pos["cost"]
                    losses += 1
                    trades_log.append({"date": today_date.strftime("%Y-%m-%d"), "desc": pos["desc"], "result": "LOSS(Exp)", "pnl": -pos["cost"]})
                continue
            
            # Simulated current Polymarket prices
            true_prob_above = model_probability(spot, pos["strike"], dte, vol, "above")
            current_pm_yes = pm_simulated_price(true_prob_above, dte)
            current_pm_no = 1.0 - current_pm_yes
            
            current_price = current_pm_yes if pos["direction"] == "BUY_YES" else current_pm_no
            current_value = pos["n"] * current_price
            
            # Fee to sell back early to Polymarket
            sell_value = current_value * (1 - BACKTEST_CONFIG["polymarket_fee_pct"]/100)
            pnl_pct = (sell_value / pos["cost"]) * 100 - 100
            
            # EARLY EXITS
            if pnl_pct >= BACKTEST_CONFIG["take_profit_pct"]:
                capital += sell_value
                pnl = sell_value - pos["cost"]
                total_pnl += pnl
                early_exits_win += 1
                trades_log.append({"date": today_date.strftime("%Y-%m-%d"), "desc": pos["desc"], "result": "TP(Early)", "pnl": pnl})
                continue
                
            if BACKTEST_CONFIG["prevent_100_loss"] and pnl_pct <= BACKTEST_CONFIG["stop_loss_pct"]:
                capital += sell_value
                pnl = sell_value - pos["cost"]
                total_pnl += pnl
                early_exits_loss += 1
                trades_log.append({"date": today_date.strftime("%Y-%m-%d"), "desc": pos["desc"], "result": "SL(Early)", "pnl": pnl})
                continue
                
            exposure += current_value
            still_open.append(pos)
            
        positions = still_open
        
        # ─── 2. Evaluate new trades (only once a week to simulate new market launches) ───
        if i % 7 == 0 and exposure < (capital * BACKTEST_CONFIG["max_exposure_pct"]):
            # Generate $20k ranges around spot
            strike_step = 2000
            min_strike = max(10000, int((spot * 0.80) / strike_step) * strike_step)
            max_strike = int((spot * 1.20) / strike_step) * strike_step
            strikes = list(range(min_strike, max_strike + strike_step, strike_step))
            
            for strike in strikes:
                # Target expiry is 14 days out
                expiry_dt = today_date + timedelta(days=14)
                
                true_prob_above = model_probability(spot, strike, 14, vol, "above")
                pm_yes = pm_simulated_price(true_prob_above, 14)
                pm_no = 1.0 - pm_yes
                
                # Check Edge for YES
                edge_yes = true_prob_above - pm_yes
                edge_no = (1-true_prob_above) - pm_no
                
                direction = None
                edge = 0
                win_prob = 0
                entry = 0
                
                if edge_yes > BACKTEST_CONFIG["min_edge_pct"]/100 and true_prob_above >= BACKTEST_CONFIG["min_win_prob"]:
                    direction = "BUY_YES"
                    edge = edge_yes
                    win_prob = true_prob_above
                    entry = pm_yes
                    desc = f"BTC > ${strike:,.0f}"
                elif edge_no > BACKTEST_CONFIG["min_edge_pct"]/100 and (1-true_prob_above) >= BACKTEST_CONFIG["min_win_prob"]:
                    direction = "BUY_NO"
                    edge = edge_no
                    win_prob = 1 - true_prob_above
                    entry = pm_no
                    desc = f"BTC < ${strike:,.0f}"

                if not direction:
                    continue
                    
                payout = (1.0 - entry) * (1 - BACKTEST_CONFIG["polymarket_fee_pct"]/100)
                cost = entry
                b = payout / cost
                kelly = max(0, (win_prob * b - (1 - win_prob)) / b) * BACKTEST_CONFIG["kelly_fraction"]
                
                if kelly <= 0: continue
                
                pos_size = min(kelly * capital, capital * BACKTEST_CONFIG["max_position_pct"])
                n = int(pos_size / entry)
                if n < 1: continue
                
                actual_cost = n * entry
                if capital > actual_cost:
                    capital -= actual_cost
                    exposure += actual_cost
                    positions.append({
                        "strike": strike, "expiry": expiry_dt, "direction": direction,
                        "desc": desc, "entry": entry, "n": n, "cost": actual_cost
                    })

        # Track Equity
        total_val = capital + exposure
        equity_curve.append(total_val)
        
    # Stats
    total_trades = wins + losses + early_exits_win + early_exits_loss
    return_pct = (capital - initial_capital) / initial_capital * 100
    win_rate = (wins + early_exits_win) / max(total_trades, 1) * 100
    peak = max(equity_curve) if equity_curve else capital
    mdd = (1 - capital/peak) * 100
    
    print("\n" + "="*60)
    print(" 🏆 ADVANCED RECORRELATION RESULTS")
    print("="*60)
    print(f" Capital:      ${initial_capital} ➡️  ${capital:.2f} ({return_pct:+.1f}%)")
    print(f" Total Trades: {total_trades}")
    print(f" Win Rate:    {win_rate:.1f}%")
    print(" Breakdowns:")
    print(f"   ✓ Expiry Wins:  {wins}")
    print(f"   ✓ Take Profit (Recorrelated): {early_exits_win}")
    print(f"   ✕ Expiry Loss:  {losses}")
    print(f"   ✕ Stop Loss (Cut early): {early_exits_loss}")
    print(f" Max Drawdown: {mdd:.1f}%")
    print("="*60)
    
    print("\n📜 5 Derniers trades:")
    for t in trades_log[-5:]:
        color = "🟩" if "WIN" in t["result"] or "TP" in t["result"] else "🟥"
        print(f"   {t['date']} | {color} {t['result']:>10s} | {t['desc']:<15s} | PnL: ${t['pnl']:+.2f}")

if __name__ == "__main__":
    run_advanced_backtest()
