import os
import csv
import time
import random
import requests
import functools
import http.client
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import robin_stocks.robinhood as rs
from dotenv import load_dotenv

# === Initialization & Config ===
load_dotenv()
USERNAME = os.getenv("ROBINHOOD_USERNAME")
PASSWORD = os.getenv("ROBINHOOD_PASSWORD")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

rs.login(username=USERNAME, password=PASSWORD)

scalper_mode = input("⚡ Scalper Mode? (y/n): ").strip().lower() == 'y'
live = input("🚀 LIVE mode? (y/n): ").strip().lower() == 'y'
LOG_SIGNALS = True

SIGNAL_LOG_PATH = "signal_log.csv"
FIELDS = [
    'timestamp', 'symbol', 'strategy', 'volatility', 'price', 'signal', 'result'
]

MAX_TICKERS = 20
TRAILING_STOP_THRESHOLD = 0.05
LOW_EFF_THRESHOLD = 0.15
CYCLE_RETRAIN = 10 if scalper_mode else 20

model = None
pending_signals = deque(maxlen=100)

# === Utilities ===
def now():
    return datetime.now().strftime("%H:%M:%S")

def retry_on_disconnect(max_retries=3, base_delay=2, relogin=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (http.client.RemoteDisconnected, ConnectionResetError):
                    print(f"Retrying {func.__name__} (attempt {attempt + 1})...")
                    time.sleep(base_delay + random.random() * base_delay)
                    if relogin and attempt == max_retries - 1:
                        print("Re-login to Robinhood...")
                        rs.logout()
                        rs.login(username=USERNAME, password=PASSWORD)
            print(f"{func.__name__} failed after {max_retries} attempts.")
            return None
        return wrapper
    return decorator

@retry_on_disconnect()
def get_price(symbol):
    if scalper_mode and FINNHUB_API_KEY:
        try:
            resp = requests.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}")
            return float(resp.json().get("c", 0))
        except Exception as e:
            print(f"Finnhub error for {symbol}: {e}")
            return 0
    else:
        try:
            return float(rs.stocks.get_quotes(symbol)[0].get('last_trade_price', 0))
        except Exception as e:
            print(f"Robinhood quote error for {symbol}: {e}")
            return 0

def log_signal(sym, strat, vol, price, signal):
    if not LOG_SIGNALS:
        return
    new = not os.path.exists(SIGNAL_LOG_PATH) or os.path.getsize(SIGNAL_LOG_PATH) == 0
    with open(SIGNAL_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'symbol': sym,
            'strategy': strat,
            'volatility': vol,
            'price': price,
            'signal': signal,
            'result': 'pending'
        })

def train_ml():
    global model
    try:
        df = pd.read_csv(SIGNAL_LOG_PATH)
        df = df[df['result'].isin(['win', 'fail'])]
        if len(df) < 10 or len(set(df['result'])) < 2:
            print("Insufficient data to train ML.")
            return
        df['result'] = df['result'].map({'win': 1, 'fail': 0})
        X = df[['volatility', 'price']].astype(float)
        y = df['result']
        model = LogisticRegression().fit(X, y)
        print(f"ML trained: {len(df)} entries, class dist: {y.value_counts().to_dict()}")
    except Exception as e:
        print(f"ML training error: {e}")

def should_trade(vol, price):
    if not model:
        return True
    try:
        return model.predict([[vol, price]])[0] == 1
    except:
        return True

def evaluate_pending(prices):
    nowt = datetime.now()
    for ps in list(pending_signals):
        if (nowt - ps['time']).total_seconds() < 600:
            continue
        cur = prices.get(ps['symbol'], 0)
        res = 'win' if (ps['signal']=='buy' and cur>ps['price']) or (ps['signal']=='sell' and cur<ps['price']) else 'fail'
        # ... implement update log logic ...
        pending_signals.remove(ps)

# === Strategies & Portfolio ===
STRATEGIES = {
    "momentum": (lambda p: np.mean(p[-3:]) > np.mean(p[-5:]), 0.85),
    "reversion": (lambda p: p[-1] < np.mean(p[-5:]), 0.85),
    "breakout": (lambda p: p[-1] > max(p[-5:-1]), 0.9),
}

class Portfolio:
    def __init__(self, cash=10000):
        self.cash = cash
        self.positions = {}
        self.history = []
        self.stats = {}

    def buy(self, sym, price, qty, strat):
        cost = price * qty
        if qty <= 0 or cost > self.cash:
            print(f"Insufficient funds for {sym} x{qty} @ {price:.2f}")
            return False
        self.cash -= cost
        pos = self.positions.get(sym, {'qty': 0, 'cost': price})
        total_qty = pos['qty'] + qty
        avg = (pos['cost']*pos['qty'] + price*qty) / total_qty
        self.positions[sym] = {'qty': total_qty, 'cost': avg}
        self.stats.setdefault(strat, {'cnt': 0, 'wins': 0})
        self.stats[strat]['cnt'] += 1
        self.history.append((now(), f"BUY {sym} {qty}@{price:.2f}"))
        return True

    def sell(self, sym, price):
        if sym not in self.positions:
            print(f"No position to sell for {sym}")
            return
        pos = self.positions.pop(sym)
        pnl = (price - pos['cost']) * pos['qty']
        self.cash += price * pos['qty']
        self.history.append((now(), f"SELL {sym} {pos['qty']}@{price:.2f} -> PnL {pnl:.2f}"))
        return pnl

    def summary(self, prices):
        total = self.cash
        print("\nPortfolio Summary:")
        for s, p in self.positions.items():
            cur = prices.get(s, get_price(s))
            total += p['qty'] * cur
            print(f" {s}: {p['qty']} @ cost {p['cost']:.2f}, cur {cur:.2f}")
        print(f" Cash: {self.cash:.2f} | Total value: {total:.2f}")
        print("Recent trades:", self.history[-5:])

# === Main Loop ===
portfolio = Portfolio()
if LOG_SIGNALS and os.path.exists(SIGNAL_LOG_PATH):
    train_ml()

cycle = 0
while True:
    print(f"\nCycle @ {now()}")
    symbols = ["AAPL","MSFT","NVDA","AMD","TSLA","GOOG",
    "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS", "TECL"]
    # Simplified symbol list for brevity
    prices = {sym: get_price(sym) for sym in symbols}
    for sym in symbols:
        try:
            hist = rs.stocks.get_stock_historicals(sym, interval='5minute', span='day', bounds='regular')
            closes = [float(h['close_price']) for h in hist if h.get('close_price')]
            if len(closes) < 6:
                continue
            vol = round(np.std(closes[-10:]), 4)
        except Exception as e:
            print(f"Hist error {sym}: {e}")
            continue

        best = (None, 0)
        for name, (func, thr) in STRATEGIES.items():
            sig = func(closes)
            if not sig:
                continue
            conf = thr  # Starting confidence simplified
            if conf > best[1]:
                best = (name, conf)

        if best[0] and best[1] >= STRATEGIES[best[0]][1]:
            price = prices[sym]
            if should_trade(vol, price):
                if not price or price <= 0:
                    print(f"Skipping {best[0]} — invalid price: {price}")
                    continue
                qty = int((portfolio.cash * best[1]) // price) or 1
                print(f"Trade {sym}: {best[0]} conf {best[1]:.2f} qty {qty} @ {price:.2f}")
                if live:
                    rs.orders.order_buy_market(sym, qty)
                else:
                    if portfolio.buy(sym, price, qty, best[0]):
                        log_signal(sym, best[0], vol, price, 'buy')

    evaluate_pending(prices)
    portfolio.summary(prices)

    cycle += 1
    if LOG_SIGNALS and cycle % CYCLE_RETRAIN == 0:
        train_ml()

    dur = time.time() - (time.time() - time.time())  # placeholder
    time.sleep(15 if scalper_mode else 300)
