import os
import csv
import time
import math
import torch
import random
import requests
import functools
import http.client
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import deque
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BartTokenizer, BartForConditionalGeneration
import robin_stocks.robinhood as rs

# === 🛂 Login ===
load_dotenv()
username = os.environ.get("ROBINHOOD_USERNAME")
password = os.environ.get("ROBINHOOD_PASSWORD")
rs.login(username=username, password=password)

# === ⚙️ Toggles & Settings ===
live_input = input("🚀 Run in LIVE mode? (y/n): ").strip().lower()
live = False
if live_input == 'y':
    confirm = input("⚠️ Are you sure? LIVE mode will use real funds. Type 'yes' to confirm: ").strip().lower()
    live = confirm == 'yes'
# === Backtest Toggle ===
ENABLE_BACKTEST = input("🧪 Run backtest simulation? (y/n): ").strip().lower() == 'y'
USE_SENTIMENT = input("💬 Use sentiment overlay? (y/n): ").strip().lower() == 'y'
LOG_SIGNALS = True
SHOW_EFFICIENCY = True
AUTOSELL_LOW_EFFICIENCY = True
TAG_DORMANT = True

# === 🛠️ Constants ===
PRICE_CAP = 2500.00  # dollars & sense
SENTIMENT_THRESHOLD = 0.20
MAX_TICKERS = 20
STALE_LIMIT = 3
TRAILING_STOP_THRESHOLD = 0.05
LOW_EFFICIENCY_THRESHOLD = 0.15
CONF_LOG_PATH = "confidence_log.csv"
SIGNAL_LOG_PATH = "signal_log.csv"
SIGNAL_LOG_FIELDS = [
    'timestamp', 'symbol', 'strategy', 'percent_change',
    'spread', 'bid', 'ask', 'mid_price',
    'volatility', 'hour', 'price', 'signal', 'result'
]
strategy_cooldowns = {}  # (symbol, strategy) → timestamp
COOLDOWN_DURATION = 1  # in cycles

# Load BART and VADER
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
vader = SentimentIntensityAnalyzer()

def next_five_min_mark(ts: datetime) -> datetime:
    ts = ts.replace(second=0, microsecond=0)
    delta = 5 - (ts.minute % 5)
    if delta == 0:
        delta = 5
    return ts + timedelta(minutes=delta)

def get_time_flair():
    hour = datetime.now().hour
    if 5 <= hour < 10:
        return "🌅", "\033[94m"  # Morning: Light blue
    elif 10 <= hour < 17:
        return "🌞", "\033[92m"  # Day: Green
    elif 17 <= hour < 21:
        return "🌆", "\033[93m"  # Evening: Yellow
    else:
        return "🌙", "\033[95m"  # Night: Purple

# === 🔁 Retry Logic ===
def retry_on_disconnect(max_retries=3, base_delay=1.5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (http.client.RemoteDisconnected, ConnectionResetError) as e:
                    error_msg = f"{datetime.now().isoformat()} — {func.__name__} failed (attempt {attempt+1}/{max_retries}): {e}"
                    print(f"⚠️ {error_msg}")
                    with open("error_log.txt", "a") as log:
                        log.write(error_msg + "\n")
                    time.sleep(base_delay + random.uniform(0, 2))
            print(f"❌ {func.__name__} failed after {max_retries} attempts.")
            return None
        return wrapper
    return decorator

# === 🧬 LLM Summarizer ===
def get_combined_sentiment(symbol, headlines):
    if not headlines:
        return "neutral", 0.0, "No new headlines found."
    
    # Use BART for summarization
    try:
        input_text = " ".join(headlines[:5])
        inputs = bart_tokenizer([input_text], return_tensors="pt", truncation=True, max_length=512)
        summary_ids = bart_model.generate(**inputs, max_length=60, num_beams=4, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        summary = f"Summarization error: {e}"

    # Use VADER on the summary
    try:
        vs = vader.polarity_scores(summary)
        compound = vs['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    except Exception as e:
        sentiment = "neutral"
        compound = 0.0

    return sentiment, compound, summary

# === 📰 Finviz Scraper ===
# Track seen headlines to prevent re-reading
seen_headlines = {}

@retry_on_disconnect(max_retries=3)
def scrape_finviz(symbol):
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        news_table = soup.find("table", class_="fullview-news-outer")
        headlines = []

        for row in news_table.find_all("tr") if news_table else []:
            td = row.find_all("td")
            if len(td) == 2:
                headline = td[1].text.strip()

                # Avoid duplicates
                if headline and headline not in seen_headlines.get(symbol, set()):
                    headlines.append(headline)
                    seen_headlines.setdefault(symbol, set()).add(headline)

        return headlines

    except Exception as e:
        print(f"⚠️ Could not scrape Finviz for {symbol}: {e}")
        return []

# === 💬 Sentiment ===
def show_sentiment_face(sentiment):
    return {
        "positive": " 😄 ",
        "neutral": " 😐 ",
        "negative": " ☹️ "
    }.get(sentiment, " ❓ ")

# === 🧠 ML Trainer ===
pending_signals = deque(maxlen=100)

# === Trailing PnL Tracking ===
cumulative_pnl = 0.0
rolling_pnl_window = deque(maxlen=10)  # adjustable window

def train_ml_from_log(path=SIGNAL_LOG_PATH):
    try:
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(k in row and row[k] for k in ('signal', 'result', 'volatility', 'price', 'hour')):
                    rows.append(row)
                else:
                    print(f"⚠️ Skipping malformed row: {row}")

        if len(rows) < 10:
            print("⚠️ Not enough clean rows to train ML model.")
            return

        df = pd.DataFrame(rows)
        y = df['result'].apply(lambda r: 1 if r == 'win' else 0)
        if y.nunique() < 2:
            print(f"⚠️ Cannot train ML model — only one class present.")
            return

        X = df[['volatility', 'price', 'hour']].astype(float)

        print("Length of X:", len(X))
        print("Length of y:", len(y))
        
        model = LogisticRegression()
        model.fit(X, y)
        print(f"🤖 ML model trained successfully — {len(df)} rows.")
    except Exception as e:
        print(f"⚠️ Error training ML model: {e}")

def ml_should_trade(row_data):
    if not model:
        return True  # fail open
    try:
        features = [[
            row_data.get('volatility', 0),
            row_data.get('last_price', 0),
            datetime.now().hour
        ]]
        pred = model.predict(features)[0]
        return pred == 1
    except:
        return True

# === 📓 Signal Logging ===
def log_signal(sym, strategy, row_data, signal):
    if not LOG_SIGNALS:
        return

    file_exists = os.path.exists(SIGNAL_LOG_PATH)
    needs_header = not file_exists or os.path.getsize(SIGNAL_LOG_PATH) == 0

    with open(SIGNAL_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'symbol': sym,
            'strategy': strategy,
            'percent_change': row_data.get('percent_change', 0),
            'spread': row_data.get('spread', 0),
            'bid': row_data.get('bid', 0),
            'ask': row_data.get('ask', 0),
            'mid_price': (row_data.get('bid', 0) + row_data.get('ask', 0)) / 2 if row_data.get('bid') and row_data.get('ask') else 0,
            'volatility': row_data.get('volatility', 0),
            'hour': datetime.now().hour,
            'price': row_data.get('last_price', 0),
            'signal': signal,
            'result': 'pending'
        })

def log_confidence(sym, strategy_confidence):
    try:
        file_exists = os.path.exists(CONF_LOG_PATH)
        with open(CONF_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'symbol', 'strategy_confidence'])
            writer.writerow([datetime.now().isoformat(), sym] + list(strategy_confidence.values()))
    except Exception as e:
        print(f"⚠️ Could not log confidence: {e}")

def update_signal_log(sym, timestamp, result):
    try:
        rows = []
        with open(SIGNAL_LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not all(k in row for k in ['symbol', 'timestamp']):
                    continue
                if row['symbol'] == sym and row['timestamp'] == timestamp.isoformat():
                    row['result'] = result
                rows.append(row)
                incremental_train(row)

        with open(SIGNAL_LOG_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"⚠️ Failed to update signal log: {e}")

def evaluate_pending_signals(prices_now):
    now_time = datetime.now()
    for ps in list(pending_signals):
        age = (now_time - ps['timestamp']).total_seconds()
        if age < 600:  # wait 10 mins (~2 cycles)
            continue

        sym = ps['symbol']
        orig_price = ps['price']
        current_price = prices_now.get(sym, 0)
        raw_signal = ps.get('signal', 'BUY')
        strategy = ps.get('strategy')  # ✅ Now tracking strategy

        # Normalize signal
        signal = str(raw_signal).strip().upper()

        # === Determine result
        if signal == 'BUY' and current_price > orig_price:
            result = 'win'
        elif signal == 'SELL' and current_price < orig_price:
            result = 'win'
        elif current_price == orig_price:
            result = 'neutral'
        else:
            result = 'loss'

        print(f"🧾 Signal evaluated: {sym} — {signal} → {result.upper()} | {orig_price} → {current_price}")
        update_signal_log(sym, ps['timestamp'], result)
        pending_signals.remove(ps)

        # ✅ Update win/loss stats
        if strategy and result in ("win", "loss"):
            stats = portfolio.strategy_stats.get(strategy)
            if stats:
                stats["wins" if result == "win" else "losses"] += 1

def show_pnl_stats():
    avg = np.mean(rolling_pnl_window) if rolling_pnl_window else 0
    print(f"\n📈 Trailing PnL: ${cumulative_pnl:.2f} | Rolling Avg (last {len(rolling_pnl_window)}): ${avg:.2f}")

# === Incremental ML ===
def incremental_train(new_row):
    try:
        if not all(k in new_row and new_row[k] for k in ('result', 'volatility', 'price', 'hour')):
            return

        result = new_row['result']
        if result not in ["win", "loss"]:
            return

        y = 1 if result == "win" else 0
        X = [[
            float(new_row['volatility']),
            float(new_row['price']),
            int(new_row['hour'])
        ]]

        if model is None:
            model = LogisticRegression()
            model.fit(X, [y])
            print("🤖 Initial incremental model trained.")
        else:
            model.partial_fit(X, [y])
            print("🔄 Incremental model updated.")

    except Exception as e:
        print(f"⚠️ Incremental training failed: {e}")

def review_confidence_and_pnl(path=SIGNAL_LOG_PATH):
    try:
        df = pd.read_csv(path)
        if len(df) < 10:
            return

        df = df[df["result"].isin(["win", "loss"])]
        if df.empty:
            return

        # Fix for float error
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].fillna(1.0)
        else:
            df["confidence"] = 1.0

        df["pnl"] = df["price"].astype(float) * df["spread"].astype(float)
        avg_pnl = df["pnl"].mean()
        avg_conf = df["confidence"].mean()
        win_rate = (df["result"] == "win").sum() / len(df) * 100

        print("\n🧠 ML Performance Summary:")
        print(f" - Samples: {len(df)}")
        print(f" - Win Rate: {win_rate:.1f}%")
        print(f" - Avg Confidence: {avg_conf:.2f}")
        print(f" - Avg PnL: ${avg_pnl:.2f}\n")

    except Exception as e:
        print(f"⚠️ Text-based ML review error: {e}")

def indicator_check(sym, prices, strategy):
    confidence_mod = 1.0
    close_prices = np.array(prices[-20:])  # Ensure enough data

    if strategy == "mean_reversion":
        rsi = compute_rsi(close_prices)
        if rsi < 30:
            confidence_mod *= 1.2  # Oversold → buy potential
        elif rsi > 70:
            confidence_mod *= 0.8  # Overbought → cautious

    elif strategy == "breakout":
        upper, lower = compute_bollinger_bands(close_prices)
        last_price = close_prices[-1]
        if last_price > upper:
            confidence_mod *= 1.15  # Breakout confirmation
        elif last_price < lower:
            confidence_mod *= 0.85  # Potential exhaustion

    return round(confidence_mod, 3)

# === 🧾 Portfolio Tracker ===
class Portfolio:
    def __init__(self, initial_cash=10000):
        self.cash = initial_cash
        self.positions = {}  # sym → {qty, cost, peak}
        self.history = []    # trade logs
        self.strategy_stats = {}  # strategy → win/loss tracking

    def buy(self, sym, price, qty, strategy=None):
        cost = price * qty
        if self.cash < cost or qty <= 0:
            print(f"❌ Not enough cash to buy {qty} x {sym}")
            return False
        self.cash -= cost
        if sym in self.positions:
            pos = self.positions[sym]
            total_qty = pos['qty'] + qty
            avg_cost = ((pos['cost'] * pos['qty']) + (price * qty)) / total_qty
            self.positions[sym] = {'qty': total_qty, 'cost': avg_cost, 'peak': price}
        else:
            self.positions[sym] = {'qty': qty, 'cost': price, 'peak': price}
        self.history.append((now(), f"BUY {qty} {sym} @ {price:.2f}"))
        if strategy:
            self.strategy_stats.setdefault(strategy, {"wins": 0, "losses": 0, "total": 0})
            self.strategy_stats[strategy]["total"] += 1
        return True

    def sell(self, sym, price):
        if sym not in self.positions:
            print(f"⚠️ Tried to sell {sym} not in holdings")
            return
        qty = self.positions[sym]['qty']
        cost = self.positions[sym]['cost']
        pnl = (price - cost) * qty
        self.cash += price * qty
        global cumulative_pnl, rolling_pnl_window
        cumulative_pnl += pnl
        rolling_pnl_window.append(pnl)
        self.history.append((now(), f"SELL {qty} {sym} @ {price:.2f} → PnL: {pnl:.2f}"))
        del self.positions[sym]
        return pnl

    def summary(self, prices):
        total = self.cash
        unrealized_pnl = 0
        realized_pnl = 0
    
        print("\n📊 Portfolio:")
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            qty = pos['qty']
            cost = round(pos['cost'], 4)
            current = round(prices.get(sym, get_price(sym)), 4)
            val = round(current * qty, 2)
            pnl = round((current - cost) * qty, 2)
            unrealized_pnl += pnl
            age = position_age.get(sym, 0)
            total += val
    
            if abs(pnl) < 0.01:
                pnl = 0.00
    
            tags = []
            if TAG_DORMANT and round(current, 2) == round(cost, 2):
                tags.append("💤")
    
            if AUTOSELL_LOW_EFFICIENCY and age >= 2:
                pnl_per_cycle = pnl / age
                if pnl_per_cycle < LOW_EFFICIENCY_THRESHOLD:
                    print(f"🔻 Auto-sell triggered for {sym} (pnl/cycle: {pnl_per_cycle:.2f})")
                    self.sell(sym, current)
                    position_age[sym] = 0
                    continue
                elif pnl_per_cycle < 2 * LOW_EFFICIENCY_THRESHOLD:
                    tags.append("⚠️")
    
            tag_str = " ".join(tags)
            if SHOW_EFFICIENCY and age > 0:
                pnl_per_cycle = pnl / age
                print(f" - {sym}: {qty} @ {cost:.2f} → {current:.2f} → PnL: {pnl:.2f} over {age} cycle(s) [{pnl_per_cycle:.2f}/cycle] {tag_str}")
            else:
                print(f" - {sym}: {qty} @ {cost:.2f} → {current:.2f} → PnL: {pnl:.2f} over {age} cycle(s) {tag_str}")
    
        # Parse realized PnL from trade history
        for t in self.history:
            if "SELL" in t[1] and "PnL" in t[1]:
                try:
                    parts = t[1].split("→ PnL:")
                    if len(parts) == 2:
                        realized_pnl += float(parts[1])
                except:
                    continue
    
        print(f"\n💰 Cash: ${round(self.cash, 2):.2f}")
        print(f"📈 Total Value: ${round(total, 2):.2f}")
        print(f"📌 Realized PnL: ${realized_pnl:.2f}")
        print(f"🧮 Unrealized PnL: ${unrealized_pnl:.2f}")
        print(f"🏁 Total PnL: ${realized_pnl + unrealized_pnl:.2f}")
    
        if self.history:
            print("\n📜 Trade History:")
            for h in self.history[-10:]:
                print(f"  - {h[0]} | {h[1]}")
    
        if self.strategy_stats:
            print("\n📚 Strategy Summary:")
            for strat, stats in self.strategy_stats.items():
                total = stats["total"]
                winrate = (stats["wins"] / total) * 100 if total else 0
                label = STRATEGIES[strat]['label']
                print(f"  - {label}: {total} trades, {winrate:.1f}% win rate")

# === ⚙️ Strategy Definitions ===
def mean_reversion_strategy(prices):
    if len(prices) < 6:
        return None
    recent = prices[-1]
    avg = np.mean(prices[-5:])
    return 'buy' if recent < 0.98 * avg else None

def momentum_ma_strategy(prices):
    if len(prices) < 6:
        return None
    ma_short = np.mean(prices[-3:])
    ma_long = np.mean(prices[-6:])
    return 'buy' if ma_short > ma_long else None

def breakout_strategy(prices):
    if len(prices) < 10:
        return None
    recent = prices[-1]
    high = max(prices[-10:])
    return 'buy' if recent >= high else None

def dip_and_rip_strategy(prices):
    if len(prices) < 6:
        return None
    return 'buy' if prices[-1] > prices[-2] < prices[-3] else None

def three_bar_breakout(prices):
    if len(prices) < 4:
        return None
    p1, p2, p3 = prices[-4:-1]
    if p1 < p2 < p3 and prices[-1] > p3:
        return 'buy'
    return None

#=== RSI & Bollinger Bands ===
def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    ups = deltas[deltas > 0].sum() / period
    downs = -deltas[deltas < 0].sum() / period
    rs = ups / downs if downs != 0 else 0
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(prices, period=20, num_std=2):
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper, lower

# Strategy registry
STRATEGIES = {
    "mean_reversion": {
        "label": "Mean Reversion",
        "func": mean_reversion_strategy,
        "threshold": 0.75,
        "weight": 1.0
    },
    "momentum_ma": {
        "label": "Momentum MA",
        "func": momentum_ma_strategy,
        "threshold": 0.75,
        "weight": 1.2
    },
    "breakout": {
        "label": "Breakout",
        "func": breakout_strategy,
        "threshold": 0.8,
        "weight": 1.3
    },
    "dip_and_rip": {
        "label": "Dip & Rip",
        "func": dip_and_rip_strategy,
        "threshold": 0.7,
        "weight": 1.1
    },
    "three_bar": {
        "label": "Three-Bar Breakout",
        "func": three_bar_breakout,
        "threshold": 0.8,
        "weight": 1.4
    }
}

# === Backtest Engine with Reporter ===
def run_backtest(symbol, strategy_func, historical_prices, buy_threshold=1.0, initial_cash=10000):
    cash = initial_cash
    position = 0
    trade_log = []
    entry_price = None

    for i in range(30, len(historical_prices)):
        window = historical_prices[i - 30:i]
        signal = strategy_func(window)

        price = historical_prices[i]
        confidence = strategy_strength.get(strategy_func.__name__, 1.0)

        if signal and cash >= price and confidence >= buy_threshold:
            if price is None or price <= 0:
                continue  # skip invalid price points
            qty = cash // price
            cash -= qty * price
            position += qty
            entry_price = price
            trade_log.append((i, "BUY", price, qty))
        elif not signal and position > 0:
            cash += position * price
            pnl = (price - entry_price) * position if entry_price else 0
            trade_log.append((i, "SELL", price, position, pnl))
            position = 0
            entry_price = None

    final_value = cash + position * historical_prices[-1]
    total_pnl = final_value - initial_cash
    sell_trades = [t for t in trade_log if t[1] == "SELL"]
    win_trades = [t for t in sell_trades if t[4] > 0]
    loss_trades = [t for t in sell_trades if t[4] <= 0]

    avg_pnl = sum(t[4] for t in sell_trades) / len(sell_trades) if sell_trades else 0
    win_rate = (len(win_trades) / len(sell_trades)) * 100 if sell_trades else 0

    print(f"\n🧪 Backtest for {symbol} using {strategy_func.__name__}:")
    print(f"💰 Final Portfolio: ${final_value:.2f} (PnL: ${total_pnl:.2f})")
    print(f"📈 Trades: {len(sell_trades)} | Wins: {len(win_trades)} | Losses: {len(loss_trades)}")
    print(f"🎯 Avg Return per Trade: ${avg_pnl:.2f} | Win Rate: {win_rate:.1f}%")

# === ⛵ Startup Bootstraps ===
def now():
    return datetime.now().strftime("%H:%M:%S")

@retry_on_disconnect(max_retries=3)
def get_price(symbol):
    q = rs.stocks.get_quotes(symbol)[0]
    return float(q['last_trade_price'] or 0)

def trailing_stop_triggered(sym, current, peak):
    drop = (peak - current) / peak
    return drop >= TRAILING_STOP_THRESHOLD

if __name__ == "__main__":
    model = None  # Initialize model here

# === ⚙️ Ticker Toggle ===
USE_STATIC_SYMBOLS = input("📦 Use static ticker list? (y/n): ").strip().lower() == 'y'

"""STATIC_TICKERS = ["TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS", "TMF", "UPRO",
    "TECL", "FAS", "TNA", "FNGU", "YINN", "UDOW", "DPST", "LABU", "NUGT", "SPXU",
    "NAIL", "URTY", "DFEN", "TZA", "ERX", "YANG", "SDOW", "TMV", "KORU", "CURE",
    "BRZU", "FAZ", "SRTY", "TECS", "EDC", "MIDU", "INDL", "FNGD", "LABD", "DRN",
    "TYD", "EURL", "DRV", "DUSL", "RETL", "OILU", "UBOT", "UTSL", "UMDD", "TTT",
    "ERY", "MEXX", "EDZ", "TPOR", "WTIU", "OILD", "TYO", "PILL", "SMDD", "WTID"
]"""
    

STATIC_TICKERS = [
    "NVDA", "TSLA", "AMD", "AMZN", "MSFT", "AAPL", "META", "NFLX", "GOOGL", "SPY",
    "PLTR", "SOFI", "BIDU", "BABA", "COIN", "SNOW", "CRM", "MARA", "RIOT", "UBER",
    "ROKU", "ARKK", "NIO", "FUBO", "CVNA", "LCID", "AFRM", "SHOP", "XYZ", "PYPL"
]

# === Dynamic Scan Size (only if not using static list)
if not USE_STATIC_SYMBOLS:
    try:
        scan_size = int(input(f"🔍 How many tickers to scan? (max {MAX_TICKERS}): "))
    except:
        scan_size = 10
    scan_size = max(1, min(scan_size, MAX_TICKERS))
else:
    scan_size = len(STATIC_TICKERS)

portfolio = Portfolio()
position_age = {}
strategy_strength = {k: 1.0 for k in STRATEGIES}

# === 📝 Log Prep ===
if LOG_SIGNALS:
    try:
        needs_header = not os.path.exists(SIGNAL_LOG_PATH) or os.stat(SIGNAL_LOG_PATH).st_size == 0
        with open(SIGNAL_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_FIELDS)
            if needs_header:
                writer.writeheader()
        print(f"📝 Logging enabled — writing to {SIGNAL_LOG_PATH}")
    except Exception as e:
        print(f"[WARN] Could not prepare log file: {e}")

if LOG_SIGNALS and os.path.exists(SIGNAL_LOG_PATH):
    train_ml_from_log()
    if input("📊 Run visual ML review? (y/n): ").strip().lower() == 'y':
        review_confidence_and_pnl()

cycle_count = 0

t_fetch = time.time()

#=== Heatmap Key ===
print("\n🎨 Visual Key — Symbol Heatmap:")
print("/" * 40)
print(" ⚪  Neutral: < ±1.0% change")
print(" 🟡  Mild: ±1.0–2.0% change")
print(" 🟠  Moderate: ±2.0–3.0% change")
print(" ❤️  High Negative: < -3.0%")
print(" 💚  High Positive: > +3.0%")
print(" 🔥  Activity marker (always shown)")
print("/" * 40)


while True:
    cycle_start = time.time()         # start timer first
    now_str = now()
    cycle_count += 1                  # increment the counter
    flair, color = get_time_flair()   # get our themed header
    print(f"{color}\n{flair} ⏳ Cycle {cycle_count} @ {now()}\033[0m")
    
    # === Symbol Fetcher (static override + multi-tag) ===
    if USE_STATIC_SYMBOLS:
        symbols = STATIC_TICKERS[:scan_size]
        print(f"📦 Using static symbols: {', '.join(symbols)}")
    else:
        #all_tags = ["top-movers"]
        all_tags = ["technology", "biopharmaceutical", "top-movers", "100-most-popular"]
        #all_tags = ["100-most-popular", "top-movers", "technology", "biopharmaceutical", "upcoming-earnings", "most-popular-under-25"]
        all_stocks = []
        for tag in all_tags:
            try:
                tag_stocks = rs.markets.get_all_stocks_from_market_tag(tag, info=None)
                if tag_stocks:
                    all_stocks.extend(tag_stocks)
                    print(f"📥 Pulled {len(tag_stocks)} symbols from tag '{tag}'")
            except Exception as e:
                print(f"⚠️ Error fetching tag '{tag}': {e}")
    
        # 🔁 Deduplicate while preserving order
        seen = set()
        unique_symbols = []
        for stock in all_stocks:
            sym = stock.get('symbol')
            if sym and sym not in seen:
                seen.add(sym)
                unique_symbols.append(sym)
    
        symbols = unique_symbols[:scan_size]
        print(f"🔍 Final symbol list ({len(symbols)}): {', '.join(symbols)}")

    quotes = rs.stocks.get_quotes(symbols)

    parsed_data = []
    prices_now = {}
    for q in quotes:
        try:
            sym = q['symbol']
            last_price = float(q.get('last_trade_price') or 0)
            previous_close = float(q.get('previous_close') or 0)
            bid = float(q.get('bid_price') or 0)
            ask = float(q.get('ask_price') or 0)
            spread = ask - bid
            pct_change = ((last_price - previous_close) / previous_close * 100) if previous_close > 0.01 else 0.0
            # 🔥 Heat Emoji Based on % Change
            if pct_change >= 3:
                heat = "💚"
            elif pct_change >= 1:
                heat = "🟡"
            elif pct_change <= -3:
                heat = "❤️"
            elif pct_change <= -1:
                heat = "🟠"
            else:
                heat = "⚪"
            print(f"🔥 {sym} {heat} — Δ {pct_change:.2f}%")
            prices_now[sym] = last_price
            if last_price > PRICE_CAP:
                print(f"💸 {sym} skipped (price ${last_price:.2f} > ${PRICE_CAP})")
                continue
            parsed_data.append({
                'symbol': sym,
                'last_price': last_price,
                'previous_close': previous_close,
                'percent_change': round(pct_change, 2),
                'bid': bid,
                'ask': ask,
                'spread': round(spread, 4),
            })
        except:
            continue

    sentiment_cache = {}
    
    # 🌀 Estimate market volatility across all fetched tickers
    market_vols = []
    for sym in prices_now:
        try:
            hist = rs.stocks.get_stock_historicals(sym, interval='5minute', span='day', bounds='regular')
            close_prices = [
                float(h['close_price'])
                for h in hist
                if h.get('close_price') and float(h['close_price']) > 0
            ]
            if len(close_prices) >= 6:
                market_vols.append(np.std(close_prices[-10:]))
        except:
            continue
    
    if market_vols:
        avg_market_vol = np.mean(market_vols)
        print(f"\n🌐 Market Volatility Avg: {avg_market_vol:.2f}")

    quote_data_by_symbol = {row['symbol']: row for row in parsed_data}

    print(f"📡 Fetch time: {time.time() - t_fetch:.2f}s")

    t_analysis = time.time()

    for sym in prices_now:
        try:
            hist = rs.stocks.get_stock_historicals(sym, interval='5minute', span='day', bounds='regular')
            close_prices = [
                float(h['close_price'])
                for h in hist
                if h.get('close_price') and float(h['close_price']) > 0
            ]
            if len(close_prices) < 6:
                continue
        except Exception as e:
            print(f"⚠️ Historical fetch failed for {sym}: {e}")
            continue

        if ENABLE_BACKTEST:
            for name, meta in STRATEGIES.items():
                run_backtest(sym, meta["func"], close_prices)
            continue  # skip rest of live logic for backtest run
    
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * 100  # % format
        current = close_prices[-1]
        best_confidence = 0
        best_strat = None
        best_signal = None
        strategy_confidence = {}
        weighted_votes = []
        
        for name, meta in STRATEGIES.items():
            signal = meta["func"](close_prices)
            if not signal:
                continue
        
            conf = strategy_strength.get(name, 1.0)
            weight = meta.get("weight", 1.0)
            base_score = conf * weight
        
            # Sentiment adjustment
            if USE_SENTIMENT and sym not in sentiment_cache:
                headlines = scrape_finviz(sym)
                sentiment, _, summary = get_combined_sentiment(sym, headlines)
                sentiment_cache[sym] = (sentiment, summary)
                print(f"\n🧠 {sym} Summary: {summary}")
                face = show_sentiment_face(sentiment)
                print(f"💬 {sym} Sentiment: {sentiment.upper()} {face}")
            elif USE_SENTIMENT:
                sentiment, summary = sentiment_cache[sym]
        
                if sentiment == "positive":
                    base_score *= (1 + SENTIMENT_THRESHOLD)
                elif sentiment == "negative":
                    base_score *= (1 - SENTIMENT_THRESHOLD)
        
            # Volatility adjustment
            if volatility < 0.2:
                base_score *= 1.1
            elif volatility > 1.0:
                base_score *= 0.9

            # RSI & Bollinger Modifiers
            indicator_mod = indicator_check(sym, close_prices, name)
            conf *= indicator_mod

            # 🎯 Sigmoid-like scaling for volatility comfort
            ideal_midpoint = 0.4 #Lower favors low-vol assets
            scale_factor = 1.3 #Higher sharpens penalty for deviation
            vol_score = max(0.5, min(1.5, 1 - abs(volatility - ideal_midpoint) * scale_factor))
            conf *= vol_score
        
            strategy_confidence[name] = round(base_score, 2)
            weighted_votes.append((name, base_score, signal))
            log_confidence(sym, strategy_confidence)
        
        if not weighted_votes:
            continue

        print(f"📊 Weighted Strategy Votes for {sym}:")
        for strat, score, signal in weighted_votes:
            sig = "📈 BUY" if signal else "📉 SELL"
            bar = "█" * int(score * 10)
            print(f"{heat} - {strat:<15}| {bar:<10} Score: ({score:.2f}) | Signal: {sig}")
            
        # Hybrid vote: pick best strategy that beats its threshold
        winning_vote = max(weighted_votes, key=lambda x: x[1])
        best_strat, best_conf, best_signal = winning_vote
        
        if best_conf < STRATEGIES[best_strat]['threshold']:
            print(f"⚠️ {sym} signal below threshold — skipping.")
            continue

        quote = quote_data_by_symbol.get(sym, {})
        row_data = {
            'symbol': sym,
            'heat': heat,
            'percent_change': quote.get('percent_change', 0),
            'spread': quote.get('spread', 0),
            'bid': quote.get('bid', 0),
            'ask': quote.get('ask', 0),
            'last_price': prices_now[sym],
            'volatility': round(volatility, 4),
        }

        if not live and prices_now[sym] == portfolio.positions.get(sym, {}).get('cost', 0):
            print(f"⏸️ {sym} skipped (unchanged price in paper mode)")
            continue

        if not ml_should_trade(row_data):
            print(f"🤖 Skipping {sym} — ML suggests low probability of success.")
            continue

        log_signal(sym, best_strat, row_data, best_signal)

        strat_meta = STRATEGIES[best_strat]
        conf = strategy_confidence[best_strat]
        threshold = strat_meta['threshold']

        # ⏳ Cooldown enforcement
        cool_key = (sym, best_strat)
        last_used = strategy_cooldowns.get(cool_key)
        if last_used and (cycle_count - last_used) < COOLDOWN_DURATION:
            print(f"🧊 Cooldown: Skipping {sym} on {best_strat} (used {cycle_count - last_used} cycles ago)")
            continue
        
        if conf >= threshold:
            #🧮 Balanced sizing: even base + confidence bias
            base_allocation = 0.25  # % of portfolio cash per trade
            bias_factor = 0.15      # Up to +15% boost for high confidence (1.0 → +15%)
            # Apply slight confidence bias (confidence is expected in range [0, 1])
            conf_boost = 1.0 + (bias_factor * min(conf, 1.0))  # capped at +15%
            max_allocation = 0.40  # Absolute cap on allocation per trade
            budget = min(portfolio.cash * base_allocation * conf_boost, portfolio.cash * max_allocation)
            if current <= 0:
                print(f"⚠️ Skipping {sym} — invalid price ({current})")
                continue
            qty = max(1, math.floor(budget / current))  # Always buys at least one if it can

            if qty > 0:
                print(f"🪙 Budget: ${budget:.2f} for {sym} @ ${current:.2f} → qty: {qty}")
                #print(f"🟢 {heat} {sym} → {strat_meta['label']} | conf: {conf:.2f} | buy {qty} @ {current:.2f}")
                print("/" * 40)
                if live:
                    rs.orders.order_buy_market(sym, qty)
                else:
                    if portfolio.buy(sym, current, qty, best_strat):
                        position_age[sym] = 0
                        pending_signals.append({
                            'timestamp': datetime.now(),
                            'symbol': sym,
                            'price': prices_now[sym],
                            'signal': 'buy' if best_signal else 'sell',
                            'strategy': best_strat
                        })
                        strategy_cooldowns[(sym, best_strat)] = cycle_count

    print(f"🧠 Analysis time: {time.time() - t_analysis:.2f}s")

    t_exec = time.time()

    # 🔁 Position Aging & Stop-Loss Logic
    for sym in list(portfolio.positions.keys()):
        position_age[sym] = position_age.get(sym, 0) + 1
        current = get_price(sym)
        peak = portfolio.positions[sym]['peak']
        portfolio.positions[sym]['peak'] = max(current, peak)

        if trailing_stop_triggered(sym, current, peak):
            print(f"🔻 TRAILING STOP triggered: {sym} @ {current:.2f}")
            if not live:
                pnl = portfolio.sell(sym, current)
                if pnl is not None:
                    pending_signals.append({
                        'timestamp': datetime.now(),
                        'symbol': sym,
                        'price': current,
                        'signal': 'sell'
                    })
                position_age.pop(sym, None)
            continue

        if position_age[sym] >= STALE_LIMIT:
            print(f"⚠️ Stale position auto-sell: {sym} after {STALE_LIMIT} cycles")
            if not live:
                pnl = portfolio.sell(sym, current)
                if pnl is not None:
                    pending_signals.append({
                        'timestamp': datetime.now(),
                        'symbol': sym,
                        'price': current,
                        'signal': 'sell'
                    })
                position_age.pop(sym, None)

    print(f"📦 Execution time: {time.time() - t_exec:.2f}s")

    # 🧠 Strategy Feedback Loop
    for sym in list(portfolio.history)[-10:]:
        if "SELL" in sym[1]:
            parts = sym[1].split("→ PnL:")
            if len(parts) == 2:
                pnl = float(parts[1])
                for strat, stats in portfolio.strategy_stats.items():
                    if strat in sym[1]:
                        stats["wins" if pnl > 0 else "losses"] += 1

    for strat, stats in portfolio.strategy_stats.items():
        total = stats["total"]
        wins = stats["wins"]
        losses = stats["losses"]
        if total > 3:
            delta = (wins - losses) / total
            strategy_strength[strat] = max(0.5, min(1.5, 1.0 + delta))

    # 📊 Evaluation & Visualization
    evaluate_pending_signals(prices_now)
    portfolio.summary(prices_now)
    show_pnl_stats()
    
    # 🧠 Incremental ML update with fresh data
    try:
        recent_rows = []
        with open(SIGNAL_LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reversed(list(reader)):
                if row.get("result") in ["win", "loss"]:
                    recent_rows.append(row)
                if len(recent_rows) >= 20:
                    break
    
        if len(recent_rows) >= 5:
            df = pd.DataFrame(recent_rows)
            y = df['result'].apply(lambda r: 1 if r == 'win' else 0)
    
            if len(set(y)) < 2:
                print("⏸️ Skipping incremental ML update — only one class present.")
            else:
                X = df[['volatility', 'price', 'hour']].astype(float)
                model = LogisticRegression()
                model.fit(X, y)
                print(f"🧠 Incremental ML model updated with {len(df)} recent signals.")
    except Exception as e:
        print(f"⚠️ Incremental ML update failed: {e}")

    # 🧠 Auto-text summary of ML performance
    review_confidence_and_pnl()

    # 💤 Sleep until next 5-minute candle
    duration = time.time() - cycle_start
    timestamp = datetime.now()
    next_five = next_five_min_mark(timestamp)
    wait_time = (next_five - timestamp).total_seconds()
    print(f"🕯️ Cycle {cycle_count} complete, duration: {duration:.2f}s - waiting {int(wait_time)}s for next candle...")
    time.sleep(wait_time)
