# Usage Guide

This document provides detailed instructions on how to set up your environment and run the trading bots in this collection.

## 1. Setup

Follow these steps to get the bots running.

### Dependencies

First, install all the required Python packages using the `requirements.txt` file. It is recommended to do this within a virtual environment.

```bash
pip install -r requirements.txt
```

### Environment Variables

The bots require API keys and credentials to function. These are managed using a `.env` file to keep them secure and out of the source code.

1.  Create a new file named `.env` in the root directory of the project.
2.  Add the following key-value pairs to the file, replacing the placeholder values with your actual credentials:

```
# Your Robinhood Credentials
ROBINHOOD_USERNAME="your_email@example.com"
ROBINHOOD_PASSWORD="your_robinhood_password"

# Finnhub API Key (for scalper bots and faster data)
# Get a free key from https://finnhub.io/
FINNHUB_API_KEY="your_finnhub_api_key"
```

**Note**: Not all scripts use all keys. However, providing all of them will ensure any bot can be run without configuration errors.

## 2. Bot Descriptions

Below is a breakdown of each bot script, its purpose, and how to run it.

---

### `main.py` - The All-in-One Bot

This is the most advanced and feature-complete script in the collection. It is designed as a powerful, all-in-one trading bot.

*   **Purpose**: To trade based on a hybrid model that combines multiple technical analysis strategies, machine learning predictions, and real-time sentiment analysis from news headlines.
*   **Features**:
    *   **Multiple Strategies**: Uses Mean Reversion, Momentum, Breakout, and other strategies simultaneously.
    *   **Sentiment Analysis**: Scrapes news from Finviz and uses both the VADER library and a BART transformer model to summarize and analyze sentiment.
    *   **Machine Learning**: Trains a `LogisticRegression` model on past trade signals to gate new trades, only executing those the model predicts will be successful.
    *   **Backtesting Engine**: Can run strategies against historical data to evaluate performance.
    *   **Extensive Logging**: Logs all trade signals and outcomes to `signal_log.csv` for analysis and ML model training.
    *   **Dynamic Ticker Selection**: Can use a static list of tickers or dynamically fetch popular/top-moving stocks.
*   **How to Run**:
    ```bash
    python main.py
    ```
    The script will prompt you with several questions at startup, such as whether to run in LIVE mode, enable backtesting, or use sentiment analysis.

---

### `scalperbot.py` - Object-Oriented Scalper

This script is a well-structured, object-oriented framework for building and running trading strategies. It is designed for extensibility.

*   **Purpose**: To provide a robust framework for trading that separates concerns like data fetching, strategy logic, and portfolio management into different classes.
*   **Features**:
    *   **Object-Oriented Design**: Uses classes for `Portfolio`, `DataSource`, and `Strategy`.
    *   **Multiple Data Sources**: Includes a `GlobalDataFetcher` that can pull data from Finnhub and fall back to `yfinance` if needed.
    *   **Complex Strategies**: Implements advanced strategies that calculate RSI, MACD, and VWAP.
    *   **Extensible**: New strategies can be easily added by inheriting from the `BaseStrategy` abstract class.
*   **How to Run**:
    ```bash
    python scalperbot.py
    ```

---

### `robin_scalper.py` / `scalp.py` - Scalping Scripts

These scripts are focused on high-frequency, short-term "scalping" strategies.

*   **Purpose**: To execute trades on very short timeframes, taking advantage of small price movements.
*   **Features**:
    *   **Scalper Mode**: Uses the Finnhub API for faster price updates than the standard Robinhood API.
    *   **ML Gating**: Uses a simple `LogisticRegression` model to approve or deny trades.
    *   **Fast Cycle Time**: Runs on a much shorter loop (e.g., 15 seconds) when in scalper mode.
*   **How to Run**:
    ```bash
    python robin_scalper.py
    ```
    You will be prompted to enable "Scalper Mode" and "LIVE mode".

---

### `robinbot_cyclic.py` - LLM-Powered Bot

This bot integrates a small Large Language Model (LLM) to enhance its sentiment analysis capabilities.

*   **Purpose**: To use an LLM to summarize news headlines for more nuanced sentiment analysis.
*   **Features**:
    *   **LLM Integration**: Uses the `T5-small` model from Hugging Face Transformers to summarize news.
    *   **Sentiment Keywords**: Scores the LLM's summary based on a list of positive and negative keywords.
    *   **ML Integration**: Also includes the `LogisticRegression` model for trade gating.
*   **How to Run**:
    ```bash
    python robinbot_cyclic.py
    ```
    The first time you run this, it may take a few minutes to download the T5-small model from Hugging Face.

---

### `robinbot-ml.py` - ML-Focused Bot

This script is a direct evolution of `goodbot.py`, adding a machine learning layer.

*   **Purpose**: To introduce a machine learning model that predicts the success of a trade signal before execution.
*   **Features**:
    *   **ML Model**: Trains a `LogisticRegression` model from `signal_log.csv`.
    *   **Signal Logging**: Collects data on trades to continuously improve the ML model.
    *   **Portfolio Management**: Includes the full `Portfolio` class from `goodbot.py`.
*   **How to Run**:
    ```bash
    python robinbot-ml.py
    ```

---

### `goodbot.py` - Intermediate Multi-Strategy Bot

This script is a great example of an intermediate-level bot with solid features.

*   **Purpose**: To trade using multiple strategies and adjust its own confidence based on their historical performance.
*   **Features**:
    *   **Portfolio Class**: Manages cash, positions, and trade history.
    *   **Strategy Registry**: Runs multiple strategies (Momentum, Mean Reversion, Breakout).
    *   **Feedback Loop**: Tracks the win/loss rate of each strategy and adjusts a "strength" score, giving more weight to successful strategies.
    *   **Guardrails**: Includes trailing stop-losses and logic to sell stale (inactive) positions.
*   **How to Run**:
    ```bash
    python goodbot.py
    ```

---

### `simplebot.py` - The Baseline

This is the most basic script and serves as an excellent starting point.

*   **Purpose**: To fetch a list of stocks, identify the top 5 daily gainers, and perform a simple "walk-forward" analysis on their recent price history.
*   **Features**:
    *   Fetches stocks from a Robinhood "market tag" (e.g., "technology").
    *   Uses `pandas` to sort and filter the stocks.
    *   Does not execute any live or paper trades.
*   **How to Run**:
    ```bash
    python simplebot.py
    ```

---

### `dualbot.py` - Simple Trading Bot

A straightforward bot that can execute both paper and live trades.

*   **Purpose**: To demonstrate a simple trading loop with a live/paper trading option.
*   **Features**:
    *   **Trader Class**: A simple class to manage a portfolio and execute trades.
    *   **Simple MA Strategy**: Generates buy/sell signals based on a moving average crossover.
*   **How to Run**:
    ```bash
    python dualbot.py
    ```
    The script will ask if you want to run in live mode.
