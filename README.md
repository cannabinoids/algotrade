Algorithmic Trading Bot Collection

This repository contains a collection of algorithmic trading bots written in Python. These bots interface with the Robinhood platform (and others, like Finnhub) to execute trades based on a variety of strategies. The scripts range from simple, single-strategy bots to complex, multi-strategy bots that incorporate machine learning and natural language processing for sentiment analysis.

Overview
The bots in this collection demonstrate an evolution in features and complexity:

Simple Bots: Start with basic strategies like identifying top-gaining stocks and running simple analyses.
Intermediate Bots: Introduce concepts like portfolio management, multiple trading strategies, and performance-based feedback loops.
Advanced Bots: Incorporate machine learning to predict trade success, sentiment analysis from financial news, and connections to multiple data APIs for more robust data gathering.
Object-Oriented Frameworks: A more structured approach using classes for strategies, data sources, and portfolio management, allowing for greater extensibility.
Getting Started
To get started with these bots, you will need to set up your environment and configure your API keys.

For detailed instructions on setup, dependencies, and how to run each bot, please refer to the following documents:

USAGE.md: Provides comprehensive setup instructions, a guide to configuring your .env file, and a breakdown of each bot's purpose and command-line execution.
requirements.txt: A list of all the necessary Python packages to run the bots.
Bot Scripts
Each primary bot has its own README file for a brief overview. For a full breakdown, see the USAGE.md file.

main.py: The most feature-rich, all-in-one trading bot.
simplebot.py: A great starting point for understanding the basics.
goodbot.py: An intermediate bot with portfolio tracking and multiple strategies.
scalperbot.py: An advanced, object-oriented bot designed for high-frequency trading strategies.
Disclaimer: Trading stocks and cryptocurrencies involves substantial risk of loss and is not suitable for every investor. All trading strategies are used at your own risk. The author of these scripts is not responsible for any financial losses you may incur.
