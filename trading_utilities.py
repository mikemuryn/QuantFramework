# -*- coding: utf-8 -*-

"""
This script provides core functions for quantitative trading.

----- Potential Areas for improvement ----
Implement reinforcement learning to optimize portfolio allocation dynamically.

Implement bayesian optimization with gaussian processes to fine-tune both
hyperparameters and feature sets.

Utilize multi-objective optimization to simultaneously maximize returns while
minimizing risks.

Integrate GARCH models to forecast volatility and adjust trading thresholds
accordingly.

Use attention mechanisms to allow the model to focus on different parts of the
input sequence.

Modify the reward function to be based on maximizing Sharpe ratio, Sortino
ratio, or simply achieving consistent portfolio growth.

Include risk parity and tail-risk hedging to mitigate large market drawdowns
and maintain consistent returns.

Integrate macroeconomic variables such as interest rates, inflation rates, and
currency movements. Create features from data like Federal Reserve meetings,
employment reports, and global geopolitical news.

Include market microstructure features such as bid-ask spread, order book
dynamics, and liquidity analysis.

Use quandll, proprietary, or alternative data sources.

@author: Mike Muryn

"""

import logging
import inspect
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import talib
from scipy.stats import zscore
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from transformers import pipeline
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_close_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical data for a given symbol within the specified date range.

    Args:
        symbol (str): Symbol (e.g. 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Dataframe containing "Close" data with additional
                        computed columns ('returns', 'volume_change'); Returns
                        None if data download fails or is empty
    """
    logger.debug(f"Entering {inspect.currentframe().f_code.co_name} method.")

    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"{symbol}: no price data found")
        data['returns'] = data['Close'].pct_change()
        data['volume_change'] = data['Volume'].pct_change()
        logger.info(f"Successfully fetched data for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to download data for {symbol}: {e}")
        return None


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical and statistical features from price data.

    Args:
        data (pd.DataFrame): Dataframe containing "Close" data

    Returns:
        pd.DataFrame: Dataframe with generated features including technical
                          indicators and statistical metrics
    """
    logger.debug(f"Entering {inspect.currentframe().f_code.co_name} method.")

    features = pd.DataFrame(index=data.index)
    features['price'] = data['Close']

    # Convert 'Close' column to NumPy array for compatibility with talib
    close_prices = data['Close'].values.ravel()

    # Technical Indicators
    features['RSI'] = talib.RSI(close_prices, timeperiod=14)
    features['MACD'], features['MACD_signal'], _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    features['EMA_10'] = talib.EMA(close_prices, timeperiod=10)
    features['EMA_50'] = talib.EMA(close_prices, timeperiod=50)
    features['BB_upper'], features['BB_middle'], features['BB_lower'] = talib.BBANDS(close_prices, timeperiod=20)

    # Statistical Features
    features['zscore'] = zscore(data['Close'])
    features['volatility'] = data['Close'].rolling(window=20).std()
    features['volume_change'] = data['volume_change']

    # Lagged Returns and Momentum
    features['return_1d'] = data['Close'].pct_change(1)
    features['return_5d'] = data['Close'].pct_change(5)
    features['momentum'] = data['Close'] / data['Close'].shift(10) - 1

    # Custom Feature: Price Oscillator (short-term vs. long-term)
    features['price_oscillator'] = (features['EMA_10'] - features['EMA_50']) / features['EMA_50']

    # Sentiment Analysis
    sentiment_pipeline = pipeline('sentiment-analysis')
    sentiment_scores = []
    for date in features.index:
        # Placeholder for actual news data / sentiment
        sample_text = "Market looks optimistic today."
        sentiment = sentiment_pipeline(sample_text)[0]
        sentiment_scores.append(1 if sentiment['label'] == 'POSITIVE' else -1)
    features['sentiment'] = sentiment_scores

    # Handle NaN values
    features.fillna(0, inplace=True)

    return features


def generate_signals(data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generates trading signals using a hybrid CNN + LSTM model.

    Args:
        data (pd.DataFrame): "Close" data
        features (Optional[pd.DataFrame]): Precomputed feature set; if None,
            features are computed from data

    Returns:
        pd.DataFrame: Dataframe containing trading signals and predicted
            positions
    """
    logger.debug(f"Entering {inspect.currentframe().f_code.co_name} method.")

    if data is None or data.empty:
        logger.warning("Data is missing. Skipping signal generation.")
        return pd.DataFrame()

    # Generate features from data if not provided
    if features is None:
        generate_features(data)

    signals = pd.DataFrame(index=features.index)
    signals['price'] = data['Close']

    # Define the target variable: 1 for Buy, 0 for Hold/Sell
    signals['signal'] = np.where(features['return_5d'].shift(-5) > 0, 1, 0)

    # Prepare data for CNN + LSTM model
    X = features[['RSI', 'MACD', 'MACD_signal', 'EMA_10', 'EMA_50', 'BB_upper',
                  'BB_middle', 'BB_lower', 'zscore', 'volatility', 'return_1d',
                  'momentum', 'price_oscillator', 'volume_change']].copy()
    y = signals['signal']

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Automatically find a valid number of timesteps
    n_features = X_train.shape[1]
    n_timesteps = max(2, min(n_features, 5))
    while n_features % n_timesteps != 0:
        n_timesteps -= 1

    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]

    # Reshape data for CNN + LSTM
    X_train = X_train.reshape((n_samples_train, n_timesteps, n_features // n_timesteps))
    X_test = X_test.reshape((n_samples_test, n_timesteps, n_features // n_timesteps))

    # Hybrid CNN + LSTM Model
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    cnn_layer = Conv1D(filters=64, kernel_size=1, activation='relu')(input_layer)  # Adjust kernel size to avoid negative dimensions
    cnn_layer = MaxPooling1D(pool_size=1)(cnn_layer)  # Adjust pool size to avoid negative dimensions
    cnn_layer = Flatten()(cnn_layer)

    lstm_layer = LSTM(50, activation='relu')(input_layer)

    concat_layer = concatenate([cnn_layer, lstm_layer])
    dropout_layer = Dropout(0.3)(concat_layer)
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(f'CNN + LSTM Model Accuracy: {accuracy * 100:.2f}%')
    logger.info(f'ROC AUC Score: {roc_auc * 100:.2f}%')

    # Predict signals for the dataset
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], n_timesteps, n_features // n_timesteps))
    signals['predicted_signal'] = (model.predict(X_scaled) > 0.5).astype(int)
    signals['position'] = signals['predicted_signal'].diff()

    return signals


def backtest(signals: pd.DataFrame, initial_capital: float = 100000.0, risk_per_trade: float = 0.02) -> pd.DataFrame:
    """
    Conducts backtesting on generated trading signals, including risk adjusted
    metrics and reinforcement learning-like reward system

    Args:
        signals (pd.DataFrame): Dataframe containing trading signals and prices
        initial_capital (float): Starting portfolio capital
        risk_per_trade (float): Percentage of capital risked per trade

    Returns:
        pd.DataFrame: Portfolio performance metrics and rewards
    """
    logger.debug(f"Entering {inspect.currentframe().f_code.co_name} method.")

    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['signals'] = signals['predicted_signal']

    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()
    portfolio['risk'] = risk_per_trade * portfolio['cash']
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['reward'] = np.where(portfolio['returns'] > 0, 1, -1)

    return portfolio


def plot_results(signals: pd.DataFrame, portfolio: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    Plots trading results and key performance metrics.

    Args:
        signals (pd.DataFrame): Dataframe containing trading signals
        portfolio (pd.DataFrame): Portfolio performance metrics
        features (pd.DataFrame): Generated features

    Returns:
        None
    """
    fig, ax = plt.subplots(6, 1, figsize=(16, 24))

    # Plot the price and moving averages
    ax[0].plot(signals.index, signals['price'], label='Price', color='blue')
    ax[0].plot(signals.index, signals['position'], label='Position', color='orange', linestyle='--')
    ax[0].set_title('Price and Position Changes')
    ax[0].legend()

    # Plot the portfolio value
    ax[1].plot(portfolio.index, portfolio['total'], label='Portfolio Value', color='purple')
    ax[1].set_title('Portfolio Value Over Time')
    ax[1].legend()

    # Plot RSI Indicator
    ax[2].plot(signals.index, features['RSI'], label='RSI', color='green')
    ax[2].axhline(70, color='red', linestyle='--', linewidth=1)
    ax[2].axhline(30, color='red', linestyle='--', linewidth=1)
    ax[2].set_title('RSI Over Time')
    ax[2].legend()

    # Plot Volatility
    ax[3].plot(signals.index, features['volatility'], label='Volatility (20-day)', color='magenta')
    ax[3].set_title('Volatility Over Time')
    ax[3].legend()

    # Plot Price Oscillator
    ax[4].plot(signals.index, features['price_oscillator'], label='Price Oscillator', color='brown')
    ax[4].set_title('Price Oscillator Over Time')
    ax[4].legend()

    # Plot Reward System
    ax[5].plot(portfolio.index, portfolio['reward'], label='Reward Signal', color='red')
    ax[5].set_title('Reinforcement Learning-like Reward Signal Over Time')
    ax[5].legend()

    plt.tight_layout()
    plt.show()
