# -*- coding: utf-8 -*-

"""
This script progresses through the typical steps involved with creating an
automated / quantitative trading algorithm.  It implements a backtesting engine
using the Kelly Criterion for dynamic risk management.

@author: Mike Muryn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from trading_utilites import get_close_data, generate_features, generate_signals, plot_results

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the symbol universe
symbol_universe = ["SPY", "QQQ", "IWM", "EEM", "EFA", "VXX",
                   "SHY", "IEF", "TLT", "JAAA", "BND",
                   "USO", "UNG", "GLD", "SLV",
                   "UUP", "BITO"]

# Define the testing period
start_date = '2018-01-01'
end_date = '2023-01-01'

# Initialize a dictionary to store backtest results for each symbol
backtest_results = {}

# Kelly Criterion parameters
win_probability = 0.65  # Estimated win rate of the strategy
avg_win = 0.05  # Estimated average win percentage
avg_loss = 0.02  # Estimated average loss percentage
reward_to_risk_ratio = avg_win / avg_loss

# Calculate the Kelly fraction
kelly_fraction = (win_probability * (reward_to_risk_ratio + 1) - 1) / reward_to_risk_ratio
kelly_fraction = max(0, min(kelly_fraction, 1))  # Ensure it's between 0 and 1

# Apply a cap of 20% of the total capital
kelly_fraction = min(kelly_fraction, 0.2)

# Loop through each symbol in the universe
for symbol in symbol_universe:
    logger.info(f"Running backtest for {symbol}...")

    try:
        # Step 1: Get Historical Data
        data = get_close_data(symbol, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"Data for {symbol} is missing or incomplete. Skipping.")
            continue

        # Handle missing data
        data.ffill(inplace=True)
        if data.isnull().values.any():
            logger.warning(f"Data for {symbol} still contains missing values after forward fill. Attempting backward fill.")
            data.bfill(inplace=True)

        missing_percentage = data.isnull().mean().mean() * 100
        if data.isnull().values.any():
            logger.warning(f"Data for {symbol} still contains missing values after forward and backward fill. Missing Percentage: {missing_percentage:.2f}%")
            if missing_percentage > 5:
                logger.warning(f"Too much missing data ({missing_percentage:.2f}%) for {symbol}. Skipping.")
                continue

        # Step 2: Generate Features
        features = generate_features(data)
        if features.isnull().values.any():
            logger.info(f"Features for {symbol} contain missing values. Filling missing values with median.")
            features.fillna(features.median(), inplace=True)

        # Step 3: Generate Trading Signals
        X = features.drop(columns=['price'])
        y = (features['price'].shift(-5) > features['price']).astype(int)  # 1 if price increases in next 5 days, else 0

        # Handle potential infinite or NaN values before scaling
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features to handle large values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train multiple models and handle errors gracefully
        models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000)),
            ('Decision Tree', DecisionTreeClassifier(random_state=42))
        ]
        best_model = None
        best_accuracy = 0

        for model_name, model in models:
            try:
                logger.info(f"Training {model_name} for {symbol}...")
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test))
                logger.info(f"{model_name} accuracy for {symbol}: {accuracy * 100:.2f}%")

                # Perform cross-validation to prevent overfitting
                cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
                logger.info(f"{model_name} cross-validation score for {symbol}: {np.mean(cross_val_scores) * 100:.2f}%")

                if accuracy > best_accuracy:
                    best_model = model
                    best_accuracy = accuracy
            except Exception as e:
                logger.error(f"Error training {model_name} for {symbol}: {e}")

        if best_model is None:
            logger.warning(f"No valid model trained for {symbol}. Skipping.")
            continue

        X_scaled = scaler.transform(X)
        signals = pd.DataFrame(index=features.index)
        signals['price'] = features['price']
        signals['predicted_signal'] = best_model.predict(X_scaled)
        signals['position'] = signals['predicted_signal'].diff().fillna(0)  # Track when positions change

        if signals.empty:
            logger.warning(f"No valid signals generated for {symbol}. Skipping.")
            continue

        # Step 4: Backtest the Strategy using Kelly Criterion
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['signals'] = signals['predicted_signal']

        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['price']
        portfolio['holdings'] = 0.0
        portfolio['cash'] = 100000.0
        portfolio['total'] = 100000.0
        portfolio['returns'] = 0.0
        portfolio['reward'] = 0.0  # Initialize reward column

        # Backtesting loop
        for i in range(1, len(signals)):
            today = signals.index[i]
            predicted_signal = signals.loc[today, 'predicted_signal']
            price = signals.loc[today, 'price']

            # If signal is 1 (buy) and cash is available
            if predicted_signal == 1 and portfolio['cash'].iloc[i - 1] > 0:
                invest_amount = kelly_fraction * portfolio['cash'].iloc[i - 1]
                portfolio.loc[today, 'holdings'] = portfolio['holdings'].iloc[i - 1] + (invest_amount / price)
                portfolio.loc[today, 'cash'] = portfolio['cash'].iloc[i - 1] - invest_amount
            else:
                portfolio.loc[today, 'holdings'] = portfolio['holdings'].iloc[i - 1]
                portfolio.loc[today, 'cash'] = portfolio['cash'].iloc[i - 1]

            # Calculate portfolio value and returns
            portfolio.loc[today, 'total'] = portfolio.loc[today, 'cash'] + portfolio.loc[today, 'holdings'] * price
            portfolio.loc[today, 'returns'] = (portfolio['total'].iloc[i] - portfolio['total'].iloc[i - 1]) / portfolio['total'].iloc[i - 1]

            # Calculate reward (positive if returns are greater than zero)
            portfolio.loc[today, 'reward'] = 1 if portfolio['returns'].iloc[i] > 0 else -1

        # Store the portfolio results in the dictionary
        backtest_results[symbol] = portfolio

        # Step 5: Plot the Results
        logger.info(f"Plotting results for {symbol}...")
        plot_results(signals, portfolio, features)

    except Exception as e:
        logger.error(f"Unexpected error occurred while processing {symbol}: {e}")
        continue

# Step 5: Summarize the backtest results
summary = []
for symbol, portfolio in backtest_results.items():
    if not portfolio.empty:
        final_value = portfolio['total'].iloc[-1]
        total_return = (final_value - 100000) / 100000 * 100
        summary.append({"Symbol": symbol, "Final Value": final_value, "Total Return (%)": total_return})

summary_df = pd.DataFrame(summary)
logger.info("Backtest Summary:")
logger.info(summary_df)

# Plot summary of total returns if there are any results to plot
if not summary_df.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Symbol'], summary_df['Total Return (%)'].fillna(0), color='blue')
    plt.xlabel('Symbol')
    plt.ylabel('Total Return (%)')
    plt.title('Total Returns for Symbol Universe')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    logger.warning("No valid backtest results to plot.")
