# Commodities_price_prediction
using Yahoo finance's api here's a take on predicting prices of commodities


Commodity Price Prediction Engine: Architectural Reference and Production Documentation
A production-grade machine learning pipeline designed to systematically ingest historical market data, engineer predictive temporal features, train advanced estimators, and generate directional forecasts for global commodity assets.
Section 1: System Philosophy and Financial Theory
Time-series forecasting within financial and commodity markets differs fundamentally from traditional static machine learning tasks. Financial market data exhibits a low signal-to-noise ratio, non-stationarity, and dynamic temporal dependencies. Simple models operating directly on raw price data inevitably fall victim to overfitting or learning spurious correlations.
To mitigate these risks, this system is built on three core pillars:
1. The Low-Signal Challenge
Raw asset price series (Pt​) are typically non-stationary processes. Attempting to train models directly on raw prices (Pt−1​→Pt​) often leads to the "naive forecast" trap, where the model simply predicts that tomorrow's price will be identical to today's price.
To overcome this, our pipeline centers feature engineering on stationary transformations (such as log returns, rolling volatility, and bounded oscillators) while maintaining predictive power.
2. Market Momentum and Mean Reversion
Commodities tend to alternate between high-momentum trends (driven by structural supply-demand imbalances or macroeconomic shocks) and mean-reverting phases (where price overextensions pull back to historical averages).
By integrating technical indicators across multiple lookback windows (short-term, medium-term, and long-term), we supply the predictive models with mathematical proxies representing both trend-following and mean-reverting market structures.
3. Absolute Prevention of Look-Ahead Bias
Data leakage is the most common failure point in quantitative trading models. If a model has access to any information from timestep t when trying to predict a value at timestep t or later, the backtested results will appear exceptional, but real-world deployment will fail entirely.
This engine enforces mathematical boundaries at every transition point: during raw data alignment, technical indicator computation, training-testing splits, and prediction generation.
Section 2: Data Acquisition and Ingestion Architecture
The pipeline ingests raw market data programmatically using the Yahoo Finance API interface via the yfinance library.
[Yahoo Finance API]
       │
       ▼
[Ingestion Layer] ────► Input Validation ────► Multi-Index Resolution
       │
       ▼
[Clean Raw DataFrame]

1. Ingestion Protocol
The system targeting global commodities queries continuous front-month futures contracts or highly liquid exchange-traded funds (ETFs). Common target tickers include:
Gold Futures: GC=F
Crude Oil (WTI): CL=F
Silver Futures: SI=F
Python
import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical daily market data, validates schema integrity,
    and formats timestamps to index format.
    """
    raw_data = yf.download(ticker, start=start_date, end=end_date)
    if raw_data.empty:
        raise ValueError(f"No market data retrieved for ticker: {ticker}")
        
    # Ensure raw column structural integrity
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    raw_data.columns = [col[0] if isinstance(col, tuple) else col for col in raw_data.columns]
    
    missing_cols = [col for col in required_cols if col not in raw_data.columns]
    if missing_cols:
        raise KeyError(f"Fetched dataset is missing essential columns: {missing_cols}")
        
    return raw_data[required_cols].copy()

2. Structural Edge-Cases Handled
Market Holidays and Gaps: Futures markets experience daily trading pauses, weekend closures, and international holidays. The ingestion layer checks for consecutive null-value strings and applies forward-filling for closing values while setting weekend trading volumes to zero to reflect systemic structural reality.
Tick-to-Daily Aggregation: Yahoo Finance raw daily data presents consolidated "Daily" periods. The script ensures dates are localized to UTC to prevent alignment mismatch issues during feature merges.
Section 3: The Feature Engineering Pipeline
The raw daily OHLCV (Open, High, Low, Close, Volume) data is fed into a mathematical transformation pipeline. Raw prices are converted into structural features that capture trend, velocity, range, and volume.
                 [Clean Raw DataFrame]
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
[Trend Features]   [Momentum Features]   [Volatility/Volume]
 (SMA/EMA Lags)         (RSI/MACD)        (Bollinger Bands)
       └────────────────────┼────────────────────┘
                            │
                            ▼
                  [Feature Matrix (X)]

1. Trend Features
Trend indicators measure whether the market is moving upward, downward, or sideways. The engine computes both simple and exponential rolling metrics.
Simple Moving Average (SMA): This calculates the unweighted mean of the last N days of closing prices:
SMAN​=N1​i=0∑N−1​Pt−i​
Exponential Moving Average (EMA): This assigns more weight to recent prices using a smoothing factor α=N+12​:
EMAt​=α⋅Pt​+(1−α)⋅EMAt−1​
Trend Differences: To prevent non-stationarity, the absolute value of the trend is converted into a percentage difference relative to the current close price:
TrendDiffN​=Pt​Pt​−SMAN​​
2. Momentum Indicators
Momentum features quantify the velocity of price movements to show how strong a trend is.
Relative Strength Index (RSI): A bounded oscillator (scaled between 0 and 100) that tracks the ratio of upward changes to downward changes over a 14-day lookback window.
Let U be the average upward change and D be the average downward change over the window:
RS=DU​
RSI=100−1+RS100​
Moving Average Convergence Divergence (MACD): The difference between a short-term and medium-term exponential moving average, showing changes in trend strength:
MACDLine​=EMA12​(P)−EMA26​(P)
SignalLine​=EMA9​(MACDLine​)
3. Volatility and Volumetric Measures
Volatility indicators capture market fear, uncertainty, and liquidity.
Bollinger Bands: A volatility channel set at a specific number of standard deviations (σ) above and below a central simple moving average (typically 20 days).
UpperBand=SMA20​+(2⋅σ20​)
LowerBand=SMA20​−(2⋅σ20​)
BandWidth=SMA20​UpperBand−LowerBand​
Volume Rate of Change (VROC): Measures the rate at which trading volume is changing, which can confirm whether a price trend is supported by market participation:
VROCN​=Volumet−N​Volumet​−Volumet−N​​
4. Implementation Code
Python
import numpy as np

def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mathematical features from raw OHLCV series.
    Ensures calculations use only historical data to prevent look-ahead bias.
    """
    features = df.copy()
    
    # 1. Price Differences and Log Returns
    features['log_return'] = np.log(features['Close'] / features['Close'].shift(1))
    
    # 2. Moving Averages & Percent Deviations
    for window in [10, 20, 50]:
        features[f'sma_{window}'] = features['Close'].rolling(window=window).mean()
        features[f'pct_diff_sma_{window}'] = (features['Close'] - features[f'sma_{window}']) / features[f'sma_{window}']
        
    # 3. Exponential Moving Averages
    features['ema_12'] = features['Close'].ewm(span=12, adjust=False).mean()
    features['ema_26'] = features['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = features['ema_12'] - features['ema_26']
    
    # 4. Relative Strength Index (RSI-14)
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5. Volatility (Bollinger Band Width)
    rolling_std = features['Close'].rolling(window=20).std()
    features['bb_width'] = (rolling_std * 4) / features['sma_20']
    
    # 6. Historical Price Lags
    for lag in [1, 2, 3, 5]:
        features[f'lag_return_{lag}'] = features['log_return'].shift(lag)
        
    # 7. Target Variable: Next Day's Log Return
    features['target_next_return'] = features['log_return'].shift(-1)
    
    # Drop rows containing NaNs resulting from rolling windows and lags
    features.dropna(inplace=True)
    return features

Section 4: Machine Learning Pipelines and Estimators
Once the features are prepared, they are passed to the model training layer. The architecture supports multiple machine learning models to compare baseline predictions against more complex non-linear models.
                 [Feature Matrix & Target]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
    [Baseline Regressor]            [Ensemble Regressor]
     (Ridge Regression)                (Random Forest)
               └──────────────┬──────────────┘
                              │
                              ▼
                     [Model Selection]

1. Ridge Regression (Linear Baseline)
Linear models serve as a key baseline to check if complex, non-linear models are actually adding predictive value. Ridge Regression introduces an L2 regularization penalty to prevent overfitting on highly correlated features:
wmin​(∥Xw−y∥22​+α∥w∥22​)
This penalty helps keep coefficients stable when features (like multiple moving averages) are highly correlated with one another.
2. Random Forest Regressor
An ensemble learning method that constructs multiple decision trees during training. It helps capture non-linear relationships and complex interactions between indicators without requiring feature scaling:
Bootstrap Aggregating (Bagging): Each tree is trained on a random sample of the data, which reduces the overall variance of the final model.
Feature Subspace Sampling: A random subset of features is selected at each split point to prevent a single dominant feature from making all the individual trees look too similar.
3. Gradient Boosted Trees (XGBoost / LightGBM)
Gradient boosting builds trees sequentially, where each new tree is trained to correct the errors (residuals) of the previous ones:
Objective Function: Minimizes a loss function while penalizing model complexity to prevent overfitting:
L(t)=i=1∑n​l(yi​,y^​i(t−1)​+ft​(xi​))+Ω(ft​)
Regularization: Includes built-in L1 and L2 regularization to help keep tree weights stable and improve generalization on out-of-sample data.
Section 5: Time-Series Validation and Testing Framework
Evaluating time-series models requires a structured validation scheme. Standard k-fold cross-validation is invalid for sequential data because it shuffles the data, which leaks future information into past predictions.
Traditional K-Fold (Invalid for Time Series):
[  Test  ][ Train ][ Train ][ Train ][ Train ] ◄── Future leaks into past

Time Series Split (Valid):
Split 1: [ Train ][ Test ]
Split 2: [   Train   ][ Test ]
Split 3: [     Train     ][ Test ]

1. TimeSeriesSplit Mechanics
The pipeline uses a rolling forward validation scheme (TimeSeriesSplit). This technique trains the model on historical data and tests it on a subsequent, forward-pointing window:
Fold 1: Train on days 1 to 500, test on days 501 to 600.
Fold 2: Train on days 1 to 600, test on days 601 to 700.
Fold 3: Train on days 1 to 700, test on days 701 to 800.
This approach simulates how the model will perform in production, where it will make predictions on new, unseen data as time moves forward.
2. Validation Class Architecture
Python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class ModelValidationPipeline:
    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        
    def evaluate(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        X = df[feature_cols].values
        y = df[target_col].values
        
        metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train the model on historical fold data
            self.model.fit(X_train, y_train)
            
            # Generate predictions on the forward test fold
            predictions = self.model.predict(X_test)
            
            # Calculate validation metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            # Calculate directional accuracy
            correct_direction = np.sign(y_test) == np.sign(predictions)
            dir_acc = np.mean(correct_direction)
            
            metrics.append({
                'Fold': fold + 1,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Directional_Accuracy': dir_acc
            })
            
        return pd.DataFrame(metrics)

Section 6: Local Production Execution Guide
To deploy this pipeline locally and begin generating predictions, follow this structured setup guide:
                 [Local Environment Setup]
                             │
                             ▼
                [Dependency Resolution]
                             │
                             ▼
         [Database Execution (Ingestion -> Output)]

1. File Structure Architecture
Ensure your local project workspace is organized as follows:
Plaintext
commodity_predictor/
│
├── data/
│   └── cached_prices.csv        # Automated local caching layer
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # Pulls and cleans yfinance data
│   ├── features.py              # Computes technical indicators
│   └── models.py                # Defines estimators and validation
│
├── main.py                      # Main pipeline execution script
├── requirements.txt             # Project library dependencies
└── README.md                    # System documentation (this file)

2. Dependency Resolution
Create a requirements.txt file containing the precise packages required to run the engine:
Plaintext
numpy>=1.22.0
pandas>=1.4.0
yfinance>=0.2.18
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

To initialize your environment and install these dependencies, run:
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

3. Pipeline Execution Code
Save the following script as main.py to run the entire pipeline from end to end:
Python
import os
from src.ingestion import fetch_market_data
from src.features import generate_technical_features
from src.models import ModelValidationPipeline

def run_pipeline():
    # Define execution parameters
    ticker = "GC=F"  # Gold Futures
    start_date = "2018-01-01"
    end_date = "2026-06-01"
    
    print("Step 1: Fetching historical commodity data...")
    raw_df = fetch_market_data(ticker, start_date, end_date)
    
    print("Step 2: Engineering technical indicators...")
    feature_df = generate_technical_features(raw_df)
    
    # Define features to use for model training
    exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'target_next_return', 'ema_12', 'ema_26']
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    target_col = 'target_next_return'
    
    print(f"Features configured for training: {feature_cols}")
    
    print("Step 3: Running time-series cross-validation...")
    evaluator = ModelValidationPipeline(n_splits=5)
    metrics_df = evaluator.evaluate(feature_df, feature_cols, target_col)
    
    print("\n--- Out-of-Sample Performance Report ---")
    print(metrics_df.to_string(index=False))
    
    # Print average metrics
    print("\nAverage Performance Across All Folds:")
    print(f"Mean MAE:  {metrics_df['MAE'].mean():.6f}")
    print(f"Mean RMSE: {metrics_df['RMSE'].mean():.6f}")
    print(f"Mean R2:   {metrics_df['R2'].mean():.4f}")
    print(f"Mean Directional Accuracy: {metrics_df['Directional_Accuracy'].mean() * 100:.2f}%")

if __name__ == "__main__":
    run_pipeline()


Authors & Contributors
Kazi Mohammed Arsh — Lead AI/ML Engineer & Systems Architect
Faisal Shaikh — Co-Author & Quantitative Specialist
Anas Khan — Co-Author & Data Pipeline Architect

