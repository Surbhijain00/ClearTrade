import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, lower_band

def calculate_stochastic(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return k

def calculate_williams_r(df, period=14):
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    return -100 * ((high_max - df['Close']) / (high_max - low_min))

def calculate_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    df['ATR'] = calculate_atr(df)
    df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MOM'] = df['Close'].diff(10)
    df['STOCH_K'] = calculate_stochastic(df)
    df['WILLR'] = calculate_williams_r(df)
    return df.dropna()

def calculate_technical_indicators_for_summary(df):
    analysis_df = df.copy()
    analysis_df['MA20'] = analysis_df['Close'].rolling(window=20).mean()
    analysis_df['MA50'] = analysis_df['Close'].rolling(window=50).mean()
    delta = analysis_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    analysis_df['RSI'] = 100 - (100 / (1 + rs))
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=20).mean()
    ma20 = analysis_df['Close'].rolling(window=20).mean()
    std20 = analysis_df['Close'].rolling(window=20).std()
    analysis_df['BB_upper'] = ma20 + (std20 * 2)
    analysis_df['BB_lower'] = ma20 - (std20 * 2)
    analysis_df['BB_middle'] = ma20
    return analysis_df