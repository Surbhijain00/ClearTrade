# ClearTrade - A Stock Predictor

ClearTrend is a Streamlit-based application that predicts future stock prices using a multi-model ensemble. It combines deep learning, machine learning, and statistical models with advanced technical indicators to provide accurate and actionable forecasts.

<br />

## Features

- **Multi-Model Ensemble:**

Combines Bidirectional LSTM, ARIMA, SVR, Random Forest, XGBoost, Gradient Boosting, and KNN to improve prediction accuracy.

- **Advanced Technical Indicators:**

Calculates Moving Averages, RSI, Bollinger Bands, to capture market trends and momentum.

- **Interactive UI:**
  
A simple Streamlit dashboard to input stock symbols, adjust the date range, and select weight configurations for ensemble predictions.

- **Data Integration:**

Retrieves historical data via Yahoo Finance (yfinance) and fetches news headlines using NewsAPI for additional market context.

- **Optimized Performance:**

Uses caching to speed up data retrieval and smooth out the user experience.

<br />

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Surbhijain00/ClearTrade.git
   cd ClearTrade
   ```

2. **Create and Activate a Virtual Environment**
  
  ```bash
    python -m venv venv
    venv\Scripts\activate    # On Windows:
    source venv/bin/activate    # On macOS/Linux:
   ```

3. **Install the Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

<br />

## Usage

1. **Configure API Keys:**
   
Create a file at ```.streamlit/secrets.toml``` and add:
```
[secrets]
NEWS_API_KEY = "your_news_api_key_here"
```

2. **Run the app:**
```bash
streamlit run stock_prediction_app.py
```

3. **Interact with the app:**
   
Enter a stock symbol (e.g., AAPL), choose the date range, select a weight configuration, and click "Generate Predictions" to view forecasts and visualizations.

<br />

## Configuration

ClearTrend supports multiple weight configurations to adjust the ensemble of predictive models:

- **Default:** Balanced weighting for all models.
- **Trend-Focused:** Optimized for growth and technology stocks.
- **Statistical:** Suited for stable, blue-chip stocks.
- **Tree-Ensemble:** Emphasizes tree-based models to capture complex market relationships.
- **Balanced:** A general-purpose configuration for varied stock characteristics.
- **Volatility-Focused:** Prioritizes models better suited for volatile and emerging market stocks.

<br />

## Technical Details

- **Backend:** Python, TensorFlow (Keras), scikit-learn, XGBoost, and statsmodels.
- **Data Processing:** Uses MinMaxScaler and sequence preparation for LSTM input.
- **Visualization:** Integrates Matplotlib charts within the Streamlit UI.
- **APIs:** Uses yfinance for historical stock data and NewsAPI for current news headlines.
