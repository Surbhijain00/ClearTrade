import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_fetcher import fetch_stock_data, get_news_headlines
from src.technical_indicators import calculate_technical_indicators_for_summary
from src.models import MultiAlgorithmStockPredictor

# Page configuration
st.set_page_config(page_title="ClearTrend", layout="wide")
st.markdown("<h1 style='text-align: center;'>ClearTrend - A Stock Predictor</h1><br><br>", unsafe_allow_html=True)

# Hyperparameter tuning UI
def hyperparameter_tuning_ui():
    st.sidebar.subheader("Model Hyperparameters")
    hyperparams = {}
    with st.sidebar.expander("LSTM Parameters"):
        hyperparams['LSTM'] = {
            'units': st.slider("LSTM Units", 50, 200, 100),
            'dropout': st.slider("LSTM Dropout", 0.0, 0.5, 0.2),
            'epochs': st.slider("Training Epochs", 10, 100, 50),
            'batch_size': st.select_slider("Batch Size", [16, 32, 64], 32)
        }
    with st.sidebar.expander("Tree-based Models"):
        hyperparams['RandomForest'] = {
            'n_estimators': st.slider("RF: Number of Trees", 50, 200, 100),
            'max_depth': st.slider("RF: Max Depth", 3, 20, 10)
        }
        hyperparams['XGBoost'] = {
            'n_estimators': st.slider("XGB: Number of Trees", 50, 200, 100),
            'learning_rate': st.slider("XGB: Learning Rate", 0.01, 0.3, 0.1),
            'max_depth': st.slider("XGB: Max Depth", 3, 10, 3)
        }
        hyperparams['GBM'] = {
            'n_estimators': st.slider("GBM: Number of Trees", 50, 200, 100),
            'learning_rate': st.slider("GBM: Learning Rate", 0.01, 0.3, 0.1),
            'max_depth': st.slider("GBM: Max Depth", 3, 10, 3)
        }
    with st.sidebar.expander("Other Models"):
        hyperparams['SVR'] = {
            'C': st.slider("SVR: C Parameter", 1, 1000, 100),
            'epsilon': st.slider("SVR: Epsilon", 0.01, 0.5, 0.1)
        }
        hyperparams['KNN'] = {
            'n_neighbors': st.slider("KNN: Neighbors", 3, 15, 5)
        }
    return hyperparams

# User inputs
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
display_days = st.slider("Select number of days to display", 30, 3650, 180)
hyperparams = hyperparameter_tuning_ui()

# Weight configurations
WEIGHT_CONFIGURATIONS = {
    "Default": {'LSTM': 0.3, 'XGBoost': 0.15, 'Random Forest': 0.15, 'ARIMA': 0.1, 'SVR': 0.1, 'GBM': 0.1, 'KNN': 0.1},
    "Trend-Focused": {'LSTM': 0.35, 'XGBoost': 0.20, 'Random Forest': 0.15, 'ARIMA': 0.10, 'SVR': 0.08, 'GBM': 0.07, 'KNN': 0.05},
    "Statistical": {'LSTM': 0.20, 'XGBoost': 0.15, 'Random Forest': 0.15, 'ARIMA': 0.20, 'SVR': 0.15, 'GBM': 0.10, 'KNN': 0.05},
    "Tree-Ensemble": {'LSTM': 0.25, 'XGBoost': 0.25, 'Random Forest': 0.20, 'ARIMA': 0.10, 'SVR': 0.08, 'GBM': 0.07, 'KNN': 0.05},
    "Balanced": {'LSTM': 0.25, 'XGBoost': 0.20, 'Random Forest': 0.15, 'ARIMA': 0.15, 'SVR': 0.10, 'GBM': 0.10, 'KNN': 0.05},
    "Volatility-Focused": {'LSTM': 0.30, 'XGBoost': 0.25, 'Random Forest': 0.20, 'ARIMA': 0.05, 'SVR': 0.10, 'GBM': 0.07, 'KNN': 0.03}
}

WEIGHT_DESCRIPTIONS = {
    "Default": "Original configuration with balanced weights",
    "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
    "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
    "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
    "Balanced": "Best for general purpose, unknown stock characteristics",
    "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks"
}

col1, col2 = st.columns([2, 1])
with col1:
    selected_weight = st.selectbox("Select Weight Configuration:", options=list(WEIGHT_CONFIGURATIONS.keys()),
                                   help="Choose different weight configurations for the prediction models")
with col2:
    st.info(WEIGHT_DESCRIPTIONS[selected_weight])

try:
    df = fetch_stock_data(symbol, display_days)
    st.subheader("Stock Price History")
    st.line_chart(df['Close'])
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                predictor = MultiAlgorithmStockPredictor(symbol, weights=WEIGHT_CONFIGURATIONS[selected_weight], hyperparams=hyperparams)
                results = predictor.predict_with_all_models()
                if results is not None:
                    last_price = float(df['Close'].iloc[-1])
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame({
                        'Model': results['individual_predictions'].keys(),
                        'Predicted Price': [v for v in results['individual_predictions'].values()]
                    })
                    model_predictions['Deviation from Ensemble'] = model_predictions['Predicted Price'] - abs(results['prediction'])
                    model_predictions = model_predictions.sort_values('Predicted Price', ascending=False)
                    st.dataframe(model_predictions.style.format({'Predicted Price': '${:.2f}', 'Deviation from Ensemble': '${:.2f}'}))

                    price_change = ((results['prediction'] - last_price) / last_price) * 100
                    fig, ax = plt.subplots(figsize=(10, 6))
                    predictions = list(results['individual_predictions'].values())
                    models = list(results['individual_predictions'].keys())
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, predictions)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.axvline(x=last_price, color='r', linestyle='--', label='Current Price')
                    ax.axvline(x=results['prediction'], color='g', linestyle='--', label='Ensemble Prediction')
                    ax.set_xlabel('Price ($)')
                    ax.set_title('Model Predictions Comparison')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.subheader("Model Consensus Analysis")
                    buy_signals = sum(1 for pred in predictions if pred > last_price)
                    sell_signals = sum(1 for pred in predictions if pred < last_price)
                    total_models = len(predictions)
                    consensus_col1, consensus_col2, consensus_col3 = st.columns(3)
                    with consensus_col1:
                        st.metric("Buy Signals", f"{buy_signals}/{total_models}")
                    with consensus_col2:
                        st.metric("Sell Signals", f"{sell_signals}/{total_models}")
                    with consensus_col3:
                        st.metric("Consensus Strength", f"{abs(buy_signals - sell_signals) / total_models:.1%}")

                    st.subheader("Risk Assessment")
                    prediction_std = np.std(predictions)
                    risk_level = "Low" if prediction_std < last_price * 0.02 else "Medium" if prediction_std < last_price * 0.05 else "High"
                    risk_col1, risk_col2 = st.columns(2)
                    with risk_col1:
                        st.metric("Prediction Volatility", f"${prediction_std:.2f}")
                    with risk_col2:
                        st.metric("Risk Level", risk_level)

        if st.sidebar.checkbox("Run Model Validation (Backtesting)"):
            with st.spinner("Running backtesting on historical data..."):
                predictor = MultiAlgorithmStockPredictor(symbol, hyperparams=hyperparams)
                validation_results = predictor.backtest_models(n_splits=3)
                st.subheader("Model Validation Results")
                val_df = pd.DataFrame(validation_results).T
                st.dataframe(val_df.style.format("{:.4f}").highlight_min(color='lightgreen').highlight_max(color='#ffcccc'))
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                val_df['RMSE'].plot(kind='bar', ax=ax[0], title='RMSE Comparison')
                val_df['MAE'].plot(kind='bar', ax=ax[1], title='MAE Comparison')
                st.pyplot(fig)

    with col2:
        st.subheader("Latest News & Market Sentiment")
        news_headlines = get_news_headlines(symbol)
        if news_headlines:
            for title, description, url in news_headlines:
                with st.expander(title):
                    st.write(description)
                    st.markdown(f"[Read full article]({url})")
        else:
            st.write("No recent news available for this stock.")

        st.subheader("Technical Analysis Summary")
        try:
            if 'df' in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                analysis_df = calculate_technical_indicators_for_summary(df)
                if len(analysis_df) >= 2:
                    latest = analysis_df.iloc[-1]
                    ma_bullish = float(latest['MA20']) > float(latest['MA50'])
                    rsi_value = float(latest['RSI'])
                    volume_high = float(latest['Volume']) > float(latest['Volume_MA'])
                    close_price = float(latest['Close'])
                    bb_upper = float(latest['BB_upper'])
                    bb_lower = float(latest['BB_lower'])

                    st.write("📊 Historical Data Analysis")
                    historical_indicators = {
                        "Moving Averages": {"value": "Bullish" if ma_bullish else "Bearish", "delta": f"{((float(latest['MA20']) - float(latest['MA50']))/float(latest['MA50']) * 100):.1f}% spread", "description": "Based on 20 & 50-day moving averages"},
                        "RSI (14)": {"value": "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral", "delta": f"{rsi_value:.1f}", "description": "Current RSI value"},
                        "Volume Trend": {"value": "Above Average" if volume_high else "Below Average", "delta": f"{((float(latest['Volume']) - float(latest['Volume_MA']))/float(latest['Volume_MA']) * 100):.1f}%", "description": "Compared to 20-day average"},
                        "Bollinger Bands": {"value": "Upper Band" if close_price > bb_upper else "Lower Band" if close_price < bb_lower else "Middle Band", "delta": f"{((close_price - bb_lower)/(bb_upper - bb_lower) * 100):.1f}%", "description": "Position within bands"}
                    }
                    for indicator, data in historical_indicators.items():
                        with st.expander(f"{indicator}: {data['value']}"):
                            st.metric(label=data['description'], value=data['value'], delta=data['delta'])

                    if 'results' in locals() and results is not None:
                        st.write("🤖 Model Predictions Analysis")
                        current_price = float(df['Close'].iloc[-1])
                        pred_price = float(results['prediction'])
                        price_change_pct = ((pred_price - current_price) / current_price) * 100
                        predictions = results['individual_predictions']
                        bullish_models = sum(1 for p in predictions.values() if p > current_price)
                        prediction_indicators = {
                            "Price Prediction": {"value": f"${pred_price:.2f}", "delta": f"{price_change_pct:+.1f}% from current", "description": "Ensemble model prediction"},
                            "Model Consensus": {"value": f"{bullish_models}/{len(predictions)} Bullish", "delta": f"{(bullish_models/len(predictions)*100):.0f}% agreement", "description": "Agreement among models"},
                            "Prediction Range": {"value": f"${abs(results['lower_bound']):.2f} - ${abs(results['upper_bound']):.2f}", "delta": f"±{((results['upper_bound'] - results['lower_bound'])/2/pred_price*100):.1f}%", "description": "Expected price range"},
                            "Confidence Score": {"value": f"{results['confidence_score']:.1%}", "delta": "Based on model agreement", "description": "Overall prediction confidence"}
                        }
                        for indicator, data in prediction_indicators.items():
                            with st.expander(f"{indicator}: {data['value']}"):
                                st.metric(label=data['description'], value=data['value'], delta=data['delta'])
                                
                else:
                    st.warning("Insufficient data points for technical analysis. Please ensure you have at least 50 days of historical data.")
            else:
                st.warning("No data available for technical analysis. Please enter a valid stock symbol.")
        except Exception as e:
            st.error(f"Error in Technical Analysis: {str(e)}")
except Exception as e:
    st.error(f"Error: {str(e)}")
st.markdown("---")