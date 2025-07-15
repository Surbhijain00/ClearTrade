import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from src.data_fetcher import fetch_stock_data
from src.technical_indicators import calculate_technical_indicators

class MetaWeightOptimizer:
    def __init__(self, model_names, lookback_window=90):
        self.model_errors = {name: [] for name in model_names}
        self.lookback = lookback_window
        self.meta_model = XGBRegressor(objective='reg:squarederror')
        
    def update_errors(self, model_name, actual, predicted):
        error = np.abs(actual - predicted)
        self.model_errors[model_name].append(error)
        # Maintain rolling window
        if len(self.model_errors[model_name]) > self.lookback:
            self.model_errors[model_name].pop(0)
            
    def get_current_weights(self):
        # Create performance dataframe
        error_df = pd.DataFrame(self.model_errors)
        if len(error_df) < 10:  # Warm-up period
            return {model: 1/len(self.model_errors) for model in self.model_errors}
            
        # Calculate relative performance
        performance = 1 / (error_df.rolling(7).mean() + 1e-8)
        X = performance.dropna().values
        y = np.ones(len(X))  # Dummy target for meta-learning
        
        # Train meta-model on recent performance
        self.meta_model.fit(X, y)
        feature_importances = self.meta_model.feature_importances_
        total = sum(feature_importances)
        return {model: imp/total for model, imp in zip(self.model_errors.keys(), feature_importances)}

class MultiAlgorithmStockPredictor:
    def __init__(self, symbol, training_years=5, weights=None, hyperparams=None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = weights if weights is not None else {
            'LSTM': 0.3, 'XGBoost': 0.15, 'Random Forest': 0.15, 'ARIMA': 0.1, 'SVR': 0.1, 'GBM': 0.1, 'KNN': 0.1
        }
        self.weight_optimizer = MetaWeightOptimizer(
            model_names=['LSTM', 'SVR', 'Random Forest', 'XGBoost', 'KNN', 'GBM', 'ARIMA']
        )
        # Define default hyperparameters
        default_hyperparams = {
            'LSTM': {'units': 100, 'dropout': 0.2, 'epochs': 50, 'batch_size': 32},
            'SVR': {'C': 100, 'epsilon': 0.1},
            'Random Forest': {'n_estimators': 100, 'max_depth': None},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'KNN': {'n_neighbors': 5},
            'GBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'ARIMA': {'order': (5, 1, 0)}
        }
        # Merge user-provided hyperparams with defaults
        if hyperparams is not None:
            self.hyperparams = default_hyperparams.copy()
            self.hyperparams.update(hyperparams)
        else:
            self.hyperparams = default_hyperparams.copy()

    def fetch_historical_data(self):
        return fetch_stock_data(self.symbol, self.training_years * 365)

    def prepare_data(self, df, seq_length=60):
        feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                           'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                           'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])
        X_other = scaled_data[seq_length:]
        return np.array(X_lstm), X_other, np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(self.hyperparams['LSTM']['units'], return_sequences=True)),
            Dropout(self.hyperparams['LSTM']['dropout']),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(self.hyperparams['LSTM']['dropout']),
            LSTM(50, return_sequences=False),
            Dropout(self.hyperparams['LSTM']['dropout']),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def backtest_models(self, n_splits=5):
        df = self.fetch_historical_data()
        df = calculate_technical_indicators(df)
        X_lstm, X_other, y = self.prepare_data(df)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []

        for train_index, test_index in tscv.split(X_other):
            X_other_train, X_other_test = X_other[train_index], X_other[test_index]
            X_lstm_train, X_lstm_test = X_lstm[train_index], X_lstm[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_metrics = {}

            models = {
                'LSTM': self.build_lstm_model((X_lstm_train.shape[1], X_lstm_train.shape[2])),
                'SVR': SVR(kernel='rbf', C=self.hyperparams['SVR']['C'], epsilon=self.hyperparams['SVR']['epsilon']),
                'Random Forest': RandomForestRegressor(n_estimators=self.hyperparams['RandomForest']['n_estimators'], max_depth=self.hyperparams['RandomForest']['max_depth'], random_state=42),
                'XGBoost': XGBRegressor(n_estimators=self.hyperparams['XGBoost']['n_estimators'], learning_rate=self.hyperparams['XGBoost']['learning_rate'], max_depth=self.hyperparams['XGBoost']['max_depth'], random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=self.hyperparams['KNN']['n_neighbors']),
                'GBM': GradientBoostingRegressor(n_estimators=self.hyperparams['GBM']['n_estimators'], learning_rate=self.hyperparams['GBM']['learning_rate'], max_depth=self.hyperparams['GBM']['max_depth'], random_state=42)
            }

            for model_name, model in models.items():
                if model_name == 'LSTM':
                    model.fit(X_lstm_train, y_train, epochs=self.hyperparams['LSTM']['epochs'], batch_size=self.hyperparams['LSTM']['batch_size'], verbose=0)
                    pred = model.predict(X_lstm_test, verbose=0).flatten()
                else:
                    model.fit(X_other_train, y_train)
                    pred = model.predict(X_other_test)
                
                # Update error tracking
                actual = y_test * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]
                predicted = pred * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]
                self.weight_optimizer.update_errors(model_name, actual.mean(), predicted.mean())
                
                rmse = np.sqrt(np.mean((pred - y_test)**2))
                mae = np.mean(np.abs(pred - y_test))
                fold_metrics[model_name] = {'RMSE': rmse, 'MAE': mae}
            metrics.append(fold_metrics)

        avg_metrics = {model: {'RMSE': np.mean([m[model]['RMSE'] for m in metrics]), 'MAE': np.mean([m[model]['MAE'] for m in metrics])} for model in models.keys()}
        return avg_metrics

    def predict_with_all_models(self, prediction_days=30, sequence_length=60):
        try:
            df = self.fetch_historical_data()
            if len(df) < sequence_length + 20:
                st.error(f"Insufficient historical data. Need at least {sequence_length + 20} days of data.")
                return None
            df = calculate_technical_indicators(df)
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None
            feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                               'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                               'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
            scaled_data = self.scaler.fit_transform(df[feature_columns])
            X_lstm, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])
            if len(X_lstm) == 0 or len(y) == 0:
                st.error("Could not create valid sequences for prediction.")
                return None
            X_other = scaled_data[sequence_length:]
            X_lstm = np.array(X_lstm)
            X_other = np.array(X_other)
            y = np.array(y)
            split_idx = int(len(y) * 0.8)
            X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
            X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            predictions = {}
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lstm_model.fit(X_lstm_train, y_train, epochs=50, batch_size=32, validation_data=(X_lstm_test, y_test), callbacks=[early_stopping], verbose=0)
            predictions['LSTM'] = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]

            svr_model = SVR(kernel='rbf', C=self.hyperparams['SVR']['C'], epsilon=self.hyperparams['SVR']['epsilon'])
            svr_model.fit(X_other_train, y_train)
            predictions['SVR'] = svr_model.predict(X_other_test[-1:])[0]

            rf_model = RandomForestRegressor(n_estimators=self.hyperparams['RandomForest']['n_estimators'], max_depth=self.hyperparams['RandomForest']['max_depth'], random_state=42)
            rf_model.fit(X_other_train, y_train)
            predictions['Random Forest'] = rf_model.predict(X_other_test[-1:])[0]

            xgb_model = XGBRegressor(n_estimators=self.hyperparams['XGBoost']['n_estimators'], learning_rate=self.hyperparams['XGBoost']['learning_rate'], max_depth=self.hyperparams['XGBoost']['max_depth'], random_state=42)
            xgb_model.fit(X_other_train, y_train)
            predictions['XGBoost'] = xgb_model.predict(X_other_test[-1:])[0]

            knn_model = KNeighborsRegressor(n_neighbors=self.hyperparams['KNN']['n_neighbors'])
            knn_model.fit(X_other_train, y_train)
            predictions['KNN'] = knn_model.predict(X_other_test[-1:])[0]

            gbm_model = GradientBoostingRegressor(n_estimators=self.hyperparams['GBM']['n_estimators'], learning_rate=self.hyperparams['GBM']['learning_rate'], max_depth=self.hyperparams['GBM']['max_depth'], random_state=42)
            gbm_model.fit(X_other_train, y_train)
            predictions['GBM'] = gbm_model.predict(X_other_test[-1:])[0]

            try:
                close_prices = df['Close'].values
                arima_model = ARIMA(close_prices, order=self.hyperparams['ARIMA']['order'])
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                close_min = self.scaler.data_min_[0]
                close_max = self.scaler.data_max_[0]
                predictions['ARIMA'] = (arima_pred - close_min) / (close_max - close_min)
            except Exception as e:
                st.warning(f"ARIMA prediction failed: {str(e)}")

            # Dynamic weighting calculation
            dynamic_weights = self.weight_optimizer.get_current_weights()
            available_models = [m for m in predictions.keys() if m in dynamic_weights]
            total_weight = sum(dynamic_weights[model] for model in available_models)
            adjusted_weights = {model: dynamic_weights[model]/total_weight 
                               for model in available_models}
            
            # Calculate ensemble prediction with proper shaping
            ensemble_array = np.zeros((1, X_other.shape[1]))
            ensemble_array[0, 0] = sum(pred * adjusted_weights[model] 
                                      for model, pred in predictions.items())
            
            # Inverse transform ensemble prediction
            final_prediction = self.scaler.inverse_transform(ensemble_array)[0, 0]

            # Calculate individual predictions with proper shaping
            individual_predictions = {}
            for model_name, pred in predictions.items():
                model_array = np.zeros((1, X_other.shape[1]))
                model_array[0, 0] = pred
                individual_predictions[model_name] = \
                    self.scaler.inverse_transform(model_array)[0, 0]

            # Update errors with proper array handling
            actual_price = df['Close'].iloc[-1]
            for model_name, pred in predictions.items():
                model_array = np.zeros((1, X_other.shape[1]))
                model_array[0, 0] = pred
                predicted_price = self.scaler.inverse_transform(model_array)[0, 0]
                self.weight_optimizer.update_errors(model_name, 
                                                   actual_price, 
                                                   predicted_price)

            # Calculate confidence metrics
            std_dev = np.std(list(individual_predictions.values()))
            print(f"Current Weights: {dynamic_weights}")
            return {
                'prediction': final_prediction,
                'lower_bound': final_prediction - std_dev,
                'upper_bound': final_prediction + std_dev,
                'confidence_score': 1 / (1 + std_dev / final_prediction),
                'individual_predictions': individual_predictions,
                'dynamic_weights': adjusted_weights
            }
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None