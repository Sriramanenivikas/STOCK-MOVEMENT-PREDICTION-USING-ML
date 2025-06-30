import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Stock Predictor", layout="wide")

class TechnicalIndicators:
    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ti = TechnicalIndicators()
    
    def fetch_data(self, symbol, period="2y"):
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    
    def create_features(self, data):
        df = data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        df['RSI'] = self.ti.rsi(df['Close'])
        macd, signal, histogram = self.ti.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        upper_bb, middle_bb, lower_bb = self.ti.bollinger_bands(df['Close'])
        df['BB_Upper'] = upper_bb
        df['BB_Middle'] = middle_bb
        df['BB_Lower'] = lower_bb
        df['BB_Width'] = (upper_bb - lower_bb) / middle_bb
        df['BB_Position'] = (df['Close'] - lower_bb) / (upper_bb - lower_bb)
        
        k_percent, d_percent = self.ti.stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k_percent
        df['Stoch_D'] = d_percent
        
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Returns'].rolling(window=period).std()
            df[f'Price_Position_{period}'] = (df['Close'] - df['Close'].rolling(window=period).min()) / (df['Close'].rolling(window=period).max() - df['Close'].rolling(window=period).min())
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        feature_columns = [
            'Returns', 'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 
            'MACD_Histogram', 'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D', 'Volume_Ratio',
            'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio', 'Volatility_5', 'Volatility_10',
            'Volatility_20', 'Price_Position_5', 'Price_Position_10', 'Price_Position_20'
        ]
        
        return df, feature_columns

class ModelTrainer:
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.scaler_xgb = StandardScaler()
        self.scaler_lstm = StandardScaler()
    
    def train_xgboost(self, X, y):
        X_scaled = self.scaler_xgb.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        train_score = self.xgb_model.score(X_train, y_train)
        test_score = self.xgb_model.score(X_test, y_test)
        
        return train_score, test_score
    
    def prepare_lstm_data(self, X_scaled, y, lookback=60):
        X_lstm, y_lstm = [], []
        for i in range(lookback, len(X_scaled)):
            X_lstm.append(X_scaled[i-lookback:i])
            y_lstm.append(y[i])
        return np.array(X_lstm), np.array(y_lstm)
    
    def train_lstm(self, X, y, lookback=60):
        # Scale features
        X_scaled = self.scaler_lstm.fit_transform(X)
        
        # Prepare LSTM data with correct target
        X_lstm, y_lstm = self.prepare_lstm_data(X_scaled, y, lookback)
        
        if len(X_lstm) == 0:
            return 0, 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = self.lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, verbose=0)
        
        # Evaluate model
        train_loss, train_acc = self.lstm_model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        
        return train_acc, test_acc
    
    def predict_xgboost(self, X):
        if self.xgb_model is None:
            return None
        X_scaled = self.scaler_xgb.transform(X)
        return self.xgb_model.predict_proba(X_scaled)[:, 1]
    
    def predict_lstm(self, X, y, lookback=60):
        if self.lstm_model is None:
            return None
        
        X_scaled = self.scaler_lstm.transform(X)
        if len(X_scaled) < lookback:
            return None
        
        # Use the last lookback sequences for prediction
        X_lstm = X_scaled[-lookback:].reshape(1, lookback, -1)
        prediction = self.lstm_model.predict(X_lstm, verbose=0)[0][0]
        return prediction

class Backtester:
    def __init__(self):
        pass
    
    def backtest_strategy(self, data, predictions, threshold=0.5):
        if len(predictions) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'volatility': 0,
                'strategy_returns': np.array([])
            }
        
        signals = (predictions > threshold).astype(int)
        positions = np.diff(np.concatenate(([0], signals)))
        
        returns = data['Returns'].values[1:len(predictions)+1]
        strategy_returns = []
        position = 0
        
        for i, pos_change in enumerate(positions):
            if pos_change == 1:
                position = 1
            elif pos_change == -1:
                position = 0
            
            if i < len(returns):
                strategy_returns.append(returns[i] * position)
            else:
                strategy_returns.append(0)
        
        strategy_returns = np.array(strategy_returns)
        
        if len(strategy_returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'volatility': 0,
                'strategy_returns': strategy_returns
            }
        
        total_return = np.prod(1 + strategy_returns) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252) if len(strategy_returns) > 1 else 0
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        cumulative_returns = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'strategy_returns': strategy_returns
        }

def main():
    st.title("🚀 Advanced ML Stock Price Prediction")
    st.sidebar.title("Configuration")
    
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    model_type = st.sidebar.selectbox("Model Type", ["XGBoost", "LSTM", "Ensemble"])
    
    if st.sidebar.button("Analyze Stock"):
        try:
            with st.spinner("Fetching data and training models..."):
                processor = DataProcessor()
                data = processor.fetch_data(symbol)
                
                if data.empty:
                    st.error("No data found for the symbol")
                    return
                
                df, feature_columns = processor.create_features(data)
                df = df.dropna()
                
                if len(df) < 100:
                    st.error("Insufficient data for analysis")
                    return
                
                X = df[feature_columns].values
                y = df['Target'].values
                
                trainer = ModelTrainer()
                backtester = Backtester()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Current Price")
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    st.metric(
                        label=f"{symbol} Price",
                        value=f"${current_price:.2f}",
                        delta=f"{change:.2f}%"
                    )
                
                xgb_predictions = None
                lstm_prediction = None
                
                if model_type in ["XGBoost", "Ensemble"]:
                    xgb_train_acc, xgb_test_acc = trainer.train_xgboost(X, y)
                    xgb_predictions = trainer.predict_xgboost(X)
                    
                    with col2:
                        st.subheader("🎯 XGBoost Model")
                        st.write(f"Training Accuracy: {xgb_train_acc:.1%}")
                        st.write(f"Test Accuracy: {xgb_test_acc:.1%}")
                        
                        if xgb_predictions is not None and len(xgb_predictions) > 0:
                            current_signal = "BUY" if xgb_predictions[-1] > 0.5 else "SELL"
                            confidence = xgb_predictions[-1] if xgb_predictions[-1] > 0.5 else 1 - xgb_predictions[-1]
                            st.write(f"Signal: **{current_signal}**")
                            st.write(f"Confidence: {confidence:.1%}")
                
                if model_type in ["LSTM", "Ensemble"]:
                    if len(X) >= 60:
                        lstm_train_acc, lstm_test_acc = trainer.train_lstm(X, y)
                        lstm_prediction = trainer.predict_lstm(X, y)
                        
                        with col3:
                            st.subheader("🧠 LSTM Model")
                            st.write(f"Training Accuracy: {lstm_train_acc:.1%}")
                            st.write(f"Test Accuracy: {lstm_test_acc:.1%}")
                            
                            if lstm_prediction is not None:
                                current_signal = "BUY" if lstm_prediction > 0.5 else "SELL"
                                confidence = lstm_prediction if lstm_prediction > 0.5 else 1 - lstm_prediction
                                st.write(f"Signal: **{current_signal}**")
                                st.write(f"Confidence: {confidence:.1%}")
                    else:
                        with col3:
                            st.subheader("🧠 LSTM Model")
                            st.write("Insufficient data for LSTM (need 60+ samples)")
                
                # Ensemble prediction
                if model_type == "Ensemble" and xgb_predictions is not None and lstm_prediction is not None:
                    ensemble_pred = (xgb_predictions[-1] + lstm_prediction) / 2
                    st.subheader("🎯 Ensemble Prediction")
                    ensemble_signal = "BUY" if ensemble_pred > 0.5 else "SELL"
                    ensemble_confidence = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred
                    st.write(f"**Ensemble Signal: {ensemble_signal}** (Confidence: {ensemble_confidence:.1%})")
                
                st.subheader("📈 Price Chart with Predictions")
                
                fig = go.Figure()
                
                recent_data = df.tail(100)
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['SMA_10'],
                    mode='lines',
                    name='SMA 10',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['SMA_30'],
                    mode='lines',
                    name='SMA 30',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🔬 Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI", yaxis_title="RSI")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red')
                    ))
                    fig_macd.update_layout(title="MACD", yaxis_title="MACD")
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                if xgb_predictions is not None:
                    st.subheader("📊 Backtest Results")
                    
                    backtest_results = backtester.backtest_strategy(df, xgb_predictions)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{backtest_results['total_return']:.1%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1%}")
                    
                    with col4:
                        st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
                    
                    if len(backtest_results['strategy_returns']) > 0:
                        strategy_cumulative = np.cumprod(1 + backtest_results['strategy_returns'])
                        buy_hold_returns = df['Returns'].fillna(0).values[1:len(backtest_results['strategy_returns'])+1]
                        buy_hold_cumulative = np.cumprod(1 + buy_hold_returns)
                        
                        fig_backtest = go.Figure()
                        
                        dates = df.index[1:len(strategy_cumulative)+1]
                        
                        fig_backtest.add_trace(go.Scatter(
                            x=dates,
                            y=strategy_cumulative,
                            mode='lines',
                            name='Strategy',
                            line=dict(color='green')
                        ))
                        
                        fig_backtest.add_trace(go.Scatter(
                            x=dates,
                            y=buy_hold_cumulative,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='blue')
                        ))
                        
                        fig_backtest.update_layout(
                            title="Strategy vs Buy & Hold Performance",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Returns"
                        )
                        
                        st.plotly_chart(fig_backtest, use_container_width=True)
                
                st.subheader("🎯 Feature Importance")
                if hasattr(trainer, 'xgb_model') and trainer.xgb_model is not None and hasattr(trainer.xgb_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': trainer.xgb_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Feature Importance"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()