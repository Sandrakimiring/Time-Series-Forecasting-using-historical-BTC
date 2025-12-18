"""
Cryptocurrency Price Predictor Dashboard
A beautiful Streamlit app for predicting crypto prices using LSTM
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import (
    CryptoPredictor, 
    SUPPORTED_CRYPTOS, 
    get_price_change_info,
    get_model_path,
    get_training_metrics,
    get_available_models
)

# ----- Page Configuration -----
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Custom CSS for Beautiful UI -----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a40 100%);
        border: 1px solid rgba(123, 44, 191, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #a0a0b0 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00d4ff !important;
        font-size: 1.8rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(123, 44, 191, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(123, 44, 191, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #1e1e30;
        border: 1px solid rgba(123, 44, 191, 0.3);
        border-radius: 10px;
        color: #e0e0e0;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e1e30;
        border-radius: 10px;
        color: #e0e0e0 !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: linear-gradient(145deg, #1e3a2e 0%, #1e1e30 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
    }
    
    /* Warning box */
    .stWarning {
        background: linear-gradient(145deg, #3a3a1e 0%, #1e1e30 100%);
        border: 1px solid rgba(255, 200, 0, 0.3);
        border-radius: 12px;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(145deg, #1a2a3a 0%, #1e1e30 100%);
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
    }
    
    .prediction-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .prediction-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #a0a0b0;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Price change indicators */
    .price-up {
        color: #00ff88 !important;
    }
    
    .price-down {
        color: #ff4757 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e1e30;
        border-radius: 10px;
        color: #a0a0b0;
        border: 1px solid rgba(123, 44, 191, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Data labels */
    .data-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #808090;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ----- Initialize Session State -----
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


# ----- Sidebar -----
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Crypto selection
    crypto_options = {f"{v['name']} ({v['symbol']})": k for k, v in SUPPORTED_CRYPTOS.items()}
    selected_crypto_display = st.selectbox(
        "🪙 Select Cryptocurrency",
        options=list(crypto_options.keys()),
        index=0
    )
    selected_crypto = crypto_options[selected_crypto_display]
    crypto_info = SUPPORTED_CRYPTOS[selected_crypto]
    
    st.markdown("---")
    
    # Date range
    st.markdown("### 📅 Data Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2020, 1, 1),
            min_value=date(2015, 1, 1),
            max_value=date.today() - timedelta(days=100)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=start_date + timedelta(days=100),
            max_value=date.today()
        )
    
    st.markdown("---")
    
    # Prediction settings
    st.markdown("### 🔮 Prediction Settings")
    forecast_days = st.slider(
        "Days to Forecast",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    st.markdown("---")
    
    # Model status
    st.markdown("### 🤖 Model Status")
    model_path = get_model_path(selected_crypto)
    
    if model_path:
        st.success(f"✅ Model available")
        st.caption(f"📁 {model_path}")
    else:
        st.warning(f"⚠️ No model found for {crypto_info['symbol']}")
        st.caption("Run training script or wait for daily auto-training")
    
    # Show training metrics if available
    training_metrics = get_training_metrics()
    if training_metrics:
        symbol = crypto_info['symbol']
        if symbol in training_metrics.get('models', {}):
            model_metrics = training_metrics['models'][symbol]
            if 'error' not in model_metrics:
                st.caption(f"📊 RMSE: {model_metrics.get('rmse_pct', 0):.2f}%")
                st.caption(f"🎯 Dir Acc: {model_metrics.get('directional_accuracy', 0):.1f}%")
                last_trained = model_metrics.get('trained_at', 'Unknown')[:10]
                st.caption(f"🕐 Trained: {last_trained}")
    
    st.markdown("---")
    
    # Load button
    load_button = st.button("🚀 Load Data & Predict", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #606070; font-size: 0.8rem;'>
        <p>Built with ❤️ using LSTM</p>
        <p>Data from Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)


# ----- Main Content -----
st.markdown(f"""
# 📈 {crypto_info['name']} Price Predictor
<p style='color: #808090; font-size: 1.1rem; font-family: Space Grotesk, sans-serif;'>
    AI-powered price predictions using LSTM neural networks
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# ----- Load Data and Make Predictions -----
@st.cache_resource
def load_predictor(model_path: str):
    """Load the prediction model"""
    try:
        predictor = CryptoPredictor(model_path=model_path)
        return predictor, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_crypto_data(ticker: str, start: str, end: str):
    """Fetch cryptocurrency data"""
    try:
        predictor = CryptoPredictor()
        df = predictor.fetch_data(ticker, start, end)
        return df, None
    except Exception as e:
        return None, str(e)


if load_button or st.session_state.data is not None:
    # Check if model exists for selected crypto
    if not model_path:
        st.error(f"❌ No trained model found for {crypto_info['name']}")
        st.info(f"""
        💡 **How to get a model:**
        1. **Wait for auto-training**: Models are retrained daily at 6 AM UTC via GitHub Actions
        2. **Train manually**: Run `python scripts/train_model.py` locally
        3. **Use BTC model**: Select Bitcoin which has a pre-trained model
        """)
    else:
        # Load model
        with st.spinner("🔄 Loading model..."):
            predictor, model_error = load_predictor(model_path)
        
        if model_error:
            st.error(f"❌ Error loading model: {model_error}")
            st.info("💡 The model file may be corrupted. Try retraining with `python scripts/train_model.py`")
        else:
            # Fetch data
            with st.spinner(f"📊 Fetching {crypto_info['name']} data..."):
                df, data_error = fetch_crypto_data(
                    selected_crypto, 
                    str(start_date), 
                    str(end_date)
                )
            
            if data_error:
                st.error(f"❌ Error fetching data: {data_error}")
            else:
                st.session_state.data = df
                st.session_state.predictor = predictor
                
                # Get price info
                price_info = get_price_change_info(df)
                
                # ----- Current Price Section -----
                st.markdown("## 💰 Current Market Data")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${price_info['current_price']:,.2f}",
                        f"{price_info['change_24h_pct']:+.2f}% (24h)"
                    )
                
                with col2:
                    st.metric(
                        "7-Day Change",
                        f"${price_info['change_7d']:+,.2f}",
                        f"{price_info['change_7d_pct']:+.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "30-Day Change",
                        f"${price_info['change_30d']:+,.2f}",
                        f"{price_info['change_30d_pct']:+.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "All-Time High",
                        f"${price_info['all_time_high']:,.2f}",
                        f"{((price_info['current_price'] / price_info['all_time_high']) - 1) * 100:.1f}% from ATH"
                    )
                
                st.markdown("---")
                
                # ----- Predictions -----
                st.markdown("## 🔮 Price Predictions")
                
                with st.spinner("🧠 Running LSTM predictions..."):
                    try:
                        # Prepare data and make predictions
                        X = predictor.prepare_prediction_data(df)
                        predictions = predictor.predict(X)
                        actual_prices = df.iloc[predictor.sequence_length:]['Close'].values
                        
                        # Calculate metrics
                        metrics = predictor.calculate_metrics(actual_prices, predictions)
                        
                        # Future predictions
                        future_preds = predictor.predict_next_days(df, days=forecast_days)
                        st.session_state.predictions = future_preds
                        
                        # Display prediction card
                        next_pred = future_preds[0]
                        pred_change = next_pred['predicted_price'] - price_info['current_price']
                        pred_change_pct = (pred_change / price_info['current_price']) * 100
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">Tomorrow's Predicted Price</div>
                            <div class="prediction-price">${next_pred['predicted_price']:,.2f}</div>
                            <div class="prediction-label {'price-up' if pred_change >= 0 else 'price-down'}">
                                {'+' if pred_change >= 0 else ''}{pred_change:,.2f} ({pred_change_pct:+.2f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Multi-day forecast table
                        if forecast_days > 1:
                            st.markdown("### 📆 Multi-Day Forecast")
                            
                            forecast_df = pd.DataFrame(future_preds)
                            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.strftime('%Y-%m-%d')
                            forecast_df['predicted_price'] = forecast_df['predicted_price'].apply(lambda x: f"${x:,.2f}")
                            forecast_df.columns = ['Date', 'Predicted Price', 'Day']
                            
                            st.dataframe(
                                forecast_df[['Day', 'Date', 'Predicted Price']],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        st.markdown("---")
                        
                        # ----- Model Performance -----
                        st.markdown("## 📊 Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("RMSE", f"${metrics['rmse']:,.2f}")
                        with col2:
                            st.metric("MAE", f"${metrics['mae']:,.2f}")
                        with col3:
                            st.metric("Error %", f"{metrics['rmse_pct']:.2f}%")
                        with col4:
                            st.metric("Direction Accuracy", f"{metrics['directional_accuracy']:.1f}%")
                        
                        st.markdown("---")
                        
                        # ----- Charts -----
                        st.markdown("## 📈 Price Charts")
                        
                        tab1, tab2, tab3 = st.tabs(["📊 Historical", "🎯 Actual vs Predicted", "🔮 Forecast"])
                        
                        with tab1:
                            # Historical price chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['Close'],
                                mode='lines',
                                name=f'{crypto_info["symbol"]} Price',
                                line=dict(color=crypto_info['color'], width=2),
                                fill='tozeroy',
                                fillcolor=f'rgba{tuple(list(bytes.fromhex(crypto_info["color"][1:])) + [0.1])}'
                            ))
                            
                            fig.update_layout(
                                title=f'{crypto_info["name"]} Historical Price',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Space Grotesk'),
                                hovermode='x unified',
                                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            # Actual vs Predicted
                            pred_dates = df.index[predictor.sequence_length:]
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=actual_prices,
                                mode='lines',
                                name='Actual',
                                line=dict(color='#00d4ff', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=predictions.flatten(),
                                mode='lines',
                                name='Predicted',
                                line=dict(color='#ff6b6b', width=2, dash='dot')
                            ))
                            
                            fig.update_layout(
                                title='Actual vs Predicted Prices',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Space Grotesk'),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            # Future forecast chart
                            # Last 60 days + forecast
                            recent_df = df.tail(60)
                            
                            fig = go.Figure()
                            
                            # Historical prices
                            fig.add_trace(go.Scatter(
                                x=recent_df.index,
                                y=recent_df['Close'],
                                mode='lines',
                                name='Historical',
                                line=dict(color='#00d4ff', width=2)
                            ))
                            
                            # Forecast
                            forecast_dates = [p['date'] for p in future_preds]
                            forecast_prices = [p['predicted_price'] for p in future_preds]
                            
                            # Connect historical to forecast
                            connection_dates = [recent_df.index[-1]] + forecast_dates
                            connection_prices = [float(recent_df['Close'].iloc[-1])] + forecast_prices
                            
                            fig.add_trace(go.Scatter(
                                x=connection_dates,
                                y=connection_prices,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#00ff88', width=3, dash='dash'),
                                marker=dict(size=8, symbol='diamond')
                            ))
                            
                            # Add prediction cone (uncertainty)
                            upper_band = [p * 1.05 for p in forecast_prices]
                            lower_band = [p * 0.95 for p in forecast_prices]
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_dates + forecast_dates[::-1],
                                y=upper_band + lower_band[::-1],
                                fill='toself',
                                fillcolor='rgba(0, 255, 136, 0.1)',
                                line=dict(color='rgba(0,0,0,0)'),
                                name='±5% Range',
                                showlegend=True
                            ))
                            
                            fig.update_layout(
                                title=f'{forecast_days}-Day Price Forecast',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Space Grotesk'),
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.info("💡 This may occur if the model was trained on different data. Try retraining the model with the current date range.")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <h2 style='color: #a0a0b0; font-family: Space Grotesk, sans-serif;'>
            👈 Configure settings and click <span style='color: #00d4ff;'>"Load Data & Predict"</span> to start
        </h2>
        <p style='color: #606070; font-size: 1.1rem; margin-top: 20px;'>
            This app uses LSTM neural networks to predict cryptocurrency prices based on historical data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1e1e30, #2a2a40); border-radius: 16px; padding: 25px; border: 1px solid rgba(123, 44, 191, 0.2);'>
            <h3 style='color: #00d4ff; font-family: Space Grotesk;'>🔮 Multi-Day Forecast</h3>
            <p style='color: #a0a0b0;'>Predict up to 30 days into the future with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1e1e30, #2a2a40); border-radius: 16px; padding: 25px; border: 1px solid rgba(123, 44, 191, 0.2);'>
            <h3 style='color: #7b2cbf; font-family: Space Grotesk;'>📊 Performance Metrics</h3>
            <p style='color: #a0a0b0;'>Track RMSE, MAE, and directional accuracy of predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #1e1e30, #2a2a40); border-radius: 16px; padding: 25px; border: 1px solid rgba(123, 44, 191, 0.2);'>
            <h3 style='color: #00ff88; font-family: Space Grotesk;'>🪙 Multi-Crypto</h3>
            <p style='color: #a0a0b0;'>Support for BTC, ETH, SOL, XRP, ADA and more</p>
        </div>
        """, unsafe_allow_html=True)


# ----- About Section -----
st.markdown("---")

with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### How It Works
    
    This application uses a **Long Short-Term Memory (LSTM)** neural network to predict cryptocurrency prices.
    
    **Model Architecture:**
    - Input: Previous 60 days of closing prices
    - LSTM layers with dropout for regularization
    - Dense output layer for price prediction
    
    **Data Source:**
    - Historical price data from Yahoo Finance via `yfinance`
    - Real-time updates when you refresh
    
    **Limitations:**
    - Predictions are based on historical patterns and may not account for sudden market events
    - The model was trained on Bitcoin data; predictions for other cryptos may be less accurate
    - This is for educational purposes only - not financial advice!
    
    ---
    
    **Tech Stack:** Python, TensorFlow/Keras, Streamlit, Plotly, yfinance
    """)

with st.expander("⚠️ Disclaimer"):
    st.markdown("""
    **IMPORTANT:** This application is for **educational and informational purposes only**.
    
    - Cryptocurrency markets are highly volatile and unpredictable
    - Past performance does not guarantee future results
    - This is **NOT** financial advice
    - Always do your own research before making investment decisions
    - Never invest more than you can afford to lose
    
    The creators of this application are not responsible for any financial losses incurred from using these predictions.
    """)
