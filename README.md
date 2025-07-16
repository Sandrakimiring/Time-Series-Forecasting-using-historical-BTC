# Time Series Forecasting Using Historical BTC Data

# 📈 Bitcoin Price Prediction Using LSTM

This project predicts the next-day closing price of Bitcoin (BTC-USD) using historical price data and a Long Short-Term Memory (LSTM) neural network.

Built as part of my time series forecasting learning journey, it covers everything from data collection and preprocessing to model training, evaluation, and dashboard deployment using Streamlit.

---


## 🧠 What You I Learnt 

- Time series forecasting fundamentals  
- Sliding window technique for sequence modeling  
- Building LSTM models with Keras & TensorFlow  
- Visualizing actual vs predicted prices    
- Deploying ML models with Streamlit  

---

## 🗂️ Project Structure

```
Time-Series-Forecasting-using-historical-BTC/
│
├── notebooks/
│   └── btc_model_training.ipynb     # EDA + model training
│
├── btc_app.py                       # Streamlit dashboard
├── .streamlit/
│   └── config.toml                  # Theme settings 
│
├── btc_lstm_model.keras             # Saved model
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 📊 Model Overview

- **Model**: LSTM (1 hidden layer, 50 units)
- **Input**: Previous 60 days of BTC prices
- **Output**: Next-day price
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam
- **Evaluation**: Root Mean Squared Error (RMSE)
- **Data Source**: Historical Bitcoin price data from Yahoo Finance (`yfinance`)

---

## ⚙️ How to Run It Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Sandrakimiring/Time-Series-Forecasting-using-historical-BTC.git
   cd Time-Series-Forecasting-using-historical-BTC
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run btc_app.py
   ```

---

## 📈 Results

| Metric               | Value         |
| -------------------- | ------------- |
| RMSE                 | ~$3,543.95    |
| Avg BTC Price (Test) | ~$66,539.78  |
| RMSE %               | ~5.33%        |

---

##  Future Improvements

- Add Ethereum (ETH-USD) toggle
- Forecast next *N* days, not just one
- Add FastAPI backend for model inference
- Dockerize for deployment

---






