# 📈 Crypto Price Predictor

> AI-powered cryptocurrency price prediction using LSTM neural networks

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

A beautiful, production-ready web application that predicts cryptocurrency prices using Long Short-Term Memory (LSTM) neural networks. Built with TensorFlow/Keras and deployed with Streamlit.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Multi-Day Forecasting** | Predict up to 30 days into the future |
| 🪙 **Multi-Crypto Support** | BTC, ETH, SOL, XRP, ADA and more |
| 📊 **Interactive Charts** | Beautiful Plotly visualizations with zoom, pan, and hover |
| 📈 **Performance Metrics** | RMSE, MAE, and directional accuracy tracking |
| 🎨 **Modern UI** | Stunning dark theme with gradient accents |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| ⚡ **Real-Time Data** | Live data from Yahoo Finance |

---

## 🖼️ Screenshots

<details>
<summary>Click to view screenshots</summary>

### Main Dashboard
The main dashboard shows current market data, predictions, and interactive charts.

### Forecast View
Multi-day price forecast with confidence intervals.

### Performance Metrics
Track model accuracy with RMSE, MAE, and directional accuracy.

</details>

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Sandrakimiring/Time-Series-Forecasting-using-historical-BTC.git
cd Time-Series-Forecasting-using-historical-BTC

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models for all cryptocurrencies (first time only)
python scripts/train_model.py

# 5. Run the app
streamlit run btc_app.py
```

### Option 2: Deploy to Streamlit Community Cloud (FREE) ⭐

1. **Push to GitHub** - Make sure your code is on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with GitHub
4. **Click** "New app"
5. **Select** your repository
6. **Set** main file path to `btc_app.py`
7. **Click** "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`!

### Option 3: Docker

```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -t crypto-predictor .
docker run -p 8501:8501 crypto-predictor
```

Then open http://localhost:8501 in your browser.

---

## 🗂️ Project Structure

```
Time-Series-Forecasting-using-historical-BTC/
│
├── 📁 src/
│   ├── __init__.py
│   └── model.py              # Core prediction utilities
│
├── 📁 models/                # Trained models (auto-generated)
│   ├── btc_lstm_model.keras
│   ├── eth_lstm_model.keras
│   ├── sol_lstm_model.keras
│   ├── xrp_lstm_model.keras
│   ├── ada_lstm_model.keras
│   └── training_metrics.json
│
├── 📁 scripts/
│   └── train_model.py        # Automated training script
│
├── 📁 notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── btc_lstm_model.keras  # Legacy pre-trained model
│
├── 📁 .github/workflows/
│   └── retrain.yml           # Daily auto-retraining
│
├── 📁 .streamlit/
│   └── config.toml           # Streamlit theme configuration
│
├── 🐳 Dockerfile             # Docker configuration
├── 🐳 docker-compose.yml     # Docker Compose setup
├── 📄 btc_app.py             # Main Streamlit application
├── 📄 requirements.txt       # Python dependencies
├── 📄 env.example            # Environment variables template
├── 📄 .gitignore             # Git ignore rules
└── 📄 README.md              # This file
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Neural Network                       │
├─────────────────────────────────────────────────────────────┤
│  Input Layer:     (60 timesteps, 1 feature)                 │
│  ↓                                                          │
│  LSTM Layer 1:    50 units, return_sequences=True           │
│  Dropout:         20%                                       │
│  ↓                                                          │
│  LSTM Layer 2:    50 units, return_sequences=False          │
│  Dropout:         20%                                       │
│  ↓                                                          │
│  Dense Layer:     25 units, ReLU activation                 │
│  ↓                                                          │
│  Output Layer:    1 unit (predicted price)                  │
└─────────────────────────────────────────────────────────────┘
```

**Training Details:**
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 20
- **Batch Size:** 32
- **Train/Test Split:** 80/20

---

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **RMSE** | ~$2,781 | Root Mean Squared Error |
| **MAE** | ~$2,100 | Mean Absolute Error |
| **RMSE %** | ~3.3% | Error as percentage of avg price |
| **Direction Accuracy** | ~55% | Correct up/down predictions |

> ⚠️ **Note:** Performance may vary based on market conditions and the date range used for training.

---

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
# Model Configuration
MODEL_PATH=notebooks/btc_lstm_model.keras
SEQUENCE_LENGTH=60

# Data Configuration
DEFAULT_CRYPTO=BTC-USD
START_DATE=2020-01-01

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
```

### Supported Cryptocurrencies

| Symbol | Name | Color |
|--------|------|-------|
| BTC-USD | Bitcoin | 🟠 |
| ETH-USD | Ethereum | 🔵 |
| SOL-USD | Solana | 🟢 |
| XRP-USD | Ripple | 🔵 |
| ADA-USD | Cardano | 🔵 |

---

## 🛠️ Development

### Training Your Own Model

1. Open `notebooks/EDA.ipynb` in Jupyter
2. Modify the date range, hyperparameters, or add features
3. Run all cells to train the model
4. The model will be saved to `notebooks/btc_lstm_model.keras`

### Adding New Cryptocurrencies

Edit `src/model.py` and add to the `SUPPORTED_CRYPTOS` dictionary:

```python
SUPPORTED_CRYPTOS = {
    "BTC-USD": {"name": "Bitcoin", "symbol": "BTC", "color": "#F7931A"},
    "YOUR-CRYPTO": {"name": "Your Crypto", "symbol": "YC", "color": "#HEXCOLOR"},
    # ...
}
```

### Running Tests

```bash
# Run tests (coming soon)
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contributions

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement sentiment analysis from news/social media
- [ ] Add backtesting functionality
- [ ] Create API endpoint with FastAPI
- [ ] Add more model architectures (GRU, Transformer)
- [ ] Implement ensemble predictions
- [x] Add email alerts for training completion
- [x] Daily automated retraining via GitHub Actions

---

## 🔄 Automated Retraining

This project includes **automated daily model retraining** via GitHub Actions!

### How It Works

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  6 AM UTC  │────▶│  Fetch     │────▶│  Train     │────▶│  Deploy    │
│  Trigger   │     │  Latest    │     │  All       │     │  Auto via  │
│            │     │  Data      │     │  Models    │     │  Commit    │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
```

### Setup Email Notifications

To receive email notifications when training completes or fails:

1. **Go to** your GitHub repository → Settings → Secrets and variables → Actions
2. **Add these secrets:**

| Secret Name | Value |
|-------------|-------|
| `EMAIL_USERNAME` | Your Gmail address |
| `EMAIL_PASSWORD` | Gmail App Password (not regular password!) |
| `NOTIFICATION_EMAIL` | Where to send notifications |

**To create a Gmail App Password:**
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Step Verification
3. Go to App passwords
4. Create a new app password for "Mail"

### Manual Retraining

You can also trigger retraining manually:
1. Go to **Actions** tab in your GitHub repo
2. Select **"🔄 Daily Model Retraining"**
3. Click **"Run workflow"**

---

## ⚠️ Disclaimer

**IMPORTANT:** This application is for **educational and informational purposes only**.

- 🚫 This is **NOT** financial advice
- 📉 Cryptocurrency markets are highly volatile and unpredictable
- 📊 Past performance does not guarantee future results
- 💡 Always do your own research (DYOR) before investing
- 💰 Never invest more than you can afford to lose

The creators of this application are not responsible for any financial losses.

---

## 📚 What I Learned

Building this project taught me:

- ✅ Time series forecasting fundamentals
- ✅ Sliding window technique for sequence modeling
- ✅ Building LSTM models with Keras & TensorFlow
- ✅ Data preprocessing with MinMaxScaler
- ✅ Creating interactive dashboards with Streamlit
- ✅ Deploying ML applications with Docker
- ✅ Building production-ready Python packages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing free financial data
- [TensorFlow](https://www.tensorflow.org/) for the amazing ML framework
- [Streamlit](https://streamlit.io/) for making ML deployment easy
- The open-source community for inspiration and support

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/Sandrakimiring">Sandra Kimiring</a></p>
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
