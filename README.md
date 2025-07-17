# Stock Price Predictor with Sentiment Analysis

A comprehensive stock price prediction application that combines machine learning models with news sentiment analysis to predict S&P 500 stock prices.

## Features

- **Multiple ML Models**: LSTM, Random Forest, XGBoost, and Ensemble predictions
- **Sentiment Analysis**: Real-time news sentiment analysis affecting stock prices
- **Technical Indicators**: 20+ technical indicators including RSI, MACD, Bollinger Bands
- **Interactive Dashboard**: Beautiful Streamlit interface with real-time visualizations
- **S&P 500 Coverage**: Pre-configured with top 50 S&P 500 stocks

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for training)
- Internet connection for data fetching

## Installation

1. **Clone or create the project directory**:
```bash
mkdir stock-predictor
cd stock-predictor
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## API Setup

### NewsAPI (Required for sentiment analysis)

1. Visit [https://newsapi.org/](https://newsapi.org/)
2. Click "Get API Key"
3. Register with your email (free tier = 100 requests/day)
4. Copy your API key

### Setting up API keys

Create a `.env` file in the project root:
```bash
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alphavantage_key_here  # Optional
```

Or modify `config.py` directly:
```python
NEWS_API_KEY = 'your_actual_api_key_here'
```

## Running the Application

1. **Start the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

## How to Use

### Step 1: Select a Stock
- Use the sidebar to select a stock symbol (e.g., UNH, AAPL, MSFT)
- Adjust the historical data range and prediction horizon

### Step 2: Fetch Data
- Click "🔄 Fetch Data" to download stock data and analyze news sentiment
- The app will display current price, volume, sentiment scores, and technical indicators

### Step 3: Train Models
- Click "🎯 Train Model" to train all ML models on the historical data
- Training takes 1-3 minutes depending on your hardware
- Model performance metrics will be displayed

### Step 4: Make Predictions
- Select your preferred model (Ensemble recommended)
- Click "🔮 Make Prediction" to generate future price predictions
- View predictions with confidence intervals

## Understanding the Output

### Metrics Displayed:
- **Current Price**: Latest closing price
- **Volume**: Trading volume with day-over-day change
- **Sentiment Score**: Aggregated news sentiment (-1 to +1)
- **Technical Indicators**: RSI, MACD, Moving Averages, etc.

### Prediction Results:
- **Price Chart**: Historical prices with future predictions
- **Confidence Interval**: Uncertainty range for predictions
- **Recommendation**: Buy/Hold/Sell signal based on predicted trend
- **Risk Metrics**: Volatility and drawdown calculations

## Model Details

### LSTM (Long Short-Term Memory)
- Deep learning model for time series
- 3-layer architecture with dropout
- Best for capturing long-term dependencies

### Random Forest
- Ensemble of decision trees
- Good for feature importance analysis
- Robust to outliers

### XGBoost
- Gradient boosting algorithm
- Fast and accurate
- Handles non-linear relationships well

### Ensemble Model
- Weighted combination of all models
- Generally provides best performance
- Reduces individual model bias

## Troubleshooting

### Common Issues:

1. **"No module named 'package_name'"**
   - Run: `pip install -r requirements.txt`

2. **NewsAPI Error**
   - Check your API key is valid
   - Ensure you haven't exceeded the free tier limit (100 requests/day)

3. **Memory Error during training**
   - Reduce the historical data range
   - Close other applications
   - Use a smaller batch size

4. **No data for symbol**
   - Ensure the symbol is correct and traded on major exchanges
   - Check your internet connection

## Performance Tips

1. **For faster training**:
   - Use shorter historical periods (90-180 days)
   - Select fewer technical indicators
   - Use Random Forest or XGBoost instead of LSTM

2. **For better accuracy**:
   - Use more historical data (1-2 years)
   - Enable all technical indicators
   - Use the Ensemble model
   - Ensure good quality news data

## Data Sources

- **Stock Data**: Yahoo Finance (via yfinance)
- **News Data**: NewsAPI and Yahoo Finance
- **No database required**: Uses local caching

## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and do your own research.

## Future Enhancements

- [ ] Add more stocks beyond S&P 500
- [ ] Include options data
- [ ] Add portfolio optimization
- [ ] Real-time streaming predictions
- [ ] Mobile app version
- [ ] Backtesting framework

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify API keys are properly configured

## License

This project is for educational purposes. Feel free to modify and enhance!