# 🤖 StockAI: AI-Powered Stock Investment Advisor

An advanced machine learning system that analyzes historical stock market data to recommend the best stocks for investment based on multiple factors including technical indicators, fundamental analysis, and ensemble machine learning predictions.

## 🌟 Features

- **10 Years of Historical Data**: Comprehensive analysis using up to 10 years of daily OHLCV data per stock
- **80+ Technical & Fundamental Indicators**: 
  - Moving averages (SMA, EMA, WMA at multiple windows)
  - Momentum indicators (RSI, MACD, Stochastic, Williams %R, CCI, Aroon)
  - Volatility measures (ATR, Bollinger Bands, standard deviation)
  - Volume analysis (OBV, PVT, AD Line, MFI, Price-Volume correlations)
  - Valuation metrics (P/E, P/B, P/S, EV/EBITDA, etc.)
  - Profitability ratios (ROE, ROA, ROIC, margins)
  - Growth metrics (Revenue growth, EPS growth, etc.)
  - Risk metrics (Beta, Sharpe ratio, drawdown)

- **Ensemble ML with 6 Algorithms**: 
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Ridge Regression
  - Support Vector Regression (SVR)
  - Neural Networks (MLP)

- **Multi-Horizon Predictions**: Predicts returns at 1-day, 5-day, 10-day, 20-day, and 60-day horizons
- **Advanced Feature Engineering**: 7,000+ training samples per analysis run
- **Cross-Validation**: Time series splits to prevent data leakage
- **Portfolio Optimization**: Allocates capital based on investment scores and risk tolerance
- **Risk Assessment**: Beta-based risk scoring and sector diversification analysis
- **Comprehensive Visualization**: Multiple charts and dashboards for analysis results

## 📋 System Overview

```
┌─────────────────────┐
│ Data Collection     │  ← Fetch 400 stocks via yfinance
├─────────────────────┤
│ Feature Engineering │  ← Compute 162 technical/fundamental features
├─────────────────────┤
│ Data Preparation    │  ← KNN imputation, outlier handling, scaling
├─────────────────────┤
│ ML Model Training   │  ← GridSearchCV for 6 models + Voting Ensemble
├─────────────────────┤
│ Investment Scoring  │  ← Combine ML + Technical + Fundamental + Risk
├─────────────────────┤
│ Recommendations     │  ← Top 15 stocks with actionable insights
├─────────────────────┤
│ Portfolio Building  │  ← Optimal allocation across sectors
└─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip
git
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StockAI.git
cd StockAI

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
python stock_investment_ai.py
```

The script will:
1. Fetch stock data for ~400 stocks (10 years of history)
2. Generate 7,000+ feature samples
3. Train ML models for 1D, 5D, and 20D horizons
4. Generate top 15 investment recommendations
5. Create an optimized portfolio allocation
6. Display visualizations and generate a comprehensive report

**Estimated runtime**: 30-60 minutes on a modern laptop (depends on network and CPU)

## 📊 Key Parameters & Customization

Edit the `main()` function in `stock_investment_ai.py`:

```python
def main():
    # Change data period
    ai = StockInvestmentAI(period='5y')  # Use 5 years instead of 10y
    
    # Adjust analysis parameters
    recommendations, portfolio = ai.run_analysis(
        investment_amount=50000,      # Change portfolio size
        risk_tolerance='aggressive',  # Options: 'conservative', 'moderate', 'aggressive'
        top_n=20,                     # Get top 20 instead of 15
        target_horizons=['5D', '20D'] # Focus on specific horizons
    )
```

### To Modify Stock Universe

Edit the `get_sp500_symbols()` method to customize the stock list:

```python
def get_sp500_symbols(self):
    return [
        # Add your custom list of tickers
        'AAPL', 'MSFT', 'GOOGL', ...
    ]
```

### To Change Prediction Horizons

In `create_features_dataset()`, modify:

```python
prediction_horizons = [1, 5, 10, 20, 60]  # Change to [1, 5, 10] for shorter horizons
```

## 📈 Output

The script produces:

1. **Console Output**: Real-time progress, model performance metrics, top recommendations
2. **Top Recommendations**: 15 stocks with:
   - Final investment score
   - Expected return (20D horizon)
   - Key metrics (P/E, ROE, Beta, Dividend Yield)
   - Score breakdown (ML + Technical + Fundamental + Risk)
   - Investment reasoning

3. **Portfolio Allocation**: 
   - Recommended holdings with allocation percentages
   - Share quantities and capital allocation
   - Cash reserve

4. **Visualizations**:
   - Top 15 stocks by score (bar chart)
   - Score breakdown by component (stacked bar chart)
   - Risk vs. Return scatter plot
   - Sector diversification (pie chart)
   - Market cap distribution (histogram)
   - Valuation analysis (P/E vs. Returns)

5. **Comprehensive Report**:
   - Analysis date and data period
   - Number of stocks and features analyzed
   - Model performance summary
   - Risk assessment
   - Key insights and improvements

## 🏗️ Architecture

### Main Class: `StockInvestmentAI`

**Key Methods:**
- `fetch_stock_data()`: Downloads historical data from yfinance
- `calculate_technical_indicators()`: Computes 150+ technical indicators
- `calculate_fundamental_features()`: Extracts 50+ fundamental metrics
- `create_features_dataset()`: Combines all features into training data (7,000+ samples)
- `train_ml_model()`: Trains and validates ensemble models with GridSearchCV
- `calculate_investment_score()`: Computes composite investment scores
- `generate_recommendations()`: Produces top stock recommendations
- `create_portfolio_allocation()`: Builds optimal portfolio
- `plot_enhanced_analysis()`: Creates comprehensive visualizations

### Data Pipeline

```
Stock Data (OHLCV)
    ↓
Technical Indicators (80+)
Fundamental Features (50+)
    ↓
Feature Dataset (7,260 samples × 162 features)
    ↓
KNN Imputation → Outlier Handling → StandardScaling
    ↓
Feature Selection (100 best features)
    ↓
Train/Test Split (80/20)
    ↓
Model Training (RF, GB, XGB, Ridge, SVR, MLP)
    ↓
Voting Ensemble
    ↓
Predictions & Confidence Scores
    ↓
Investment Scores (ML + Technical + Fundamental + Risk)
    ↓
Recommendations & Portfolio Allocation
```

## 📊 Model Performance

Typical R² scores achieved:
- **1-Day Horizon**: 0.06-0.08
- **5-Day Horizon**: 0.07-0.10
- **20-Day Horizon**: 0.15-0.25 (best predictability)

Note: Lower R² on shorter horizons is expected due to market noise and limited predictability.

## 🎯 Scoring Methodology

**Final Score = Base Score × ML Confidence**

Where **Base Score** = 
- 40% × ML Score (predicted return)
- 25% × Technical Score (RSI, MACD, MA, Bollinger Bands)
- 20% × Fundamental Score (P/E, ROE, Growth, Margins)
- 15% × Risk Score (Volatility, Beta, Sharpe Ratio)

**ML Confidence** = 1 / (1 + std_dev of model predictions)

## ⚙️ Dependencies

See `requirements.txt`:
- yfinance: Stock data retrieval
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: ML models and preprocessing
- xgboost: Gradient boosting
- ta: Technical analysis indicators
- matplotlib, seaborn: Visualization
- scipy: Statistical functions

## 🔧 Performance Optimization

For faster runs on limited hardware:

```python
# Reduce lookback periods (fewer samples)
lookback_periods = [30, 60, 120, 252]  # Instead of 20 different periods

# Reduce stock universe
symbols = ai.get_sp500_symbols()[:100]  # First 100 instead of 400

# Reduce GridSearchCV complexity
# In train_ml_model(), reduce parameter grids:
rf_params = {
    'n_estimators': [100],
    'max_depth': [10],
}

# Use shorter historical period
ai = StockInvestmentAI(period='2y')  # 2 years instead of 10y

# Disable less important models
# Remove SVR and MLP from ensemble in train_ml_model()
```

## 📝 Example Output

```
🤖 ENHANCED AI STOCK INVESTMENT RECOMMENDATIONS
================================================================
📊 Based on analysis of 363 stocks
🎯 Prediction horizon: 20D
📅 Analysis date: 2025-11-17 18:30:29
================================================================

 1.    FTI - Final Score: 0.519
    💰 Current Price: $20.70
    📈 Expected Return (20D): +1.2%
    📊 Market Cap: $17.3B (Large Cap)
    🏭 Sector: Energy
    📈 Key Metrics: P/E: 19.2 | ROE: 29.3% | Risk: Low (β=0.68) | Div: 46.0%
    🎯 Score Breakdown - ML: 0.012 | Tech: 0.867 | Fund: 0.889 | Risk: 0.858
    💡 Why recommended: Positive technical signals, Strong fundamentals, Favorable risk profile
```

## 🚨 Important Notes

1. **Disclaimer**: This is an educational and research tool. Not financial advice. Use at your own risk.
2. **Past Performance**: Historical performance does not guarantee future results.
3. **Data Quality**: Depends on yfinance data availability and accuracy.
4. **Market Conditions**: Models are trained on historical data and may not adapt quickly to regime changes.
5. **Risk**: Always conduct your own due diligence before investing.

## 🐛 Troubleshooting

### UTF-8 Encoding Error on Windows
```bash
# Run this command in PowerShell before executing the script
chcp 65001
```

### Memory Issues with Large Datasets
- Reduce number of stocks: `symbols[:100]`
- Reduce lookback periods
- Use shorter historical period
- Consider using SimpleImputer instead of KNNImputer

### Slow Execution
- Reduce GridSearchCV complexity
- Use RandomizedSearchCV instead
- Reduce feature set size
- Run on GPU-enabled machine (optional)

## 📚 Resources & References

- yfinance Documentation: https://github.com/ranaroussi/yfinance
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Technical Analysis Library (TA): https://technical-analysis-library-in-python.readthedocs.io/

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

StockAI Development Team

---

**Last Updated**: November 17, 2025  
**Version**: 1.0.0
