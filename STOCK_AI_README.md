# 🤖 AI-Powered Stock Investment Advisor

An intelligent stock analysis and investment recommendation system that uses machine learning to analyze historical market data and suggest optimal stock investments based on multiple factors.

## 🌟 Features

- **Real-time Data Analysis**: Fetches live stock data from Yahoo Finance
- **Machine Learning Predictions**: Uses Gradient Boosting to predict future returns
- **Technical Analysis**: Moving averages, volatility, price trends
- **Fundamental Analysis**: P/E ratios, ROE, revenue growth, debt ratios
- **Risk Assessment**: Beta analysis, volatility scoring
- **Portfolio Optimization**: Automated allocation based on investment amount
- **Multi-factor Scoring**: Combines ML predictions with technical and fundamental analysis

## 📁 Files

- `simple_stock_ai.py` - Main AI system (recommended for use)
- `stock_investment_ai.py` - Advanced version with more features
- `ai_stock_guide.py` - Complete usage guide and examples
- `test_stock_ai.py` - Simple test script
- `demo_stock_ai.py` - Comprehensive demo with multiple scenarios

## 🚀 Quick Start

```python
from simple_stock_ai import SimpleStockAI

# Create AI instance with your chosen stocks
ai = SimpleStockAI(symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], period='1y')

# Run complete analysis
recommendations, portfolio = ai.run_analysis(investment_amount=10000, top_n=5)
```

## 📊 Sample Output

```
🤖 AI STOCK RECOMMENDATIONS
==================================================

1. GOOGL - Score: 0.578
   💰 Price: $239.71
   📊 P/E: 25.5
   ⚖️  Beta: 1.01

2. JNJ - Score: 0.561
   💰 Price: $177.07
   📊 P/E: 19.0
   ⚖️  Beta: 0.39

💼 PORTFOLIO ALLOCATION ($10,000)
 GOOGL:   8 shares × $ 239.71 = $ 1,917.72 (20.0%)
   JNJ:  11 shares × $ 177.07 = $ 1,947.77 (20.0%)
```

## 🧮 Scoring System

The AI uses a comprehensive 4-factor scoring system:

### 1. Machine Learning Score (40%)
- Predicts 30-day future returns using historical patterns
- Considers price movements, volatility, volume trends
- Uses Gradient Boosting algorithm

### 2. Technical Analysis Score (30%)
- Price vs moving averages (SMA 20, SMA 50)
- Recent price stability and momentum
- Volume patterns

### 3. Fundamental Analysis Score (20%)
- P/E ratio optimization (5-30 range preferred)
- Return on Equity (ROE > 10% preferred)
- Revenue growth trends
- Debt-to-equity ratios

### 4. Risk Assessment Score (10%)
- Beta analysis (0.5-1.5 optimal range)
- Volatility measurement
- Market stability factors

## 💡 Usage Examples

### Basic Analysis
```python
# Simple 5-stock analysis
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
ai = SimpleStockAI(symbols=stocks, period='1y')
recommendations, portfolio = ai.run_analysis(investment_amount=5000, top_n=3)
```

### Sector-Specific Analysis
```python
# Technology sector focus
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE']
ai = SimpleStockAI(symbols=tech_stocks, period='2y')
recommendations, portfolio = ai.run_analysis(investment_amount=15000, top_n=4)

# Healthcare sector focus  
health_stocks = ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO']
ai = SimpleStockAI(symbols=health_stocks, period='1y')
recommendations, portfolio = ai.run_analysis(investment_amount=10000, top_n=3)
```

### Step-by-Step Analysis
```python
ai = SimpleStockAI(symbols=['AAPL', 'MSFT', 'GOOGL'])

# Step 1: Fetch data
ai.fetch_stock_data()

# Step 2: Calculate features  
features = ai.calculate_features()
print(f"Analyzed {len(features)} stocks with {len(features.columns)} features")

# Step 3: Train model
model = ai.train_model()

# Step 4: Generate recommendations
recommendations = ai.generate_recommendations(top_n=5)

# Step 5: Create portfolio
portfolio = ai.create_portfolio(investment_amount=10000, top_n=3)
```

## ⚙️ Requirements

```bash
pip install yfinance pandas numpy matplotlib scikit-learn
```

## 🔧 Installation

1. Download all Python files to your directory
2. Install required packages:
   ```bash
   pip install yfinance pandas numpy matplotlib scikit-learn
   ```
3. Run the system:
   ```bash
   python simple_stock_ai.py
   ```

## ⚠️ Important Disclaimers

- **Educational Purpose**: This system is for learning and research only
- **Not Financial Advice**: Always consult with qualified financial advisors
- **Past Performance**: Historical data doesn't guarantee future results
- **Market Risk**: All investments carry risk of loss
- **Data Limitations**: Analysis based on available historical data only

**Remember**: This is a powerful tool for learning about quantitative finance and machine learning in investing, but always do your own research and consult professionals for actual investment decisions!
