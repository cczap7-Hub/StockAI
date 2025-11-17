"""
AI Stock Investment Advisor - User Guide
=======================================

OVERVIEW:
This AI-powered stock investment system analyzes historical market data using machine learning
to recommend the best stocks for investment. It combines technical analysis, fundamental 
analysis, and predictive modeling to generate investment recommendations.

FEATURES:
✅ Real-time stock data fetching (via Yahoo Finance)
✅ Technical analysis (moving averages, volatility, price trends)
✅ Fundamental analysis (P/E ratios, ROE, revenue growth, etc.)
✅ Machine learning predictions (Gradient Boosting)
✅ Risk assessment and scoring
✅ Automated portfolio allocation
✅ Multiple risk tolerance strategies

HOW TO USE:
===========

1. BASIC USAGE:
--------------
from simple_stock_ai import SimpleStockAI

# Create AI instance
ai = SimpleStockAI(symbols=['AAPL', 'MSFT', 'GOOGL'], period='1y')

# Run complete analysis
recommendations, portfolio = ai.run_analysis(investment_amount=10000, top_n=5)

2. CUSTOM STOCK SELECTION:
-------------------------
# Tech stocks
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
ai = SimpleStockAI(symbols=tech_stocks, period='2y')

# Healthcare stocks
health_stocks = ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO']
ai = SimpleStockAI(symbols=health_stocks, period='1y')

3. STEP-BY-STEP ANALYSIS:
------------------------
ai = SimpleStockAI(symbols=['AAPL', 'MSFT', 'GOOGL'])

# Step 1: Fetch data
ai.fetch_stock_data()

# Step 2: Calculate features
features = ai.calculate_features()
print(features.head())

# Step 3: Train model
model = ai.train_model()

# Step 4: Get recommendations
recommendations = ai.generate_recommendations(top_n=5)

# Step 5: Create portfolio
portfolio = ai.create_portfolio(investment_amount=15000, top_n=3)

SCORING SYSTEM:
==============

The AI uses a comprehensive scoring system that combines:

1. MACHINE LEARNING SCORE (40%):
   - Predicts future 30-day returns using historical patterns
   - Uses gradient boosting algorithm
   - Considers price movements, volatility, and volume

2. TECHNICAL SCORE (30%):
   - Price vs moving averages (SMA 20, SMA 50)
   - Recent price stability
   - Trend momentum

3. FUNDAMENTAL SCORE (20%):
   - P/E ratio (optimal range: 5-30)
   - Return on Equity (ROE > 10% preferred)
   - Revenue growth (positive growth preferred)

4. RISK SCORE (10%):
   - Beta (0.5-1.5 is optimal)
   - Volatility (lower is better)
   - Market stability

EXAMPLE OUTPUTS:
===============

RECOMMENDATIONS:
1. GOOGL - Score: 0.578
   💰 Price: $239.86
   📊 P/E: 25.5
   ⚖️  Beta: 1.01

2. AAPL - Score: 0.557
   💰 Price: $233.64
   📊 P/E: 35.5
   ⚖️  Beta: 1.11

PORTFOLIO ALLOCATION:
 GOOGL:   6 shares × $ 239.86 = $ 1439.16 (33.3%)
  AAPL:   7 shares × $ 233.64 = $ 1635.51 (33.3%)
  AMZN:   6 shares × $ 238.70 = $ 1432.17 (33.3%)

ADVANCED USAGE:
==============

1. SECTOR ANALYSIS:
# Compare different sectors
sectors = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS'],
    'healthcare': ['JNJ', 'PFE', 'UNH', 'ABT']
}

results = {}
for sector, stocks in sectors.items():
    ai = SimpleStockAI(symbols=stocks)
    recs, portfolio = ai.run_analysis(investment_amount=10000)
    results[sector] = recs

2. TIME PERIOD COMPARISON:
# Compare different time periods
periods = ['6mo', '1y', '2y']
for period in periods:
    ai = SimpleStockAI(symbols=['AAPL', 'MSFT'], period=period)
    recs, _ = ai.run_analysis()
    print(f"Period {period}: Top stock = {recs.iloc[0]['Symbol']}")

3. RISK ANALYSIS:
# Analyze high beta (risky) vs low beta (stable) stocks
high_beta = ['TSLA', 'NVDA', 'AMD']  # Typically higher volatility
low_beta = ['KO', 'PG', 'JNJ']       # Typically more stable

for stocks, risk_type in [(high_beta, 'HIGH'), (low_beta, 'LOW')]:
    ai = SimpleStockAI(symbols=stocks)
    recs, _ = ai.run_analysis()
    avg_beta = recs['Beta'].mean()
    print(f"{risk_type} RISK - Average Beta: {avg_beta:.2f}")

TIPS FOR BEST RESULTS:
=====================

1. USE DIVERSE STOCKS: Include stocks from different sectors
2. LONGER PERIODS: Use 1-2 year periods for more stable analysis
3. REGULAR UPDATES: Re-run analysis weekly/monthly for fresh data
4. CONSIDER MARKET CONDITIONS: Results may vary in bull vs bear markets
5. COMBINE WITH RESEARCH: Use AI recommendations as starting point, not final decision

LIMITATIONS:
===========

⚠️  Past performance doesn't guarantee future results
⚠️  Model accuracy depends on data quality and market conditions
⚠️  External factors (news, events) not included in analysis
⚠️  Requires internet connection for real-time data
⚠️  Yahoo Finance API rate limits may affect large analyses

TROUBLESHOOTING:
===============

PROBLEM: "No data available"
SOLUTION: Check internet connection, verify stock symbols are correct

PROBLEM: "Insufficient data" 
SOLUTION: Use longer time periods or different stocks

PROBLEM: Low R² score
SOLUTION: Normal for stock prediction; focus on relative rankings

PROBLEM: API rate limits
SOLUTION: Reduce number of stocks or add delays between requests

EXAMPLE COMPLETE WORKFLOW:
=========================
"""

# Example: Complete investment analysis workflow
from simple_stock_ai import SimpleStockAI
import pandas as pd

def complete_investment_analysis():
    """Example of a complete investment analysis workflow with extended history and more stocks"""
    
    print("🏆 COMPLETE INVESTMENT ANALYSIS WORKFLOW")
    print("=" * 50)
    
    # 1. Define investment parameters
    investment_amount = 25000
    risk_tolerance = 'moderate'  # conservative, moderate, aggressive
    
    # 2. Use expanded stock universe (can specify custom list or use default 100)
    # Option 1: Use default 100 stocks from SimpleStockAI
    # Option 2: Define custom diverse portfolio
    custom_stocks = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM',
        # Finance  
        'JPM', 'BAC', 'V', 'MA', 'BLK', 'GS',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'MRNA',
        # Consumer & Retail
        'WMT', 'KO', 'PG', 'NKE', 'HD', 'DIS',
        # Energy & Utilities
        'XOM', 'CVX', 'NEE', 'DUK',
        # Industrial
        'BA', 'CAT', 'MMM', 'HON'
    ]
    
    # 3. Run analysis with 10 years of historical data for ultimate depth
    print("Using 10 years of historical data for comprehensive long-term analysis...")
    ai = SimpleStockAI(symbols=custom_stocks, period='10y')  # or use default 100 stocks
    recommendations, portfolio = ai.run_analysis(
        investment_amount=investment_amount, 
        top_n=12
    )
    
    # 4. Display detailed results
    print(f"\n📊 DETAILED ANALYSIS RESULTS:")
    print(f"Investment Amount: ${investment_amount:,}")
    print(f"Stocks Analyzed: {len(ai.stock_data)}")
    print(f"Top Recommendation: {recommendations.iloc[0]['Symbol']}")
    print(f"Portfolio Diversity: {len(portfolio)} stocks")
    
    # 5. Risk assessment
    avg_beta = recommendations.head(5)['Beta'].mean()
    risk_level = "LOW" if avg_beta < 1 else "HIGH" if avg_beta > 1.3 else "MEDIUM"
    print(f"Portfolio Risk Level: {risk_level} (avg β={avg_beta:.2f})")
    
    # 6. Return the results for further analysis
    return ai, recommendations, portfolio

if __name__ == "__main__":
    # Run the complete analysis
    ai, recommendations, portfolio = complete_investment_analysis()
    
    print("\n🎯 ANALYSIS COMPLETE!")
    print("Use the 'recommendations' and 'portfolio' dataframes for detailed review.")
    print("Remember: This is for educational purposes. Always consult a financial advisor!")
