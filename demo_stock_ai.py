"""
Demo Script for AI Stock Investment Advisor
==========================================
This script demonstrates how to use the AI Stock Investment Advisor
with different configurations and examples.
"""

from stock_investment_ai import StockInvestmentAI
import pandas as pd

def demo_basic_analysis():
    """Demo: Basic stock analysis with default settings"""
    print("🔥 DEMO 1: Basic Analysis with Top Tech Stocks")
    print("=" * 50)
    
    # Create AI with specific tech stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
    ai = StockInvestmentAI(symbols=tech_stocks, period='1y')
    
    # Run analysis
    recommendations, portfolio = ai.run_analysis(
        investment_amount=10000,
        risk_tolerance='moderate',
        top_n=5
    )
    
    return ai, recommendations, portfolio

def demo_conservative_portfolio():
    """Demo: Conservative investment strategy"""
    print("\n\n💼 DEMO 2: Conservative Portfolio (Blue Chip Stocks)")
    print("=" * 50)
    
    # Conservative blue chip stocks
    blue_chip_stocks = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'V', 'JPM', 'WMT', 'PFE', 'HD']
    ai = StockInvestmentAI(symbols=blue_chip_stocks, period='2y')
    
    # Run conservative analysis
    recommendations, portfolio = ai.run_analysis(
        investment_amount=50000,
        risk_tolerance='conservative',
        top_n=8
    )
    
    return ai, recommendations, portfolio

def demo_aggressive_portfolio():
    """Demo: Aggressive growth investment strategy"""
    print("\n\n🚀 DEMO 3: Aggressive Growth Portfolio")
    print("=" * 50)
    
    # Growth and emerging stocks
    growth_stocks = ['TSLA', 'NVDA', 'AMD', 'PYPL', 'SQ', 'ROKU', 'ZM', 'PTON', 'SNOW', 'PLTR']
    ai = StockInvestmentAI(symbols=growth_stocks, period='1y')
    
    # Run aggressive analysis
    recommendations, portfolio = ai.run_analysis(
        investment_amount=25000,
        risk_tolerance='aggressive',
        top_n=4
    )
    
    return ai, recommendations, portfolio

def demo_custom_analysis():
    """Demo: Custom analysis with user-defined parameters"""
    print("\n\n⚙️ DEMO 4: Custom Analysis")
    print("=" * 50)
    
    # Mixed portfolio of different sectors
    mixed_stocks = ['AAPL', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'NKE', 'DIS', 'BA', 'GE', 'F']
    ai = StockInvestmentAI(symbols=mixed_stocks, period='2y')
    
    # Custom step-by-step analysis
    print("Step 1: Fetching data...")
    ai.fetch_stock_data()
    
    print("Step 2: Creating features...")
    features_df = ai.create_features_dataset()
    print(f"Features created: {features_df.shape}")
    print(f"Sample features:\n{features_df[['Symbol', 'Current_Price', 'RSI', 'PE_Ratio']].head()}")
    
    print("Step 3: Training ML model...")
    model = ai.train_ml_model()
    
    print("Step 4: Generating recommendations...")
    recommendations = ai.generate_recommendations(top_n=6)
    
    print("Step 5: Creating portfolio...")
    portfolio = ai.create_portfolio_allocation(investment_amount=15000, risk_tolerance='moderate')
    
    return ai, recommendations, portfolio

def demo_sector_analysis():
    """Demo: Sector-specific analysis"""
    print("\n\n🏭 DEMO 5: Sector Analysis - Healthcare vs Technology")
    print("=" * 50)
    
    # Healthcare stocks
    healthcare_stocks = ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB']
    ai_healthcare = StockInvestmentAI(symbols=healthcare_stocks, period='1y')
    
    print("Healthcare Sector Analysis:")
    health_recs, health_portfolio = ai_healthcare.run_analysis(
        investment_amount=20000,
        risk_tolerance='conservative',
        top_n=5
    )
    
    # Technology stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'IBM']
    ai_tech = StockInvestmentAI(symbols=tech_stocks, period='1y')
    
    print("\n\nTechnology Sector Analysis:")
    tech_recs, tech_portfolio = ai_tech.run_analysis(
        investment_amount=20000,
        risk_tolerance='moderate',
        top_n=5
    )
    
    return ai_healthcare, ai_tech

def demo_risk_comparison():
    """Demo: Compare different risk tolerance strategies"""
    print("\n\n⚖️ DEMO 6: Risk Tolerance Comparison")
    print("=" * 50)
    
    # Same stocks, different risk strategies
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'JNJ', 'V', 'WMT']
    investment_amount = 10000
    
    risk_levels = ['conservative', 'moderate', 'aggressive']
    results = {}
    
    for risk_level in risk_levels:
        print(f"\n--- {risk_level.upper()} STRATEGY ---")
        ai = StockInvestmentAI(symbols=stocks, period='1y')
        
        # Simplified analysis (no plots for comparison)
        ai.fetch_stock_data()
        ai.create_features_dataset()
        ai.train_ml_model()
        recommendations = ai.generate_recommendations(top_n=8)
        portfolio = ai.create_portfolio_allocation(investment_amount, risk_level)
        
        results[risk_level] = {
            'recommendations': recommendations,
            'portfolio': portfolio,
            'ai': ai
        }
    
    return results

def print_summary_comparison(results):
    """Print a summary comparison of different strategies"""
    print("\n\n📊 STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    
    for risk_level, data in results.items():
        portfolio = data['portfolio']
        recommendations = data['recommendations']
        
        print(f"\n{risk_level.upper()} STRATEGY:")
        print(f"  • Number of stocks: {len(portfolio)}")
        print(f"  • Top stock: {recommendations.iloc[0]['Symbol']} (Score: {recommendations.iloc[0]['Final_Score']:.3f})")
        print(f"  • Largest allocation: {portfolio.iloc[0]['Allocation_%']:.1f}% to {portfolio.iloc[0]['Symbol']}")
        print(f"  • Total invested: ${portfolio['Amount'].sum():,.2f}")

def main():
    """Run all demos"""
    print("🤖 AI STOCK INVESTMENT ADVISOR - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases different ways to use the AI system\n")
    
    # Run different demos
    try:
        # Basic analysis
        demo1_ai, demo1_recs, demo1_portfolio = demo_basic_analysis()
        
        # Conservative portfolio
        demo2_ai, demo2_recs, demo2_portfolio = demo_conservative_portfolio()
        
        # Aggressive portfolio  
        demo3_ai, demo3_recs, demo3_portfolio = demo_aggressive_portfolio()
        
        # Custom analysis
        demo4_ai, demo4_recs, demo4_portfolio = demo_custom_analysis()
        
        # Sector analysis
        health_ai, tech_ai = demo_sector_analysis()
        
        # Risk comparison
        risk_results = demo_risk_comparison()
        print_summary_comparison(risk_results)
        
        print("\n\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key Features Demonstrated:")
        print("✅ Basic stock analysis and recommendations")
        print("✅ Different risk tolerance strategies")
        print("✅ Sector-specific analysis")
        print("✅ Custom portfolio allocation")
        print("✅ Machine learning-based predictions")
        print("✅ Technical and fundamental analysis")
        print("✅ Risk assessment and scoring")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        print("This might be due to network issues or API rate limits.")
        print("Try running individual demos or with fewer stocks.")

if __name__ == "__main__":
    main()
