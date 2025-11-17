"""
Simple Test of AI Stock Investment System
========================================
A basic test to ensure the system works correctly.
"""

from stock_investment_ai import StockInvestmentAI

def simple_test():
    """Simple test with just a few stocks"""
    print("🧪 SIMPLE TEST: AI Stock Analysis")
    print("=" * 40)
    
    # Test with just 5 popular stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"Testing with stocks: {', '.join(test_stocks)}")
    print("This may take a few minutes to download data and analyze...")
    
    try:
        # Initialize AI
        ai = StockInvestmentAI(symbols=test_stocks, period='1y')
        
        # Run analysis
        recommendations, portfolio = ai.run_analysis(
            investment_amount=5000,
            risk_tolerance='moderate',
            top_n=3
        )
        
        print("\n✅ Test completed successfully!")
        print(f"Analyzed {len(ai.stock_data)} stocks")
        print(f"Top recommendation: {recommendations.iloc[0]['Symbol']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 The AI Stock Investment System is working correctly!")
    else:
        print("\n⚠️ There was an issue with the system. Check your internet connection.")
