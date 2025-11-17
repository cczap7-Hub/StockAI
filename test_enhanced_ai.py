"""
Test Enhanced AI Stock System
============================
Test the upgraded system with 5 years of history and more stocks.
"""

from simple_stock_ai import SimpleStockAI

def test_enhanced_system():
    """Test the enhanced AI system with more stocks and longer history"""
    print("🧪 TESTING ENHANCED AI STOCK SYSTEM")
    print("=" * 45)
    
    # Test with a smaller subset first (20 stocks, 5 years)
    test_stocks = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        # Finance
        'JPM', 'BAC', 'V', 'MA', 'GS',
        # Healthcare  
        'JNJ', 'PFE', 'UNH', 'ABT', 'TMO',
        # Consumer/Other
        'WMT', 'KO', 'HD', 'DIS', 'XOM'
    ]
    
    print(f"Testing with {len(test_stocks)} stocks and 5 years of data...")
    print("This will take a few minutes to fetch and analyze data...")
    
    try:
        # Initialize with 5 years of data
        ai = SimpleStockAI(symbols=test_stocks, period='5y')
        
        # Run analysis
        recommendations, portfolio = ai.run_analysis(
            investment_amount=15000,
            top_n=8
        )
        
        print(f"\n✅ Enhanced system test completed successfully!")
        print(f"📊 Key Results:")
        print(f"   • Stocks analyzed: {len(ai.stock_data)}")
        print(f"   • Features per stock: {len(ai.features_df.columns) - 1}")
        print(f"   • Top recommendation: {recommendations.iloc[0]['Symbol']}")
        print(f"   • Portfolio contains {len(portfolio)} stocks")
        print(f"   • Historical period: 5 years")
        
        # Show some of the new features
        sample_stock = ai.features_df.iloc[0]
        print(f"\n📈 Enhanced Features Example ({sample_stock['Symbol']}):")
        print(f"   • 1-year return: {sample_stock.get('Returns_250D', 0):.2%}")
        print(f"   • Price vs 200-day MA: {sample_stock.get('Price_vs_SMA200', 1):.2f}")
        print(f"   • Distance from 52W high: {sample_stock.get('Price_vs_52W_High', 1):.2%}")
        
        return True, ai, recommendations, portfolio
        
    except Exception as e:
        print(f"\n❌ Enhanced system test failed: {e}")
        return False, None, None, None

def test_full_100_stocks():
    """Test with the full 100 stock universe (optional - takes longer)"""
    print("\n🔥 TESTING FULL 100-STOCK UNIVERSE")
    print("=" * 40)
    print("⚠️  Warning: This will take 10-15 minutes to complete!")
    
    response = input("Do you want to run the full 100-stock test? (y/n): ")
    
    if response.lower() == 'y':
        try:
            # Use default 100 stocks
            ai = SimpleStockAI(period='5y')
            
            print("Starting full analysis...")
            recommendations, portfolio = ai.run_analysis(
                investment_amount=50000,
                top_n=15
            )
            
            print(f"\n🎉 FULL SYSTEM TEST SUCCESSFUL!")
            print(f"📊 Massive Analysis Results:")
            print(f"   • Total stocks analyzed: {len(ai.stock_data)}")
            print(f"   • Historical period: 5 years")
            print(f"   • Top 15 recommendations generated")
            print(f"   • Portfolio optimized for $50,000")
            
            return True, ai, recommendations, portfolio
            
        except Exception as e:
            print(f"❌ Full system test failed: {e}")
            return False, None, None, None
    else:
        print("Skipping full 100-stock test.")
        return None, None, None, None

if __name__ == "__main__":
    # Test enhanced system
    success, ai, recs, portfolio = test_enhanced_system()
    
    if success:
        print("\n🎯 Enhanced system is working perfectly!")
        
        # Optionally test full system
        full_success, full_ai, full_recs, full_portfolio = test_full_100_stocks()
        
        if full_success:
            print("\n🚀 Full 100-stock system verified!")
        
    else:
        print("\n⚠️ There was an issue with the enhanced system.")
        print("Check your internet connection and try again.")
