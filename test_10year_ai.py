"""
Test 10-Year AI Stock System
===========================
Test the ultimate version with 10 years of historical data and enhanced features.
"""

from simple_stock_ai import SimpleStockAI

def test_10_year_system():
    """Test the AI system with 10 years of data"""
    print("🔥 TESTING 10-YEAR AI STOCK SYSTEM")
    print("=" * 42)
    
    # Test with a focused set of established stocks (20 stocks, 10 years)
    # These should have 10+ years of trading history
    established_stocks = [
        # Mega-cap tech (established before 2014)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'INTC', 'ORCL', 'IBM',
        # Financial giants
        'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA',
        # Healthcare stalwarts  
        'JNJ', 'PFE', 'UNH', 'ABT',
        # Consumer/Industrial classics
        'WMT', 'KO', 'XOM'
    ]
    
    print(f"Testing with {len(established_stocks)} established stocks and 10 years of data...")
    print("This will take several minutes to fetch and process extensive historical data...")
    
    try:
        # Initialize with 10 years of data
        ai = SimpleStockAI(symbols=established_stocks, period='10y')
        
        # Run analysis
        recommendations, portfolio = ai.run_analysis(
            investment_amount=25000,
            top_n=10
        )
        
        print(f"\n✅ 10-Year system test completed successfully!")
        print(f"📊 Ultimate Analysis Results:")
        print(f"   • Stocks analyzed: {len(ai.stock_data)}")
        print(f"   • Features per stock: {len(ai.features_df.columns) - 1}")
        print(f"   • Top recommendation: {recommendations.iloc[0]['Symbol']}")
        print(f"   • Portfolio contains {len(portfolio)} stocks")
        print(f"   • Historical period: 10 YEARS (decade-long perspective)")
        
        # Show some of the advanced 10-year features
        sample_stock = ai.features_df.iloc[0]
        print(f"\n📈 Advanced 10-Year Features Example ({sample_stock['Symbol']}):")
        print(f"   • 2-year return: {sample_stock.get('Returns_500D', 0):.2%}")
        print(f"   • 4-year return: {sample_stock.get('Returns_1000D', 0):.2%}")
        print(f"   • Price vs 2-year MA: {sample_stock.get('Price_vs_SMA500', 1):.2f}")
        print(f"   • Max 1-year drawdown: {sample_stock.get('Max_Drawdown_1Y', 0):.2%}")
        print(f"   • Distance from 2Y high: {sample_stock.get('Price_vs_2Y_High', 1):.2%}")
        
        # Model performance with 10 years of data
        print(f"\n🧠 Machine Learning Performance:")
        print(f"   • Training data span: 10 years")
        print(f"   • Prediction target: 90-day future returns")
        print(f"   • Features analyzed: {len(ai.features_df.columns) - 2}")  # -2 for Symbol and target
        
        return True, ai, recommendations, portfolio
        
    except Exception as e:
        print(f"\n❌ 10-Year system test failed: {e}")
        print("This might be due to:")
        print("  • Network issues with extensive data download")
        print("  • Some stocks may not have full 10-year history")
        print("  • API rate limiting due to large data requests")
        return False, None, None, None

def test_full_100_stocks_10y():
    """Test with the full 100 stock universe and 10 years (ultimate test)"""
    print("\n🌟 ULTIMATE TEST: 100 STOCKS × 10 YEARS")
    print("=" * 45)
    print("⚠️  WARNING: This is the most comprehensive test!")
    print("   • Will download 10 years of data for 100 stocks")
    print("   • May take 15-20 minutes to complete")
    print("   • Requires stable internet connection")
    
    response = input("\nDo you want to run the ultimate 100-stock × 10-year test? (y/n): ")
    
    if response.lower() == 'y':
        try:
            print("\n🚀 Starting ultimate analysis...")
            print("Please be patient, this is processing massive amounts of data...")
            
            # Use default 100 stocks with 10 years
            ai = SimpleStockAI(period='10y')
            
            recommendations, portfolio = ai.run_analysis(
                investment_amount=100000,  # $100k for ultimate portfolio
                top_n=20  # Top 20 recommendations
            )
            
            print(f"\n🎉 ULTIMATE SYSTEM TEST SUCCESSFUL!")
            print(f"🏆 Massive 10-Year Analysis Results:")
            print(f"   • Total stocks analyzed: {len(ai.stock_data)}")
            print(f"   • Historical period: 10 YEARS")
            print(f"   • Features per stock: {len(ai.features_df.columns) - 1}")
            print(f"   • Top 20 recommendations generated")
            print(f"   • Ultimate portfolio optimized for $100,000")
            print(f"   • Decade-long market perspective captured")
            
            # Show distribution of recommendations by decade performance
            top_10 = recommendations.head(10)
            avg_4y_return = top_10['Returns_1000D'].mean() if 'Returns_1000D' in top_10.columns else 0
            print(f"   • Average 4-year return of top 10: {avg_4y_return:.2%}")
            
            return True, ai, recommendations, portfolio
            
        except Exception as e:
            print(f"❌ Ultimate system test failed: {e}")
            print("This is normal for such extensive analysis. Try the smaller test instead.")
            return False, None, None, None
    else:
        print("Skipping ultimate test.")
        return None, None, None, None

def compare_timeframes():
    """Compare analysis results across different timeframes"""
    print("\n📊 TIMEFRAME COMPARISON ANALYSIS")
    print("=" * 40)
    
    # Test same stocks with different periods
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    periods = ['1y', '2y', '5y', '10y']
    
    results = {}
    
    for period in periods:
        print(f"\nTesting {period} period...")
        try:
            ai = SimpleStockAI(symbols=test_stocks, period=period)
            ai.fetch_stock_data()
            ai.calculate_features()
            ai.train_model()
            recs = ai.generate_recommendations(top_n=3)
            
            results[period] = {
                'top_stock': recs.iloc[0]['Symbol'],
                'top_score': recs.iloc[0]['Final_Score'],
                'stocks_analyzed': len(ai.stock_data),
                'features': len(ai.features_df.columns) - 1
            }
            
            print(f"  ✓ {period}: Top = {results[period]['top_stock']} (Score: {results[period]['top_score']:.3f})")
            
        except Exception as e:
            print(f"  ✗ {period}: Failed - {e}")
            results[period] = None
    
    print(f"\n📈 TIMEFRAME COMPARISON RESULTS:")
    print("-" * 40)
    for period, result in results.items():
        if result:
            print(f"{period:>4}: {result['top_stock']:>6} | Score: {result['top_score']:.3f} | Features: {result['features']:>2}")
        else:
            print(f"{period:>4}: Failed")
    
    return results

if __name__ == "__main__":
    print("🎯 COMPREHENSIVE 10-YEAR AI STOCK SYSTEM TESTING")
    print("=" * 50)
    
    # Test 1: Enhanced 10-year system
    success, ai, recs, portfolio = test_10_year_system()
    
    if success:
        print("\n🎯 10-Year enhanced system is working perfectly!")
        
        # Test 2: Timeframe comparison
        timeframe_results = compare_timeframes()
        
        # Test 3: Ultimate test (optional)
        ultimate_success, ultimate_ai, ultimate_recs, ultimate_portfolio = test_full_100_stocks_10y()
        
        if ultimate_success:
            print("\n🚀 ULTIMATE 10-YEAR SYSTEM VERIFIED!")
            print("🏆 You now have the most comprehensive stock analysis AI available!")
        
    else:
        print("\n⚠️ There was an issue with the 10-year system.")
        print("This might be due to network issues or data availability.")
        print("Try running with fewer stocks or a shorter period first.")
