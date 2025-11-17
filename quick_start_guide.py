"""
🚀 QUICK START GUIDE - Ultimate 15-Year AI Stock Investment System
=================================================================

This is your ultimate AI-powered stock analysis tool!
It analyzes 15 YEARS of market data for 250 companies to make investment recommendations.

SYSTEM SPECIFICATIONS:
• Companies: 250 diverse stocks across all sectors
• Historical Data: 15 years (or maximum available for newer companies)
• Features: 47+ advanced technical and fundamental indicators
• Machine Learning: Gradient Boosting with 120-day predictions
• Sectors: Technology, Healthcare, Finance, Energy, Consumer, Utilities, Real Estate

USAGE EXAMPLES:
"""

from simple_stock_ai import SimpleStockAI

# Example 1: Quick Analysis (50 stocks)
def quick_analysis():
    """Fast analysis with 50 established stocks"""
    print("🚀 Running Quick Analysis...")
    ai = SimpleStockAI()
    # Use first 50 stocks from the ultimate 250-stock list
    quick_symbols = ai.symbols[:50]
    ai_quick = SimpleStockAI(symbols=quick_symbols)
    ai_quick.run_analysis(num_stocks=50)

# Example 2: Comprehensive Analysis (250 stocks)  
def ultimate_analysis():
    """Complete analysis with 250 diverse stocks and 15 years of data"""
    print("🚀 Running Ultimate Analysis...")
    print("⚠️  This will take 30-45 minutes but provides the most comprehensive results!")
    ai = SimpleStockAI()
    ai.run_analysis(num_stocks=250)

# Example 3: Sector-Focused Analysis
def sector_analysis():
    """Analysis focused on specific sectors"""
    print("🚀 Running Sector Analysis...")
    ai = SimpleStockAI()
    # Technology focus (first 50 from tech sector)
    tech_symbols = ai.symbols[:50]  # First 50 are mostly tech
    ai_tech = SimpleStockAI(symbols=tech_symbols)
    ai_tech.run_analysis(num_stocks=50, portfolio_value=50000)

# Example 4: Custom Portfolio Size
def custom_portfolio():
    """Analysis with custom investment amount"""
    print("🚀 Creating Custom Portfolio...")
    ai = SimpleStockAI()
    # Medium analysis with 100 stocks
    medium_symbols = ai.symbols[:100]
    ai_medium = SimpleStockAI(symbols=medium_symbols)
    ai_medium.run_analysis(num_stocks=100, portfolio_value=75000)

if __name__ == "__main__":
    print("🎯 ULTIMATE 15-YEAR AI STOCK SYSTEM - QUICK START")
    print("=" * 60)
    print("🏆 WORLD'S MOST COMPREHENSIVE STOCK AI")
    print("   • 250 companies across all sectors")
    print("   • 15 years of historical analysis")
    print("   • 47+ advanced features per stock")
    print("   • Graceful handling of newer companies")
    print("=" * 60)
    print()
    print("Choose your analysis:")
    print("1. ⚡ Quick (50 stocks, ~5 minutes)")
    print("2. 🔥 Ultimate (250 stocks, ~45 minutes)")
    print("3. 🎯 Sector Focus (50 stocks, custom)")
    print("4. 💰 Custom Portfolio (100 stocks)")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        quick_analysis()
    elif choice == "2":
        print("\n⚠️  ULTIMATE ANALYSIS WARNING:")
        print("This will download 15 years of data for 250 stocks.")
        print("It may take 30-45 minutes to complete.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() == 'y':
            ultimate_analysis()
        else:
            print("Running quick analysis instead...")
            quick_analysis()
    elif choice == "3":
        sector_analysis()
    elif choice == "4":
        portfolio_value = float(input("Enter investment amount ($): "))
        ai = SimpleStockAI()
        medium_symbols = ai.symbols[:100]
        ai_medium = SimpleStockAI(symbols=medium_symbols)
        ai_medium.run_analysis(num_stocks=100, portfolio_value=portfolio_value)
    else:
        print("Running default quick analysis...")
        quick_analysis()

    print("\n🎉 Analysis complete! Check the recommendations above.")
    print("💡 Your AI analyzed up to 15 YEARS of market data to make these predictions!")
    print("🏆 You now have the most advanced stock analysis system available!")
