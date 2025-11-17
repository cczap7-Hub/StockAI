"""
Simplified AI Stock Investment Advisor
====================================
A simpler, more robust version that focuses on core functionality
without complex technical indicators that might cause data issues.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

class SimpleStockAI:
    def __init__(self, symbols=None, period='15y'):
        """Initialize the Simple Stock AI"""
        self.period = period
        self.symbols = symbols or self.get_sp500_stocks()
        self.stock_data = {}
        self.features_df = pd.DataFrame()
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def get_sp500_stocks(self):
        """Get a comprehensive list of 250 diverse stocks from various sectors"""
        return [
            # Technology (50 stocks)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM',
            'NFLX', 'ORCL', 'IBM', 'INTC', 'AMD', 'PYPL', 'UBER', 'LYFT', 'SNAP', 'SHOP',
            'TWLO', 'SQ', 'ROKU', 'ZM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'CRWD', 'NET',
            'DDOG', 'MDB', 'WDAY', 'VEEV', 'SPLK', 'VMW', 'INTU', 'NOW', 'TEAM', 'ATLASSIAN',
            'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'ADI', 'MRVL', 'KLAC', 'SWKS',
            
            # Healthcare (40 stocks)
            'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB',
            'CVS', 'CI', 'HUM', 'MRNA', 'REGN', 'VRTX', 'ISRG', 'ILMN', 'DXCM', 'ZTS',
            'ELV', 'MCK', 'CAH', 'ABC', 'HSIC', 'ANTM', 'CNC', 'MOH', 'HCA', 'UHS',
            'DVA', 'FMS', 'VEEV', 'TDOC', 'PTON', 'CERNER', 'EPIC', 'ALLSCRIPTS', 'ATHM', 'DOCS',
            
            # Financial Services (35 stocks)
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'BLK',
            'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'ALLY', 'FITB', 'RF', 'ZION', 'CFG',
            'KEY', 'BBT', 'STI', 'HBAN', 'CMA', 'MTB', 'PBCT', 'NTRS', 'STT', 'BK',
            'SPGI', 'MCO', 'ICE', 'CME', 'NDAQ',
            
            # Consumer Discretionary (25 stocks)
            'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'DIS', 'LOW', 'TJX', 'BKNG',
            'CMG', 'YUM', 'ORLY', 'AZO', 'BBY', 'ETSY', 'EBAY', 'LULU', 'RH', 'ULTA',
            'GPS', 'ANF', 'AEO', 'URBN', 'EXPR',
            
            # Consumer Staples (20 stocks)
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'WBA', 'KR', 'CL', 'GIS', 'K',
            'CPB', 'HSY', 'MKC', 'CHD', 'CLX', 'SJM', 'HRL', 'TSN', 'CAG', 'KHC',
            
            # Energy (20 stocks)
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'KMI', 'OKE',
            'WMB', 'EPD', 'ET', 'MPLX', 'PAA', 'BKR', 'HAL', 'FTI', 'NBR', 'RIG',
            
            # Industrials (25 stocks)
            'BA', 'CAT', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'GE', 'DE',
            'ITW', 'PH', 'ROK', 'EMR', 'ETN', 'JCI', 'IR', 'DOV', 'FLS', 'XYL',
            'WAB', 'NSC', 'UNP', 'CSX', 'KSU',
            
            # Utilities (15 stocks)
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'SRE', 'AEP', 'XEL', 'WEC', 'ES',
            'PPL', 'FE', 'ETR', 'CNP', 'NI',
            
            # Real Estate (10 stocks)
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS',
            
            # Communication Services (10 stocks)
            'GOOGL', 'META', 'NFLX', 'DIS', 'VZ', 'T', 'CMCSA', 'CHTR', 'TMUS', 'DISH'
        ]
        
    def fetch_stock_data(self):
        """Fetch historical stock data with graceful handling for newer companies"""
        print(f"Fetching stock data for {len(self.symbols)} stocks...")
        print("This may take several minutes due to the large number of stocks and 15-year history...")
        
        successful_stocks = 0
        failed_stocks = 0
        
        for i, symbol in enumerate(self.symbols):
            try:
                # Progress indicator
                if (i + 1) % 25 == 0:
                    print(f"Progress: {i + 1}/{len(self.symbols)} stocks processed...")
                
                stock = yf.Ticker(symbol)
                
                # Try 15y first, if not available try max available data
                hist = None
                try:
                    hist = stock.history(period=self.period)
                except:
                    # If 15y fails, try maximum available data
                    try:
                        hist = stock.history(period='max')
                    except:
                        hist = stock.history(period='10y')  # Fallback to 10y
                
                info = stock.info
                
                # Accept stocks with at least 250 days of data (1 year minimum)
                if hist is not None and len(hist) > 250:
                    self.stock_data[symbol] = {
                        'history': hist,
                        'info': info
                    }
                    
                    # Show how much data we got
                    years = len(hist) / 252  # Approximate trading days per year
                    if years >= 15:
                        print(f"✓ {symbol} ({len(hist)} days - 15+ years)")
                    elif years >= 10:
                        print(f"✓ {symbol} ({len(hist)} days - {years:.1f} years)")
                    else:
                        print(f"✓ {symbol} ({len(hist)} days - {years:.1f} years, newer company)")
                    
                    successful_stocks += 1
                else:
                    print(f"✗ {symbol} - Insufficient data ({len(hist) if hist is not None else 0} days)")
                    failed_stocks += 1
                    
            except Exception as e:
                print(f"✗ {symbol} - Error: {str(e)}")
                failed_stocks += 1
        
        print(f"\nData collection complete:")
        print(f"✓ Successfully fetched: {successful_stocks} stocks")
        print(f"✗ Failed to fetch: {failed_stocks} stocks")
        print(f"📊 Total usable stocks: {len(self.stock_data)}")
        print(f"📈 Historical period: 15 years (or max available for newer companies)")
        
        return len(self.stock_data)
    
    def calculate_features(self):
        """Calculate simplified features for each stock"""
        print("Calculating features...")
        
        all_features = []
        
        for symbol, data in self.stock_data.items():
            try:
                df = data['history'].copy()
                info = data['info']
                
                # Basic price features
                current_price = df['Close'].iloc[-1]
                
                # Returns over different periods (comprehensive for 15-year history)
                returns_1d = df['Close'].pct_change(1).iloc[-1] if len(df) > 1 else 0
                returns_5d = df['Close'].pct_change(5).iloc[-1] if len(df) > 5 else 0
                returns_20d = df['Close'].pct_change(20).iloc[-1] if len(df) > 20 else 0
                returns_60d = df['Close'].pct_change(60).iloc[-1] if len(df) > 60 else 0
                returns_250d = df['Close'].pct_change(250).iloc[-1] if len(df) > 250 else 0  # ~1 year
                returns_500d = df['Close'].pct_change(500).iloc[-1] if len(df) > 500 else 0  # ~2 years
                returns_1000d = df['Close'].pct_change(1000).iloc[-1] if len(df) > 1000 else 0  # ~4 years
                returns_1250d = df['Close'].pct_change(1250).iloc[-1] if len(df) > 1250 else 0  # ~5 years
                returns_1750d = df['Close'].pct_change(1750).iloc[-1] if len(df) > 1750 else 0  # ~7 years
                returns_2500d = df['Close'].pct_change(2500).iloc[-1] if len(df) > 2500 else 0  # ~10 years
                
                # Volatility over different periods (extended for 15-year analysis)
                volatility_20d = df['Close'].pct_change().rolling(20).std().iloc[-1] if len(df) > 20 else 0
                volatility_60d = df['Close'].pct_change().rolling(60).std().iloc[-1] if len(df) > 60 else 0
                volatility_250d = df['Close'].pct_change().rolling(250).std().iloc[-1] if len(df) > 250 else 0
                volatility_500d = df['Close'].pct_change().rolling(500).std().iloc[-1] if len(df) > 500 else 0
                volatility_1000d = df['Close'].pct_change().rolling(1000).std().iloc[-1] if len(df) > 1000 else 0
                
                # Moving averages (comprehensive for long-term analysis)
                sma_20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) > 20 else current_price
                sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) > 50 else current_price
                sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) > 200 else current_price
                sma_500 = df['Close'].rolling(500).mean().iloc[-1] if len(df) > 500 else current_price  # ~2 year MA
                sma_1000 = df['Close'].rolling(1000).mean().iloc[-1] if len(df) > 1000 else current_price  # ~4 year MA
                
                # Price relative to moving averages
                price_vs_sma20 = current_price / sma_20 if sma_20 > 0 else 1
                price_vs_sma50 = current_price / sma_50 if sma_50 > 0 else 1
                price_vs_sma200 = current_price / sma_200 if sma_200 > 0 else 1
                price_vs_sma500 = current_price / sma_500 if sma_500 > 0 else 1
                price_vs_sma1000 = current_price / sma_1000 if sma_1000 > 0 else 1
                
                # Volume features (enhanced)
                avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else df['Volume'].iloc[-1]
                avg_volume_60 = df['Volume'].rolling(60).mean().iloc[-1] if len(df) > 60 else df['Volume'].iloc[-1]
                avg_volume_250 = df['Volume'].rolling(250).mean().iloc[-1] if len(df) > 250 else df['Volume'].iloc[-1]
                volume_ratio_20 = df['Volume'].iloc[-1] / avg_volume_20 if avg_volume_20 > 0 else 1
                volume_ratio_60 = df['Volume'].iloc[-1] / avg_volume_60 if avg_volume_60 > 0 else 1
                volume_ratio_250 = df['Volume'].iloc[-1] / avg_volume_250 if avg_volume_250 > 0 else 1
                
                # Long-term price trend analysis
                price_trend_20 = (sma_20 - df['Close'].rolling(20).mean().iloc[-21]) / df['Close'].rolling(20).mean().iloc[-21] if len(df) > 40 else 0
                price_trend_60 = (sma_50 - df['Close'].rolling(50).mean().iloc[-61]) / df['Close'].rolling(50).mean().iloc[-61] if len(df) > 110 else 0
                price_trend_250 = (sma_200 - df['Close'].rolling(200).mean().iloc[-251]) / df['Close'].rolling(200).mean().iloc[-251] if len(df) > 450 else 0
                
                # Multi-timeframe High/Low analysis (extended to 15 years)
                high_52w = df['High'].rolling(252).max().iloc[-1] if len(df) > 252 else df['High'].max()
                low_52w = df['Low'].rolling(252).min().iloc[-1] if len(df) > 252 else df['Low'].min()
                high_2y = df['High'].rolling(504).max().iloc[-1] if len(df) > 504 else df['High'].max()
                low_2y = df['Low'].rolling(504).min().iloc[-1] if len(df) > 504 else df['Low'].min()
                high_5y = df['High'].rolling(1260).max().iloc[-1] if len(df) > 1260 else df['High'].max()
                low_5y = df['Low'].rolling(1260).min().iloc[-1] if len(df) > 1260 else df['Low'].min()
                high_10y = df['High'].rolling(2520).max().iloc[-1] if len(df) > 2520 else df['High'].max()
                low_10y = df['Low'].rolling(2520).min().iloc[-1] if len(df) > 2520 else df['Low'].min()
                
                price_vs_52w_high = current_price / high_52w if high_52w > 0 else 1
                price_vs_52w_low = current_price / low_52w if low_52w > 0 else 1
                price_vs_2y_high = current_price / high_2y if high_2y > 0 else 1
                price_vs_2y_low = current_price / low_2y if low_2y > 0 else 1
                price_vs_5y_high = current_price / high_5y if high_5y > 0 else 1
                price_vs_5y_low = current_price / low_5y if low_5y > 0 else 1
                price_vs_10y_high = current_price / high_10y if high_10y > 0 else 1
                price_vs_10y_low = current_price / low_10y if low_10y > 0 else 1
                
                # Long-term performance metrics (extended for 15-year analysis)
                max_drawdown_1y = self.calculate_max_drawdown(df['Close'], 252) if len(df) > 252 else 0
                max_drawdown_2y = self.calculate_max_drawdown(df['Close'], 504) if len(df) > 504 else 0
                max_drawdown_5y = self.calculate_max_drawdown(df['Close'], 1260) if len(df) > 1260 else 0
                max_drawdown_10y = self.calculate_max_drawdown(df['Close'], 2520) if len(df) > 2520 else 0
                
                # Fundamental features (with safe extraction)
                pe_ratio = info.get('trailingPE', np.nan)
                market_cap = info.get('marketCap', np.nan)
                beta = info.get('beta', 1.0)
                roe = info.get('returnOnEquity', np.nan)
                debt_to_equity = info.get('debtToEquity', np.nan)
                revenue_growth = info.get('revenueGrowth', np.nan)
                profit_margin = info.get('profitMargins', np.nan)
                dividend_yield = info.get('dividendYield', 0)
                
                # Target for training (extended lookback for 15-year history)
                future_return = 0
                if len(df) > 120:  # Use 120-day future return for 15-year data
                    past_price = df['Close'].iloc[-121]
                    future_return = (current_price - past_price) / past_price
                
                features = {
                    'Symbol': symbol,
                    'Current_Price': current_price,
                    'Returns_1D': returns_1d,
                    'Returns_5D': returns_5d,
                    'Returns_20D': returns_20d,
                    'Returns_60D': returns_60d,
                    'Returns_250D': returns_250d,
                    'Returns_500D': returns_500d,
                    'Returns_1000D': returns_1000d,
                    'Returns_1250D': returns_1250d,
                    'Returns_1750D': returns_1750d,
                    'Returns_2500D': returns_2500d,
                    'Volatility_20D': volatility_20d,
                    'Volatility_60D': volatility_60d,
                    'Volatility_250D': volatility_250d,
                    'Volatility_500D': volatility_500d,
                    'Volatility_1000D': volatility_1000d,
                    'Price_vs_SMA20': price_vs_sma20,
                    'Price_vs_SMA50': price_vs_sma50,
                    'Price_vs_SMA200': price_vs_sma200,
                    'Price_vs_SMA500': price_vs_sma500,
                    'Price_vs_SMA1000': price_vs_sma1000,
                    'Volume_Ratio_20': volume_ratio_20,
                    'Volume_Ratio_60': volume_ratio_60,
                    'Volume_Ratio_250': volume_ratio_250,
                    'Price_Trend_20': price_trend_20,
                    'Price_Trend_60': price_trend_60,
                    'Price_Trend_250': price_trend_250,
                    'Price_vs_52W_High': price_vs_52w_high,
                    'Price_vs_52W_Low': price_vs_52w_low,
                    'Price_vs_2Y_High': price_vs_2y_high,
                    'Price_vs_2Y_Low': price_vs_2y_low,
                    'Price_vs_5Y_High': price_vs_5y_high,
                    'Price_vs_5Y_Low': price_vs_5y_low,
                    'Price_vs_10Y_High': price_vs_10y_high,
                    'Price_vs_10Y_Low': price_vs_10y_low,
                    'Max_Drawdown_1Y': max_drawdown_1y,
                    'Max_Drawdown_2Y': max_drawdown_2y,
                    'Max_Drawdown_5Y': max_drawdown_5y,
                    'Max_Drawdown_10Y': max_drawdown_10y,
                    'PE_Ratio': pe_ratio,
                    'Market_Cap': market_cap,
                    'Beta': beta,
                    'ROE': roe,
                    'Debt_to_Equity': debt_to_equity,
                    'Revenue_Growth': revenue_growth,
                    'Profit_Margin': profit_margin,
                    'Dividend_Yield': dividend_yield,
                    'Future_Return_120D': future_return
                }
                
                all_features.append(features)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        self.features_df = pd.DataFrame(all_features)
        
        # Handle missing values
        numeric_columns = self.features_df.select_dtypes(include=[np.number]).columns
        self.features_df[numeric_columns] = self.features_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created features for {len(self.features_df)} stocks")
        return self.features_df
    
    def calculate_max_drawdown(self, price_series, window):
        """Calculate maximum drawdown over a specified window"""
        if len(price_series) < window:
            return 0
        
        # Get the rolling window data
        windowed_prices = price_series.tail(window)
        
        # Calculate running maximum
        running_max = windowed_prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (windowed_prices - running_max) / running_max
        
        # Return maximum drawdown (most negative value)
        return abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    def train_model(self):
        """Train the ML model"""
        print("Training model...")
        
        # Prepare features (updated for 15-year model)
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['Symbol', 'Future_Return_120D']]
        
        X = self.features_df[feature_columns]
        y = self.features_df['Future_Return_120D']
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=feature_columns)
        
        # Split data
        if len(X_imputed) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=0.3, random_state=42)
        else:
            X_train = X_test = X_imputed
            y_train = y_test = y
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            print(f"Model R² Score: {r2:.4f}")
        
        return self.model
    
    def generate_recommendations(self, top_n=5):
        """Generate stock recommendations"""
        print("Generating recommendations...")
        
        if self.model is None:
            self.train_model()
        
        # Prepare features for prediction (updated for 15-year model)
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['Symbol', 'Future_Return_120D']]
        
        X = self.features_df[feature_columns]
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict returns
        predicted_returns = self.model.predict(X_scaled)
        
        # Calculate scores (enhanced for 15-year analysis)
        scores = []
        for idx, row in self.features_df.iterrows():
            # Base ML score (40%)
            ml_score = predicted_returns[idx]
            
            # Technical score (30%) - enhanced with 15-year indicators
            tech_score = 0
            if row['Price_vs_SMA20'] > 1:
                tech_score += 0.10
            if row['Price_vs_SMA50'] > 1:
                tech_score += 0.10
            if row['Price_vs_SMA200'] > 1:
                tech_score += 0.10
            if row['Price_vs_SMA500'] > 1:  # Long-term trend
                tech_score += 0.10
            if row['Price_vs_SMA1000'] > 1:  # Ultra long-term trend
                tech_score += 0.10
            if abs(row['Returns_20D']) < 0.1:  # Recent stability
                tech_score += 0.05
            if row['Price_vs_52W_High'] > 0.8:  # Near 52-week high
                tech_score += 0.05
            if row['Price_vs_2Y_High'] > 0.7:  # Strong 2-year performance
                tech_score += 0.05
            if row['Price_vs_5Y_High'] > 0.6:  # Strong 5-year performance
                tech_score += 0.05
            if row['Price_vs_10Y_High'] > 0.5:  # Strong 10-year performance
                tech_score += 0.05
            if row['Volume_Ratio_20'] > 1:  # Above average volume
                tech_score += 0.05
            if row['Max_Drawdown_1Y'] < 0.3:  # Controlled 1Y risk
                tech_score += 0.05
            if row['Max_Drawdown_5Y'] < 0.5:  # Controlled 5Y risk
                tech_score += 0.05
            if row['Returns_1250D'] > 0:  # Positive 5-year return
                tech_score += 0.05
            if row['Returns_2500D'] > 0:  # Positive 10-year return
                tech_score += 0.05
            
            # Fundamental score (20%)
            fund_score = 0
            if not pd.isna(row['PE_Ratio']) and 5 < row['PE_Ratio'] < 30:
                fund_score += 0.4
            if not pd.isna(row['ROE']) and row['ROE'] > 0.1:
                fund_score += 0.3
            if not pd.isna(row['Revenue_Growth']) and row['Revenue_Growth'] > 0:
                fund_score += 0.3
            
            # Risk score (10%)
            risk_score = 0
            if not pd.isna(row['Beta']) and 0.5 < row['Beta'] < 1.5:
                risk_score += 0.5
            if row['Volatility_20D'] < 0.05:
                risk_score += 0.5
            
            # Combined score
            final_score = (ml_score * 0.4 + tech_score * 0.3 + 
                          fund_score * 0.2 + risk_score * 0.1)
            
            scores.append({
                'Symbol': row['Symbol'],
                'Final_Score': final_score,
                'ML_Score': ml_score,
                'Technical_Score': tech_score,
                'Fundamental_Score': fund_score,
                'Risk_Score': risk_score,
                'Current_Price': row['Current_Price'],
                'PE_Ratio': row['PE_Ratio'],
                'Beta': row['Beta'],
                'Market_Cap': row['Market_Cap']
            })
        
        scores_df = pd.DataFrame(scores).sort_values('Final_Score', ascending=False)
        
        # Display results
        print(f"\n{'='*50}")
        print("🤖 AI STOCK RECOMMENDATIONS")
        print(f"{'='*50}")
        
        top_stocks = scores_df.head(top_n)
        for idx, stock in top_stocks.iterrows():
            print(f"\n{idx+1}. {stock['Symbol']} - Score: {stock['Final_Score']:.3f}")
            print(f"   💰 Price: ${stock['Current_Price']:.2f}")
            if not pd.isna(stock['PE_Ratio']):
                print(f"   📊 P/E: {stock['PE_Ratio']:.1f}")
            if not pd.isna(stock['Beta']):
                print(f"   ⚖️  Beta: {stock['Beta']:.2f}")
        
        return scores_df
    
    def create_portfolio(self, investment_amount=10000, top_n=5):
        """Create portfolio allocation"""
        if len(self.features_df) == 0:
            print("No data available. Run analysis first.")
            return None
        
        recommendations = self.generate_recommendations(top_n)
        top_stocks = recommendations.head(top_n)
        
        print(f"\n💼 PORTFOLIO ALLOCATION (${investment_amount:,})")
        print("-" * 40)
        
        # Simple equal weight allocation
        weight_per_stock = 1.0 / len(top_stocks)
        
        portfolio = []
        total_invested = 0
        
        for _, stock in top_stocks.iterrows():
            allocation = investment_amount * weight_per_stock
            shares = int(allocation / stock['Current_Price'])
            actual_amount = shares * stock['Current_Price']
            
            portfolio.append({
                'Symbol': stock['Symbol'],
                'Shares': shares,
                'Price': stock['Current_Price'],
                'Amount': actual_amount,
                'Weight': weight_per_stock * 100
            })
            
            total_invested += actual_amount
            print(f"{stock['Symbol']:>6}: {shares:>3} shares × ${stock['Current_Price']:>7.2f} = ${actual_amount:>8.2f} ({weight_per_stock*100:.1f}%)")
        
        cash_remaining = investment_amount - total_invested
        print(f"{'CASH':>6}: ${cash_remaining:>8.2f}")
        print(f"{'TOTAL':>6}: ${total_invested:>8.2f}")
        
        return pd.DataFrame(portfolio)
    
    def run_analysis(self, investment_amount=10000, top_n=5, num_stocks=None, portfolio_value=None):
        """Run complete analysis with flexible parameters"""
        print("🚀 Starting Stock Analysis...")
        
        # Handle legacy parameter names
        if portfolio_value is not None:
            investment_amount = portfolio_value
        
        # Optionally limit number of stocks to analyze
        if num_stocks is not None and num_stocks < len(self.symbols):
            print(f"Limiting analysis to first {num_stocks} stocks from the universe...")
            original_symbols = self.symbols
            self.symbols = self.symbols[:num_stocks]
        
        self.fetch_stock_data()
        self.calculate_features()
        self.train_model()
        recommendations = self.generate_recommendations(top_n)
        portfolio = self.create_portfolio(investment_amount, top_n)
        
        print("\n✅ Analysis Complete!")
        return recommendations, portfolio


def main():
    """Simple demo with 100 stocks and 10 years of history"""
    print("🚀 AI Stock Analysis with 100 Stocks & 10 Years History")
    print("=" * 56)
    
    # Create AI with default 100 stocks and 10 years of data
    ai = SimpleStockAI(period='10y')
    
    # Run analysis - this will take several minutes due to extensive data volume
    recommendations, portfolio = ai.run_analysis(investment_amount=15000, top_n=12)
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY:")
    print(f"Historical Period: 10 years")
    print(f"Stocks Analyzed: {len(ai.stock_data)}")
    print(f"Features per Stock: {len(ai.features_df.columns) - 1}")  # -1 for Symbol column
    print(f"Top 12 Recommendations Generated")
    print(f"Portfolio Optimized for $15,000 investment")
    print(f"Long-term Analysis: Multi-decade perspective")
    
    return ai, recommendations, portfolio

if __name__ == "__main__":
    ai, recs, portfolio = main()
