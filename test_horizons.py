#!/usr/bin/env python
"""Test if each horizon produces different recommendations"""

import pandas as pd
from stock_investment_ai import StockInvestmentAI
import sys
sys.path.insert(0, '.')

# Don't fetch data, just load existing data
ai = StockInvestmentAI(period='20y')

# Load existing features if available
try:
    ai.features_df = pd.read_csv(
        'features_data.csv') if False else pd.DataFrame()  # Skip for now
except:
    pass

# Just create features once
if ai.features_df.empty:
    print("Creating features...")
    ai.fetch_stock_data()
    ai.create_features_dataset()

print('\nChecking recommendations for each horizon:')
print('=' * 80)

all_horizons_data = {}
for horizon in ['20D', '60D', '90D', '365D', '730D']:
    scores = ai.calculate_investment_score(horizon)
    if not scores.empty:
        top_5 = scores.head(
            5)[['Symbol', 'Final_Score', 'Expected_Return']].copy()
        all_horizons_data[horizon] = top_5
        print(f'\n{horizon} Horizon - Top 5 Stocks:')
        for idx, row in top_5.iterrows():
            ret = row["Expected_Return"] * 100
            print(
                f'  {row["Symbol"]}: Score={row["Final_Score"]:.3f}, Return={ret:+.1f}%')
    else:
        print(f'\n{horizon}: No scores available')

# Compare if all horizons are the same
print('\n\nComparison - Are all horizons the same?')
print('=' * 80)
first_symbols = set(all_horizons_data.get('20D', pd.DataFrame())[
                    'Symbol'].values) if '20D' in all_horizons_data else set()
same = True
for horizon in ['60D', '90D', '365D', '730D']:
    if horizon in all_horizons_data:
        symbols = set(all_horizons_data[horizon]['Symbol'].values)
        if symbols == first_symbols:
            print(f'{horizon} has SAME top 5 symbols as 20D')
        else:
            print(f'{horizon} has DIFFERENT top 5 symbols from 20D')
            same = False

if same:
    print('\n⚠️  WARNING: All horizons have the same top 5 stocks!')
else:
    print('\n✓ Good: Different horizons have different top stocks!')
