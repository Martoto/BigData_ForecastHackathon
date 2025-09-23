# September 2022 Analysis - Investigating the Golden Month
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading transaction data...")
transactions_file = './data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
df_transactions = pd.read_parquet(transactions_file)

# Convert date column
df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])

print("🚀 SEPTEMBER 2022 ANALYSIS - THE GOLDEN MONTH")
print("="*60)

# Filter for September 2022
sep_2022 = df_transactions[
    (df_transactions['transaction_date'].dt.year == 2022) & 
    (df_transactions['transaction_date'].dt.month == 9)
].copy()

print(f"📊 September 2022 Overview:")
print(f"   Total Transactions: {len(sep_2022):,}")
print(f"   Total Revenue: ${sep_2022['gross_value'].sum():,.2f}")
print(f"   Average Transaction: ${sep_2022['gross_value'].mean():.2f}")
print(f"   Median Transaction: ${sep_2022['gross_value'].median():.2f}")
print(f"   Total Quantity: {sep_2022['quantity'].sum():,.0f}")
print(f"   Total Gross Profit: ${sep_2022['gross_profit'].sum():,.2f}")

# Daily breakdown
daily_breakdown = sep_2022.groupby(sep_2022['transaction_date'].dt.day).agg({
    'gross_value': 'sum',
    'quantity': 'sum',
    'gross_profit': 'sum'
}).round(2)

print(f"\n📅 Daily Revenue in September 2022:")
print(f"{'Day':<5} {'Revenue ($)':<15} {'Quantity':<12} {'Profit ($)':<12}")
print("-" * 50)
for day, row in daily_breakdown.iterrows():
    revenue = row['gross_value']
    quantity = row['quantity']
    profit = row['gross_profit']
    print(f"{day:<5} ${revenue:<14,.0f} {quantity:<12,.0f} ${profit:<11,.0f}")

# September 11th deep dive
sep_11 = sep_2022[sep_2022['transaction_date'].dt.day == 11]
print(f"\n🎯 SEPTEMBER 11, 2022 - THE MEGA DAY!")
print(f"   Revenue: ${sep_11['gross_value'].sum():,.2f}")
print(f"   Transactions: {len(sep_11):,}")
print(f"   % of September total: {(sep_11['gross_value'].sum() / sep_2022['gross_value'].sum() * 100):.1f}%")
print(f"   Average transaction: ${sep_11['gross_value'].mean():.2f}")
print(f"   Quantity sold: {sep_11['quantity'].sum():,.0f}")
print(f"   Gross profit: ${sep_11['gross_profit'].sum():,.2f}")

# What made Sep 11 special?
print(f"\n🔍 What made September 11th special?")
avg_daily_revenue = sep_2022['gross_value'].sum() / len(daily_breakdown)
multiplier = sep_11['gross_value'].sum() / avg_daily_revenue
print(f"   Sep 11 revenue was {multiplier:.1f}x the average daily revenue in September")

# Compare with other months
print(f"\n📊 Monthly comparison for 2022:")
monthly_revenue = df_transactions[df_transactions['transaction_date'].dt.year == 2022].groupby(
    df_transactions['transaction_date'].dt.month)['gross_value'].sum()

for month, revenue in monthly_revenue.items():
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    multiplier_vs_sep = revenue / monthly_revenue[9] if month != 9 else 1.0
    print(f"   {month_names[month]}: ${revenue:,.0f} ({multiplier_vs_sep:.1f}x Sep revenue)")

# Top performing products on Sep 11
print(f"\n🛍️ Top 5 Products on September 11, 2022:")
sep_11_products = sep_11.groupby('internal_product_id')['gross_value'].sum().sort_values(ascending=False).head(5)
for product_id, revenue in sep_11_products.items():
    transactions_count = sep_11[sep_11['internal_product_id'] == product_id].shape[0]
    print(f"   Product {product_id}: ${revenue:,.2f} ({transactions_count:,} transactions)")

# Top performing stores on Sep 11
print(f"\n🏪 Top 5 Stores on September 11, 2022:")
sep_11_stores = sep_11.groupby('internal_store_id')['gross_value'].sum().sort_values(ascending=False).head(5)
for store_id, revenue in sep_11_stores.items():
    transactions_count = sep_11[sep_11['internal_store_id'] == store_id].shape[0]
    print(f"   Store {store_id}: ${revenue:,.2f} ({transactions_count:,} transactions)")

# Distributor analysis for Sep 11
print(f"\n🚚 Distributor Performance on September 11, 2022:")
sep_11_distributors = sep_11.groupby('distributor_id')['gross_value'].sum().sort_values(ascending=False)
for dist_id, revenue in sep_11_distributors.items():
    transactions_count = sep_11[sep_11['distributor_id'] == dist_id].shape[0]
    print(f"   Distributor {dist_id}: ${revenue:,.2f} ({transactions_count:,} transactions)")

print("="*60)
print("Analysis complete!")