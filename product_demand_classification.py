# Product Demand Pattern Classification
# Classifying products as Lumpy, Smooth, Erratic, or Intermittent based on demand patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

print("🔍 PRODUCT DEMAND PATTERN CLASSIFICATION")
print("="*60)
print("Analyzing products to classify as: Smooth, Erratic, Intermittent, or Lumpy")
print()

# Load the transaction data
print("📊 Loading transaction data...")
transactions_file = './data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
df_transactions = pd.read_parquet(transactions_file)

# Convert date column
df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])

print(f"Total transactions loaded: {len(df_transactions):,}")
print(f"Unique products: {df_transactions['internal_product_id'].nunique():,}")
print(f"Date range: {df_transactions['transaction_date'].min().strftime('%Y-%m-%d')} to {df_transactions['transaction_date'].max().strftime('%Y-%m-%d')}")
print()

# Create daily demand data by product
print("📈 Creating daily demand series by product...")
# Get all dates in the range
date_range = pd.date_range(
    start=df_transactions['transaction_date'].min(),
    end=df_transactions['transaction_date'].max(),
    freq='D'
)

# Aggregate daily demand by product
daily_demand = df_transactions.groupby(['internal_product_id', 'transaction_date']).agg({
    'quantity': 'sum',
    'gross_value': 'sum'
}).reset_index()

print(f"Daily demand records created: {len(daily_demand):,}")
print()

# Calculate demand pattern metrics for each product
print("📊 Calculating demand pattern metrics...")

def calculate_demand_metrics(product_id, transactions_df, date_range):
    """Calculate demand pattern metrics for a single product"""
    # Get product transactions
    product_data = transactions_df[transactions_df['internal_product_id'] == product_id].copy()
    
    if len(product_data) == 0:
        return None
    
    # Create complete daily series (including zeros)
    product_daily = product_data.groupby('transaction_date')['quantity'].sum().reindex(date_range, fill_value=0)
    
    # Basic statistics
    total_days = len(product_daily)
    demand_days = (product_daily > 0).sum()
    zero_days = total_days - demand_days
    
    # Average demand per period (only considering non-zero periods)
    avg_demand = product_daily[product_daily > 0].mean() if demand_days > 0 else 0
    
    # Coefficient of Variation (CV) - variability of demand
    std_demand = product_daily[product_daily > 0].std() if demand_days > 1 else 0
    cv = std_demand / avg_demand if avg_demand > 0 else 0
    
    # Average Demand Interval (ADI) - average time between demand occurrences
    if demand_days <= 1:
        adi = total_days  # If only one or no demand periods, ADI = total periods
    else:
        # Calculate intervals between demand periods
        demand_dates = product_daily[product_daily > 0].index
        intervals = [(demand_dates[i] - demand_dates[i-1]).days for i in range(1, len(demand_dates))]
        adi = np.mean(intervals) if intervals else total_days
    
    # Additional metrics
    total_quantity = product_data['quantity'].sum()
    total_revenue = product_data['gross_value'].sum()
    transaction_count = len(product_data)
    
    return {
        'product_id': product_id,
        'total_days': total_days,
        'demand_days': demand_days,
        'zero_days': zero_days,
        'demand_frequency': demand_days / total_days,
        'avg_demand': avg_demand,
        'cv': cv,
        'adi': adi,
        'total_quantity': total_quantity,
        'total_revenue': total_revenue,
        'transaction_count': transaction_count
    }

# Get unique products and calculate metrics
unique_products = df_transactions['internal_product_id'].unique()
print(f"Analyzing {len(unique_products):,} unique products...")

# Calculate metrics for all products (sample first 1000 for performance)
sample_size = min(1000, len(unique_products))
sample_products = unique_products[:sample_size]

metrics_list = []
for i, product_id in enumerate(sample_products):
    if i % 100 == 0:
        print(f"  Processed {i}/{sample_size} products...")
    
    metrics = calculate_demand_metrics(product_id, df_transactions, date_range)
    if metrics:
        metrics_list.append(metrics)

# Create metrics DataFrame
metrics_df = pd.DataFrame(metrics_list)
print(f"✅ Calculated metrics for {len(metrics_df):,} products")
print()

# Classification logic based on CV and ADI
print("🏷️ Classifying products based on demand patterns...")

def classify_demand_pattern(cv, adi):
    """
    Classify demand pattern based on Coefficient of Variation (CV) and Average Demand Interval (ADI)
    
    Classification rules:
    - Smooth: Low CV (< 0.5) and Low ADI (< 1.32) - Regular, predictable demand
    - Erratic: High CV (>= 0.5) and Low ADI (< 1.32) - Variable demand but frequent
    - Intermittent: Low CV (< 0.5) and High ADI (>= 1.32) - Regular amounts but infrequent
    - Lumpy: High CV (>= 0.5) and High ADI (>= 1.32) - Variable and infrequent
    """
    cv_threshold = 0.5  # Coefficient of variation threshold
    adi_threshold = 1.32  # Average demand interval threshold (industry standard)
    
    if cv < cv_threshold and adi < adi_threshold:
        return 'Smooth'
    elif cv >= cv_threshold and adi < adi_threshold:
        return 'Erratic'
    elif cv < cv_threshold and adi >= adi_threshold:
        return 'Intermittent'
    else:  # cv >= cv_threshold and adi >= adi_threshold
        return 'Lumpy'

# Apply classification
metrics_df['pattern_type'] = metrics_df.apply(lambda row: classify_demand_pattern(row['cv'], row['adi']), axis=1)

# Summary statistics by pattern type
pattern_summary = metrics_df.groupby('pattern_type').agg({
    'product_id': 'count',
    'cv': ['mean', 'median', 'std'],
    'adi': ['mean', 'median', 'std'],
    'demand_frequency': ['mean', 'median'],
    'avg_demand': ['mean', 'median'],
    'total_quantity': 'sum',
    'total_revenue': 'sum'
}).round(3)

print("📋 DEMAND PATTERN CLASSIFICATION SUMMARY")
print("="*50)
print(f"Total products analyzed: {len(metrics_df):,}")
print()

for pattern in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
    count = (metrics_df['pattern_type'] == pattern).sum()
    percentage = (count / len(metrics_df)) * 100
    print(f"{pattern:12}: {count:,} products ({percentage:.1f}%)")

print()
print("📊 Detailed Statistics by Pattern Type:")
print("="*50)

for pattern in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
    pattern_data = metrics_df[metrics_df['pattern_type'] == pattern]
    if len(pattern_data) > 0:
        print(f"\n{pattern.upper()} PRODUCTS ({len(pattern_data):,} products):")
        print(f"  Avg CV: {pattern_data['cv'].mean():.3f} (Std: {pattern_data['cv'].std():.3f})")
        print(f"  Avg ADI: {pattern_data['adi'].mean():.2f} (Std: {pattern_data['adi'].std():.2f})")
        print(f"  Demand Frequency: {pattern_data['demand_frequency'].mean():.3f}")
        print(f"  Avg Daily Demand: {pattern_data['avg_demand'].mean():.1f}")
        print(f"  Total Revenue: ${pattern_data['total_revenue'].sum():,.0f}")

# Show examples of each pattern type
print("\n🔍 EXAMPLES OF EACH PATTERN TYPE:")
print("="*50)

for pattern in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
    pattern_products = metrics_df[metrics_df['pattern_type'] == pattern]
    if len(pattern_products) > 0:
        # Get top 3 products by revenue for this pattern
        top_products = pattern_products.nlargest(3, 'total_revenue')
        
        print(f"\n{pattern.upper()} Products (Top 3 by Revenue):")
        print(f"{'Product ID':<20} {'CV':<8} {'ADI':<8} {'Freq':<8} {'Revenue':<12}")
        print("-" * 65)
        
        for _, product in top_products.iterrows():
            print(f"{str(product['product_id']):<20} {product['cv']:<8.3f} {product['adi']:<8.2f} "
                  f"{product['demand_frequency']:<8.3f} ${product['total_revenue']:<11,.0f}")

# Classification matrix
print(f"\n📈 CLASSIFICATION MATRIX:")
print("="*40)
print(f"{'Pattern':<12} {'CV Range':<15} {'ADI Range':<15} {'Description'}")
print("-" * 70)
print(f"{'Smooth':<12} {'< 0.5':<15} {'< 1.32':<15} {'Regular, predictable demand'}")
print(f"{'Erratic':<12} {'>= 0.5':<15} {'< 1.32':<15} {'Variable demand, frequent'}")
print(f"{'Intermittent':<12} {'< 0.5':<15} {'>= 1.32':<15} {'Regular amounts, infrequent'}")
print(f"{'Lumpy':<12} {'>= 0.5':<15} {'>= 1.32':<15} {'Variable and infrequent'}")

# Save detailed results
output_file = 'product_demand_classification_results.csv'
metrics_df.to_csv(output_file, index=False)
print(f"\n💾 Detailed results saved to: {output_file}")

print("\n" + "="*60)
print("🎯 CLASSIFICATION COMPLETE!")
print("="*60)

# Create visualizations
print("\n📊 Creating visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Product Demand Pattern Classification Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of pattern types (pie chart)
pattern_counts = metrics_df['pattern_type'].value_counts()
colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1']
axes[0, 0].pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[0, 0].set_title('Distribution of Demand Patterns')

# 2. CV vs ADI scatter plot
scatter_colors = {'Smooth': '#2E8B57', 'Erratic': '#FF6B6B', 'Intermittent': '#4ECDC4', 'Lumpy': '#45B7D1'}
for pattern in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
    pattern_data = metrics_df[metrics_df['pattern_type'] == pattern]
    axes[0, 1].scatter(pattern_data['adi'], pattern_data['cv'], 
                      label=pattern, alpha=0.6, s=30, color=scatter_colors[pattern])

axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='CV threshold')
axes[0, 1].axvline(x=1.32, color='red', linestyle='--', alpha=0.7, label='ADI threshold')
axes[0, 1].set_xlabel('Average Demand Interval (ADI)')
axes[0, 1].set_ylabel('Coefficient of Variation (CV)')
axes[0, 1].set_title('Demand Pattern Classification Matrix')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Revenue distribution by pattern type
pattern_revenue = metrics_df.groupby('pattern_type')['total_revenue'].sum().sort_values(ascending=True)
bars = axes[1, 0].barh(pattern_revenue.index, pattern_revenue.values, 
                       color=[scatter_colors[p] for p in pattern_revenue.index])
axes[1, 0].set_xlabel('Total Revenue ($)')
axes[1, 0].set_title('Revenue Distribution by Pattern Type')
axes[1, 0].ticklabel_format(style='plain', axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, pattern_revenue.values)):
    axes[1, 0].text(value + max(pattern_revenue.values) * 0.01, i, f'${value:,.0f}', 
                    va='center', fontsize=9)

# 4. Demand frequency vs Average demand
for pattern in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
    pattern_data = metrics_df[metrics_df['pattern_type'] == pattern]
    axes[1, 1].scatter(pattern_data['demand_frequency'], pattern_data['avg_demand'], 
                      label=pattern, alpha=0.6, s=30, color=scatter_colors[pattern])

axes[1, 1].set_xlabel('Demand Frequency (proportion of days with demand)')
axes[1, 1].set_ylabel('Average Daily Demand (quantity)')
axes[1, 1].set_title('Demand Frequency vs Average Demand')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('product_demand_classification_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed example plot for each pattern type
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('Example Demand Patterns for Each Category', fontsize=16, fontweight='bold')

pattern_examples = {}
for i, pattern in enumerate(['Smooth', 'Erratic', 'Intermittent', 'Lumpy']):
    pattern_products = metrics_df[metrics_df['pattern_type'] == pattern]
    if len(pattern_products) > 0:
        # Get the product with median CV and ADI for this pattern (most representative)
        median_product = pattern_products.iloc[(pattern_products['cv'] - pattern_products['cv'].median()).abs().argsort().iloc[0]]
        pattern_examples[pattern] = median_product['product_id']

# Plot example demand patterns (this would require daily data for selected products)
print(f"\n📈 Pattern examples identified:")
for pattern, product_id in pattern_examples.items():
    product_metrics = metrics_df[metrics_df['product_id'] == product_id].iloc[0]
    print(f"  {pattern}: Product {product_id} (CV: {product_metrics['cv']:.3f}, ADI: {product_metrics['adi']:.2f})")

plt.tight_layout()
plt.savefig('demand_pattern_examples.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualizations saved as PNG files")
print("📊 Analysis complete! Check the generated CSV file and visualizations.")