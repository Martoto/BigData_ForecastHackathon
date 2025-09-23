# Seasonal Product Volume Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("🌊 SEASONAL PRODUCT VOLUME ANALYSIS")
print("="*60)

# Load product demand classification data
print("\n📊 Loading product demand classification data...")
classification_df = pd.read_csv('./product_demand_classification_results.csv')
classification_df['product_id'] = classification_df['product_id'].astype(str)

# Load transaction data
print("📈 Loading transaction data...")
transactions_file = './data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
df_transactions = pd.read_parquet(transactions_file)
df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])
df_transactions['internal_product_id'] = df_transactions['internal_product_id'].astype(str)

# Add time-based features
df_transactions['year_month'] = df_transactions['transaction_date'].dt.to_period('M')
df_transactions['month'] = df_transactions['transaction_date'].dt.month
df_transactions['quarter'] = df_transactions['transaction_date'].dt.quarter
df_transactions['week'] = df_transactions['transaction_date'].dt.isocalendar().week

print(f"Analyzing {len(classification_df):,} classified products across {len(df_transactions):,} transactions")

# Calculate monthly volumes for all products
print("\n🔄 Calculating monthly volumes by product...")
monthly_product_volumes = df_transactions.groupby(['internal_product_id', 'year_month']).agg({
    'quantity': 'sum',
    'gross_value': 'sum',
    'transaction_date': 'count'
}).rename(columns={'transaction_date': 'transaction_count'})

# Reset index for easier manipulation
monthly_volumes_df = monthly_product_volumes.reset_index()
monthly_volumes_df['month_num'] = monthly_volumes_df['year_month'].dt.month

# Merge with classification data
monthly_volumes_df = monthly_volumes_df.merge(
    classification_df[['product_id', 'pattern_type', 'avg_demand', 'cv', 'demand_frequency', 'total_revenue']], 
    left_on='internal_product_id', 
    right_on='product_id', 
    how='inner'
)

print(f"Successfully matched {len(monthly_volumes_df['internal_product_id'].unique()):,} products")

# Select top products by pattern type for meaningful visualization
def select_top_products_by_pattern(df, pattern, n_products=5):
    """Select top products by total revenue for a given pattern"""
    pattern_products = df[df['pattern_type'] == pattern].copy()
    top_products = pattern_products.groupby('internal_product_id')['total_revenue'].first().nlargest(n_products)
    return pattern_products[pattern_products['internal_product_id'].isin(top_products.index)]

# Create comprehensive seasonal plots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Seasonal Volume Patterns by Product Demand Type', fontsize=16, fontweight='bold')

patterns = ['Erratic', 'Lumpy', 'Smooth', 'Intermittent']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for idx, pattern in enumerate(patterns):
    ax = axes[idx // 2, idx % 2]
    
    # Get top products for this pattern
    pattern_data = select_top_products_by_pattern(monthly_volumes_df, pattern, 8)
    
    if len(pattern_data) == 0:
        ax.text(0.5, 0.5, f'No {pattern} products found', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{pattern} Products (0 found)', fontweight='bold')
        continue
    
    # Plot each product's seasonal pattern
    product_count = 0
    for product_id in pattern_data['internal_product_id'].unique():
        if product_count >= 5:  # Limit to 5 products per plot for clarity
            break
            
        product_data = pattern_data[pattern_data['internal_product_id'] == product_id].copy()
        product_data = product_data.sort_values('month_num')
        
        # Calculate relative volume (normalized to show seasonal patterns)
        if len(product_data) > 1:
            avg_volume = product_data['quantity'].mean()
            product_data['relative_volume'] = product_data['quantity'] / avg_volume
            
            ax.plot(product_data['month_num'], product_data['relative_volume'], 
                   marker='o', linewidth=2.5, markersize=6, alpha=0.8,
                   label=f'Product {product_id[:8]}...')
            product_count += 1
    
    # Customize the subplot
    ax.set_title(f'{pattern} Products - Seasonal Patterns', fontweight='bold', fontsize=12)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Relative Volume (vs avg)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Add horizontal line at 1.0 (average)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    if product_count > 0:
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, f'Insufficient data for {pattern} products', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('seasonal_patterns_by_type.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed analysis of seasonal coefficients
print("\n📊 Calculating seasonal variation coefficients...")

seasonal_stats = []
for product_id in monthly_volumes_df['internal_product_id'].unique():
    product_data = monthly_volumes_df[monthly_volumes_df['internal_product_id'] == product_id]
    
    if len(product_data) >= 6:  # Need at least 6 months of data
        monthly_avg = product_data.groupby('month_num')['quantity'].mean()
        overall_avg = product_data['quantity'].mean()
        
        # Calculate seasonal coefficients
        seasonal_coeffs = monthly_avg / overall_avg
        seasonal_variation = seasonal_coeffs.std()
        peak_month = seasonal_coeffs.idxmax()
        trough_month = seasonal_coeffs.idxmin()
        peak_to_trough_ratio = seasonal_coeffs.max() / seasonal_coeffs.min()
        
        pattern_type = product_data['pattern_type'].iloc[0]
        total_revenue = product_data['total_revenue'].iloc[0]
        
        seasonal_stats.append({
            'product_id': product_id,
            'pattern_type': pattern_type,
            'seasonal_variation': seasonal_variation,
            'peak_month': peak_month,
            'trough_month': trough_month,
            'peak_to_trough_ratio': peak_to_trough_ratio,
            'total_revenue': total_revenue,
            'data_months': len(product_data)
        })

seasonal_df = pd.DataFrame(seasonal_stats)

# Top seasonal products by pattern
print("\n🏆 Most Seasonal Products by Pattern Type:")
print("="*50)

for pattern in patterns:
    pattern_products = seasonal_df[seasonal_df['pattern_type'] == pattern]
    if len(pattern_products) > 0:
        top_seasonal = pattern_products.nlargest(3, 'seasonal_variation')
        print(f"\n{pattern} Products:")
        for _, product in top_seasonal.iterrows():
            print(f"  Product {product['product_id'][:12]}...")
            print(f"    Seasonal Variation: {product['seasonal_variation']:.3f}")
            print(f"    Peak: Month {product['peak_month']} | Trough: Month {product['trough_month']}")
            print(f"    Peak/Trough Ratio: {product['peak_to_trough_ratio']:.2f}x")
            print(f"    Revenue: ${product['total_revenue']:,.0f}")

# Monthly seasonality heatmap
print("\n🔥 Creating monthly seasonality heatmap...")

# Calculate average seasonal patterns by product type
pattern_seasonality = []
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for pattern in patterns:
    pattern_data = monthly_volumes_df[monthly_volumes_df['pattern_type'] == pattern]
    if len(pattern_data) > 0:
        monthly_pattern = pattern_data.groupby('month_num')['quantity'].sum()
        total_volume = monthly_pattern.sum()
        if total_volume > 0:
            monthly_percentages = (monthly_pattern / total_volume * 100).reindex(range(1, 13), fill_value=0)
            pattern_seasonality.append([pattern] + monthly_percentages.tolist())

if pattern_seasonality:
    seasonality_df = pd.DataFrame(pattern_seasonality, 
                                 columns=['Pattern'] + month_names)
    seasonality_df = seasonality_df.set_index('Pattern')
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(seasonality_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Volume %'})
    plt.title('Monthly Volume Distribution by Product Pattern Type', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Product Pattern Type', fontweight='bold')
    plt.tight_layout()
    plt.savefig('seasonality_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# Individual high-impact product deep dive
print("\n🎯 Deep Dive: Top 3 Most Seasonal High-Revenue Products")
print("="*60)

top_seasonal_products = seasonal_df.nlargest(3, 'seasonal_variation')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Top 3 Most Seasonal Products - Monthly Volume Patterns', 
             fontsize=14, fontweight='bold')

for idx, (_, product) in enumerate(top_seasonal_products.iterrows()):
    product_id = product['product_id']
    product_data = monthly_volumes_df[monthly_volumes_df['internal_product_id'] == product_id]
    
    # Calculate monthly averages
    monthly_avg = product_data.groupby('month_num')['quantity'].mean().reindex(range(1, 13), fill_value=0)
    
    ax = axes[idx]
    bars = ax.bar(range(1, 13), monthly_avg.values, color=colors[idx % len(colors)], alpha=0.8)
    
    # Highlight peak and trough months
    peak_idx = int(product['peak_month']) - 1
    trough_idx = int(product['trough_month']) - 1
    bars[peak_idx].set_color('gold')
    bars[peak_idx].set_edgecolor('red')
    bars[peak_idx].set_linewidth(3)
    bars[trough_idx].set_color('lightblue')
    bars[trough_idx].set_edgecolor('blue')
    bars[trough_idx].set_linewidth(3)
    
    ax.set_title(f'Product {product_id[:8]}...\n{product["pattern_type"]} Pattern', 
                fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Quantity')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.grid(True, alpha=0.3)
    
    # Add stats text
    stats_text = f'Variation: {product["seasonal_variation"]:.3f}\nPeak/Trough: {product["peak_to_trough_ratio"]:.1f}x'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"Product {product_id}:")
    print(f"  Pattern: {product['pattern_type']}")
    print(f"  Seasonal Variation: {product['seasonal_variation']:.3f}")
    print(f"  Peak Month: {month_names[int(product['peak_month'])-1]} | Trough: {month_names[int(product['trough_month'])-1]}")
    print(f"  Peak/Trough Ratio: {product['peak_to_trough_ratio']:.2f}x")
    print(f"  Total Revenue: ${product['total_revenue']:,.0f}")
    print()

plt.tight_layout()
plt.savefig('top_seasonal_products.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Analysis complete! Generated visualizations:")
print("  📈 seasonal_patterns_by_type.png - Seasonal patterns by product type")
print("  🔥 seasonality_heatmap.png - Monthly volume heatmap by pattern")
print("  🎯 top_seasonal_products.png - Deep dive on most seasonal products")