# High-Impact Seasonal Product Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 HIGH-IMPACT SEASONAL PRODUCT ANALYSIS")
print("="*60)

# Load data
classification_df = pd.read_csv('./product_demand_classification_results.csv')
classification_df['product_id'] = classification_df['product_id'].astype(str)

transactions_file = './data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
df_transactions = pd.read_parquet(transactions_file)
df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])
df_transactions['internal_product_id'] = df_transactions['internal_product_id'].astype(str)
df_transactions['month'] = df_transactions['transaction_date'].dt.month

# Focus on high-revenue products with meaningful seasonal patterns
print("\n💰 Focusing on high-revenue products for meaningful seasonal analysis...")

# Get top revenue products by pattern type
top_products_by_pattern = {}
for pattern in ['Erratic', 'Lumpy', 'Smooth', 'Intermittent']:
    pattern_products = classification_df[classification_df['pattern_type'] == pattern]
    if len(pattern_products) > 0:
        # Select top 10 by total revenue
        top_products = pattern_products.nlargest(10, 'total_revenue')['product_id'].tolist()
        top_products_by_pattern[pattern] = top_products
        print(f"  {pattern}: {len(top_products)} top revenue products")

# Calculate detailed monthly patterns for top products
monthly_product_data = []

for pattern, product_list in top_products_by_pattern.items():
    for product_id in product_list:
        product_transactions = df_transactions[df_transactions['internal_product_id'] == product_id]
        
        if len(product_transactions) > 0:
            monthly_summary = product_transactions.groupby('month').agg({
                'quantity': ['sum', 'mean', 'count'],
                'gross_value': ['sum', 'mean'],
                'transaction_date': lambda x: x.nunique()  # unique days
            }).round(2)
            
            # Flatten column names
            monthly_summary.columns = ['total_qty', 'avg_qty', 'transaction_count', 
                                     'total_revenue', 'avg_revenue', 'active_days']
            
            # Calculate seasonality metrics
            if len(monthly_summary) >= 6:  # Need at least 6 months
                avg_monthly_qty = monthly_summary['total_qty'].mean()
                seasonal_variation = monthly_summary['total_qty'].std() / avg_monthly_qty if avg_monthly_qty > 0 else 0
                
                peak_month = monthly_summary['total_qty'].idxmax()
                trough_month = monthly_summary['total_qty'].idxmin()
                peak_qty = monthly_summary['total_qty'].max()
                trough_qty = monthly_summary['total_qty'].min()
                peak_to_trough = peak_qty / trough_qty if trough_qty > 0 else float('inf')
                
                # Get product info
                product_info = classification_df[classification_df['product_id'] == product_id].iloc[0]
                
                # Store monthly data with product info
                for month in monthly_summary.index:
                    monthly_product_data.append({
                        'product_id': product_id,
                        'pattern_type': pattern,
                        'month': month,
                        'total_qty': monthly_summary.loc[month, 'total_qty'],
                        'total_revenue': monthly_summary.loc[month, 'total_revenue'],
                        'active_days': monthly_summary.loc[month, 'active_days'],
                        'seasonal_variation': seasonal_variation,
                        'peak_month': peak_month,
                        'trough_month': trough_month,
                        'peak_to_trough_ratio': peak_to_trough,
                        'annual_revenue': product_info['total_revenue'],
                        'cv': product_info['cv'],
                        'demand_frequency': product_info['demand_frequency']
                    })

monthly_df = pd.DataFrame(monthly_product_data)

if len(monthly_df) > 0:
    print(f"\n📊 Analyzed {monthly_df['product_id'].nunique()} high-revenue products")
    
    # Create comprehensive seasonal visualizations
    print("\n📈 Creating detailed seasonal volume charts...")
    
    # 1. Top seasonal products by revenue impact
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('High-Revenue Products: Seasonal Volume Patterns', fontsize=16, fontweight='bold')
    
    patterns = ['Erratic', 'Lumpy', 'Smooth', 'Intermittent']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx // 2, idx % 2]
        pattern_data = monthly_df[monthly_df['pattern_type'] == pattern]
        
        if len(pattern_data) > 0:
            # Get top 5 most seasonal products by revenue
            top_seasonal = pattern_data.groupby('product_id').agg({
                'seasonal_variation': 'first',
                'annual_revenue': 'first'
            }).sort_values(['seasonal_variation', 'annual_revenue'], ascending=[False, False]).head(5)
            
            for i, (product_id, _) in enumerate(top_seasonal.iterrows()):
                product_monthly = pattern_data[pattern_data['product_id'] == product_id].sort_values('month')
                
                if len(product_monthly) > 0:
                    # Normalize to show relative seasonal pattern
                    avg_qty = product_monthly['total_qty'].mean()
                    if avg_qty > 0:
                        relative_qty = product_monthly['total_qty'] / avg_qty
                        
                        ax.plot(product_monthly['month'], relative_qty, 
                               marker='o', linewidth=3, markersize=8, alpha=0.8,
                               label=f'{product_id[:8]}... (${product_monthly["annual_revenue"].iloc[0]/1000:.0f}K)',
                               color=colors[i % len(colors)])
            
            ax.set_title(f'{pattern} Products - Seasonal Patterns\n(Normalized to Average = 1.0)', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Month', fontweight='bold')
            ax.set_ylabel('Relative Volume (vs avg)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            if len(top_seasonal) > 0:
                ax.legend(loc='best', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'No {pattern} products with sufficient data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{pattern} Products', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('high_revenue_seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Seasonal intensity vs Revenue scatter plot
    print("\n💎 Creating seasonal intensity analysis...")
    
    # Get product-level summary
    product_summary = monthly_df.groupby(['product_id', 'pattern_type']).agg({
        'seasonal_variation': 'first',
        'annual_revenue': 'first',
        'peak_to_trough_ratio': 'first',
        'cv': 'first',
        'demand_frequency': 'first'
    }).reset_index()
    
    # Filter out extreme outliers for better visualization
    product_summary = product_summary[
        (product_summary['seasonal_variation'] < 5) & 
        (product_summary['peak_to_trough_ratio'] < 100) &
        (product_summary['annual_revenue'] > 1000)  # Focus on meaningful revenue
    ]
    
    plt.figure(figsize=(14, 10))
    
    pattern_colors = {'Erratic': '#E74C3C', 'Lumpy': '#3498DB', 'Smooth': '#2ECC71', 'Intermittent': '#F39C12'}
    
    for pattern in patterns:
        pattern_data = product_summary[product_summary['pattern_type'] == pattern]
        if len(pattern_data) > 0:
            plt.scatter(pattern_data['seasonal_variation'], 
                       pattern_data['annual_revenue'], 
                       c=pattern_colors[pattern], 
                       alpha=0.7, 
                       s=100, 
                       label=f'{pattern} ({len(pattern_data)} products)',
                       edgecolors='black',
                       linewidth=0.5)
    
    plt.xlabel('Seasonal Variation (Standard Deviation)', fontweight='bold', fontsize=12)
    plt.ylabel('Annual Revenue ($)', fontweight='bold', fontsize=12)
    plt.title('Seasonal Intensity vs Revenue by Product Pattern Type', fontweight='bold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations for interesting outliers
    high_seasonal_high_revenue = product_summary[
        (product_summary['seasonal_variation'] > 1) & 
        (product_summary['annual_revenue'] > 100000)
    ]
    
    for _, product in high_seasonal_high_revenue.iterrows():
        plt.annotate(f'{product["product_id"][:6]}...', 
                    (product['seasonal_variation'], product['annual_revenue']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('seasonal_vs_revenue_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Business insights summary
    print("\n🎯 KEY SEASONAL INSIGHTS:")
    print("="*50)
    
    # Most seasonal high-revenue products
    top_seasonal_revenue = product_summary.nlargest(5, 'seasonal_variation')
    print("\n🌊 Most Seasonal High-Revenue Products:")
    for _, product in top_seasonal_revenue.iterrows():
        print(f"  Product {product['product_id'][:10]}... ({product['pattern_type']})")
        print(f"    Annual Revenue: ${product['annual_revenue']:,.0f}")
        print(f"    Seasonal Variation: {product['seasonal_variation']:.2f}")
        print(f"    Peak/Trough Ratio: {product['peak_to_trough_ratio']:.1f}x")
        print()
    
    # Pattern-wise seasonal summary
    print("📊 Pattern-wise Seasonal Characteristics:")
    pattern_summary = product_summary.groupby('pattern_type').agg({
        'seasonal_variation': ['mean', 'std', 'max'],
        'annual_revenue': ['mean', 'sum'],
        'product_id': 'count'
    }).round(2)
    
    for pattern in patterns:
        if pattern in pattern_summary.index:
            stats = pattern_summary.loc[pattern]
            print(f"\n{pattern} Products:")
            print(f"  Count: {stats[('product_id', 'count')]:.0f}")
            print(f"  Avg Seasonal Variation: {stats[('seasonal_variation', 'mean')]:.3f}")
            print(f"  Max Seasonal Variation: {stats[('seasonal_variation', 'max')]:.3f}")
            print(f"  Avg Annual Revenue: ${stats[('annual_revenue', 'mean')]:,.0f}")
            print(f"  Total Revenue: ${stats[('annual_revenue', 'sum')]:,.0f}")
    
    print("\n✅ Analysis complete! Generated:")
    print("  📈 high_revenue_seasonal_patterns.png - Detailed seasonal patterns")
    print("  💎 seasonal_vs_revenue_analysis.png - Seasonality vs revenue analysis")
    
else:
    print("❌ No sufficient data found for seasonal analysis")