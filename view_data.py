import pandas as pd

def view_weekly():
    df = pd.read_csv('sales_predictions_v3.csv', sep=';')
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nWeeks: {df['semana'].min()} to {df['semana'].max()}")
    print(f"Stores: {df['pdv'].nunique():,}")
    print(f"Products: {df['produto'].nunique():,}")
    print(f"Total quantity: {df['quantidade'].sum():,}")
    
    print("\nWeekly breakdown:")
    summary = df.groupby('semana').agg({
        'quantidade': ['count', 'sum', 'mean']
    })
    summary.columns = ['predictions', 'total_qty', 'avg_qty']
    print(summary)
    
    print("\nTop 10 predictions:")
    print(df.nlargest(10, 'quantidade'))
    
    print("\nSample by week:")
    for week in sorted(df['semana'].unique()):
        week_data = df[df['semana'] == week]
        print(f"Week {week}: {len(week_data):,} predictions")

def view_daily():
    df = pd.read_csv('sales_predictions_daily_v3.csv', sep=';')
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total quantity: {df['quantidade_diaria'].sum():,}")
    
    print("\nDaily breakdown:")
    daily_summary = df.groupby('date').agg({
        'quantidade_diaria': ['count', 'sum']
    })
    daily_summary.columns = ['predictions', 'total_qty']
    print(daily_summary.head(10))

if __name__ == "__main__":
    print("=== WEEKLY DATA ===")
    view_weekly()
    print("\n=== DAILY DATA ===")
    view_daily()
