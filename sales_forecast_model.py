#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import os
import sys

class SalesForecastModel:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.transactions = None
        self.products = None
        self.stores = None
        self.model = None
        self.label_encoders = {}
        self.validation_metrics = {}
        
    def load_data(self):
        print("Loading data...")
        parquet_files = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        print(f"Found {len(parquet_files)} parquet files")
        
        for file in parquet_files:
            df = pd.read_parquet(os.path.join(self.data_path, file))
            print(f"{file}: Shape {df.shape}")
            
            if 'internal_store_id' in df.columns and 'quantity' in df.columns:
                self.transactions = df
                print("-> Identified as TRANSACTIONS data")
            elif 'produto' in df.columns and 'categoria' in df.columns:
                self.products = df
                print("-> Identified as PRODUCTS data")
            elif 'pdv' in df.columns and 'premise' in df.columns:
                self.stores = df
                print("-> Identified as STORES data")
        
        print(f"Data loaded successfully:")
        print(f"- Transactions: {self.transactions.shape[0]:,} rows")
        print(f"- Products: {self.products.shape[0]:,} rows")
        print(f"- Stores: {self.stores.shape[0]:,} rows")
        
    def cleanse_data(self):
        print("\nCleansing data...")
        
        initial_rows = len(self.transactions)
        self.transactions = self.transactions.dropna(subset=['internal_store_id', 'internal_product_id', 'quantity', 'transaction_date'])
        print(f"Removed {initial_rows - len(self.transactions):,} rows with null values")
        
        self.transactions = self.transactions[self.transactions['quantity'] > 0]
        print(f"Kept {len(self.transactions):,} rows with positive quantities")
        
        self.transactions[['transaction_date', 'reference_date']] = self.transactions[['transaction_date', 'reference_date']].apply(pd.to_datetime)
        
        self.transactions = self.transactions[
            (self.transactions['transaction_date'].dt.year == 2022)
        ]
        print(f"Filtered to 2022 data: {len(self.transactions):,} rows")
        
        print("Cleaning products and stores...")
        self.products['descricao'] = self.products['descricao'].fillna('Unknown')
        self.products['categoria'] = self.products['categoria'].fillna('Other')
        self.products['marca'] = self.products['marca'].fillna('Unknown')
        
        self.stores['categoria_pdv'] = self.stores['categoria_pdv'].fillna('Other')
        self.stores['premise'] = self.stores['premise'].fillna('Unknown')
        
    def merge_data(self):
        print("\nMerging data...")
        
        merged_data = self.transactions.merge(
            self.products, 
            left_on='internal_product_id', 
            right_on='produto', 
            how='left'
        )
        print(f"After product merge: {len(merged_data):,} rows")
        
        merged_data = merged_data.merge(
            self.stores,
            left_on='internal_store_id',
            right_on='pdv',
            how='left'
        )
        print(f"After store merge: {len(merged_data):,} rows")
        
        self.merged_data = merged_data
        print("Data merge completed")
        
    def create_weekly_aggregations(self):
        print("\nCreating weekly aggregations...")
        
        dt_info = self.merged_data['transaction_date'].dt
        self.merged_data['year'] = dt_info.year
        self.merged_data['week'] = dt_info.isocalendar().week
        self.merged_data['year_week'] = self.merged_data['year'].astype(str) + '_' + self.merged_data['week'].astype(str).str.zfill(2)
        
        weekly_data = self.merged_data.groupby([
            'year_week', 'week', 'internal_store_id', 'internal_product_id',
            'categoria', 'marca', 'premise', 'categoria_pdv'
        ]).agg({
            'quantity': ['sum', 'mean', 'count'],
            'gross_value': ['sum', 'mean'],
            'net_value': ['sum', 'mean'],
            'gross_profit': ['sum', 'mean']
        }).reset_index()
        
        weekly_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in weekly_data.columns.values]
        
        column_mapping = {
            'quantity_sum': 'total_quantity',
            'quantity_mean': 'avg_quantity_per_transaction',
            'quantity_count': 'num_transactions',
            'gross_value_sum': 'total_gross_value',
            'gross_value_mean': 'avg_gross_value',
            'net_value_sum': 'total_net_value',
            'net_value_mean': 'avg_net_value',
            'gross_profit_sum': 'total_gross_profit',
            'gross_profit_mean': 'avg_gross_profit'
        }
        weekly_data.rename(columns=column_mapping, inplace=True)
        
        self.weekly_data = weekly_data
        print(f"Created weekly aggregations: {len(self.weekly_data):,} rows")
        
    def build_features(self):
        print("\nBuilding features...")
        
        self.weekly_data = self.weekly_data.sort_values(['internal_store_id', 'internal_product_id', 'week'])
        
        print("Creating lag features...")
        grouped = self.weekly_data.groupby(['internal_store_id', 'internal_product_id'])['total_quantity']
        
        for lag in [1, 2, 3, 4]:
            self.weekly_data[f'quantity_lag_{lag}'] = grouped.shift(lag)
        
        print("Creating rolling averages...")
        for window in [2, 4, 8]:
            self.weekly_data[f'quantity_rolling_avg_{window}'] = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        
        print("Creating seasonal features...")
        self.weekly_data['week_sin'] = np.sin(2 * np.pi * self.weekly_data['week'] / 52)
        self.weekly_data['week_cos'] = np.cos(2 * np.pi * self.weekly_data['week'] / 52)
        
        print("Creating store-product interaction features...")
        agg_dict = {
            'total_quantity': ['mean', 'std', 'min', 'max'],
            'num_transactions': 'mean'
        }
        store_product_stats = self.weekly_data.groupby(['internal_store_id', 'internal_product_id']).agg(agg_dict)
        
        store_product_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in store_product_stats.columns.values]
        store_product_stats.rename(columns={
            'total_quantity_mean': 'store_product_avg_quantity',
            'total_quantity_std': 'store_product_std_quantity',
            'total_quantity_min': 'store_product_min_quantity',
            'total_quantity_max': 'store_product_max_quantity',
            'num_transactions_mean': 'store_product_avg_transactions'
        }, inplace=True)
        
        self.weekly_data = self.weekly_data.merge(
            store_product_stats,
            left_on=['internal_store_id', 'internal_product_id'],
            right_index=True,
            how='left'
        )
        
        self.weekly_data = self.weekly_data.fillna(0)
        print("Feature engineering completed")
        
    def prepare_training_data(self):
        print("\nPreparing training data...")
        
        categorical_features = ['categoria', 'marca', 'premise', 'categoria_pdv']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                self.weekly_data[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
                    self.weekly_data[feature].astype(str)
                )
        
        feature_columns = [
            'week', 'week_sin', 'week_cos',
            'num_transactions', 'avg_quantity_per_transaction',
            'total_gross_value', 'avg_gross_value',
            'total_net_value', 'avg_net_value',
            'total_gross_profit', 'avg_gross_profit',
            'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_4',
            'quantity_rolling_avg_2', 'quantity_rolling_avg_4', 'quantity_rolling_avg_8',
            'store_product_avg_quantity', 'store_product_std_quantity',
            'store_product_min_quantity', 'store_product_max_quantity',
            'store_product_avg_transactions'
        ] + [f'{cat}_encoded' for cat in categorical_features]
        
        train_data = self.weekly_data[self.weekly_data['week'] >= 5].copy()
        
        X = train_data[feature_columns]
        y = train_data['total_quantity']
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        return X, y, feature_columns, train_data
    
    def calculate_wmape(self, y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    
    def evaluate_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
        val_mape = mean_absolute_percentage_error(y_val, val_pred) * 100
        
        train_wmape = self.calculate_wmape(y_train, train_pred)
        val_wmape = self.calculate_wmape(y_val, val_pred)
        
        self.validation_metrics = {
            'train_mape': train_mape,
            'val_mape': val_mape,
            'train_wmape': train_wmape,
            'val_wmape': val_wmape,
            'train_samples': len(y_train),
            'val_samples': len(y_val)
        }
        
        print(f"Training MAPE: {train_mape:.2f}%")
        print(f"Validation MAPE: {val_mape:.2f}%")
        print(f"Training WMAPE: {train_wmape:.2f}%")
        print(f"Validation WMAPE: {val_wmape:.2f}%")
        
        return self.validation_metrics
        
    def train_model(self, X, y):
        print("\nTraining CatBoost model...")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            random_seed=42,
            verbose=100
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=100
        )
        
        self.evaluate_model(X, y)
        
    def generate_predictions(self, feature_columns, train_data, max_rows=1500000, weeks_to_predict=5, recent_weeks=8):
        print("\nGenerating predictions for January 2023...")
        
        # Discover unique pairs
        store_product_combinations = train_data[['internal_store_id', 'internal_product_id']].drop_duplicates()
        print(f"Found {len(store_product_combinations):,} unique store-product combinations")
        
        # Build latest snapshot per pair
        latest_records = train_data.loc[train_data.groupby(['internal_store_id', 'internal_product_id'])['week'].idxmax()].copy()
        
        # Select top active pairs to respect the 1.5M-row portal limit
        try:
            pairs_limit = max_rows // weeks_to_predict
            max_week = int(self.weekly_data['week'].max())
            start_week = max(1, max_week - int(recent_weeks) + 1)
            recent_slice = self.weekly_data[self.weekly_data['week'] >= start_week]
            activity = recent_slice.groupby(['internal_store_id', 'internal_product_id']).agg(
                recent_total_qty=('total_quantity', 'sum'),
                weeks_with_sales=('total_quantity', lambda s: int((s > 0).sum())),
                last_week_seen=('week', 'max')
            ).reset_index()
            activity = activity.sort_values(
                by=['recent_total_qty', 'weeks_with_sales', 'last_week_seen'],
                ascending=[False, False, False]
            )
            selected_pairs = activity.head(pairs_limit)[['internal_store_id', 'internal_product_id']]
            before = len(latest_records)
            latest_records = latest_records.merge(selected_pairs, on=['internal_store_id', 'internal_product_id'], how='inner')
            after = len(latest_records)
            print(f"Selected top {after:,} active pairs out of {before:,} (recent_weeks={recent_weeks})")
        except Exception as e:
            print(f"Pair selection step skipped due to error: {e}")
        
        all_predictions = []
        
        for week in range(1, weeks_to_predict + 1):
            print(f"Predicting week {week}...")
            
            week_data = latest_records.copy()
            week_data['week'] = week
            week_data['week_sin'] = np.sin(2 * np.pi * week / 52)
            week_data['week_cos'] = np.cos(2 * np.pi * week / 52)
            
            features_matrix = week_data[feature_columns].values
            predictions = self.model.predict(features_matrix)
            predictions = np.maximum(0, predictions)
            
            week_predictions = pd.DataFrame({
                'semana': week,
                'pdv': week_data['internal_store_id'].astype(int),
                'produto': week_data['internal_product_id'].astype(int),
                'quantidade': predictions.round().astype(int)
            })
            
            all_predictions.append(week_predictions)
            print(f"Generated {len(week_predictions):,} predictions for week {week}")
        
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Final safeguard
        if len(predictions_df) > max_rows:
            print(f"Limiting predictions to {max_rows:,} rows (was {len(predictions_df):,})")
            predictions_df = predictions_df.head(max_rows)
        
        print(f"Total predictions generated: {len(predictions_df):,}")
        self.analyze_predictions(predictions_df)
        return predictions_df
    
    def analyze_predictions(self, predictions_df):
        print("\nPrediction Analysis:")
        print(f"Total predictions: {len(predictions_df):,}")
        print(f"Zero predictions: {(predictions_df['quantidade'] == 0).sum():,}")
        print(f"Non-zero predictions: {(predictions_df['quantidade'] > 0).sum():,}")
        print(f"Mean prediction: {predictions_df['quantidade'].mean():.2f}")
        print(f"Median prediction: {predictions_df['quantidade'].median():.2f}")
        print(f"Max prediction: {predictions_df['quantidade'].max():,}")
        print(f"Std prediction: {predictions_df['quantidade'].std():.2f}")
        
        quantiles = predictions_df['quantidade'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        print("Prediction quantiles:")
        for q, val in quantiles.items():
            print(f"  {q*100:.0f}%: {val:.2f}")
        
        weekly_stats = predictions_df.groupby('semana')['quantidade'].agg(['count', 'mean', 'sum']).round(2)
        print("\nWeekly prediction summary:")
        print(weekly_stats)
        
    def save_predictions(self, predictions_df, filename="sales_predictions.csv"):
        print(f"\nSaving predictions to {filename}...")
        
        predictions_df.to_csv(filename, sep=';', index=False, encoding='utf-8')
        
        print(f"Predictions saved successfully!")
        print(f"File: {filename}")
        print(f"Rows: {len(predictions_df):,}")
        print(f"Sample:")
        print(predictions_df.head(10))
        
    def run_complete_pipeline(self):
        print("=== Sales Forecast Model - Big Data Hackathon 2025 ===\n")
        
        self.load_data()
        self.cleanse_data()
        self.merge_data()
        self.create_weekly_aggregations()
        self.build_features()
        
        X, y, feature_columns, train_data = self.prepare_training_data()
        self.train_model(X, y)
        
        predictions_df = self.generate_predictions(feature_columns, train_data, max_rows=1500000, weeks_to_predict=5, recent_weeks=8)
        
        self.save_predictions(predictions_df)
        self.print_performance_report()
        
        return predictions_df
    
    def print_performance_report(self):
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        
        if self.validation_metrics:
            print("Model Validation Metrics:")
            print(f"  Training MAPE: {self.validation_metrics['train_mape']:.2f}%")
            print(f"  Validation MAPE: {self.validation_metrics['val_mape']:.2f}%")
            print(f"  Training WMAPE: {self.validation_metrics['train_wmape']:.2f}%")
            print(f"  Validation WMAPE: {self.validation_metrics['val_wmape']:.2f}%")
            print(f"  Training samples: {self.validation_metrics['train_samples']:,}")
            print(f"  Validation samples: {self.validation_metrics['val_samples']:,}")
            
            wmape_diff = abs(self.validation_metrics['val_wmape'] - self.validation_metrics['train_wmape'])
            mape_diff = abs(self.validation_metrics['val_mape'] - self.validation_metrics['train_mape'])
            
            print(f"\nOverfitting Check:")
            print(f"  MAPE difference: {mape_diff:.2f}%")
            print(f"  WMAPE difference: {wmape_diff:.2f}%")
            
            if wmape_diff < 5 and mape_diff < 5:
                print("  Status: Good generalization")
            elif wmape_diff < 10 and mape_diff < 10:
                print("  Status: Moderate overfitting")
            else:
                print("  Status: High overfitting risk")
        
        print("="*50)

def main():
    try:
        model = SalesForecastModel()
        predictions = model.run_complete_pipeline()
        
        print("\n=== Pipeline completed successfully! ===")
        print(f"Generated {len(predictions):,} predictions for 5 weeks of January 2023")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
