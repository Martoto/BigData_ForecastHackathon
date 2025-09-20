#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import os
import sys
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

class SalesForecastModelV2:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.transactions = None
        self.products = None
        self.stores = None
        self.model = None
        self.label_encoders = {}
        self.validation_metrics = {}
        self.best_params = None
        self.target_log_transformed = True
        
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
        
        # Remove extreme value outliers (likely data errors)
        print("Removing extreme value outliers...")
        initial_rows = len(self.transactions)
        
        # Calculate value per unit to detect unrealistic transactions
        value_per_unit = self.transactions['gross_value'] / self.transactions['quantity']
        q01 = value_per_unit.quantile(0.005)
        q99 = value_per_unit.quantile(0.995)
        
        # Remove transactions with extreme value per unit
        valid_value_mask = (value_per_unit >= q01) & (value_per_unit <= q99)
        self.transactions = self.transactions[valid_value_mask]
        
        # Also cap extreme quantities (likely bulk orders or errors)
        quantity_q99 = self.transactions['quantity'].quantile(0.995)
        extreme_qty_mask = self.transactions['quantity'] <= quantity_q99
        self.transactions = self.transactions[extreme_qty_mask]
        
        print(f"Removed {initial_rows - len(self.transactions):,} outlier transactions ({((initial_rows - len(self.transactions))/initial_rows)*100:.2f}%)")
        
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
        self.merged_data['month'] = dt_info.month
        self.merged_data['quarter'] = dt_info.quarter
        self.merged_data['year_week'] = self.merged_data['year'].astype(str) + '_' + self.merged_data['week'].astype(str).str.zfill(2)
        
        weekly_data = self.merged_data.groupby([
            'year_week', 'week', 'month', 'quarter', 'internal_store_id', 'internal_product_id',
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
        
        # Cap weekly quantity outliers per store-product pair
        print("Capping weekly quantity outliers...")
        self.cap_weekly_outliers()
        
    def build_features_v2(self):
        print("\nBuilding enhanced features...")
        
        self.weekly_data = self.weekly_data.sort_values(['internal_store_id', 'internal_product_id', 'week'])
        
        print("Creating temporal features...")
        self.weekly_data['week_sin'] = np.sin(2 * np.pi * self.weekly_data['week'] / 52)
        self.weekly_data['week_cos'] = np.cos(2 * np.pi * self.weekly_data['week'] / 52)
        self.weekly_data['month_sin'] = np.sin(2 * np.pi * self.weekly_data['month'] / 12)
        self.weekly_data['month_cos'] = np.cos(2 * np.pi * self.weekly_data['month'] / 12)
        
        print("Creating trend features...")
        grouped = self.weekly_data.groupby(['internal_store_id', 'internal_product_id'])
        
        self.weekly_data['quantity_lag_1'] = grouped['total_quantity'].shift(1)
        self.weekly_data['quantity_lag_2'] = grouped['total_quantity'].shift(2)
        
        for window in [2, 4]:
            past_total = grouped['total_quantity'].shift(1)
            self.weekly_data[f'quantity_rolling_avg_{window}'] = past_total.rolling(window=window, min_periods=1).mean()
        
        self.weekly_data['quantity_trend_2w'] = (self.weekly_data['quantity_lag_1'] - self.weekly_data['quantity_lag_2']) / (self.weekly_data['quantity_lag_2'] + 1)
        self.weekly_data['quantity_lag_3'] = grouped['total_quantity'].shift(3)
        self.weekly_data['quantity_growth_rate'] = (self.weekly_data['quantity_lag_1'] - self.weekly_data['quantity_lag_3']) / (self.weekly_data['quantity_lag_3'] + 1)
        
        print("Creating lifecycle features...")
        first_sale = grouped['week'].transform('min')
        last_sale = grouped['week'].transform('max')
        self.weekly_data['weeks_since_first_sale'] = self.weekly_data['week'] - first_sale
        self.weekly_data['weeks_since_last_sale'] = self.weekly_data['week'] - last_sale
        
        print("Creating store-product interaction features...")
        store_product_stats = self.weekly_data.groupby(['internal_store_id', 'internal_product_id']).agg({
            'total_quantity': ['mean', 'std', 'count'],
            'num_transactions': 'mean'
        })
        
        store_product_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in store_product_stats.columns.values]
        store_product_stats.rename(columns={
            'total_quantity_mean': 'store_product_avg_quantity',
            'total_quantity_std': 'store_product_std_quantity',
            'total_quantity_count': 'store_product_weeks_active',
            'num_transactions_mean': 'store_product_avg_transactions'
        }, inplace=True)
        
        self.weekly_data = self.weekly_data.merge(
            store_product_stats,
            left_on=['internal_store_id', 'internal_product_id'],
            right_index=True,
            how='left'
        )
        
        print("Creating category performance features...")
        self.weekly_data = self.weekly_data.sort_values(['categoria', 'week'])
        self.weekly_data['category_weekly_avg'] = (
            self.weekly_data.groupby('categoria')['total_quantity']
            .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        
        self.weekly_data = self.weekly_data.sort_values(['internal_store_id', 'week'])
        self.weekly_data['store_weekly_avg'] = (
            self.weekly_data.groupby('internal_store_id')['total_quantity']
            .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        
        self.weekly_data = self.weekly_data.fillna(0)
        print("Enhanced feature engineering completed")
        
    def cap_weekly_outliers(self):
        """Cap extreme weekly quantities using robust per-group statistics"""
        
        # Calculate caps per store-product pair (99.5th percentile)
        store_product_caps = self.weekly_data.groupby(['internal_store_id', 'internal_product_id'])['total_quantity'].quantile(0.995).reset_index()
        store_product_caps.rename(columns={'total_quantity': 'sp_quantity_cap'}, inplace=True)
        
        # Calculate global cap as backup (99.8th percentile)
        global_cap = self.weekly_data['total_quantity'].quantile(0.998)
        
        # Merge caps
        self.weekly_data = self.weekly_data.merge(
            store_product_caps, 
            on=['internal_store_id', 'internal_product_id'], 
            how='left'
        )
        
        # Apply caps (use store-product cap, fallback to global cap)
        before_sum = self.weekly_data['total_quantity'].sum()
        before_max = self.weekly_data['total_quantity'].max()
        
        self.weekly_data['sp_quantity_cap'] = self.weekly_data['sp_quantity_cap'].fillna(global_cap)
        self.weekly_data['total_quantity'] = np.minimum(
            self.weekly_data['total_quantity'], 
            self.weekly_data['sp_quantity_cap']
        )
        
        after_sum = self.weekly_data['total_quantity'].sum()
        after_max = self.weekly_data['total_quantity'].max()
        
        # Clean up temporary column
        self.weekly_data.drop('sp_quantity_cap', axis=1, inplace=True)
        
        print(f"  Before capping: max={before_max:,.0f}, total={before_sum:,.0f}")
        print(f"  After capping:  max={after_max:,.0f}, total={after_sum:,.0f}")
        print(f"  Reduction: {((before_sum - after_sum)/before_sum)*100:.2f}% of total volume capped")
        
    def prepare_training_data_v2(self):
        print("\nPreparing training data...")
        
        categorical_features = ['categoria', 'marca', 'premise', 'categoria_pdv']
        
        # Keep original categorical columns for CatBoost native handling
        for feature in categorical_features:
            self.weekly_data[feature] = self.weekly_data[feature].astype(str)
        
        feature_columns = [
            'week', 'month', 'quarter',
            'week_sin', 'week_cos', 'month_sin', 'month_cos',
            'num_transactions', 'avg_quantity_per_transaction',
            'total_gross_value', 'avg_gross_value',
            'total_net_value', 'avg_net_value',
            'total_gross_profit', 'avg_gross_profit',
            'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
            'quantity_rolling_avg_2', 'quantity_rolling_avg_4',
            'quantity_trend_2w', 'quantity_growth_rate',
            'weeks_since_first_sale', 'weeks_since_last_sale',
            'store_product_avg_quantity', 'store_product_std_quantity',
            'store_product_weeks_active', 'store_product_avg_transactions',
            'category_weekly_avg', 'store_weekly_avg'
        ] + categorical_features
        
        self.categorical_features = categorical_features
        
        train_data = self.weekly_data[self.weekly_data['week'] >= 5].copy()
        
        X = train_data[feature_columns]
        y = train_data['total_quantity']
        
        if self.target_log_transformed:
            y = np.log1p(y)
            print("Applied log1p transformation to target")
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        return X, y, feature_columns, train_data
    
    def calculate_wmape(self, y_true, y_pred):
        if self.target_log_transformed:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        y_pred = np.maximum(0, y_pred)
        denom = np.sum(y_true)
        return 0.0 if denom == 0 else np.sum(np.abs(y_true - y_pred)) / denom * 100
    
    def time_series_split(self, X, y, test_weeks=6):
        train_data = self.weekly_data[self.weekly_data['week'] >= 5].copy()
        max_week = train_data['week'].max()
        split_week = max_week - test_weeks + 1
        
        train_mask = train_data['week'] < split_week
        val_mask = train_data['week'] >= split_week
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        
        print(f"Time-based split: Train weeks 5-{split_week-1}, Validation weeks {split_week}-{max_week}")
        print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
        
        return X_train, X_val, y_train, y_val
    
    def optimize_hyperparameters_v2(self, X, y, n_trials=150):
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, using default parameters")
            return None
            
        print(f"\nOptimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'loss_function': 'MAE',
                'eval_metric': 'MAE',
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 150
            }
            
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = self.time_series_split(X, y)
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=150,
                verbose=False,
                use_best_model=True
            )
            
            val_pred = model.predict(X_val_fold)
            wmape = self.calculate_wmape(y_val_fold, val_pred)
            return wmape
        
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nHyperparameter optimization completed!")
        print(f"Best WMAPE: {study.best_value:.3f}%")
        print(f"Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params
        
    def train_model_v2(self, X, y, optimize_hyperparams=False, n_trials=150):
        print("\nTraining enhanced CatBoost model...")
        
        X_train, X_val, y_train, y_val = self.time_series_split(X, y)
        
        # if optimize_hyperparams and OPTUNA_AVAILABLE:
        #     self.optimize_hyperparameters_v2(X, y, n_trials=n_trials)
        #     model_params = self.best_params.copy()
        # else:
        model_params = {
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 15,
            'bagging_temperature': 0.8,
            'random_strength': 1.5,
            'border_count': 200,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 200
        }
        
        print(f"Using parameters: {model_params}")
        self.model = CatBoostRegressor(**model_params)
        
        cat_feature_indices = [X_train.columns.get_loc(col) for col in self.categorical_features if col in X_train.columns]
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_feature_indices,
            early_stopping_rounds=150,
            verbose=100,
            use_best_model=True
        )
        
        self.evaluate_model_v2(X_train, X_val, y_train, y_val)
        
    def evaluate_model_v2(self, X_train, X_val, y_train, y_val):
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_wmape = self.calculate_wmape(y_train, train_pred)
        val_wmape = self.calculate_wmape(y_val, val_pred)
        
        if self.target_log_transformed:
            y_train_orig = np.expm1(y_train)
            y_val_orig = np.expm1(y_val)
            train_pred_orig = np.expm1(train_pred)
            val_pred_orig = np.expm1(val_pred)
        else:
            y_train_orig = y_train
            y_val_orig = y_val
            train_pred_orig = train_pred
            val_pred_orig = val_pred
        
        train_pred_orig = np.maximum(0, train_pred_orig)
        val_pred_orig = np.maximum(0, val_pred_orig)
        
        train_mape = mean_absolute_percentage_error(y_train_orig, train_pred_orig) * 100
        val_mape = mean_absolute_percentage_error(y_val_orig, val_pred_orig) * 100
        
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
        
    def generate_predictions_v2(self, feature_columns, train_data, max_rows=1500000, weeks_to_predict=5, recent_weeks=8):
        print("\nGenerating predictions for January 2023...")
        
        store_product_combinations = train_data[['internal_store_id', 'internal_product_id']].drop_duplicates()
        print(f"Found {len(store_product_combinations):,} unique store-product combinations")
        
        latest_records = train_data.loc[train_data.groupby(['internal_store_id', 'internal_product_id'])['week'].idxmax()].copy()
        
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
        current_data = latest_records.copy()
        
        for week in range(1, weeks_to_predict + 1):
            print(f"Predicting week {week}...")
            
            week_data = current_data.copy()
            week_data['week'] = week
            week_data['month'] = 1
            week_data['quarter'] = 1
            week_data['week_sin'] = np.sin(2 * np.pi * week / 52)
            week_data['week_cos'] = np.cos(2 * np.pi * week / 52)
            week_data['month_sin'] = np.sin(2 * np.pi * 1 / 12)
            week_data['month_cos'] = np.cos(2 * np.pi * 1 / 12)
            
            features_matrix = week_data[feature_columns].values
            predictions = self.model.predict(features_matrix)
            
            if self.target_log_transformed:
                predictions = np.expm1(predictions)
            
            predictions = np.maximum(0, predictions)
            
            # Apply reasonable caps to predictions (prevent extreme outliers)
            prediction_cap = np.percentile(predictions, 99.5)
            predictions = np.minimum(predictions, prediction_cap)
            
            week_predictions = pd.DataFrame({
                'semana': week,
                'pdv': week_data['internal_store_id'].astype(int),
                'produto': week_data['internal_product_id'].astype(int),
                'quantidade': predictions.round().astype(int)
            })
            
            all_predictions.append(week_predictions)
            print(f"Generated {len(week_predictions):,} predictions for week {week}")
            
            # Update lag features for next week prediction
            if week < weeks_to_predict:
                current_data['quantity_lag_3'] = current_data['quantity_lag_2']
                current_data['quantity_lag_2'] = current_data['quantity_lag_1']
                current_data['quantity_lag_1'] = predictions if not self.target_log_transformed else np.log1p(predictions)
                
                # Update rolling averages
                for window in [2, 4]:
                    if week == 1:
                        current_data[f'quantity_rolling_avg_{window}'] = (
                            current_data[['quantity_lag_1', 'quantity_lag_2']].mean(axis=1) if window >= 2 else current_data['quantity_lag_1']
                        )
                    else:
                        # Simple approximation - in practice would need full rolling window logic
                        current_data[f'quantity_rolling_avg_{window}'] = (
                            (current_data[f'quantity_rolling_avg_{window}'] * (window-1) + current_data['quantity_lag_1']) / window
                        )
                
                # Update trend features
                current_data['quantity_trend_2w'] = (current_data['quantity_lag_1'] - current_data['quantity_lag_2']) / (current_data['quantity_lag_2'] + 1)
                current_data['quantity_growth_rate'] = (current_data['quantity_lag_1'] - current_data['quantity_lag_3']) / (current_data['quantity_lag_3'] + 1)
        
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
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
        
    def save_predictions(self, predictions_df, filename="sales_predictions_v2.csv"):
        print(f"\nSaving predictions to {filename}...")
        predictions_df.to_csv(filename, sep=';', index=False, encoding='utf-8')
        
        parquet_filename = filename.replace('.csv', '.parquet')
        predictions_df.to_parquet(parquet_filename, index=False)
        
        print(f"Predictions saved successfully!")
        print(f"CSV File: {filename}")
        print(f"Parquet File: {parquet_filename}")
        print(f"Rows: {len(predictions_df):,}")
        print(f"Sample:")
        print(predictions_df.head(10))
        
    def run_complete_pipeline(self):
        print("=== Sales Forecast Model V2 - Enhanced ===\n")
        
        self.load_data()
        self.cleanse_data()
        self.merge_data()
        self.create_weekly_aggregations()
        self.build_features_v2()
        
        X, y, feature_columns, train_data = self.prepare_training_data_v2()
        self.train_model_v2(X, y, optimize_hyperparams=False, n_trials=150)
        
        predictions_df = self.generate_predictions_v2(feature_columns, train_data)
        self.save_predictions(predictions_df)
        self.print_performance_report()
        
        return predictions_df
    
    def print_performance_report(self):
        print("\n" + "="*50)
        print("PERFORMANCE REPORT V2")
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
            
            if wmape_diff < 2 and mape_diff < 10:
                print("  Status: Good generalization")
            elif wmape_diff < 5 and mape_diff < 20:
                print("  Status: Moderate overfitting")
            else:
                print("  Status: High overfitting risk")
        
        print("="*50)

def main():
    try:
        model = SalesForecastModelV2()
        predictions = model.run_complete_pipeline()
        
        print("\n=== V2 Pipeline completed successfully! ===")
        print(f"Generated {len(predictions):,} predictions for 5 weeks of January 2023")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
