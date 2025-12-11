# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import logging

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[INFO] Added project root to sys.path: {project_root}")

# Now imports work
from src.config import PROJECT_ROOT, DATA_PATH, PLOT_STYLE, PALETTE
from src.data_loader import DataLoader

# Suppress the specific matplotlib category INFO messages
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
plt.style.use(PLOT_STYLE)
sns.set_palette(PALETTE)


class FeatureEngineer:
    def __init__(self, snapshot_date: str = "2019-03-01"):
        self.snapshot_date = pd.to_datetime(snapshot_date)
        self.pipeline = None
        self.woe = None
        self.iv_table = None

    # 1. Filter only debit transactions (Amount > 0)
    @staticmethod
    def filter_debits(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering debit transactions (Amount > 0)")
        return df[df['Amount'] > 0].copy()

    # 2. Extract time features
    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting time-based features")
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['hour'] = df['TransactionStartTime'].dt.hour
        df['day'] = df['TransactionStartTime'].dt.day
        df['month'] = df['TransactionStartTime'].dt.month
        # For recency later
        df['year'] = df['TransactionStartTime'].dt.year
        df['weekday'] = df['TransactionStartTime'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        return df

    # 1. Aggregate features per CustomerId
    @staticmethod
    def aggregate_per_customer(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Aggregating features per CustomerId")
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'count', 'max'],
            'Value': ['sum'],
            'hour': ['mean', 'std'],
            'day': ['mean'],
            'month': ['nunique'],
            'is_weekend': 'mean',
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'PricingStrategy': 'mean'
        }

        agg_df = df.groupby('CustomerId').agg(agg_dict)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
        agg_df = agg_df.reset_index()

        # Rename columns
        rename_map = {
            'Amount_sum': 'total_transaction_amount',
            'Amount_mean': 'average_transaction_amount',
            'Amount_std': 'std_transaction_amount',
            'Amount_count': 'transaction_count',
            'Amount_max': 'max_transaction_amount',
            'Value_sum': 'total_value',
            'hour_mean': 'avg_transaction_hour',
            'hour_std': 'std_transaction_hour',
            'day_mean': 'avg_transaction_day',
            'month_nunique': 'active_months',
            'is_weekend_mean': 'weekend_ratio',
            'ProductCategory_<lambda>': 'most_frequent_product',
            'ChannelId_<lambda>': 'most_frequent_channel',
            'PricingStrategy_mean': 'avg_pricing_strategy'
        }
        agg_df = agg_df.rename(columns=rename_map)

        # 4. Handle Missing Values
        agg_df['std_transaction_amount'] = agg_df['std_transaction_amount'].fillna(0)
        agg_df['std_transaction_hour'] = agg_df['std_transaction_hour'].fillna(0)

        return agg_df

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_features = [
            'total_transaction_amount', 'average_transaction_amount',
            'std_transaction_amount', 'transaction_count', 'max_transaction_amount',
            'total_value', 'avg_transaction_hour', 'std_transaction_hour',
            'avg_transaction_day', 'active_months', 'weekend_ratio', 'avg_pricing_strategy'
        ]

        categorical_features = ['most_frequent_product', 'most_frequent_channel']

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())  # 5. Standardization
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 3. One-Hot Encoding
        ])

        return ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    def build_full_pipeline(self) -> Pipeline:
        """Build the complete sklearn Pipeline"""
        self.pipeline = Pipeline(steps=[
            ('filter_debits', FunctionTransformer(self.filter_debits)),
            ('time_features', FunctionTransformer(self.extract_time_features)),
            ('aggregate', FunctionTransformer(self.aggregate_per_customer)),
            ('preprocess', self.build_preprocessor())
        ])
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit pipeline and optionally apply WoE"""
        if self.pipeline is None:
            self.build_full_pipeline()

        X = self.pipeline.fit_transform(df)

        # Convert to DataFrame with proper feature names
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        X_df = pd.DataFrame(X, columns=feature_names)

        # 6. WoE + IV (only if target is provided
        if y is not None:
            logger.info("Applying Weight of Evidence (WoE) transformation")
            woe_features = ['most_frequent_product', 'most_frequent_channel', 'transaction_count']
            self.woe = WOE()
            self.woe.fit(X_df[woe_features], y)
            X_woe = self.woe.transform(X_df[woe_features])
            X_df = pd.concat([X_df.drop(columns=woe_features), X_woe.add_prefix('woe_')], axis=1)

            # Save IV table
            self.iv_table = self.woe.iv_df.sort_values('Information_Value', ascending=False)
            print("\nInformation Value (IV) Table:")
            print(self.iv_table[['Variable_Name', 'Information_Value']])

        logger.info(f"Feature engineering complete. Final shape: {X_df.shape}")
        return X_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet.")
        X = self.pipeline.transform(df)
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        return pd.DataFrame(X, columns=feature_names)

# Test


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df)
    
    print("\nModel-ready features:")
    print(X.head())
    print(f"\nFinal shape: {X.shape}")