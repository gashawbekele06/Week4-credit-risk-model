# tests/test_data_processing.py
import pandas as pd
import pytest
from src.data_processing import FeatureEngineer
from src.data_loader import DataLoader

@pytest.fixture
def sample_data():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'Amount': [1000, 2000, 1500, 3000],
        'TransactionStartTime': ['2018-12-01', '2019-01-15', '2019-02-20', '2019-03-10'],
        'ProductCategory': ['airtime', 'data_bundles', 'airtime', 'data_bundles']
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_aggregate_features_returns_expected_columns(sample_data):
    engineer = FeatureEngineer()
    processed = engineer.aggregate_features(sample_data)
    expected_cols = [
        'CustomerId', 'total_transaction_amount', 'average_transaction_amount',
        'standard_deviation_transaction_amounts', 'transaction_count'
    ]
    assert all(col in processed.columns for col in expected_cols)
    assert processed.shape[0] == 3  # 3 unique customers

def test_pipeline_produces_valid_output(sample_data):
    engineer = FeatureEngineer()
    pipeline = engineer.build_full_pipeline()
    X = pipeline.fit_transform(sample_data)
    assert isinstance(X, pd.DataFrame)
    assert X.shape[1] > 0
    assert X.isna().sum().sum() == 0  # No missing values after pipeline