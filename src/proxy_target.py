# src/proxy_target.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple
import sys
import os

# -------------------------------------------------
# PATH FIX: Add project root to sys.path when running as script
# -------------------------------------------------
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[INFO] Added project root to sys.path: {project_root}")

from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProxyTargetEngineer:
    """
    Task 4: Create proxy target variable using RFM + KMeans clustering.
    Labels disengaged (low frequency, low monetary, high recency) customers as high-risk.
    """
    
    def __init__(self, snapshot_date: str = "2019-03-01", n_clusters: int = 3, random_state: int = 42):
        self.snapshot_date = pd.to_datetime(snapshot_date)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.high_risk_cluster = None
        logger.info(f"ProxyTargetEngineer initialized with snapshot_date={snapshot_date}, n_clusters={n_clusters}")

    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """1. Calculate RFM metrics per CustomerId using only debit transactions."""
        logger.info("Calculating RFM metrics from debit transactions...")
        
        # Use only debits (positive Amount = actual spending)
        debits = df[df['Amount'] > 0].copy()
        debits['TransactionStartTime'] = pd.to_datetime(debits['TransactionStartTime'], utc=True)  # Force UTC
        
        # Make snapshot_date timezone-aware (UTC)
        snapshot = self.snapshot_date
        if snapshot.tz is None:
            snapshot = snapshot.tz_localize('UTC')
        
        rfm = debits.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot - x.max()).days,  # Recency
            'TransactionId': 'count',                                               # Frequency
            'Amount': 'sum'                                                         # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
        
        # Handle edge cases
        rfm['recency'] = rfm['recency'].clip(lower=0)
        rfm['frequency'] = rfm['frequency'].clip(lower=1)
        rfm['monetary'] = rfm['monetary'].clip(lower=1)
        
        # Log transform monetary to reduce skew
        rfm['monetary_log'] = np.log1p(rfm['monetary'])
        
        logger.info(f"RFM calculated for {len(rfm)} customers")
        return rfm

    def cluster_customers(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """2. Cluster customers using KMeans on scaled RFM features."""
        logger.info("Clustering customers with KMeans...")
        
        features = ['recency', 'frequency', 'monetary_log']
        X = rfm[features].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        rfm['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Show cluster centers (unscaled for interpretability)
        centers_scaled = self.kmeans.cluster_centers_
        centers_raw = self.scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_raw, columns=features)
        centers_df['monetary'] = np.expm1(centers_df['monetary_log'])
        centers_df = centers_df[['recency', 'frequency', 'monetary']]
        centers_df['cluster'] = range(self.n_clusters)
        
        print("\nRFM Cluster Centers (raw values):")
        print(centers_df.round(2))
        
        return rfm

    def assign_high_risk_label(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """3. Identify and label the high-risk (least engaged) cluster."""
        logger.info("Identifying high-risk cluster...")
        
        # Reconstruct raw centers for ranking
        centers_scaled = self.kmeans.cluster_centers_
        centers_raw = self.scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_raw, columns=['recency', 'frequency', 'monetary_log'])
        centers_df['monetary'] = np.expm1(centers_df['monetary_log'])
        
        # Risk score: high recency + low frequency + low monetary = high risk
        centers_df['risk_score'] = (
            centers_df['recency'].rank(ascending=False) +
            centers_df['frequency'].rank(ascending=True) +
            centers_df['monetary'].rank(ascending=True)
        )
        
        self.high_risk_cluster = centers_df['risk_score'].idxmax()
        
        print(f"\nHigh-risk cluster identified: Cluster {self.high_risk_cluster}")
        print(f"Reason: Highest recency, lowest frequency/monetary")
        
        rfm['is_high_risk'] = (rfm['cluster'] == self.high_risk_cluster).astype(int)
        
        bad_rate = rfm['is_high_risk'].mean()
        print(f"Proxy bad rate: {bad_rate:.2%} ({rfm['is_high_risk'].sum()} high-risk customers out of {len(rfm)})")
        
        return rfm

    def create_proxy_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: RFM → Clustering → High-Risk Label"""
        logger.info("Starting proxy target creation...")
        
        rfm = self.calculate_rfm(df)
        rfm = self.cluster_customers(rfm)
        target_df = self.assign_high_risk_label(rfm)
        
        logger.info("Proxy target creation complete.")
        return target_df[['CustomerId', 'is_high_risk', 'cluster', 'recency', 'frequency', 'monetary']]

    def merge_with_features(self, features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """4. Merge is_high_risk back into processed feature dataset"""
        logger.info("Merging proxy target with feature matrix...")
        merged = features_df.merge(target_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        merged['is_high_risk'] = merged['is_high_risk'].fillna(0).astype(int)
        return merged

# Standalone test


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    
    proxy = ProxyTargetEngineer()
    target_df = proxy.create_proxy_target(df)
    
    print("\nProxy Target Sample:")
    print(target_df.head(10))
    print(f"\nFinal bad rate: {target_df['is_high_risk'].mean():.2%}")