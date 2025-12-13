# src/train.py
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pandas as pd
import numpy as np
import logging
import sys
import os

# -------------------------------------------------
# PATH FIX
# -------------------------------------------------
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.data_processing import FeatureEngineer
from src.proxy_target import ProxyTargetEngineer
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("BatiBank_CreditRisk_Proxy")


def load_data():
    loader = DataLoader()
    df_raw = loader.load()
    
    # Feature engineering
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df_raw)
    
    # Proxy target
    proxy = ProxyTargetEngineer()
    target_df = proxy.create_proxy_target(df_raw)
    
    # Merge target
    final_df = X.merge(target_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    final_df['is_high_risk'] = final_df['is_high_risk'].fillna(0).astype(int)
    
    y = final_df['is_high_risk']
    X = final_df.drop(columns=['CustomerId', 'is_high_risk'])
    
    logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Bad rate: {y.mean():.2%}")
    
    return X, y


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


if __name__ == "__main__":
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}

    # 1. Logistic Regression (baseline, interpretable)
    with mlflow.start_run(run_name="LogisticRegression"):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        
        mlflow.log_params({"model": "LogisticRegression"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "model", signature=infer_signature(X_train, y_pred))
        logger.info(f"Logistic Regression - AUC: {metrics['roc_auc']:.4f}")

    # 2. XGBoost (high performance)
    with mlflow.start_run(run_name="XGBoost_Tuned"):
        param_dist = {
            'n_estimators': [200, 400, 600],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        search = RandomizedSearchCV(
            xgb, param_distributions=param_dist,
            n_iter=40, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        
        best_xgb = search.best_estimator_
        y_pred = best_xgb.predict(X_test)
        y_prob = best_xgb.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        
        mlflow.log_params({"model": "XGBoost", **search.best_params_})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_xgb, "model", signature=infer_signature(X_train, y_pred))
        mlflow.log_metric("best_cv_auc", search.best_score_)
        logger.info(f"XGBoost - AUC: {metrics['roc_auc']:.4f}")

    # Register best model
    best_model = "XGBoost" if metrics['roc_auc'] > 0.85 else "LogisticRegression"
    logger.info(f"Best model: {best_model}")
    
    # Register in MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "BatiBankCreditRiskModel")
    