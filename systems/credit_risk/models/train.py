"""
Model Training and Temporal Splitting for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR, SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def temporal_split(df):
    """Split data into train, val, and test sets based on time."""
    logger.info("Performing temporal split...")
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    
    # Define splits
    # Training: 2007-2015
    # Validation: 2016-2017
    # Testing: 2018
    train_mask = df['issue_d'].dt.year <= 2015
    val_mask = (df['issue_d'].dt.year >= 2016) & (df['issue_d'].dt.year <= 2017)
    test_mask = df['issue_d'].dt.year >= 2018
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train size: {len(train_df)} (up to 2015)")
    logger.info(f"Val size: {len(val_df)} (2016-2017)")
    logger.info(f"Test size: {len(test_df)} (2018)")
    
    return train_df, val_df, test_df

def prepare_xy(train_df, val_df, test_df):
    """Separate features and target, and scale for Logistic Regression."""
    X_train = train_df.drop(columns=['target', 'issue_d'])
    y_train = train_df['target']
    
    X_val = val_df.drop(columns=['target', 'issue_d'])
    y_val = val_df['target']
    
    X_test = test_df.drop(columns=['target', 'issue_d'])
    y_test = test_df['target']
    
    # Scale for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_scaled, X_val_scaled, X_test_scaled), scaler

def train_models(X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled):
    """Train LR, XGB, and LGBM."""
    models = {}
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # 2. XGBoost
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, 
                        random_state=SEED, n_jobs=-1, eval_metric='logloss',
                        early_stopping_rounds=20)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['XGBoost'] = xgb
    
    # 3. LightGBM
    logger.info("Training LightGBM...")
    lgb = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, 
                         random_state=SEED, n_jobs=-1)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            callbacks=[early_stopping(stopping_rounds=20)])
    models['LightGBM'] = lgb
    
    return models

if __name__ == "__main__":
    data_path = OUTPUT_DIR / "engineered_data.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        train_df, val_df, test_df = temporal_split(df)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_scaled, X_val_scaled, X_test_scaled), scaler = prepare_xy(train_df, val_df, test_df)
        
        models = train_models(X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled)
        
        # Save models and data
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
        
        # Save processed sets for evaluation
        X_train.to_parquet(OUTPUT_DIR / "X_train.parquet")
        y_train.to_frame().to_parquet(OUTPUT_DIR / "y_train.parquet")
        X_val.to_parquet(OUTPUT_DIR / "X_val.parquet")
        y_val.to_frame().to_parquet(OUTPUT_DIR / "y_val.parquet")
        X_test.to_parquet(OUTPUT_DIR / "X_test.parquet")
        y_test.to_frame().to_parquet(OUTPUT_DIR / "y_test.parquet")
        
        logger.info("Models trained and data saved.")
    else:
        logger.error("Engineered data not found.")
