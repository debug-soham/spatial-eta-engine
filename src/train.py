import os
import sys
# Add parent directory to path to enable importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from lightgbm import LGBMRegressor, LGBMClassifier

from src.dataset import save_synthetic_data
from src.spatial_features import H3SpatialAggregator
from src.temporal_features import CyclicTemporalEncoder

def train_and_evaluate(data_path='data/delivery_records.csv', model_dir='models'):
    # Ensure data exists
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating synthetic data...")
        save_synthetic_data(num_records=15000)
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} records.")

    # Define targets
    # Continuous target for regression: ETA in minutes
    y_reg = df['total_delivery_time_min']
    
    # Binary target for classification: Is order delayed? (Delay threshold: > 35 minutes)
    delay_threshold = 35.0
    y_clf = (y_reg > delay_threshold).astype(int)
    
    # Define features
    feature_cols = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'order_timestamp',
        'distance_km', 'merchant_prep_time_min',
        'active_driver_ratio', 'rain_level_mm_hr'
    ]
    X = df[feature_cols]

    # Split dataset
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Base rate of delays in training: {y_train_clf.mean():.2%}")

    # Build Pipeline Component: Column Transformer
    coords_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    time_col = ['order_timestamp']
    numeric_cols = ['distance_km', 'merchant_prep_time_min', 'active_driver_ratio', 'rain_level_mm_hr']

    preprocessor = ColumnTransformer(
        transformers=[
            ('coords', H3SpatialAggregator(resolution=8), coords_cols),
            ('time', CyclicTemporalEncoder(timestamp_col='order_timestamp'), time_col),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='drop'
    )

    # Fit preprocessor
    print("Fitting spatial-temporal feature engineering pipeline...")
    X_train_proc = preprocessor.fit_transform(X_train, y_train_reg)
    X_test_proc = preprocessor.transform(X_test)
    
    # Collect feature names out
    # Get feature names from transformers
    coords_names = preprocessor.named_transformers_['coords'].get_feature_names_out()
    time_names = preprocessor.named_transformers_['time'].get_feature_names_out()
    num_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_cols)
    
    all_feature_names = list(coords_names) + list(time_names) + list(num_names)
    print(f"Generated {len(all_feature_names)} features: {all_feature_names}")

    # Train ETA Regressor
    print("Training LightGBM Regressor for ETA estimation...")
    reg_model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    reg_model.fit(X_train_proc, y_train_reg)
    
    # Train Delay Probability Classifier
    print("Training LightGBM Classifier for delay probability estimation...")
    clf_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    clf_model.fit(X_train_proc, y_train_clf)

    # Evaluate Regressor
    reg_preds = reg_model.predict(X_test_proc)
    mae = mean_absolute_error(y_test_reg, reg_preds)
    rmse = root_mean_squared_error(y_test_reg, reg_preds)
    mape = mean_absolute_percentage_error(y_test_reg, reg_preds)

    print("\n" + "="*40)
    print("REGRESSION MODEL METRICS (ETA prediction)")
    print(f"Mean Absolute Error (MAE): {mae:.3f} mins")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} mins")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
    print("="*40)

    # Evaluate Classifier
    clf_preds = clf_model.predict(X_test_proc)
    clf_probs = clf_model.predict_proba(X_test_proc)[:, 1]
    roc_auc = roc_auc_score(y_test_clf, clf_probs)
    f1 = f1_score(y_test_clf, clf_preds)

    print("\n" + "="*40)
    print("CLASSIFIER MODEL METRICS (Delay prediction)")
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_clf, clf_preds))
    print("="*40)

    # Save Pipeline artifacts
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, 'eta_pipeline.joblib')
    
    payload = {
        'preprocessor': preprocessor,
        'regressor': reg_model,
        'classifier': clf_model,
        'feature_names': all_feature_names,
        'delay_threshold_min': delay_threshold,
        'metrics': {
            'regressor': {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)},
            'classifier': {'roc_auc': float(roc_auc), 'f1': float(f1)}
        }
    }
    
    joblib.dump(payload, pipeline_path)
    print(f"\nModel pipeline successfully saved to {pipeline_path}")
    return pipeline_path

if __name__ == '__main__':
    train_and_evaluate()
