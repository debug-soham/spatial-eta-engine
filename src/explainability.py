import numpy as np
import pandas as pd

def get_global_importance(pipeline_payload):
    """
    Extracts global feature importances from the trained LightGBM regressor.
    """
    regressor = pipeline_payload['regressor']
    feature_names = pipeline_payload['feature_names']
    
    importances = regressor.feature_importances_
    # Normalize importances
    if importances.sum() > 0:
        importances = importances / importances.sum()
        
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    return df_importance

def explain_prediction(pipeline_payload, raw_input_df):
    """
    Computes local feature attributions for a single prediction using 
    marginal perturbation analysis relative to a standardized baseline.
    """
    preprocessor = pipeline_payload['preprocessor']
    regressor = pipeline_payload['regressor']
    
    # Transform raw input to processed feature space
    x_target = preprocessor.transform(raw_input_df)[0]
    
    # Establish a baseline processed representation
    # Coords: global mean and volume
    coords_trans = preprocessor.named_transformers_['coords']
    global_mean = coords_trans.global_mean_delay_
    global_vol = coords_trans.global_volume_
    
    # Standard Scaled features have a mean of 0.0. 
    # Cyclic time features average to 0.0 over a full cycle.
    x_baseline = np.array([
        global_mean,  # pickup_h3_mean_delay
        global_vol,   # pickup_h3_volume
        global_mean,  # dropoff_h3_mean_delay
        global_vol,   # dropoff_h3_volume
        0.0,          # hour_sin
        0.0,          # hour_cos
        0.0,          # dow_sin
        0.0,          # dow_cos
        0.0,          # distance_km
        0.0,          # merchant_prep_time_min
        0.0,          # active_driver_ratio
        0.0           # rain_level_mm_hr
    ])
    
    # Predictions
    base_pred = float(regressor.predict(x_baseline.reshape(1, -1))[0])
    target_pred = float(regressor.predict(x_target.reshape(1, -1))[0])
    
    # Compute marginal impact of each feature group
    attributions = {}
    
    # 1. Spatial Coordinates (H3)
    x_spatial = x_baseline.copy()
    x_spatial[0:4] = x_target[0:4]
    attributions['Spatial Congestion (H3)'] = float(regressor.predict(x_spatial.reshape(1, -1))[0]) - base_pred
    
    # 2. Temporal (Hour / Day of Week cyclic)
    x_temporal = x_baseline.copy()
    x_temporal[4:8] = x_target[4:8]
    attributions['Time of Day & Week'] = float(regressor.predict(x_temporal.reshape(1, -1))[0]) - base_pred
    
    # 3. Distance
    x_dist = x_baseline.copy()
    x_dist[8] = x_target[8]
    attributions['Distance (Travel Time)'] = float(regressor.predict(x_dist.reshape(1, -1))[0]) - base_pred
    
    # 4. Merchant Prep Time
    x_prep = x_baseline.copy()
    x_prep[9] = x_target[9]
    attributions['Merchant Prep Time'] = float(regressor.predict(x_prep.reshape(1, -1))[0]) - base_pred
    
    # 5. Driver Density (Active Driver Ratio)
    x_driver = x_baseline.copy()
    x_driver[10] = x_target[10]
    attributions['Active Driver Density'] = float(regressor.predict(x_driver.reshape(1, -1))[0]) - base_pred
    
    # 6. Weather (Rain level)
    x_weather = x_baseline.copy()
    x_weather[11] = x_target[11]
    attributions['Weather (Rain Impact)'] = float(regressor.predict(x_weather.reshape(1, -1))[0]) - base_pred
    
    # Check for residual due to non-linear interactions
    calculated_sum = sum(attributions.values())
    actual_diff = target_pred - base_pred
    residual = actual_diff - calculated_sum
    
    # Distribute the residual proportionally, or keep as a separate group
    if abs(actual_diff) > 0.01:
        attributions['Interactions & Base Overhead'] = base_pred + residual
    else:
        attributions['Interactions & Base Overhead'] = base_pred
        
    return {
        'base_prediction_min': base_pred,
        'target_prediction_min': target_pred,
        'net_impact_min': actual_diff,
        'attributions_min': attributions
    }
