import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicTemporalEncoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that extracts time of day and day of week
    from a timestamp column and transforms them into cyclic features 
    using sine and cosine transformations.
    """
    def __init__(self, timestamp_col='order_timestamp'):
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Determine the source series of timestamps
        if isinstance(X, pd.DataFrame):
            # If a DataFrame is provided, extract the configured timestamp column
            if self.timestamp_col in X.columns:
                ts_series = X[self.timestamp_col]
            else:
                # If column name not found, try the first column
                ts_series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            ts_series = X
        else:
            # 1D numpy array or list
            ts_series = pd.Series(np.array(X).ravel())

        # Ensure datetime format
        dt_series = pd.to_datetime(ts_series)

        # Extract components
        hours = dt_series.dt.hour + dt_series.dt.minute / 60.0  # continuous hour representation
        dows = dt_series.dt.dayofweek

        # Compute sine and cosine encodings
        hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
        
        dow_sin = np.sin(2.0 * np.pi * dows / 7.0)
        dow_cos = np.cos(2.0 * np.pi * dows / 7.0)

        return np.column_stack((hour_sin, hour_cos, dow_sin, dow_cos))

    def get_feature_names_out(self, input_features=None):
        return [
            'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos'
        ]
