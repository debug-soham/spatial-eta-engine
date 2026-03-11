import h3
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def coord_to_h3(lat, lng, resolution=8):
    """
    Convert lat/lng to H3 cell index. Handles both H3 v3 and v4 APIs.
    """
    if hasattr(h3, 'latlng_to_cell'):
        return h3.latlng_to_cell(lat, lng, resolution)
    else:
        return h3.geo_to_h3(lat, lng, resolution)

def h3_to_boundary(h3_cell):
    """
    Get coordinates of the boundary of an H3 cell. Handles both H3 v3 and v4 APIs.
    """
    if hasattr(h3, 'cell_to_boundary'):
        return h3.cell_to_boundary(h3_cell)
    else:
        return h3.h3_to_geo_boundary(h3_cell)

def get_hexagon_geojson(h3_cell, properties=None):
    """
    Generate GeoJSON feature dict for a given H3 cell boundary.
    """
    boundary = h3_to_boundary(h3_cell)
    # GeoJSON expects coordinates as [longitude, latitude]
    geojson_coords = [[lng, lat] for lat, lng in boundary]
    if geojson_coords:
        geojson_coords.append(geojson_coords[0])  # Close the polygon loop
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [geojson_coords]
        },
        "properties": properties or {}
    }

class H3SpatialAggregator(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that converts latitude and longitude coordinates
    into H3 spatial hexagons and calculates target-encoded historical statistics
    (mean delivery time and volume of orders) for pickup and dropoff locations.
    """
    def __init__(self, resolution=8, 
                 pickup_lat_col='pickup_latitude', pickup_lng_col='pickup_longitude',
                 dropoff_lat_col='dropoff_latitude', dropoff_lng_col='dropoff_longitude'):
        self.resolution = resolution
        self.pickup_lat_col = pickup_lat_col
        self.pickup_lng_col = pickup_lng_col
        self.dropoff_lat_col = dropoff_lat_col
        self.dropoff_lng_col = dropoff_lng_col
        
        # Aggregation lookup tables
        self.pickup_stats_ = {}
        self.dropoff_stats_ = {}
        self.global_mean_delay_ = 30.0
        self.global_volume_ = 1.0

    def fit(self, X, y=None):
        # Convert X to dataframe if it is a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[
                self.pickup_lat_col, self.pickup_lng_col,
                self.dropoff_lat_col, self.dropoff_lng_col
            ])
        
        # Calculate target variable (y) if it is provided
        if y is not None:
            y = pd.Series(y)
        else:
            # Fallback if target is missing (fit on features alone isn't standard for target encoding,
            # but we need it to prevent errors)
            y = pd.Series(np.full(len(X), 30.0))
            
        self.global_mean_delay_ = float(y.mean())
        self.global_volume_ = float(len(X) / 100.0) # normalized volume baseline

        # Map to H3
        pickup_cells = X.apply(
            lambda r: coord_to_h3(r[self.pickup_lat_col], r[self.pickup_lng_col], self.resolution), 
            axis=1
        )
        dropoff_cells = X.apply(
            lambda r: coord_to_h3(r[self.dropoff_lat_col], r[self.dropoff_lng_col], self.resolution), 
            axis=1
        )

        # Build stats
        temp_df = pd.DataFrame({
            'pickup_h3': pickup_cells,
            'dropoff_h3': dropoff_cells,
            'target': y
        })

        # Calculate pickup stats
        pickup_grouped = temp_df.groupby('pickup_h3')['target'].agg(['mean', 'count'])
        self.pickup_stats_ = {
            cell: {
                'mean_delay': float(row['mean']),
                'volume': float(row['count'])
            }
            for cell, row in pickup_grouped.iterrows()
        }

        # Calculate dropoff stats
        dropoff_grouped = temp_df.groupby('dropoff_h3')['target'].agg(['mean', 'count'])
        self.dropoff_stats_ = {
            cell: {
                'mean_delay': float(row['mean']),
                'volume': float(row['count'])
            }
            for cell, row in dropoff_grouped.iterrows()
        }

        return self

    def transform(self, X):
        # Check if X is dataframe, else convert
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[
                self.pickup_lat_col, self.pickup_lng_col,
                self.dropoff_lat_col, self.dropoff_lng_col
            ])
            
        features = []
        for _, row in X.iterrows():
            p_cell = coord_to_h3(row[self.pickup_lat_col], row[self.pickup_lng_col], self.resolution)
            d_cell = coord_to_h3(row[self.dropoff_lat_col], row[self.dropoff_lng_col], self.resolution)
            
            p_stats = self.pickup_stats_.get(p_cell, {'mean_delay': self.global_mean_delay_, 'volume': 0.0})
            d_stats = self.dropoff_stats_.get(d_cell, {'mean_delay': self.global_mean_delay_, 'volume': 0.0})
            
            features.append([
                p_stats['mean_delay'],
                p_stats['volume'],
                d_stats['mean_delay'],
                d_stats['volume']
            ])
            
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return [
            'pickup_h3_mean_delay',
            'pickup_h3_volume',
            'dropoff_h3_mean_delay',
            'dropoff_h3_volume'
        ]
