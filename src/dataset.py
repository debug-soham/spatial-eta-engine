import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    return R * c

def generate_synthetic_data(num_records=10000, seed=42):
    """
    Generates synthetic delivery records centered around New York City.
    """
    np.random.seed(seed)
    
    # NYC Center (Midtown Manhattan)
    nyc_lat, nyc_lng = 40.7589, -73.9851
    
    # Generate pickup and dropoff points around NYC
    # standard deviation of ~0.04 degrees (~4.5 km)
    pickup_lats = np.random.normal(nyc_lat, 0.035, num_records)
    pickup_lngs = np.random.normal(nyc_lng, 0.035, num_records)
    
    # Dropoffs are usually within a short distance of pickups
    dropoff_lats = pickup_lats + np.random.normal(0, 0.02, num_records)
    dropoff_lngs = pickup_lngs + np.random.normal(0, 0.02, num_records)
    
    # Calculate Haversine distance
    distances_km = haversine_distance(pickup_lats, pickup_lngs, dropoff_lats, dropoff_lngs)
    
    # Filter/Adjust extremely close or far points
    # Ensure distance is at least 0.2 km and at most 15 km
    distances_km = np.clip(distances_km, 0.2, 15.0)
    
    # Generate timestamps over March 2026
    start_date = datetime(2026, 3, 1)
    end_date = datetime(2026, 3, 31, 23, 59, 59)
    total_seconds = int((end_date - start_date).total_seconds())
    
    random_seconds = np.random.randint(0, total_seconds, num_records)
    timestamps = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
    timestamps = pd.Series(timestamps)
    
    # Merchant preparation times (normal distribution around 15 mins)
    merchant_prep = np.random.normal(15, 5, num_records)
    # dinner hours (18:00 - 21:00) increase prep time
    hours = timestamps.dt.hour
    is_dinner = (hours >= 18) & (hours <= 21)
    merchant_prep[is_dinner] += np.random.normal(8, 3, sum(is_dinner))
    merchant_prep = np.clip(merchant_prep, 5, 45)  # Clip between 5 and 45 mins
    
    # Operational metrics
    # Active driver ratio (drivers / orders): low ratio = delay in dispatch
    # Generally higher during middle of the day, lower during rush hour
    active_driver_ratio = np.random.uniform(0.1, 2.0, num_records)
    # Simulate rush hour driver shortages
    is_rush = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 20))
    active_driver_ratio[is_rush] = np.clip(active_driver_ratio[is_rush] - np.random.uniform(0.1, 0.5, sum(is_rush)), 0.05, 2.0)
    
    # Weather conditions (rain level in mm/hr)
    # 85% of time no rain, 15% rain
    rain_occurred = np.random.binomial(1, 0.15, num_records)
    rain_level = rain_occurred * np.random.uniform(1.0, 12.0, num_records)
    
    # Now compute target delivery time
    # Base overhead (finding parking, walking up stairs, handing over)
    base_overhead = 8.0
    
    # Travel speed (normally ~20 km/hr -> 3 minutes per km)
    # Traffic increases time per km during rush hour
    min_per_km = np.full(num_records, 3.0)
    min_per_km[is_rush] += np.random.uniform(2.5, 5.0, sum(is_rush))
    
    # Rain slows traffic and increases travel time per km, and increases overhead
    min_per_km[rain_level > 0] += rain_level[rain_level > 0] * 0.4
    weather_overhead = rain_level * 1.5
    
    # Dispatch delay due to driver density
    # low active driver ratio = longer dispatch queue time
    # delay starts to increase rapidly when driver ratio < 0.5
    dispatch_delay = np.zeros(num_records)
    low_driver_mask = active_driver_ratio < 0.8
    dispatch_delay[low_driver_mask] = (0.8 - active_driver_ratio[low_driver_mask]) * 15.0
    
    # Spatial congestion bias (let's simulate a traffic hot zone in Midtown Manhattan)
    # Midtown is around 40.748 <= lat <= 40.768 and -74.000 <= lng <= -73.970
    in_midtown = (pickup_lats >= 40.748) & (pickup_lats <= 40.768) & (pickup_lngs >= -74.000) & (pickup_lngs <= -73.970)
    spatial_delay = np.zeros(num_records)
    spatial_delay[in_midtown] += np.random.uniform(5.0, 12.0, sum(in_midtown))
    
    # Total Delivery Time
    travel_time = distances_km * min_per_km
    noise = np.random.normal(0, 3.0, num_records)
    
    total_delivery_time = base_overhead + travel_time + merchant_prep + weather_overhead + dispatch_delay + spatial_delay + noise
    
    # Clip delivery time to a minimum of 10 minutes
    total_delivery_time = np.clip(total_delivery_time, 10.0, 120.0)
    
    # Build dataframe
    df = pd.DataFrame({
        'pickup_latitude': pickup_lats,
        'pickup_longitude': pickup_lngs,
        'dropoff_latitude': dropoff_lats,
        'dropoff_longitude': dropoff_lngs,
        'order_timestamp': timestamps.dt.strftime('%Y-%m-%d %H:%M:%S'),
        'distance_km': distances_km,
        'merchant_prep_time_min': merchant_prep,
        'active_driver_ratio': active_driver_ratio,
        'rain_level_mm_hr': rain_level,
        'total_delivery_time_min': total_delivery_time
    })
    
    return df

def save_synthetic_data(output_dir='data', filename='delivery_records.csv', num_records=10000):
    """
    Generates and saves the synthetic dataset to the specified folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = generate_synthetic_data(num_records=num_records)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_records} delivery records and saved to {output_path}")
    return output_path

if __name__ == '__main__':
    save_synthetic_data(num_records=15000)
