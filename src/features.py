import pandas as pd
import numpy as np
from src.mappings import get_airport_code  

def engineer_features(flights):
    print(f"\n[3/6] Engineering features...")
    print("Mapping numeric airport IDs to 3-letter codes...")
    flights['origin_airport'] = flights['ORIGIN_AIRPORT'].apply(get_airport_code)
    
    flights = flights.dropna(subset=['origin_airport'])
    
    print(f"Kept {len(flights):,} flights after mapping")
    
    flights['dep_hour'] = flights['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2].astype(int)
    flights['dep_hour_sin'] = np.sin(2 * np.pi * flights['dep_hour'] / 24)
    flights['dep_hour_cos'] = np.cos(2 * np.pi * flights['dep_hour'] / 24)
    
    flights['day_of_week_sin'] = np.sin(2 * np.pi * flights['DAY_OF_WEEK'] / 7)
    flights['day_of_week_cos'] = np.cos(2 * np.pi * flights['DAY_OF_WEEK'] / 7)
    
    flights['month_sin'] = np.sin(2 * np.pi * flights['MONTH'] / 12)
    flights['month_cos'] = np.cos(2 * np.pi * flights['MONTH'] / 12)
    
    flights['is_weekend'] = flights['DAY_OF_WEEK'].isin([6, 7]).astype(int)

    airline_mapping = {airline: i for i, airline in enumerate(flights['AIRLINE'].unique())}
    flights['airline_encoded'] = flights['AIRLINE'].map(airline_mapping)

    flights['distance'] = flights['DISTANCE'].fillna(flights['DISTANCE'].median())

    feature_cols = [
        'dep_hour_sin', 'dep_hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'is_weekend',
        'airline_encoded',
        'distance'
    ]

    flights_clean = flights[feature_cols + ['significant_delay', 'total_delay', 'origin_airport']].dropna()

    X = flights_clean[feature_cols].values
    y_cls = flights_clean['significant_delay'].values
    y_reg = flights_clean['total_delay'].values
    airports_data = flights_clean['origin_airport'].values

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
    print(f"Unique airports: {len(np.unique(airports_data))}")

    return X, y_cls, y_reg, airports_data