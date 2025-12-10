import pandas as pd

def preprocess_data(flights):

    print(f"\n[2/6] Preprocessing data...")
    
    flights['SCHEDULED_DEPARTURE'] = flights['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4)

    flights['significant_delay'] = (flights['ARRIVAL_DELAY'] > 30).astype(int)
    flights['total_delay'] = flights['ARRIVAL_DELAY'].fillna(0)

    if 'CANCELLED' in flights.columns:
        flights.loc[flights['CANCELLED'] == 1, 'total_delay'] = 300
        flights.loc[flights['CANCELLED'] == 1, 'significant_delay'] = 1
        
        cancelled_count = flights['CANCELLED'].sum()
        print(f" Found {cancelled_count:,} cancelled flights")

    print(f"Created {flights['significant_delay'].sum():,} significant delays")
    print(f"Average delay: {flights['total_delay'].mean():.1f} minutes")
    
    return flights