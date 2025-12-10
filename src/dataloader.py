import pandas as pd
import os

def load_data(data_dir='data', nrows=None):
    print(f"\n[1/6] Loading data...")
    
    flights_path = os.path.join(data_dir, 'flights.csv')
    airlines_path = os.path.join(data_dir, 'airlines.csv')
    airports_path = os.path.join(data_dir, 'airports.csv')

    try:
        flights = pd.read_csv(flights_path, low_memory=False, nrows=nrows)
        if nrows:
            limit_msg = f"(Limit set to {nrows:,})"
        else:
            limit_msg = "(Full Dataset)"
            
        print(f"Loaded flights: {flights.shape[0]:,} rows{limit_msg}")

    except FileNotFoundError:
        print(f"Error: {flights_path} not found")
        return None, None, None

    try:
        airlines = pd.read_csv(airlines_path)
        airports = pd.read_csv(airports_path)
    except FileNotFoundError:
        print("Missing airlines/airports files, continuing without them...")
        airlines = pd.DataFrame()
        airports = pd.DataFrame()

    return flights, airlines, airports