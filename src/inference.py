import numpy as np
import joblib
import os
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from src.model import FlightDelayModel

class FlightPredictor:
    def __init__(self, model_dir='output', data_dir='data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.metadata = None
        self.airport_coords = {}
        self.loaded = False

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3956 
        return c * r

    def load(self):
        try:
            meta_path = os.path.join(self.model_dir, 'metadata.pkl')
            if not os.path.exists(meta_path):
                return False
            self.metadata = joblib.load(meta_path)
            
            self.model = FlightDelayModel(input_dim=self.metadata['input_dim'])
            model_path = os.path.join(self.model_dir, 'flight_delay_model.h5')
            if not os.path.exists(model_path):
                return False
            self.model.load(model_path)
            
            airports_path = os.path.join(self.data_dir, 'airports.csv')
            if os.path.exists(airports_path):
                df = pd.read_csv(airports_path)
                for _, row in df.iterrows():
                    self.airport_coords[row['IATA_CODE']] = {
                        'lat': float(row['LATITUDE']), 
                        'lon': float(row['LONGITUDE'])
                    }
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error loading inference model: {e}")
            return False

    def predict(self, data):
        if not self.loaded:
            if not self.load():
                return {"error": "Model not trained yet."}

        try:
            origin = str(data.get('ORIGIN_AIRPORT', '')).strip().upper()
            dest = str(data.get('DESTINATION_AIRPORT', '')).strip().upper()

            if origin not in self.airport_coords or dest not in self.airport_coords:
                distance = 1000.0
            else:
                coord1 = self.airport_coords[origin]
                coord2 = self.airport_coords[dest]
                distance = self.calculate_distance(coord1['lat'], coord1['lon'], coord2['lat'], coord2['lon'])

            hour = int(data.get('HOUR', 12))
            dep_hour_sin = np.sin(2 * np.pi * hour / 24)
            dep_hour_cos = np.cos(2 * np.pi * hour / 24)

            day = int(data.get('DAY_OF_WEEK', 1))
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)

            month = int(data.get('MONTH', 1))
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            is_weekend = 1 if day in [6, 7] else 0

            airline = str(data.get('AIRLINE', '')).strip().upper()
            airline_encoded = self.metadata['airline_mapping'].get(airline, 0)

            features = np.array([[
                dep_hour_sin, dep_hour_cos,
                day_sin, day_cos,
                month_sin, month_cos,
                is_weekend,
                airline_encoded,
                distance
            ]])

            preds = self.model.predict(features)
            
            prob_delay = float(preds[0][0])
            raw_delay_pred = float(preds[1][0])
            
            if raw_delay_pred < 1:
                est_delay = prob_delay * 50
            else:
                est_delay = raw_delay_pred

            if prob_delay > 0.25:
                risk_level = "High"
            elif prob_delay > 0.15:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                "probability_percent": round(prob_delay * 100, 1),
                "estimated_delay_minutes": round(est_delay, 1),
                "risk_level": risk_level,
                "airline_used": airline,
                "calculated_distance": round(distance, 1),
                "route": f"{origin} ➝ {dest}"
            }
        except Exception as e:
            return {"error": f"Prediction logic error: {str(e)}"}