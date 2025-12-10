from flask import Flask, render_template, send_file, jsonify, request
import pandas as pd
import os
import time
from src.inference import FlightPredictor

app = Flask(__name__)

predictor = FlightPredictor()

class AirportDashboard:
    def __init__(self):
        self.output_dir = 'output/'
        self.cache = {
            'scorecard': None,
            'predictions': None,
            'last_updated': 0,
            'cache_duration': 300
        }
        
    def load_scorecard(self, force_reload=False):
        current_time = time.time()
        if (not force_reload and 
            self.cache['scorecard'] is not None and 
            (current_time - self.cache['last_updated']) < self.cache['cache_duration']):
            return self.cache['scorecard']
        
        try:
            file_path = f'{self.output_dir}airport_scorecard.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.cache['scorecard'] = df
                self.cache['last_updated'] = current_time
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            if self.cache['scorecard'] is not None:
                return self.cache['scorecard']
            return pd.DataFrame()
    
    def load_predictions(self):
        try:
            file_path = f'{self.output_dir}predictions.csv'
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            return pd.DataFrame()
        except:
            return pd.DataFrame()

dashboard = AirportDashboard()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/api/scorecard')
def get_scorecard():
    df = dashboard.load_scorecard()
    if df.empty:
        return jsonify({"error": "Scorecard not found. Run the prediction model first."})
    
    summary = {
        'best_airport': df.iloc[0]['Airport'],
        'best_score': float(df.iloc[0]['Score']),
        'worst_airport': df.iloc[-1]['Airport'],
        'worst_score': float(df.iloc[-1]['Score']),
        'avg_delay': float(df['Avg_Delay'].mean()),
        'avg_score': float(df['Score'].mean()),
        'total_airports': len(df)
    }
    
    return jsonify({
        'airports': df.to_dict('records'),
        'total_airports': len(df),
        'summary': summary
    })

@app.route('/api/predict_flight', methods=['POST'])
def predict_flight():
    try:
        data = request.json
        result = predictor.predict(data)
        if "error" in result:
            return jsonify(result), 500
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/refresh')
def refresh_data():
    df = dashboard.load_scorecard(force_reload=True)
    if df.empty:
        return jsonify({"error": "Failed to reload data"})
    
    return jsonify({
        "success": True,
        "message": f"Data refreshed. Loaded {len(df)} airports."
    })

@app.route('/api/predictions')
def get_predictions():
    df = dashboard.load_predictions()
    if df.empty:
        return jsonify({"error": "Predictions not found."})
    
    return jsonify({
        'total_predictions': len(df),
        'sample': df.head(100).to_dict('records')
    })

@app.route('/api/status')
def get_status():
    file_exists = os.path.exists(f'{dashboard.output_dir}airport_scorecard.csv')
    return jsonify({
        'data_available': file_exists,
        'last_updated': dashboard.cache['last_updated']
    })

@app.route('/download/scorecard')
def download_scorecard():
    return send_file(f'{dashboard.output_dir}airport_scorecard.csv', as_attachment=True)

@app.route('/visualization')
def show_visualization():
    return send_file(f'{dashboard.output_dir}scorecard_visualization.png')

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True) 
    
    print("=" * 60)
    print("Airport Scorecard Dashboard")
    print("=" * 60)
    print("Access the dashboard at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)