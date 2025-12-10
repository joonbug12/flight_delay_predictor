import warnings
warnings.filterwarnings('ignore')
import time
import os
import joblib  
import pandas as pd

from src import dataloader, preprocessing, features, evaluation, scorecard
from src.model import FlightDelayModel

def main():
    total_start = time.time()
    print("=" * 60)
    print("FLIGHT DELAY PREDICTION & AIRPORT SCORECARD SYSTEM")
    print("=" * 60)
    print("Using Neural Network (Multi-Task Learning)")
    print("=" * 60)

    try:
        start_time = time.time()
        flights, airlines, airports = dataloader.load_data(nrows=None) 
        if flights is None: return
        print(f"Time: {time.time() - start_time:.1f} seconds")

        start_time = time.time()
        flights = preprocessing.preprocess_data(flights)
        print(f"Time: {time.time() - start_time:.1f} seconds")

        start_time = time.time()
        X, y_cls, y_reg, airports_data = features.engineer_features(flights)
        
        print("Saving metadata for inference...")
        airline_mapping = {airline: i for i, airline in enumerate(flights['AIRLINE'].unique())}
        metadata = {
            'airline_mapping': airline_mapping,
            'input_dim': X.shape[1]
        }
        os.makedirs('output', exist_ok=True)
        joblib.dump(metadata, 'output/metadata.pkl')
        print("Metadata saved to output/metadata.pkl")
        
        print(f"Time: {time.time() - start_time:.1f} seconds")

        print(f"\n[4/6] Training Neural Network...")
        train_start = time.time()
        
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_cls_train, y_cls_val = y_cls[:split_idx], y_cls[split_idx:]
        y_reg_train, y_reg_val = y_reg[:split_idx], y_reg[split_idx:]
        airports_train, airports_val = airports_data[:split_idx], airports_data[split_idx:]
        
        print(f"Training on {len(X_train):,} samples")
        print(f"Validating on {len(X_val):,} samples")

        model = FlightDelayModel(input_dim=X.shape[1])
        model.train(X_train, y_cls_train, y_reg_train, epochs=20, batch_size=256)
        print(f"Time: {time.time() - train_start:.1f} seconds")

        
        start_time = time.time()
        cls_preds, reg_preds, auc, mae = evaluation.evaluate_model(model, X_val, y_cls_val, y_reg_val)
        print(f"Time: {time.time() - start_time:.1f} seconds")

        start_time = time.time()
        
        model.save('output/flight_delay_model.h5')
        print(f"Saved model to output/flight_delay_model.h5")

        scorecard_df = scorecard.create_scorecard_dataframe(
            airports_val, y_cls_val, y_reg_val, cls_preds, reg_preds
        )
        scorecard_df.to_csv('output/airport_scorecard.csv', index=False)
        print(f"Saved scorecard for {len(scorecard_df)} airports")

        scorecard.save_visualizations(scorecard_df, 'output')

        predictions_df = pd.DataFrame({
            'Airport': airports_val,
            'True_Significant_Delay': y_cls_val,
            'Pred_Significant_Delay_Prob': cls_preds,
            'True_Total_Delay': y_reg_val,
            'Pred_Total_Delay': reg_preds
        })
        predictions_df.to_csv('output/predictions.csv', index=False)
        print(f"Saved predictions for {len(predictions_df)} flights")

        scorecard.save_summary(
            scorecard_df, auc, mae, len(flights), X.shape[0], 'output'
        )
        print(f"Time: {time.time() - start_time:.1f} seconds")

        total_time = time.time() - total_start
        print(f"\n{'=' * 60}")
        print(f"NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nTop 3 Airports:")
        for i, row in scorecard_df.head(3).iterrows():
            print(f" {i+1}. {row['Airport']} - Score: {row['Score']}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()