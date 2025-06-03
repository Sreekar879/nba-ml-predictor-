import argparse
import time
from collect_data import collect_data
from feature_engineering import create_features, prepare_train_test_data
from train_model import train_model, evaluate_model
from predict import predict_points_line  # ✅ Fixed import
from player_stats import get_player_features
import pandas as pd
import joblib

def run_pipeline():
    """
    Run the full data pipeline to get fresh data and train a new model
    """
    print("Starting real-time prediction pipeline...")
    print("-" * 50)
    
    # Step 1: Collect data
    print("\nStep 1: Collecting latest data...")
    start_time = time.time()
    game_logs = collect_data()
    print(f"Data collection completed in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Engineer features
    print("\nStep 2: Engineering features...")
    start_time = time.time()
    features_df = create_features(game_logs)
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    start_time = time.time()
    X_train, X_test, y_train, y_test = prepare_train_test_data(features_df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the newly trained model
    joblib.dump(model, 'nba_points_predictor.pkl')
    print("\nPipeline completed successfully!")
    print("Model saved as 'nba_points_predictor.pkl'")
    
    return model

def make_prediction(player_name: str, points_line: float = 20.5):
    """
    Make a real-time prediction for a player
    """
    try:
        # Get latest player features
        print(f"\nFetching latest stats for {player_name}...")
        player_stats = get_player_features(player_name)
        
        # Load the newly trained model
        model = joblib.load('nba_points_predictor.pkl')
        
        # Make prediction
        pred, prob = predict_points_line(player_stats, points_line)
        
        # Print results
        print(f"\nReal-time Prediction for {player_name} ({points_line} points line):")
        print(f"Latest stats:")
        for key, value in player_stats.items():
            print(f"  {key}: {value:.2f}")
        print(f"\nPrediction: {'OVER' if pred == 1 else 'UNDER'}")
        print(f"Confidence: {prob:.2%}")
        
    except ValueError as e:
        print(f"Error: {e}")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Make real-time NBA player points over/under predictions')
    parser.add_argument('--player_name', type=str, required=True,
                      help='Full name of the NBA player (e.g., "LeBron James")')
    parser.add_argument('--points_line', type=float, default=20.5,
                      help='Points line to predict over/under (default: 20.5)')
    parser.add_argument('--skip_pipeline', action='store_true',
                      help='Skip the data collection and model training pipeline')
    
    args = parser.parse_args()
    
    if not args.skip_pipeline:
        # Run the full pipeline to get fresh data and train a new model
        run_pipeline()
    
    # Make prediction for the specified player
    make_prediction(args.player_name, args.points_line)

if __name__ == "__main__":
    main()
