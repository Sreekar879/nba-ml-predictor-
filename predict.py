import joblib
import numpy as np
import argparse
from player_stats import get_player_features

def predict_points_line(player_stats, points_line=20.5):
    """
    Predict whether a player will go over or under a points line
    Args:
        player_stats: Dictionary containing player's recent stats
        points_line: Points line to predict over/under
    Returns:
        Prediction (1 for over, 0 for under) and probability
    """
    # Load the trained model
    model = joblib.load('nba_points_predictor.pkl')
    
    # Create feature vector
    features = np.array([
        player_stats['points_3game_avg'],
        player_stats['points_5game_avg'],
        player_stats['fga_3game_avg'],
        player_stats['game_number']
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Predict NBA player points over/under')
    parser.add_argument('--player_name', type=str, required=True,
                      help='Full name of the NBA player (e.g., "LeBron James")')
    parser.add_argument('--points_line', type=float, default=20.5,
                      help='Points line to predict over/under (default: 20.5)')
    
    args = parser.parse_args()
    
    try:
        # Get player features
        player_stats = get_player_features(args.player_name)
        
        # Make prediction
        pred, prob = predict_points_line(player_stats, args.points_line)
        
        # Print results
        print(f"\nPrediction for {args.player_name} ({args.points_line} points line):")
        print(f"Recent stats:")
        for key, value in player_stats.items():
            print(f"  {key}: {value:.2f}")
        print(f"\nPrediction: {'OVER' if pred == 1 else 'UNDER'}")
        print(f"Confidence: {prob:.2%}")
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 