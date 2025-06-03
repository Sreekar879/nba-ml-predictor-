import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from player_stats import get_player_features

# Define available stats and their default lines
AVAILABLE_STATS = {
    'points': {
        'name': 'Points',
        'default_line': 20.5,
        'features': ['PTS_3G_AVG', 'PTS_5G_AVG', 'FGA_3G_AVG', 'FGM_3G_AVG', 'FG_PCT_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_points_predictor.pkl'
    },
    'assists': {
        'name': 'Assists',
        'default_line': 5.5,
        'features': ['AST_3G_AVG', 'AST_5G_AVG', 'PTS_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_assists_predictor.pkl'
    },
    'rebounds': {
        'name': 'Rebounds',
        'default_line': 7.5,
        'features': ['REB_3G_AVG', 'REB_5G_AVG', 'OREB_3G_AVG', 'DREB_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_rebounds_predictor.pkl'
    },
    'blocks': {
        'name': 'Blocks',
        'default_line': 1.5,
        'features': ['BLK_3G_AVG', 'BLK_5G_AVG', 'REB_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_blocks_predictor.pkl'
    },
    'steals': {
        'name': 'Steals',
        'default_line': 1.5,
        'features': ['STL_3G_AVG', 'STL_5G_AVG', 'AST_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_steals_predictor.pkl'
    },
    'threes': {
        'name': '3-Pointers',
        'default_line': 2.5,
        'features': ['FG3M_3G_AVG', 'FG3M_5G_AVG', 'FG3_PCT_3G_AVG', 'FGA_3G_AVG', 'MIN_3G_AVG', 'PLUS_MINUS_3G_AVG', 'HOME_GAME_3G_AVG'],
        'model_file': 'nba_threes_predictor.pkl'
    }
}

def predict_stat(player_stats: Dict, stat_type: str, stat_line: float) -> Tuple[int, float]:
    """
    Predict whether a player will go over or under a stat line
    
    Args:
        player_stats: Dictionary containing player's recent stats
        stat_type: Type of stat to predict (e.g., 'points', 'assists')
        stat_line: Line to predict over/under
        
    Returns:
        Tuple of (prediction, probability)
        - prediction: 1 for over, 0 for under
        - probability: Confidence in the prediction
    """
    if stat_type not in AVAILABLE_STATS:
        raise ValueError(f"Unsupported stat type: {stat_type}")
    
    stat_config = AVAILABLE_STATS[stat_type]
    
    try:
        # Load the appropriate model
        model = joblib.load(stat_config['model_file'])
    except FileNotFoundError:
        # If model doesn't exist, use a simple heuristic based on recent averages
        feature_name = f"{stat_type}_3game_avg"
        if feature_name in player_stats:
            avg = player_stats[feature_name]
            prob = 0.5 + (avg - stat_line) / (2 * stat_line)  # Simple linear scaling
            prob = max(0.1, min(0.9, prob))  # Clamp between 0.1 and 0.9
            return (1 if avg > stat_line else 0), prob
        else:
            raise ValueError(f"No model available for {stat_type} and no fallback calculation possible")
    
    # Build feature vector as DataFrame with correct columns
    features = pd.DataFrame([{feature: player_stats.get(feature, 0) for feature in stat_config['features']}])
    print(f"Prediction DataFrame columns: {features.columns.tolist()}")
    print(f"Prediction DataFrame shape: {features.shape}")
    print(f"Model expects: {model.named_steps['scaler'].n_features_in_}")
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def predict_all_stats(player_stats: Dict, stat_lines: Dict[str, float] = None) -> Dict[str, Tuple[int, float]]:
    """
    Predict over/under for all available stats
    
    Args:
        player_stats: Dictionary containing player's recent stats
        stat_lines: Dictionary of stat lines to predict (optional)
        
    Returns:
        Dictionary mapping stat types to (prediction, probability) tuples
    """
    if stat_lines is None:
        stat_lines = {stat: config['default_line'] for stat, config in AVAILABLE_STATS.items()}
    
    predictions = {}
    for stat_type in AVAILABLE_STATS:
        if stat_type in stat_lines:
            try:
                pred, prob = predict_stat(player_stats, stat_type, stat_lines[stat_type])
                predictions[stat_type] = (pred, prob)
            except Exception as e:
                print(f"Warning: Could not predict {stat_type}: {str(e)}")
    
    return predictions

if __name__ == "__main__":
    # Example usage
    try:
        player_name = "LeBron James"
        player_stats, _ = get_player_features(player_name)
        
        # Example stat lines
        stat_lines = {
            'points': 25.5,
            'assists': 7.5,
            'rebounds': 7.5,
            'blocks': 1.5,
            'steals': 1.5,
            'threes': 2.5
        }
        
        predictions = predict_all_stats(player_stats, stat_lines)
        
        print(f"\nPredictions for {player_name}:")
        for stat_type, (pred, prob) in predictions.items():
            stat_name = AVAILABLE_STATS[stat_type]['name']
            line = stat_lines[stat_type]
            print(f"{stat_name} ({line}): {'OVER' if pred == 1 else 'UNDER'} ({prob:.1%} confidence)")
            
    except ValueError as e:
        print(f"Error: {e}") 