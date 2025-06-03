import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from nba_api.stats.static import players
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

# Define the stats we want to predict
STATS_CONFIG = {
    'points': {
        'name': 'Points',
        'target_col': 'PTS',
        'features': [
            'PTS', 'FGA', 'FGM', 'FG_PCT',
            'MIN', 'PLUS_MINUS', 'HOME_GAME'
        ],
        'model_file': 'nba_points_predictor.pkl'
    },
    'assists': {
        'name': 'Assists',
        'target_col': 'AST',
        'features': [
            'AST', 'PTS', 'MIN', 'PLUS_MINUS',
            'HOME_GAME', 'TOV'
        ],
        'model_file': 'nba_assists_predictor.pkl'
    },
    'rebounds': {
        'name': 'Rebounds',
        'target_col': 'REB',
        'features': [
            'REB', 'OREB', 'DREB', 'MIN',
            'PLUS_MINUS', 'HOME_GAME'
        ],
        'model_file': 'nba_rebounds_predictor.pkl'
    },
    'blocks': {
        'name': 'Blocks',
        'target_col': 'BLK',
        'features': [
            'BLK', 'REB', 'MIN', 'PLUS_MINUS',
            'HOME_GAME'
        ],
        'model_file': 'nba_blocks_predictor.pkl'
    },
    'steals': {
        'name': 'Steals',
        'target_col': 'STL',
        'features': [
            'STL', 'AST', 'MIN', 'PLUS_MINUS',
            'HOME_GAME', 'TOV'
        ],
        'model_file': 'nba_steals_predictor.pkl'
    },
    'threes': {
        'name': '3-Pointers',
        'target_col': 'FG3M',
        'features': [
            'FG3M', 'FG3A', 'FG3_PCT', 'MIN',
            'PLUS_MINUS', 'HOME_GAME'
        ],
        'model_file': 'nba_threes_predictor.pkl'
    }
}

def get_historical_games(season: str = '2023-24', min_games: int = 20) -> pd.DataFrame:
    """
    Get historical game data for all active players
    
    Args:
        season: NBA season to get data for
        min_games: Minimum number of games a player must have played
        
    Returns:
        DataFrame containing game logs for all active players
    """
    logging.info(f"Fetching historical games for {season}")
    
    # Get all active players
    active_players = [p for p in players.get_players() if p['is_active']]
    all_games = []
    
    for player in tqdm(active_players, desc="Fetching player game logs"):
        try:
            # Get game logs for the player
            gamelog = playergamelog.PlayerGameLog(
                player_id=player['id'],
                season=season
            ).get_data_frames()[0]
            
            if len(gamelog) >= min_games:
                gamelog['PLAYER_NAME'] = player['full_name']
                all_games.append(gamelog)
                
        except Exception as e:
            logging.warning(f"Error fetching games for {player['full_name']}: {str(e)}")
            continue
    
    if not all_games:
        raise ValueError("No game data found")
    
    # Combine all game logs
    games_df = pd.concat(all_games, ignore_index=True)
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df.sort_values(['PLAYER_NAME', 'GAME_DATE'])
    
    logging.info(f"Found {len(games_df)} games for {len(all_games)} players")
    return games_df

def calculate_rolling_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling averages and other features for each player
    
    Args:
        games_df: DataFrame containing game logs
        
    Returns:
        DataFrame with additional features
    """
    logging.info("Calculating rolling statistics")
    
    # Group by player and calculate rolling stats
    features_df = games_df.copy()
    
    # Add home game indicator
    features_df['HOME_GAME'] = (features_df['MATCHUP'].str.contains('vs')).astype(int)
    
    # Calculate days between games
    features_df['DAYS_REST'] = features_df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days
    
    # Calculate rolling averages for each stat
    for stat in ['PTS', 'AST', 'REB', 'BLK', 'STL', 'FG3M', 'FGA', 'FGM', 'FG3A', 'MIN']:
        # 3-game average
        features_df[f'{stat}_3G_AVG'] = features_df.groupby('PLAYER_NAME')[stat].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        # 5-game average
        features_df[f'{stat}_5G_AVG'] = features_df.groupby('PLAYER_NAME')[stat].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
    
    # Fill NaN values
    features_df = features_df.fillna(method='bfill')
    
    return features_df

def prepare_training_data(features_df: pd.DataFrame, stat_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for a specific stat type
    
    Args:
        features_df: DataFrame with all features
        stat_type: Type of stat to predict
        
    Returns:
        Tuple of (X, y) for training
    """
    config = STATS_CONFIG[stat_type]
    target_col = config['target_col']
    features = config['features']
    
    # Create target variable (1 if over median, 0 if under)
    median_value = features_df[target_col].median()
    y = (features_df[target_col] > median_value).astype(int)
    
    # Select features and add rolling averages
    X = features_df[features].copy()
    
    # Add rolling averages for the target stat
    rolling_cols = [f'{target_col}_3G_AVG', f'{target_col}_5G_AVG']
    for col in rolling_cols:
        if col in features_df.columns:
            X[col] = features_df[col]
    
    # Remove rows with NaN values
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y

def train_model(X: np.ndarray, y: np.ndarray, stat_type: str) -> Pipeline:
    """
    Train a model for a specific stat type
    
    Args:
        X: Feature matrix
        y: Target vector
        stat_type: Type of stat being predicted
        
    Returns:
        Trained model pipeline
    """
    logging.info(f"Training model for {STATS_CONFIG[stat_type]['name']}")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logging.info(f"Model performance for {STATS_CONFIG[stat_type]['name']}:")
    logging.info(f"Accuracy: {accuracy:.3f}")
    logging.info(f"AUC-ROC: {auc:.3f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    return best_model

def main():
    """Main function to train models for all stats"""
    try:
        # Get historical data
        games_df = get_historical_games()
        
        # Calculate features
        features_df = calculate_rolling_stats(games_df)
        
        # Train models for each stat type
        for stat_type in STATS_CONFIG:
            try:
                # Prepare training data
                X, y = prepare_training_data(features_df, stat_type)
                
                # Train model
                model = train_model(X, y, stat_type)
                
                # Save model
                model_file = STATS_CONFIG[stat_type]['model_file']
                joblib.dump(model, model_file)
                logging.info(f"Saved model to {model_file}")
                
            except Exception as e:
                logging.error(f"Error training model for {stat_type}: {str(e)}")
                continue
        
        logging.info("Model training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main training process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 