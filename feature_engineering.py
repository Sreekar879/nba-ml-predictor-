import pandas as pd
import numpy as np
import random
from opponent_context import get_opponent_context

def create_features(df, points_line=20.5):
    # Fix date parsing warning by specifying the format
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format="%b %d, %Y", errors='coerce')
    df = df.sort_values(['player_id', 'GAME_DATE'])

    features = []
    context_cache = {}  # Cache to avoid duplicate API calls

    for player_id in df['player_id'].unique():
        player_df = df[df['player_id'] == player_id].copy()
        player_name = player_df['player_name'].iloc[0]
        print(f"\n[PROCESSING] {player_name} ({player_id})")

        if len(player_df) < 5:
            print(f"[SKIP] {player_name} has only {len(player_df)} games")
            continue

        try:
            # Rolling averages for the player's statistics
            player_df['points_3game_avg'] = player_df['PTS'].rolling(window=3, min_periods=1).mean()
            player_df['points_5game_avg'] = player_df['PTS'].rolling(window=5, min_periods=1).mean()
            player_df['fga_3game_avg'] = player_df['FGA'].rolling(window=3, min_periods=1).mean()

            # Simulated dynamic betting line and target variable (whether the player went over or under the line)
            player_df['line'] = player_df['points_5game_avg'] + np.random.uniform(-1.5, 1.5, size=len(player_df))
            player_df['target'] = (player_df['PTS'] > player_df['line']).astype(int)
            player_df['game_number'] = range(1, len(player_df) + 1)

            # Initialize context columns
            player_df['def_rating'] = np.nan
            player_df['opp_pts_allowed'] = np.nan
            player_df['pace'] = np.nan
            player_df['is_home_game'] = np.nan

            # Fetch and assign opponent context for each row
            for idx, row in player_df.iterrows():
                try:
                    if row['player_name'] not in context_cache:
                        context_cache[row['player_name']] = get_opponent_context(row['player_name'])

                    context = context_cache[row['player_name']]
                    player_df.at[idx, 'def_rating'] = context['def_rating']
                    player_df.at[idx, 'opp_pts_allowed'] = context.get('opp_pts_allowed', 0)
                    player_df.at[idx, 'pace'] = context['pace']
                    player_df.at[idx, 'is_home_game'] = 1.0 if context['is_home_game'] else 0.0

                except Exception as e:
                    print(f"[WARN] Skipping context for {row['player_name']} index {idx}: {e}")
                    # Fill with means (or neutral values) in case of failure to retrieve context
                    player_df.at[idx, 'def_rating'] = player_df['def_rating'].mean()
                    player_df.at[idx, 'opp_pts_allowed'] = player_df['opp_pts_allowed'].mean()
                    player_df.at[idx, 'pace'] = player_df['pace'].mean()
                    player_df.at[idx, 'is_home_game'] = 0.5

            features.append(player_df)
        except Exception as e:
            print(f"[ERROR] Failed to process {player_name}: {e}")

    # Check if any features were collected
    if not features:
        raise ValueError("No valid player features generated. Check your data or context code.")

    # Combine all player features and remove rows with NaN values
    all_features = pd.concat(features, ignore_index=True).dropna()

    # Select the relevant feature columns
    feature_columns = [
        'points_3game_avg',
        'points_5game_avg',
        'fga_3game_avg',
        'game_number',
        'def_rating',
        'opp_pts_allowed',
        'pace',
        'is_home_game',
        'target'
    ]

    return all_features[feature_columns]

def prepare_train_test_data(features_df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    # Split the data into features and target
    X = features_df.drop('target', axis=1)
    y = features_df['target']
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    # Load the game log data
    df = pd.read_csv('nba_game_logs.csv')
    
    # Create features
    features_df = create_features(df)
    
    # Save features to a CSV file
    features_df.to_csv('nba_features.csv', index=False)
    
    print("\n✅ Features created and saved to nba_features.csv")
