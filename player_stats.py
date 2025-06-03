from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
from typing import Dict, Optional, List, Tuple

def get_player_features(player_name: str, num_games: int = 5) -> Tuple[Dict, pd.DataFrame]:
    """
    Get feature values and recent game logs for a specific NBA player.
    
    Args:
        player_name: Full name of the NBA player (e.g., 'LeBron James')
        num_games: Number of recent games to consider (default: 5)
        
    Returns:
        Tuple containing:
        1. Dictionary of feature values:
            - points_3game_avg: Average points over last 3 games
            - points_5game_avg: Average points over last 5 games
            - fga_3game_avg: Average field goal attempts over last 3 games
            - ast_3game_avg: Average assists over last 3 games
            - reb_3game_avg: Average rebounds over last 3 games
            - blk_3game_avg: Average blocks over last 3 games
            - stl_3game_avg: Average steals over last 3 games
            - fg3m_3game_avg: Average 3-pointers made over last 3 games
            - fgm_3game_avg: Average field goals made over last 3 games
            - game_number: Total games played this season
        2. DataFrame containing recent game logs
        
    Raises:
        ValueError: If player not found or insufficient data
    """
    # Find player ID from name
    player_list = players.find_players_by_full_name(player_name)
    
    if not player_list:
        raise ValueError(f"Player '{player_name}' not found in NBA database")
    
    player_id = player_list[0]['id']
    
    # Get player's game logs from the correct season
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
    except Exception as e:
        raise ValueError(f"Error fetching game logs for {player_name}: {str(e)}")
    
    if df.empty:
        raise ValueError(f"No game logs found for {player_name} in the 2024-25 season")
    
    # Sort by date (most recent first)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=False)
    
    # Check if we have enough games
    if len(df) < num_games:
        raise ValueError(f"Not enough games found for {player_name}. Found {len(df)}, need at least {num_games}")
    
    # Calculate features
    most_recent = df.iloc[0]
    features = {
        # Points
        'PTS_3G_AVG': df['PTS'].head(3).mean(),
        'PTS_5G_AVG': df['PTS'].head(5).mean(),
        'FGA_3G_AVG': df['FGA'].head(3).mean(),
        'FGM_3G_AVG': df['FGM'].head(3).mean(),
        'FG_PCT_3G_AVG': df['FG_PCT'].head(3).mean(),
        'MIN_3G_AVG': df['MIN'].head(3).mean(),
        'PLUS_MINUS_3G_AVG': df['PLUS_MINUS'].head(3).mean(),
        'HOME_GAME_3G_AVG': (df['MATCHUP'].head(3).str.contains('vs.').mean()),
        # Assists
        'AST_3G_AVG': df['AST'].head(3).mean(),
        'AST_5G_AVG': df['AST'].head(5).mean(),
        # Rebounds
        'REB_3G_AVG': df['REB'].head(3).mean(),
        'REB_5G_AVG': df['REB'].head(5).mean(),
        'OREB_3G_AVG': df['OREB'].head(3).mean(),
        'DREB_3G_AVG': df['DREB'].head(3).mean(),
        # Blocks
        'BLK_3G_AVG': df['BLK'].head(3).mean(),
        'BLK_5G_AVG': df['BLK'].head(5).mean(),
        # Steals
        'STL_3G_AVG': df['STL'].head(3).mean(),
        'STL_5G_AVG': df['STL'].head(5).mean(),
        # Threes
        'FG3M_3G_AVG': df['FG3M'].head(3).mean(),
        'FG3M_5G_AVG': df['FG3M'].head(5).mean(),
        'FG3_PCT_3G_AVG': df['FG3_PCT'].head(3).mean(),
        # Other
        'AST_3G_AVG': df['AST'].head(3).mean(),
        'AST_5G_AVG': df['AST'].head(5).mean(),
        'FGA_3G_AVG': df['FGA'].head(3).mean(),
        'MIN_3G_AVG': df['MIN'].head(3).mean(),
        'PLUS_MINUS_3G_AVG': df['PLUS_MINUS'].head(3).mean(),
        'HOME_GAME_3G_AVG': (df['MATCHUP'].head(3).str.contains('vs.').mean()),
        'game_number': len(df)
    }
    
    # Get recent game logs (last 5 games)
    recent_games = df.head(5)[[
        'GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'REB', 'BLK', 'STL', 
        'FGM', 'FGA', 'FG3M', 'FG3A', 'FG_PCT', 'FG3_PCT', 'PLUS_MINUS'
    ]].copy()
    
    # Format the date
    recent_games['GAME_DATE'] = recent_games['GAME_DATE'].dt.strftime('%Y-%m-%d')
    
    return features, recent_games

if __name__ == "__main__":
    # Example usage
    try:
        player_name = "LeBron James"
        features, recent_games = get_player_features(player_name)
        print(f"\nFeatures for {player_name}:")
        for key, value in features.items():
            print(f"{key}: {value:.2f}")
        print("\nRecent Games:")
        print(recent_games)
    except ValueError as e:
        print(f"Error: {e}")
