from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time
import json

def get_active_players():
    """Get list of active NBA players"""
    all_players = players.get_players()
    active_players = [player for player in all_players if player['is_active']]
    return active_players

def get_player_game_logs(player_id, season='2023-24'):
    """Get game logs for a specific player"""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching data for player {player_id}: {e}")
        return pd.DataFrame()

def collect_data(num_players=50):
    """Collect game logs for specified number of players"""
    active_players = get_active_players()
    all_game_logs = []
    
    # Get game logs for first num_players active players
    for i, player in enumerate(active_players[:num_players]):
        print(f"Fetching data for {player['full_name']} ({i+1}/{num_players})")
        game_logs = get_player_game_logs(player['id'])
        
        if not game_logs.empty:
            game_logs['player_id'] = player['id']
            game_logs['player_name'] = player['full_name']
            all_game_logs.append(game_logs)
        
        # Add delay to avoid rate limiting
        time.sleep(0.6)
    
    # Combine all game logs
    if all_game_logs:
        combined_logs = pd.concat(all_game_logs, ignore_index=True)
        combined_logs.to_csv('nba_game_logs.csv', index=False)
        print(f"Successfully collected data for {len(all_game_logs)} players")
        return combined_logs
    else:
        print("No data was collected")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Starting data collection...")
    data = collect_data()
    print("Data collection complete!") 