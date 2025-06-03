from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
import pandas as pd
from typing import Dict, Optional
import time

# Build a static abbreviation → full team name lookup
team_lookup = {team['abbreviation']: team['full_name'] for team in teams.get_teams()}

def get_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    """Helper function to get player game logs for a specific season"""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"[DEBUG] Error fetching {season} logs: {str(e)}")
        return pd.DataFrame()

def get_opponent_context(player_name: str) -> Optional[Dict]:
    """
    Get opponent context for a player's most recent game.
    Returns opponent defensive stats and game context.
    """
    try:
        print(f"\n[DEBUG] Processing player: {player_name}")
        
        # Get player ID
        player_list = players.find_players_by_full_name(player_name)
        if not player_list:
            raise ValueError(f"Player '{player_name}' not found")
        
        player_id = player_list[0]['id']
        print(f"[DEBUG] Found player ID: {player_id}")

        # Try current season first, then fall back to previous season
        seasons_to_try = ['2024-25', '2023-24']
        df = pd.DataFrame()
        used_season = None
        
        for season in seasons_to_try:
            print(f"[DEBUG] Trying season {season}...")
            df = get_player_game_logs(player_id, season)
            if not df.empty:
                used_season = season
                break
        
        if df.empty:
            print(f"[DEBUG] No game logs found for {player_name} in any recent season")
            print("[DEBUG] Checking if player exists in active players list...")
            active_players = [p for p in players.get_players() if p['is_active']]
            is_active = any(p['full_name'] == player_name for p in active_players)
            print(f"[DEBUG] Player active status: {is_active}")
            raise ValueError(f"No games found for {player_name} in recent seasons")

        print(f"[DEBUG] Found {len(df)} games from {used_season} season")
        print(f"[DEBUG] Most recent game date: {df['GAME_DATE'].iloc[0]}")
        
        most_recent_game = df.iloc[0]
        matchup = most_recent_game['MATCHUP']
        print(f"[DEBUG] Most recent matchup: {matchup}")

        # Parse opponent team abbreviation
        if '@' in matchup:
            opponent_abbr = matchup.split('@')[1].strip()
            is_home_game = False
        elif 'vs.' in matchup:
            opponent_abbr = matchup.split('vs.')[1].strip()
            is_home_game = True
        else:
            raise ValueError(f"Unexpected matchup format: {matchup}")

        print(f"[DEBUG] Parsed opponent: {opponent_abbr}, Home game: {is_home_game}")

        # Convert abbreviation to full name
        opponent_name = team_lookup.get(opponent_abbr)
        if not opponent_name:
            print(f"[DEBUG] Available team abbreviations: {list(team_lookup.keys())}")
            raise ValueError(f"Could not resolve team abbreviation '{opponent_abbr}'")

        print(f"[DEBUG] Found opponent full name: {opponent_name}")

        # Sleep to avoid rate limiting
        time.sleep(1)

        # Get team stats for the same season as the game logs
        print(f"[DEBUG] Fetching team stats for {used_season}...")
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=used_season,
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        
        print(f"[DEBUG] Found stats for {len(team_stats)} teams")
        print(f"[DEBUG] Available teams: {team_stats['TEAM_NAME'].tolist()}")

        # Match based on TEAM_NAME
        opponent_stats = team_stats[team_stats['TEAM_NAME'] == opponent_name]
        if opponent_stats.empty:
            print(f"[DEBUG] Could not find stats for team: {opponent_name}")
            print("[DEBUG] Available team names:", team_stats['TEAM_NAME'].tolist())
            raise ValueError(f"No stats found for opponent team '{opponent_name}'")

        print(f"[DEBUG] Found opponent stats with columns: {opponent_stats.columns.tolist()}")

        required_columns = ['DEF_RATING', 'PACE']
        for col in required_columns:
            if col not in opponent_stats.columns:
                print(f"[ERROR] Column '{col}' missing from API response")
                print("[DEBUG] Available columns:", opponent_stats.columns.tolist())
                raise ValueError(f"Opponent stats missing column: {col}")

        result = {
            'opponent_team': opponent_abbr,
            'def_rating': float(opponent_stats['DEF_RATING'].iloc[0]),
            'pace': float(opponent_stats['PACE'].iloc[0]),
            'is_home_game': is_home_game,
            'season_used': used_season  # Add this to track which season's data we used
        }
        
        print(f"[DEBUG] Successfully created context: {result}")
        return result

    except Exception as e:
        print(f"[ERROR] Exception in get_opponent_context: {str(e)}")
        raise ValueError(f"Error getting opponent context: {str(e)}")

if __name__ == "__main__":
    try:
        print("\nTesting with Saddiq Bey...")
        context = get_opponent_context("Saddiq Bey")
        print("\nOpponent Context:")
        for key, value in context.items():
            print(f"{key}: {value}")
    except ValueError as e:
        print(f"Error: {e}")
