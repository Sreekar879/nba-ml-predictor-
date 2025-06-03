import streamlit as st
import pandas as pd
import joblib
from nba_api.stats.static import players, teams
from player_stats import get_player_features
from multi_stat_predict import predict_stat, AVAILABLE_STATS
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="NBA Player Stats Predictor",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .confidence-high {
        color: #4CAF50;
    }
    .confidence-medium {
        color: #FFC107;
    }
    .confidence-low {
        color: #F44336;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_active_player_names():
    return sorted([p['full_name'] for p in players.get_players() if p['is_active']])

def get_team_logo(team_id):
    """Get team logo URL"""
    team = teams.find_team_by_id(team_id)
    if team:
        return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    return None

def create_trend_chart(player_stats):
    """Create a trend chart for player's recent performance"""
    # Create sample data (replace with actual data from player_stats)
    dates = pd.date_range(end=datetime.now(), periods=10)
    points = [20, 22, 18, 25, 23, 19, 21, 24, 20, 22]  # Replace with actual data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=points,
        mode='lines+markers',
        name='Points',
        line=dict(color='#1e88e5', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Recent Performance Trend',
        xaxis_title='Date',
        yaxis_title='Points',
        template='plotly_white',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_stats_chart(player_stats, stat_name, title):
    """Create a bar chart for a specific stat"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['3-Game Avg'],
        y=[player_stats[f'{stat_name}_3game_avg']],
        name=title,
        marker_color='#1e88e5'
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white'
    )
    
    return fig

def create_recent_games_chart(recent_games):
    """Create a line chart for recent game performance"""
    fig = go.Figure()
    
    # Add points line
    fig.add_trace(go.Scatter(
        x=recent_games['GAME_DATE'],
        y=recent_games['PTS'],
        mode='lines+markers',
        name='Points',
        line=dict(color='#1e88e5', width=3),
        marker=dict(size=8)
    ))
    
    # Add assists line
    fig.add_trace(go.Scatter(
        x=recent_games['GAME_DATE'],
        y=recent_games['AST'],
        mode='lines+markers',
        name='Assists',
        line=dict(color='#4caf50', width=3),
        marker=dict(size=8)
    ))
    
    # Add rebounds line
    fig.add_trace(go.Scatter(
        x=recent_games['GAME_DATE'],
        y=recent_games['REB'],
        mode='lines+markers',
        name='Rebounds',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Recent Game Performance',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    st.title("🏀 NBA Player Stats Predictor")
    st.markdown("Predict player performance for the 2024-25 NBA season")

    # Sidebar for player selection and stat line
    with st.sidebar:
        st.header("Player Selection")
        player_names = get_active_player_names()
        player_name = st.selectbox(
            "Select player",
            player_names,
            index=player_names.index("LeBron James") if "LeBron James" in player_names else 0
        )
        
        st.header("Stat Prediction")
        stat_options = [(v['name'], k) for k, v in AVAILABLE_STATS.items()]
        stat_label, stat_type = st.selectbox(
            "Select Stat to Predict",
            stat_options,
            format_func=lambda x: x[0],
            index=0
        )
        stat_config = AVAILABLE_STATS[stat_type]
        stat_line = st.number_input(
            f"{stat_config['name']} Line",
            value=stat_config['default_line'],
            step=0.5,
            format="%.1f"
        )
        predict_button = st.button("Predict", type="primary")

    # Main content
    try:
        # Get player stats and recent games
        player_stats, recent_games = get_player_features(player_name)
        
        # Layout: two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Performance")
            if recent_games is not None and not recent_games.empty:
                fig = go.Figure()
                for k, v in AVAILABLE_STATS.items():
                    stat_name = v['name']
                    if stat_name.lower() in recent_games.columns:
                        fig.add_trace(go.Scatter(
                            x=recent_games['date'],
                            y=recent_games[stat_name.lower()],
                            name=stat_name,
                            mode='lines+markers'
                        ))
                fig.update_layout(
                    title="Recent Game Performance",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    recent_games.style.format({
                        'points': '{:.1f}',
                        'assists': '{:.1f}',
                        'rebounds': '{:.1f}',
                        'blocks': '{:.1f}',
                        'steals': '{:.1f}',
                        '3-pointers': '{:.1f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No recent games data available")
        
        with col2:
            st.subheader("Prediction")
            if predict_button:
                try:
                    pred, prob = predict_stat(player_stats, stat_type, stat_line)
                    if prob >= 0.7:
                        conf_class = "confidence-high"
                    elif prob >= 0.6:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3>{stat_config['name']} ({stat_line})</h3>
                            <p class="stat-value">
                                {'OVER' if pred == 1 else 'UNDER'}
                                <span class="{conf_class}">({prob:.1%} confidence)</span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.info("Select a stat, enter a line, and click Predict.")
        
        # Display season averages
        st.subheader("Season Averages")
        avg_cols = st.columns(len(AVAILABLE_STATS))
        for i, (k, v) in enumerate(AVAILABLE_STATS.items()):
            with avg_cols[i]:
                avg_key = f"{k}_3game_avg"
                if avg_key in player_stats:
                    st.metric(
                        v['name'],
                        f"{player_stats[avg_key]:.1f}",
                        f"Last 3 games"
                    )

    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Data from NBA API | Last updated: {}</p>
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 