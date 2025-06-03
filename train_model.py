import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train):
    """
    Train a calibrated Random Forest Classifier
    Args:
        X_train: Training features
        y_train: Training labels
    Returns:
        Trained and calibrated model
    """
    # Base model
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    # Calibrate the model using Platt scaling
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='sigmoid',  # Use Platt scaling
        cv=5,  # 5-fold cross-validation
        n_jobs=-1  # Use all available cores
    )
    
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'brier_score': np.mean((y_prob - y_test) ** 2)  # Add Brier score for probability calibration
    }
    
    print("\nModel Performance:")
    for metric, score in metrics.items():
        print(f"{metric.capitalize()}: {score:.3f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Under', 'Over'],
                yticklabels=['Under', 'Over'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.base_estimator.feature_importances_  # Get importance from base model
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def predict_player_game(model, player_stats, points_line=20.5):
    """
    Predict whether a player will go over or under the points line
    Args:
        model: Trained model
        player_stats: Dictionary containing player's recent stats
        points_line: Points line to predict over/under
    Returns:
        Prediction (1 for over, 0 for under) and probability
    """
    # Create feature vector
    features = np.array([
        player_stats['points_3game_avg'],
        player_stats['points_5game_avg'],
        player_stats['fga_3game_avg'],
        player_stats['game_number'],
        player_stats.get('def_rating', 110.0),  # Default values if not provided
        player_stats.get('opp_pts_allowed', 110.0),
        player_stats.get('pace', 100.0),
        player_stats.get('is_home_game', 0.5)
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    # Load features
    features_df = pd.read_csv('nba_features.csv')
    
    # Split data
    from feature_engineering import prepare_train_test_data
    X_train, X_test, y_train, y_test = prepare_train_test_data(features_df)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    joblib.dump(model, 'nba_points_predictor.pkl')
    print("\nModel saved as 'nba_points_predictor.pkl'")
    
    # Example prediction
    example_player = {
        'points_3game_avg': 22.3,
        'points_5game_avg': 21.8,
        'fga_3game_avg': 18.5,
        'game_number': 45,
        'def_rating': 110.5,
        'opp_pts_allowed': 112.3,
        'pace': 98.7,
        'is_home_game': 1
    }
    
    pred, prob = predict_player_game(model, example_player)
    print(f"\nExample Prediction:")
    print(f"Over/Under 20.5 points: {'OVER' if pred == 1 else 'UNDER'}")
    print(f"Probability: {prob:.2%}") 