import os
import time

def run_pipeline():
    """
    Run the full NBA player prop bet predictor pipeline:
    1. Collect data
    2. Engineer features
    3. Train and evaluate model
    """
    print("Starting NBA Player Prop Bet Predictor Pipeline...")
    print("-" * 50)
    
    # Step 1: Collect data
    print("\nStep 1: Collecting data...")
    start_time = time.time()
    os.system('python collect_data.py')
    print(f"Data collection completed in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Engineer features
    print("\nStep 2: Engineering features...")
    start_time = time.time()
    os.system('python feature_engineering.py')
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    start_time = time.time()
    os.system('python train_model.py')
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    print("\nPipeline completed successfully!")
    print("Model saved as 'nba_points_predictor.pkl'")
    print("Visualizations saved as 'confusion_matrix.png' and 'feature_importance.png'")

if __name__ == "__main__":
    run_pipeline() 