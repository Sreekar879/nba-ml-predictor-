# NBA Points Over/Under Predictor

This project predicts whether an NBA player will score over or under a given points line in a game using machine learning.

## Features
- Collects player game logs from the 2023 NBA season
- Computes rolling averages for points and field goal attempts
- Trains a Random Forest Classifier to predict Over/Under outcomes
- Supports custom points lines (e.g., 20.5 points)

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the data collection script:
```bash
python collect_data.py
```

3. Train the model:
```bash
python train_model.py
```

## Project Structure
- `collect_data.py`: Collects and processes NBA player data
- `feature_engineering.py`: Creates features for the model
- `train_model.py`: Trains and evaluates the prediction model
- `requirements.txt`: Project dependencies

## Future Enhancements
- Add opponent defense statistics
- Incorporate game pace metrics
- Develop a web interface using Flask or Streamlit 